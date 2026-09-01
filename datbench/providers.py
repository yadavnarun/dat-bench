"""Model registry and one chat client for every OpenAI-compatible provider.

CONTRACT.md section 2 is the authority for ModelSpec, ChatResult, load_models,
available_models and ChatClient.

Three additions the contract predates, because cli.py needs the non-model blocks
of models.yaml:

    load_config(path)         -> the whole parsed yaml
    load_scoring_config(path) -> the 'scoring' block, layered over module
                                 defaults and the yaml top-level 'defaults'
    load_run_config(path)     -> the 'run' block, same layering

ChatResult also carries a trailing `notes` field (default ""), used to record
that a call had to be re-sent with a parameter dropped -- otherwise a silent
provider quirk is indistinguishable from a clean call.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable

import yaml

log = logging.getLogger(__name__)

_MAX_ATTEMPTS = 3
_BACKOFF_BASE = 0.5

# The SDK refuses to construct without a non-empty api_key, but LM Studio (and
# anything else with api_key_env: null) ignores auth entirely.
_NO_AUTH_KEY = "lm-studio"

# Statuses that will never succeed on a retry: a bad model id, a bad key or a
# malformed request. Retrying these three times per run burns the whole
# factorial's wall-clock for nothing.
_NO_RETRY_STATUS = frozenset({400, 401, 403, 404, 405, 409, 413, 422})

_ERROR_MAX_CHARS = 180

# Applied to any scoring/run key the yaml omits, so cli.py can index these
# without a .get() dance on a trimmed config.
SCORING_DEFAULTS: dict[str, Any] = {
    "embedders": "auto",
    "embed_base_url": "http://localhost:1234/v1",
    "n_use": 7,
    "policies": ["strict", "lenient"],
    "min_words_lenient": 4,
    "rare_zipf_threshold": 2.5,
    "baseline_draws": 1000,
    "baseline_seed": 0,
}

RUN_DEFAULTS: dict[str, Any] = {
    "n": 10,
    "temperatures": [0.0, 0.7, 1.0],
    "prompts": ["verbatim", "terse", "cot", "maxcreative"],
}


@dataclass(frozen=True)
class ModelSpec:
    id: str
    model: str
    base_url: str
    api_key_env: str | None
    enabled: bool = True
    max_tokens: int = 512
    supports_temperature: bool = True
    max_concurrency: int = 1
    notes: str = ""


@dataclass(frozen=True)
class ChatResult:
    text: str
    model_reported: str
    finish_reason: str
    usage: dict[str, int]
    latency_ms: int
    error: str | None = None
    notes: str = ""


# --------------------------------------------------------------------------- env

_ENV_LOADED = False


def load_dotenv(path: Path | None = None, *, override: bool = False) -> dict[str, str]:
    """Minimal KEY=VALUE loader. Returns what it actually put into os.environ.

    A real env var always wins over the file, so `OPENAI_API_KEY=... python -m
    datbench` behaves the way anyone would expect. python-dotenv would be a
    dependency for thirty lines.
    """
    target = Path(path) if path is not None else _repo_root() / ".env"
    if not target.is_file():
        return {}

    applied: dict[str, str] = {}
    for raw in target.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not override and os.environ.get(key):
            continue
        os.environ[key] = value
        applied[key] = value
    return applied


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _ensure_env() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    applied = load_dotenv()
    if applied:
        log.debug("loaded %d var(s) from .env: %s", len(applied), ", ".join(sorted(applied)))


# ------------------------------------------------------------------------ config


def load_config(path: Path) -> dict:
    """The whole parsed models.yaml. An empty file is an empty config, not a crash."""
    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping, got {type(data).__name__}")
    return data


def _block(cfg: dict, name: str, builtin: dict[str, Any]) -> dict:
    # Three layers: module builtins, then the yaml 'defaults' block (so a shared
    # value like max_tokens can be set once), then the block itself.
    merged: dict[str, Any] = dict(builtin)
    defaults = cfg.get("defaults") or {}
    if isinstance(defaults, dict):
        merged.update(defaults)
    block = cfg.get(name) or {}
    if not isinstance(block, dict):
        raise ValueError(f"'{name}' block must be a mapping, got {type(block).__name__}")
    merged.update(block)
    return merged


def load_scoring_config(path: Path) -> dict:
    return _block(load_config(path), "scoring", SCORING_DEFAULTS)


def load_run_config(path: Path) -> dict:
    return _block(load_config(path), "run", RUN_DEFAULTS)


_SPEC_FIELDS = {f.name for f in fields(ModelSpec)}
_REQUIRED_FIELDS = ("id", "model", "base_url")


def _coerce(entry: dict) -> ModelSpec:
    kwargs: dict[str, Any] = {}
    for key, value in entry.items():
        if key in _SPEC_FIELDS:
            kwargs[key] = value
        else:
            # A typo in models.yaml would otherwise be silently inert.
            log.warning("models.yaml: ignoring unknown key %r on entry %r", key, entry.get("id"))

    api_key_env = kwargs.get("api_key_env")
    return ModelSpec(
        id=str(kwargs["id"]),
        model=str(kwargs["model"]),
        base_url=str(kwargs["base_url"]).rstrip("/"),
        api_key_env=None if api_key_env in (None, "", "null") else str(api_key_env),
        enabled=bool(kwargs.get("enabled", True)),
        max_tokens=int(kwargs.get("max_tokens", 512)),
        supports_temperature=bool(kwargs.get("supports_temperature", True)),
        max_concurrency=max(1, int(kwargs.get("max_concurrency", 1))),
        notes=str(kwargs.get("notes", "") or ""),
    )


def _parse_specs(path: Path) -> list[ModelSpec]:
    cfg = load_config(path)
    raw_defaults = cfg.get("defaults") or {}
    if not isinstance(raw_defaults, dict):
        raise ValueError("'defaults' block must be a mapping")
    defaults = {k: v for k, v in raw_defaults.items() if k in _SPEC_FIELDS}

    entries = cfg.get("models") or []
    if not isinstance(entries, list):
        raise ValueError("'models' must be a list")

    specs: list[ModelSpec] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"models[{index}] must be a mapping, got {type(entry).__name__}")
        # Key-presence, not truthiness: `api_key_env: null` is a deliberate
        # "no auth needed" and must not be overwritten by a default.
        merged = {**defaults, **entry}
        missing = [f for f in _REQUIRED_FIELDS if merged.get(f) in (None, "")]
        if missing:
            raise ValueError(
                f"models[{index}] ({merged.get('id', '?')}) is missing: {', '.join(missing)}"
            )
        spec = _coerce(merged)
        if spec.id in seen:
            raise ValueError(f"duplicate model id {spec.id!r} in {path}")
        seen.add(spec.id)
        specs.append(spec)
    return specs


def _key_reason(env: str | None) -> str | None:
    """None when the key is usable. One wording, so the reason the `models`
    listing prints is the same one a failed call reports."""
    if env is None:
        return None
    value = os.environ.get(env)
    if value is None:
        return f"{env} is not set"
    if not value.strip():
        # An empty var is not usable either, and the user-facing reason is the
        # same; the log says which of the two it was, since the fix differs.
        log.debug("%s is set but empty", env)
        return f"{env} is not set"
    return None


def _skip_reason(spec: ModelSpec) -> str | None:
    if not spec.enabled:
        return "disabled in models.yaml"
    return _key_reason(spec.api_key_env)


def available_models(path: Path) -> tuple[list[ModelSpec], list[tuple[str, str]]]:
    _ensure_env()
    usable: list[ModelSpec] = []
    skipped: list[tuple[str, str]] = []
    for spec in _parse_specs(path):
        reason = _skip_reason(spec)
        if reason is None:
            usable.append(spec)
        else:
            skipped.append((spec.id, reason))
    return usable, skipped


def load_models(path: Path) -> list[ModelSpec]:
    usable, skipped = available_models(path)
    for model_id, reason in skipped:
        log.info("skipping %s: %s", model_id, reason)
    return usable


# ------------------------------------------------------------------------ client


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _make_client(base_url: str, api_key: str, timeout: float) -> Any:
    from openai import OpenAI

    return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)


def _status_of(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _is_timeout(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    name = type(exc).__name__.lower()
    return "timeout" in name


def _is_connection_error(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    return "connection" in name or isinstance(exc, (ConnectionError, OSError))


def _retryable(exc: BaseException) -> bool:
    """429, 5xx and timeouts only. Everything else is a permanent condition."""
    if _is_timeout(exc):
        return True
    status = _status_of(exc)
    if status is None:
        # No status at all means the request never landed -- a dropped socket or
        # LM Studio mid-restart, both worth another go. An arbitrary exception
        # with no status is not retried.
        return _is_connection_error(exc)
    if status == 429 or status >= 500:
        return True
    if status in _NO_RETRY_STATUS:
        return False
    return False


def _error_blob(exc: BaseException) -> str:
    body = getattr(exc, "body", None)
    return f"{exc} {body if body is not None else ''}".lower()


def _error_string(exc: BaseException, attempts: int) -> str:
    message = str(exc).strip() or type(exc).__name__
    status = _status_of(exc)
    prefix = f"http_{status}" if status is not None else type(exc).__name__
    text = f"{prefix}: {message}"
    if len(text) > _ERROR_MAX_CHARS:
        text = text[: _ERROR_MAX_CHARS - 3] + "..."
    if attempts > 1:
        return f"after {attempts} attempts, {text}"
    return text


def _usage_dict(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    out: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key) if isinstance(usage, dict) else getattr(usage, key, None)
        if isinstance(value, int):
            out[key] = value
    return out


def _first_choice(response: Any) -> tuple[str, str] | None:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return None
    choice = choices[0]
    message = getattr(choice, "message", None)
    content = getattr(message, "content", None) if message is not None else None
    if content is None and isinstance(message, dict):
        content = message.get("content")
    finish = getattr(choice, "finish_reason", None) or ""
    return (content or ""), str(finish)


def _drop_unsupported(kwargs: dict[str, Any], exc: BaseException, applied: set[str]) -> str | None:
    """Rewrite one request param a provider rejected outright. Returns a note.

    Some reasoning models 400 on any temperature (or on max_tokens rather than
    max_completion_tokens) without being flagged in models.yaml. Re-sending once
    without the offending key is cheaper than a hand-maintained quirk table --
    but each rewrite is applied at most once, so this cannot loop.
    """
    status = _status_of(exc)
    if status not in (400, 422):
        return None
    blob = _error_blob(exc)
    if "temperature" not in blob and "max_tokens" not in blob:
        return None

    if "temperature" in blob and "temperature" in kwargs and "temperature" not in applied:
        applied.add("temperature")
        kwargs.pop("temperature")
        return "provider rejected temperature; re-sent without it"

    if "max_tokens" in blob and "max_tokens" in kwargs and "max_tokens" not in applied:
        applied.add("max_tokens")
        kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
        return "provider rejected max_tokens; re-sent as max_completion_tokens"

    return None


class ChatClient:
    """One openai client per (base_url, resolved key), reused across the factorial."""

    def __init__(
        self,
        *,
        client_factory: Callable[[str, str, float], Any] | None = None,
        max_attempts: int = _MAX_ATTEMPTS,
        backoff_base: float = _BACKOFF_BASE,
    ) -> None:
        _ensure_env()
        self._factory = client_factory
        self._max_attempts = max(1, int(max_attempts))
        self._backoff_base = float(backoff_base)
        self._clients: dict[tuple[str, str], Any] = {}
        # Request-shape quirks learned from a provider's own 400s, remembered per
        # model id. Without this the discovery 400 is re-paid on every call: a
        # 1000-call factorial against a model that rejects `temperature` would
        # spend 1000 wasted round trips relearning the same fact, and burn
        # rate-limit budget doing it.
        self._quirks: dict[str, set[str]] = {}
        self._quirks_lock = threading.Lock()

    def _client_for(self, base_url: str, api_key: str, timeout: float) -> Any:
        # Keyed on the resolved key too: two entries can share a base_url and
        # read different env vars.
        cache_key = (base_url, api_key)
        client = self._clients.get(cache_key)
        if client is None:
            factory = self._factory or _make_client
            client = factory(base_url, api_key, timeout)
            self._clients[cache_key] = client
        return client

    def resolve_key(self, spec: ModelSpec) -> tuple[str | None, str | None]:
        """-> (key to send, reason it is unusable). Exactly one is None."""
        reason = _key_reason(spec.api_key_env)
        if reason is not None:
            return None, reason
        if spec.api_key_env is None:
            return _NO_AUTH_KEY, None
        return os.environ[spec.api_key_env], None

    def complete(
        self,
        spec: ModelSpec,
        prompt: str,
        *,
        temperature: float | None,
        max_tokens: int | None = None,
        timeout: float = 120.0,
    ) -> ChatResult:
        api_key, reason = self.resolve_key(spec)
        if api_key is None:
            return ChatResult("", "", "", {}, 0, error=reason)

        kwargs: dict[str, Any] = {
            "model": spec.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(max_tokens if max_tokens is not None else spec.max_tokens),
        }
        # Omitted, not sent as None: temperature=None still serialises the key
        # for some providers and 400s.
        if spec.supports_temperature and temperature is not None:
            kwargs["temperature"] = float(temperature)

        notes: list[str] = []
        with self._quirks_lock:
            rewrites = set(self._quirks.get(spec.id, ()))
        # Apply what this model has already taught us, so the first attempt is
        # the corrected shape rather than a known-bad one.
        if "temperature" in rewrites:
            kwargs.pop("temperature", None)
        if "max_tokens" in rewrites and "max_tokens" in kwargs:
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
        attempts = 0
        elapsed_ms = 0

        while True:
            attempts += 1
            started = time.perf_counter()
            try:
                client = self._client_for(spec.base_url, api_key, timeout)
                response = client.chat.completions.create(**kwargs, timeout=timeout)
            except BaseException as exc:  # noqa: BLE001 -- a failed run is a row, not a crash
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                elapsed_ms += _ms_since(started)

                note = _drop_unsupported(kwargs, exc, rewrites)
                if note is not None:
                    notes.append(note)
                    log.info("%s: %s", spec.id, note)
                    with self._quirks_lock:
                        self._quirks.setdefault(spec.id, set()).update(rewrites)
                    attempts = 0  # a different request shape gets its own budget
                    continue

                if _retryable(exc) and attempts < self._max_attempts:
                    delay = self._backoff_base * (2 ** (attempts - 1))
                    log.warning(
                        "%s: attempt %d/%d failed (%s); retrying in %.1fs",
                        spec.id, attempts, self._max_attempts, exc, delay,
                    )
                    _sleep(delay)
                    continue

                # Latency on a failure row is the time actually spent in
                # requests, not including the backoff waits between them.
                return ChatResult(
                    "", "", "", {}, elapsed_ms,
                    error=_error_string(exc, attempts),
                    notes="; ".join(notes),
                )

            latency_ms = _ms_since(started)
            parsed = _first_choice(response)
            if parsed is None:
                return ChatResult(
                    "", str(getattr(response, "model", "") or ""), "", _usage_dict(response),
                    latency_ms, error="empty response: no choices",
                    notes="; ".join(notes),
                )
            text, finish_reason = parsed
            return ChatResult(
                text=text,
                # Provenance: whatever the API says it served, never spec.id.
                model_reported=str(getattr(response, "model", "") or ""),
                finish_reason=finish_reason,
                usage=_usage_dict(response),
                latency_ms=latency_ms,
                error=None,
                notes="; ".join(notes),
            )

    def close(self) -> None:
        for client in self._clients.values():
            closer = getattr(client, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:  # noqa: BLE001 -- teardown must not mask results
                    log.debug("client close failed", exc_info=True)
        self._clients.clear()


def _ms_since(started: float) -> int:
    return int(round((time.perf_counter() - started) * 1000))
