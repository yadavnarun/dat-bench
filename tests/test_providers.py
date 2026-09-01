from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import openai
import pytest

from datbench import providers as prov
from datbench.providers import ChatClient, ChatResult, ModelSpec, available_models, load_models

LOCAL = """
defaults:
  max_tokens: 512
  max_concurrency: 1

models:
  - id: local
    model: vendor/local-3b
    base_url: http://localhost:1234/v1
    api_key_env: null
    enabled: true
"""


def write_yaml(tmp_path: Path, text: str, name: str = "models.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def response(
    text: str = "1. cat\n2. thimble",
    *,
    model: str = "vendor/local-3b-q4",
    finish_reason: str = "stop",
    usage: object | None = SimpleNamespace(prompt_tokens=118, completion_tokens=44),
    choices: object | None = None,
):
    if choices is None:
        choices = [
            SimpleNamespace(
                message=SimpleNamespace(content=text), finish_reason=finish_reason
            )
        ]
    return SimpleNamespace(model=model, choices=choices, usage=usage)


def status_error(cls, code: int, message: str = "boom", body: object = None):
    """A real openai exception -- no httpx needed, APIStatusError only reads
    status_code/headers/request off the response."""
    return cls(
        message,
        response=SimpleNamespace(status_code=code, headers={}, request=None),
        body=body,
    )


class FakeClient:
    """Stands in for openai.OpenAI. Never touches a socket."""

    def __init__(self, script: list[object] | None = None) -> None:
        self.calls: list[dict] = []
        self.script = list(script or [])
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0) if self.script else response()
        if isinstance(item, BaseException):
            raise item
        return item


class Factory:
    def __init__(self, *scripts: list[object]) -> None:
        self.constructed: list[tuple[str, str, float]] = []
        self._scripts = list(scripts)
        self.clients: list[FakeClient] = []

    def __call__(self, base_url: str, api_key: str, timeout: float) -> FakeClient:
        self.constructed.append((base_url, api_key, timeout))
        script = self._scripts.pop(0) if self._scripts else None
        client = FakeClient(script)
        self.clients.append(client)
        return client

    @property
    def client(self) -> FakeClient:
        assert len(self.clients) == 1, f"{len(self.clients)} clients constructed"
        return self.clients[0]


@pytest.fixture(autouse=True)
def no_real_client(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("a real openai client was constructed")

    monkeypatch.setattr(prov, "_make_client", boom)


@pytest.fixture(autouse=True)
def sleeps(monkeypatch):
    recorded: list[float] = []
    monkeypatch.setattr(prov, "_sleep", recorded.append)
    return recorded


@pytest.fixture(autouse=True)
def sealed_env(monkeypatch):
    # The repo root may hold a real .env; a test that asserts "OPENAI_API_KEY is
    # not set" must not depend on the developer's machine.
    monkeypatch.setattr(prov, "_ENV_LOADED", True)
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def local_spec(**kw) -> ModelSpec:
    base = dict(
        id="local",
        model="vendor/local-3b",
        base_url="http://localhost:1234/v1",
        api_key_env=None,
    )
    base.update(kw)
    return ModelSpec(**base)


# ----------------------------------------------------------------- config parsing


def test_defaults_fill_omitted_fields(tmp_path):
    path = write_yaml(tmp_path, LOCAL)
    (spec,) = load_models(path)
    assert (spec.max_tokens, spec.max_concurrency) == (512, 1)
    assert spec.supports_temperature is True
    assert spec.notes == ""


def test_entry_overrides_defaults(tmp_path):
    path = write_yaml(
        tmp_path,
        LOCAL
        + """
  - id: cloudy
    model: big
    base_url: https://api.example.com/v1
    api_key_env: null
    max_tokens: 64
    max_concurrency: 4
    supports_temperature: false
    notes: reasoning model
""",
    )
    specs = {s.id: s for s in load_models(path)}
    assert specs["cloudy"].max_tokens == 64
    assert specs["cloudy"].max_concurrency == 4
    assert specs["cloudy"].supports_temperature is False
    assert specs["cloudy"].notes == "reasoning model"
    assert specs["local"].max_tokens == 512


def test_api_key_env_null_needs_no_env(tmp_path):
    usable, skipped = available_models(write_yaml(tmp_path, LOCAL))
    assert [s.id for s in usable] == ["local"]
    assert usable[0].api_key_env is None
    assert skipped == []


def test_disabled_entry_skipped_with_reason(tmp_path):
    path = write_yaml(
        tmp_path,
        LOCAL
        + """
  - id: routed
    model: routed/thing
    base_url: https://openrouter.ai/api/v1
    api_key_env: OPENROUTER_API_KEY
    enabled: false
""",
    )
    usable, skipped = available_models(path)
    assert [s.id for s in usable] == ["local"]
    assert skipped == [("routed", "disabled in models.yaml")]


def test_disabled_wins_over_missing_key(tmp_path):
    # Both wrong: report the one the user must fix first.
    path = write_yaml(
        tmp_path,
        """
models:
  - id: routed
    model: m
    base_url: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
    enabled: false
""",
    )
    _, skipped = available_models(path)
    assert skipped == [("routed", "disabled in models.yaml")]


CLOUD = """
models:
  - id: gpt
    model: gpt-5.1
    base_url: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
    enabled: true
"""


def test_missing_key_env_skipped_with_named_reason(tmp_path):
    usable, skipped = available_models(write_yaml(tmp_path, CLOUD))
    assert usable == []
    assert skipped == [("gpt", "OPENAI_API_KEY is not set")]


def test_empty_key_env_skipped_with_named_reason(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    usable, skipped = available_models(write_yaml(tmp_path, CLOUD))
    assert usable == []
    assert skipped == [("gpt", "OPENAI_API_KEY is not set")]


def test_entry_activates_when_key_is_added(tmp_path, monkeypatch):
    path = write_yaml(tmp_path, CLOUD)
    assert available_models(path)[0] == []
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    usable, skipped = available_models(path)
    assert [s.id for s in usable] == ["gpt"]
    assert skipped == []


def test_load_models_returns_only_usable(tmp_path):
    path = write_yaml(tmp_path, LOCAL + CLOUD.split("models:", 1)[1])
    assert [s.id for s in load_models(path)] == ["local"]


def test_duplicate_id_raises(tmp_path):
    path = write_yaml(tmp_path, LOCAL + LOCAL.split("models:", 1)[1])
    with pytest.raises(ValueError, match="duplicate model id"):
        load_models(path)


def test_missing_required_field_raises(tmp_path):
    path = write_yaml(
        tmp_path,
        """
models:
  - id: broken
    base_url: http://localhost:1234/v1
    api_key_env: null
""",
    )
    with pytest.raises(ValueError, match="missing: model"):
        load_models(path)


def test_unknown_key_is_ignored(tmp_path):
    path = write_yaml(tmp_path, LOCAL + "    tempreature: 0.7\n")
    (spec,) = load_models(path)
    assert spec.id == "local"


def test_empty_yaml_is_empty_config(tmp_path):
    path = write_yaml(tmp_path, "")
    assert prov.load_config(path) == {}
    assert load_models(path) == []


def test_scoring_and_run_config_expose_blocks(tmp_path):
    path = write_yaml(
        tmp_path,
        """
defaults:
  max_tokens: 256
scoring:
  n_use: 5
  policies: [strict]
run:
  n: 3
  temperatures: [0.0]
""",
    )
    scoring = prov.load_scoring_config(path)
    assert scoring["n_use"] == 5
    assert scoring["policies"] == ["strict"]
    assert scoring["embedders"] == "auto"          # module default
    assert scoring["baseline_seed"] == 0
    assert scoring["max_tokens"] == 256            # yaml defaults block

    run = prov.load_run_config(path)
    assert run["n"] == 3
    assert run["temperatures"] == [0.0]
    assert run["prompts"] == list(prov.RUN_DEFAULTS["prompts"])


def test_config_blocks_fall_back_entirely(tmp_path):
    path = write_yaml(tmp_path, LOCAL)
    assert prov.load_scoring_config(path)["n_use"] == 7
    assert prov.load_run_config(path)["n"] == 10


# --------------------------------------------------------------------------- .env


def test_load_dotenv_parses_and_does_not_clobber(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "# a comment\n\nOPENAI_API_KEY=sk-from-file\n"
        'export ANTHROPIC_API_KEY="sk-quoted"\n'
        "DEEPSEEK_API_KEY=already-set\n"
        "junk-line\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-from-shell")
    applied = prov.load_dotenv(env)
    assert applied == {"OPENAI_API_KEY": "sk-from-file", "ANTHROPIC_API_KEY": "sk-quoted"}
    import os

    assert os.environ["DEEPSEEK_API_KEY"] == "sk-from-shell"


def test_load_dotenv_missing_file_is_noop(tmp_path):
    assert prov.load_dotenv(tmp_path / "nope.env") == {}


# ------------------------------------------------------------------- happy path


def test_complete_returns_fields_from_response():
    factory = Factory()
    client = ChatClient(client_factory=factory)
    result = client.complete(local_spec(), "go", temperature=0.7)

    assert isinstance(result, ChatResult)
    assert result.error is None
    assert result.text == "1. cat\n2. thimble"
    # Provenance: what the server said it served, not our label.
    assert result.model_reported == "vendor/local-3b-q4"
    assert result.finish_reason == "stop"
    assert result.usage == {"prompt_tokens": 118, "completion_tokens": 44}
    assert isinstance(result.latency_ms, int) and result.latency_ms >= 0


def test_no_auth_entry_gets_dummy_key():
    factory = Factory()
    ChatClient(client_factory=factory).complete(local_spec(), "go", temperature=0.0)
    base_url, api_key, _ = factory.constructed[0]
    assert base_url == "http://localhost:1234/v1"
    assert api_key and api_key != ""


def test_key_is_read_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-live")
    factory = Factory()
    spec = local_spec(id="gpt", base_url="https://api.openai.com/v1", api_key_env="OPENAI_API_KEY")
    ChatClient(client_factory=factory).complete(spec, "go", temperature=None)
    assert factory.constructed[0][1] == "sk-live"


def test_missing_key_at_call_time_is_an_error_row():
    factory = Factory()
    spec = local_spec(api_key_env="OPENAI_API_KEY")
    result = ChatClient(client_factory=factory).complete(spec, "go", temperature=0.7)
    assert result.error == "OPENAI_API_KEY is not set"
    assert result.text == ""
    assert factory.constructed == []


def test_client_cached_per_base_url_and_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-a")
    factory = Factory()
    client = ChatClient(client_factory=factory)
    a = local_spec(id="a")
    b = local_spec(id="b", model="other")
    cloud = local_spec(id="c", base_url="https://api.openai.com/v1", api_key_env="OPENAI_API_KEY")
    for spec in (a, b, a, cloud, cloud):
        client.complete(spec, "go", temperature=0.0)
    # Two distinct (base_url, key) pairs over five calls.
    assert len(factory.constructed) == 2


# ------------------------------------------------------------------- temperature


def test_supports_temperature_false_omits_the_kwarg():
    factory = Factory()
    client = ChatClient(client_factory=factory)
    client.complete(local_spec(supports_temperature=False), "go", temperature=0.7)
    (call,) = factory.client.calls
    assert "temperature" not in call


def test_temperature_none_omits_the_kwarg():
    factory = Factory()
    ChatClient(client_factory=factory).complete(local_spec(), "go", temperature=None)
    assert "temperature" not in factory.client.calls[0]


def test_temperature_sent_when_supported():
    factory = Factory()
    ChatClient(client_factory=factory).complete(local_spec(), "go", temperature=0.7)
    assert factory.client.calls[0]["temperature"] == 0.7


def test_max_tokens_from_spec_then_override():
    factory = Factory()
    client = ChatClient(client_factory=factory)
    client.complete(local_spec(max_tokens=99), "go", temperature=0.0)
    client.complete(local_spec(max_tokens=99), "go", temperature=0.0, max_tokens=7)
    assert [c["max_tokens"] for c in factory.client.calls] == [99, 7]


def test_unsupported_temperature_400_is_retried_without_it():
    err = status_error(
        openai.BadRequestError,
        400,
        "Unsupported value: 'temperature' does not support 0.7",
        body={"error": {"code": "unsupported_value", "param": "temperature"}},
    )
    factory = Factory([err, response()])
    result = ChatClient(client_factory=factory).complete(local_spec(), "go", temperature=0.7)

    calls = factory.client.calls
    assert len(calls) == 2
    assert calls[0]["temperature"] == 0.7
    assert "temperature" not in calls[1]
    assert result.error is None
    assert "temperature" in result.notes


def test_unsupported_max_tokens_400_is_retried_renamed():
    err = status_error(
        openai.BadRequestError,
        400,
        "Unsupported parameter: 'max_tokens' is not supported; use 'max_completion_tokens'",
    )
    factory = Factory([err, response()])
    result = ChatClient(client_factory=factory).complete(local_spec(), "go", temperature=None)

    calls = factory.client.calls
    assert len(calls) == 2
    assert "max_tokens" not in calls[1]
    assert calls[1]["max_completion_tokens"] == 512
    assert result.error is None
    assert "max_completion_tokens" in result.notes


def test_unsupported_param_rewrite_is_not_infinite():
    err = status_error(openai.BadRequestError, 400, "'temperature' unsupported")
    factory = Factory([err] * 8)
    result = ChatClient(client_factory=factory).complete(local_spec(), "go", temperature=0.7)
    # One original + one without temperature; a 400 is never retried as-is.
    assert len(factory.client.calls) == 2
    assert result.error is not None


# ------------------------------------------------------------------------ retries


def test_500_retries_then_succeeds(sleeps):
    boom = status_error(openai.InternalServerError, 500, "server had a moment")
    factory = Factory([boom, boom, response()])
    result = ChatClient(client_factory=factory).complete(local_spec(), "go", temperature=0.0)

    assert result.error is None
    assert result.text == "1. cat\n2. thimble"
    assert len(factory.client.calls) == 3
    assert sleeps == [0.5, 1.0]          # exponential


def test_429_retries_then_succeeds(sleeps):
    factory = Factory([status_error(openai.RateLimitError, 429, "slow down"), response()])
    result = ChatClient(client_factory=factory).complete(local_spec(), "go", temperature=0.0)
    assert result.error is None
    assert len(factory.client.calls) == 2
    assert sleeps == [0.5]


def test_timeout_retries_then_succeeds():
    factory = Factory([openai.APITimeoutError(request=None), response()])
    result = ChatClient(client_factory=factory).complete(local_spec(), "go", temperature=0.0)
    assert result.error is None
    assert len(factory.client.calls) == 2


def test_connection_error_retries():
    factory = Factory([openai.APIConnectionError(request=None), response()])
    result = ChatClient(client_factory=factory).complete(local_spec(), "go", temperature=0.0)
    assert result.error is None
    assert len(factory.client.calls) == 2


@pytest.mark.parametrize(
    "exc, retryable",
    [
        (status_error(openai.InternalServerError, 500), True),
        (status_error(openai.APIStatusError, 503), True),
        (status_error(openai.RateLimitError, 429), True),
        (openai.APITimeoutError(request=None), True),
        (status_error(openai.BadRequestError, 400), False),
        (status_error(openai.AuthenticationError, 401), False),
        (status_error(openai.NotFoundError, 404), False),
        (status_error(openai.UnprocessableEntityError, 422), False),
        (RuntimeError("who knows"), False),
    ],
)
def test_retry_classification(exc, retryable):
    assert prov._retryable(exc) is retryable


@pytest.mark.parametrize(
    "exc",
    [
        status_error(openai.BadRequestError, 400, "malformed"),
        status_error(openai.AuthenticationError, 401, "bad key"),
        status_error(openai.NotFoundError, 404, "no such model: gpt-6"),
    ],
)
def test_permanent_status_is_not_retried(exc, sleeps):
    factory = Factory([exc, response()])
    result = ChatClient(client_factory=factory).complete(local_spec(), "go", temperature=None)
    assert len(factory.client.calls) == 1, "a permanent failure must not be retried"
    assert sleeps == []
    assert result.error is not None
    assert result.text == ""


def test_total_failure_is_a_result_not_an_exception(sleeps):
    boom = status_error(openai.InternalServerError, 500, "still down")
    factory = Factory([boom, boom, boom, response()])
    result = ChatClient(client_factory=factory).complete(local_spec(), "go", temperature=0.7)

    assert isinstance(result, ChatResult)
    assert result.error is not None
    assert "500" in result.error
    assert result.text == ""
    assert result.model_reported == ""
    assert result.usage == {}
    assert len(factory.client.calls) == 3
    assert sleeps == [0.5, 1.0]


def test_unexpected_exception_is_recorded_not_raised():
    factory = Factory([RuntimeError("json decode blew up")])
    result = ChatClient(client_factory=factory).complete(local_spec(), "go", temperature=0.0)
    assert result.error is not None
    assert "RuntimeError" in result.error
    assert len(factory.client.calls) == 1


def test_keyboard_interrupt_still_propagates():
    factory = Factory([KeyboardInterrupt()])
    with pytest.raises(KeyboardInterrupt):
        ChatClient(client_factory=factory).complete(local_spec(), "go", temperature=0.0)


# ------------------------------------------------------------------ odd responses


def test_missing_usage_gives_empty_dict():
    factory = Factory([response(usage=None)])
    result = ChatClient(client_factory=factory).complete(local_spec(), "go", temperature=0.0)
    assert result.usage == {}
    assert result.error is None


def test_partial_usage_keeps_what_is_there():
    factory = Factory([response(usage=SimpleNamespace(prompt_tokens=5))])
    result = ChatClient(client_factory=factory).complete(local_spec(), "go", temperature=0.0)
    assert result.usage == {"prompt_tokens": 5}


def test_empty_choices_is_an_error_row():
    factory = Factory([response(choices=[])])
    result = ChatClient(client_factory=factory).complete(local_spec(), "go", temperature=0.0)
    assert result.error == "empty response: no choices"
    assert result.text == ""


def test_null_content_becomes_empty_string():
    factory = Factory([response(text=None, finish_reason="length")])
    result = ChatClient(client_factory=factory).complete(local_spec(), "go", temperature=0.0)
    assert result.text == ""
    assert result.finish_reason == "length"
    assert result.error is None


def test_timeout_is_passed_through_to_the_request():
    factory = Factory()
    ChatClient(client_factory=factory).complete(local_spec(), "go", temperature=0.0, timeout=7.5)
    assert factory.client.calls[0]["timeout"] == 7.5
    assert factory.constructed[0][2] == 7.5


def test_prompt_becomes_a_single_user_message():
    factory = Factory()
    ChatClient(client_factory=factory).complete(local_spec(), "name 10 words", temperature=0.0)
    assert factory.client.calls[0]["messages"] == [
        {"role": "user", "content": "name 10 words"}
    ]


def test_close_is_safe_and_clears_the_cache():
    factory = Factory()
    client = ChatClient(client_factory=factory)
    client.complete(local_spec(), "go", temperature=0.0)
    client.close()
    client.complete(local_spec(), "go", temperature=0.0)
    assert len(factory.constructed) == 2


# ------------------------------------------------- learned request quirks ----
# A provider that rejects `temperature` (or wants max_completion_tokens) teaches
# the client once. Re-discovering it per call means a wasted 400 on every request
# of the factorial -- ~1000 of them on a real run, against the same rate limit.

QUIRK_SPEC = ModelSpec(
    id="reasoner", model="vendor/reasoner",
    base_url="https://api.example.com/v1", api_key_env=None,
)


def _rejecting_factory(bad_key: str, message: str, n: int = 6):
    """A factory whose clients 400 while `bad_key` is present, then succeed."""
    class Rejector(FakeClient):
        def _create(self, **kwargs):
            self.calls.append(kwargs)
            if bad_key in kwargs:
                raise status_error(openai.BadRequestError, 400, message)
            return response()

    class F(Factory):
        def __call__(self, base_url, api_key, timeout):
            self.constructed.append((base_url, api_key, timeout))
            c = Rejector()
            self.clients.append(c)
            return c
    return F()


def test_client_learns_a_temperature_rejection_once(monkeypatch):
    """gpt-5.5 / o4-mini / gpt-5.6-sol all 400 on any temperature."""
    factory = _rejecting_factory(
        "temperature", "Unsupported value: 'temperature' does not support 0.7")
    monkeypatch.setattr(prov, "_make_client", factory)
    client = ChatClient()

    first = client.complete(QUIRK_SPEC, "p", temperature=0.7)
    assert first.error is None
    calls = [c for cl in factory.clients for c in cl.calls]
    assert len(calls) == 2, "expected one rejected call, then one corrected call"

    second = client.complete(QUIRK_SPEC, "p", temperature=0.7)
    assert second.error is None
    calls2 = [c for cl in factory.clients for c in cl.calls]
    assert len(calls2) - len(calls) == 1, "the client re-paid the discovery 400"
    assert "temperature" not in calls2[-1]


def test_client_learns_max_completion_tokens_once(monkeypatch):
    factory = _rejecting_factory(
        "max_tokens", "Unsupported parameter: 'max_tokens' is not supported")
    monkeypatch.setattr(prov, "_make_client", factory)
    client = ChatClient()

    client.complete(QUIRK_SPEC, "p", temperature=None)
    n = len([c for cl in factory.clients for c in cl.calls])
    client.complete(QUIRK_SPEC, "p", temperature=None)
    calls = [c for cl in factory.clients for c in cl.calls]
    assert len(calls) - n == 1, "max_tokens rewrite was not remembered"
    assert "max_completion_tokens" in calls[-1] and "max_tokens" not in calls[-1]


def test_quirks_are_per_model_not_global(monkeypatch):
    """One model's quirk must not silently reshape another model's requests."""
    factory = _rejecting_factory(
        "temperature", "Unsupported value: 'temperature' does not support 0.7")
    monkeypatch.setattr(prov, "_make_client", factory)
    client = ChatClient()
    client.complete(QUIRK_SPEC, "p", temperature=0.7)

    other = ModelSpec(id="normal", model="vendor/normal",
                      base_url="https://api.example.com/v1", api_key_env=None)
    before = len([c for cl in factory.clients for c in cl.calls])
    client.complete(other, "p", temperature=0.7)
    calls = [c for cl in factory.clients for c in cl.calls]
    # It must still TRY temperature for a model that never rejected it.
    assert "temperature" in calls[before]
