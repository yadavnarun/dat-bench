"""Local embedding client with a sqlite vector cache.

Points at LM Studio's OpenAI-compatible surface. Requests are issued one at a
time on purpose -- see Embedder.embed.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS emb (
  model TEXT NOT NULL, word TEXT NOT NULL,
  dim INTEGER NOT NULL, vec BLOB NOT NULL,
  ts TEXT NOT NULL, PRIMARY KEY (model, word)
)
"""

_MAX_ATTEMPTS = 3
_BACKOFF_BASE = 0.5

# Bounds on the per-word fallback and on a run of dead batches. Without them a
# down endpoint turns one 32-word batch into ~100 requests that each sit until
# the 120s timeout -- on the box that is also serving the chat models.
_MAX_SOLO_FAILURES = 3
_MAX_DEAD_BATCHES = 2

# sqlite caps host parameters per statement (999 on older builds), so a very
# large word list is read in a handful of IN-queries rather than one.
_SQL_VAR_LIMIT = 900

# float32 little-endian. Fixed by the schema comment in CONTRACT.md; a cache
# written by one build must be readable by the next.
_DTYPE = "<f4"


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _make_client(base_url: str, timeout: float) -> Any:
    from openai import OpenAI

    # LM Studio ignores auth entirely, but the SDK refuses to construct without
    # a non-empty key.
    return OpenAI(base_url=base_url, api_key="lm-studio", timeout=timeout)


def _field(item: Any, name: str) -> Any:
    # LM Studio responses arrive as SDK models, but a raw dict shows up when the
    # server returns a shape the SDK does not model.
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


def _chunks(items: Sequence[str], size: int) -> Iterator[Sequence[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _decode(blob: bytes, dim: int) -> list[float] | None:
    """None when the stored bytes disagree with the recorded dim."""
    if dim <= 0 or len(blob) != dim * 4:
        return None
    return np.frombuffer(blob, dtype=_DTYPE).tolist()


def _quantize(vec: Sequence[float]) -> list[float]:
    # Round-trip through float32 before returning, so a score computed off a
    # cold cache matches one computed off a warm cache bit for bit.
    return np.asarray(vec, dtype=_DTYPE).tolist()


class Embedder:
    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:1234/v1",
        cache_path: Path = Path("cache/embeddings.sqlite"),
        batch_size: int = 32,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.cache_path = Path(cache_path)
        self.batch_size = max(1, int(batch_size))
        self.timeout = timeout

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: runner.py may hand the same Embedder to a
        # worker thread. Writes are still serialised by embed() itself.
        self._conn: sqlite3.Connection | None = sqlite3.connect(
            self.cache_path, check_same_thread=False
        )
        self._conn.execute(_SCHEMA)
        self._conn.commit()

        # Built on first miss, so a fully-cached run never constructs a client
        # and tests can substitute a fake by assigning this attribute.
        self._client: Any | None = None

    def __enter__(self) -> Embedder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def embed(self, words: Sequence[str]) -> dict[str, list[float]]:
        wanted = self._dedupe(words)
        if not wanted:
            return {}

        found = self._read_cache(wanted)
        misses = [w for w in wanted if w not in found]
        if not misses:
            return found

        fetched: dict[str, list[float]] = {}
        dead_batches = 0
        for batch in _chunks(misses, self.batch_size):
            # Strictly sequential: the same local box serves the chat models
            # being benchmarked, and parallel embedding requests would contend
            # with the generation runs and distort their latency measurements.
            got = self._fetch_with_retry(list(batch))
            fetched.update(got)
            # A batch that yields nothing at all almost always means the
            # endpoint is down, not that every word in it is unembeddable.
            dead_batches = 0 if got else dead_batches + 1
            if dead_batches >= _MAX_DEAD_BATCHES:
                log.warning(
                    "embedder %s: %d batches in a row returned nothing; skipping "
                    "the remaining words in this call",
                    self.model,
                    dead_batches,
                )
                break

        if fetched:
            self._write_cache(fetched)
        found.update(fetched)

        dropped = [w for w in misses if w not in fetched]
        if dropped:
            # score.py tolerates missing vectors; a zero vector would not be
            # tolerated, it would quietly poison every distance.
            log.warning(
                "embedder %s returned no vector for %d word(s): %s",
                self.model,
                len(dropped),
                ", ".join(repr(w) for w in dropped[:10]),
            )
        return found

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @staticmethod
    def _dedupe(words: Sequence[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for w in words:
            if not isinstance(w, str) or not w.strip():
                continue  # a blank input is a request error at most servers
            if w in seen:
                continue
            seen.add(w)
            out.append(w)
        return out

    @property
    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Embedder cache connection is closed")
        return self._conn

    def _client_or_build(self) -> Any:
        if self._client is None:
            self._client = _make_client(self.base_url, self.timeout)
        return self._client

    def _read_cache(self, words: Sequence[str]) -> dict[str, list[float]]:
        found: dict[str, list[float]] = {}
        for chunk in _chunks(words, _SQL_VAR_LIMIT):
            placeholders = ",".join("?" * len(chunk))
            rows = self._db.execute(
                f"SELECT word, dim, vec FROM emb WHERE model = ? AND word IN ({placeholders})",
                (self.model, *chunk),
            ).fetchall()
            for word, dim, blob in rows:
                vec = _decode(bytes(blob), int(dim))
                if vec is None:
                    log.warning(
                        "cache row %s/%r is corrupt (dim=%s, %d bytes); refetching",
                        self.model,
                        word,
                        dim,
                        len(blob),
                    )
                    continue
                found[word] = vec
        return found

    def _write_cache(self, vecs: dict[str, list[float]]) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        rows = [
            (self.model, w, len(v), np.asarray(v, dtype=_DTYPE).tobytes(), ts)
            for w, v in vecs.items()
        ]
        with self._db:  # one transaction for the whole batch of misses
            self._db.executemany(
                "INSERT OR REPLACE INTO emb (model, word, dim, vec, ts) VALUES (?, ?, ?, ?, ?)",
                rows,
            )

    def _attempt(self, batch: list[str]) -> tuple[dict[str, list[float]], Exception | None]:
        last: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                return self._fetch_batch(batch), None
            except Exception as exc:  # noqa: BLE001 -- any failure is retryable here
                last = exc
                if attempt < _MAX_ATTEMPTS - 1:
                    _sleep(_BACKOFF_BASE * 2**attempt)
        return {}, last

    def _fetch_with_retry(self, batch: list[str]) -> dict[str, list[float]]:
        got, err = self._attempt(batch)
        if err is None:
            return got
        log.warning(
            "embedding request failed %dx for %d word(s) on %s: %s",
            _MAX_ATTEMPTS,
            len(batch),
            self.model,
            err,
        )
        if len(batch) == 1:
            return {}

        # One poison word must not cost the whole batch, so a failed multi-word
        # batch is retried a word at a time -- abandoning the rest once several
        # in a row fail, which reads as a dead endpoint rather than bad words.
        out: dict[str, list[float]] = {}
        consecutive = 0
        for i, word in enumerate(batch):
            solo, solo_err = self._attempt([word])
            if solo_err is None:
                out.update(solo)
                consecutive = 0
                continue
            consecutive += 1
            if consecutive >= _MAX_SOLO_FAILURES:
                log.warning(
                    "embedder %s: %d words failed in a row; abandoning %d more "
                    "in this batch",
                    self.model,
                    consecutive,
                    len(batch) - i - 1,
                )
                break
        return out

    def _fetch_batch(self, batch: list[str]) -> dict[str, list[float]]:
        resp = self._client_or_build().embeddings.create(model=self.model, input=batch)
        data = list(_field(resp, "data") or [])
        if len(data) != len(batch):
            raise ValueError(f"expected {len(batch)} embeddings, got {len(data)}")

        out: dict[str, list[float]] = {}
        for pos, item in enumerate(data):
            vec = _field(item, "embedding")
            if not vec:
                raise ValueError("response item carried no embedding")
            # Trust the server's index when it sends one: the spec does not
            # promise input order back.
            idx = _field(item, "index")
            i = pos if idx is None else int(idx)
            if not 0 <= i < len(batch):
                raise ValueError(f"embedding index {i} outside the batch")
            out[batch[i]] = _quantize(vec)
        if len(out) != len(batch):
            raise ValueError("response indices did not cover the batch")
        return out


def list_embedding_models(base_url: str = "http://localhost:1234/v1") -> list[str]:
    client = _make_client(base_url, 30.0)
    ids = []
    for m in client.models.list().data:
        mid = _field(m, "id")
        if mid and "embed" in str(mid).lower():
            ids.append(str(mid))
    # Sorted so `--embedders auto` picks the same order on every run and the
    # report rows stay stable.
    return sorted(set(ids))
