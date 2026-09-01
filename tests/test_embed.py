from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from datbench import embed as embed_mod
from datbench.embed import Embedder, list_embedding_models


def vector_for(word: str) -> list[float]:
    # Values that are NOT exactly representable in float32, so the round-trip
    # test has something to catch.
    return [0.1, 1.0 / 3.0, len(word) / 7.0, ord(word[0]) / 1000.0]


class FakeClient:
    """Stands in for openai.OpenAI: records every request, never touches a socket."""

    def __init__(self, *, fail_words: set[str] | None = None, fail_first: int = 0,
                 shuffle: bool = False) -> None:
        self.calls: list[list[str]] = []
        self.fail_words = fail_words or set()
        self.fail_first = fail_first
        self.shuffle = shuffle
        self.embeddings = SimpleNamespace(create=self._create)

    def _create(self, *, model: str, input: list[str], **kw):  # noqa: A002
        self.calls.append(list(input))
        if self.fail_first > 0:
            self.fail_first -= 1
            raise RuntimeError("boom")
        bad = self.fail_words.intersection(input)
        if bad:
            raise RuntimeError(f"cannot embed {sorted(bad)}")
        items = [
            SimpleNamespace(index=i, embedding=vector_for(w)) for i, w in enumerate(input)
        ]
        if self.shuffle:
            items = list(reversed(items))
        return SimpleNamespace(data=items, model=model)


class ExplodingClient:
    def __init__(self) -> None:
        self.embeddings = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        raise AssertionError("network was touched")


@pytest.fixture(autouse=True)
def no_backoff_sleep(monkeypatch):
    monkeypatch.setattr(embed_mod, "_sleep", lambda s: None)


@pytest.fixture(autouse=True)
def no_real_client(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("a real openai client was constructed")

    monkeypatch.setattr(embed_mod, "_make_client", boom)


def make_embedder(tmp_path, model="m1", client=None, **kw) -> tuple[Embedder, FakeClient]:
    e = Embedder(model, cache_path=tmp_path / "cache" / "embeddings.sqlite", **kw)
    fake = client if client is not None else FakeClient()
    e._client = fake
    return e, fake


def rows(tmp_path) -> list[tuple]:
    con = sqlite3.connect(tmp_path / "cache" / "embeddings.sqlite")
    try:
        return con.execute("SELECT model, word, dim, length(vec) FROM emb ORDER BY model, word").fetchall()
    finally:
        con.close()


def test_empty_input_is_no_op(tmp_path):
    e, _ = make_embedder(tmp_path, client=ExplodingClient())
    try:
        assert e.embed([]) == {}
        assert e.embed(["", "   "]) == {}
    finally:
        e.close()
    assert rows(tmp_path) == []


def test_cache_miss_populates_db(tmp_path):
    e, fake = make_embedder(tmp_path)
    try:
        out = e.embed(["cat", "thimble"])
    finally:
        e.close()

    assert sorted(out) == ["cat", "thimble"]
    assert fake.calls == [["cat", "thimble"]]
    assert rows(tmp_path) == [("m1", "cat", 4, 16), ("m1", "thimble", 4, 16)]


def test_cache_hit_avoids_the_network(tmp_path):
    path = tmp_path / "cache" / "embeddings.sqlite"
    with Embedder("m1", cache_path=path) as warm:
        warm._client = FakeClient()
        first = warm.embed(["cat", "dog"])

    with Embedder("m1", cache_path=path) as cold:
        cold._client = ExplodingClient()
        second = cold.embed(["dog", "cat"])  # order-independent

    assert second == first


def test_different_model_does_not_read_other_models_rows(tmp_path):
    path = tmp_path / "cache" / "embeddings.sqlite"
    with Embedder("m1", cache_path=path) as a:
        a._client = FakeClient()
        a.embed(["cat"])

    fake_b = FakeClient()
    with Embedder("m2", cache_path=path) as b:
        b._client = fake_b
        b.embed(["cat"])

    assert fake_b.calls == [["cat"]], "m2 must not be served m1's vector"
    assert rows(tmp_path) == [("m1", "cat", 4, 16), ("m2", "cat", 4, 16)]


def test_dedupe_means_one_request_per_word(tmp_path):
    e, fake = make_embedder(tmp_path)
    try:
        out = e.embed(["cat"] * 5 + ["dog", "cat"])
    finally:
        e.close()

    assert fake.calls == [["cat", "dog"]]
    assert sorted(out) == ["cat", "dog"]


def test_float32_roundtrip_is_exact(tmp_path):
    path = tmp_path / "cache" / "embeddings.sqlite"
    with Embedder("m1", cache_path=path) as warm:
        warm._client = FakeClient()
        fresh = warm.embed(["cat"])["cat"]

    with Embedder("m1", cache_path=path) as cold:
        cold._client = ExplodingClient()
        cached = cold.embed(["cat"])["cat"]

    expected = np.asarray(vector_for("cat"), dtype="<f4").tolist()
    assert fresh == expected, "fresh vectors must be quantised like cached ones"
    assert cached == fresh


def test_corrupt_blob_is_treated_as_a_miss(tmp_path):
    path = tmp_path / "cache" / "embeddings.sqlite"
    with Embedder("m1", cache_path=path) as warm:
        warm._client = FakeClient()
        good = warm.embed(["cat"])["cat"]

    con = sqlite3.connect(path)
    with con:
        con.execute("UPDATE emb SET vec = ? WHERE word = 'cat'", (b"\x00\x00\x00",))
    con.close()

    fake = FakeClient()
    with Embedder("m1", cache_path=path) as e:
        e._client = fake
        out = e.embed(["cat"])

    assert fake.calls == [["cat"]]
    assert out["cat"] == good
    assert rows(tmp_path) == [("m1", "cat", 4, 16)]


def test_batches_are_sequential_and_capped(tmp_path):
    e, fake = make_embedder(tmp_path, batch_size=2)
    words = ["a1", "b2", "c3", "d4", "e5"]
    try:
        out = e.embed(words)
    finally:
        e.close()

    assert fake.calls == [["a1", "b2"], ["c3", "d4"], ["e5"]]
    assert sorted(out) == words


def test_response_index_is_honoured(tmp_path):
    e, _ = make_embedder(tmp_path, client=FakeClient(shuffle=True))
    try:
        out = e.embed(["cat", "dog"])
    finally:
        e.close()

    assert out["cat"] == np.asarray(vector_for("cat"), dtype="<f4").tolist()
    assert out["dog"] == np.asarray(vector_for("dog"), dtype="<f4").tolist()


def test_request_is_retried_then_succeeds(tmp_path):
    e, fake = make_embedder(tmp_path, client=FakeClient(fail_first=2))
    try:
        out = e.embed(["cat"])
    finally:
        e.close()

    assert len(fake.calls) == 3
    assert "cat" in out


def test_permanently_failing_word_is_omitted_not_zeroed(tmp_path):
    e, fake = make_embedder(tmp_path, client=FakeClient(fail_words={"poison"}))
    try:
        out = e.embed(["cat", "poison", "dog"])
    finally:
        e.close()

    assert sorted(out) == ["cat", "dog"], "no zero vector, no raise"
    assert rows(tmp_path) == [("m1", "cat", 4, 16), ("m1", "dog", 4, 16)]


def test_short_batch_failure_does_not_lose_the_healthy_words(tmp_path):
    # The whole batch fails 3x because one word is poison; the per-word fallback
    # must still return the other two.
    e, fake = make_embedder(tmp_path, batch_size=8, client=FakeClient(fail_words={"poison"}))
    try:
        out = e.embed(["cat", "poison", "dog"])
    finally:
        e.close()

    assert sorted(out) == ["cat", "dog"]
    assert ["poison"] in fake.calls


def test_cache_dir_is_created(tmp_path):
    target = tmp_path / "deep" / "nested" / "embeddings.sqlite"
    with Embedder("m1", cache_path=target) as e:
        e._client = FakeClient()
        e.embed(["cat"])
    assert target.exists()


def test_close_closes_the_connection(tmp_path):
    e, _ = make_embedder(tmp_path)
    e.embed(["cat"])
    e.close()
    e.close()  # idempotent
    with pytest.raises(RuntimeError):
        e.embed(["dog"])


def test_list_embedding_models_filters_and_sorts(monkeypatch):
    ids = [
        "google/gemma-4-e4b",
        "text-embedding-qwen3-embedding-4b",
        "text-embedding-embeddinggemma-300m",
        "liquid/lfm2.5-1.2b",
        "text-embedding-nomic-embed-text-v1.5",
    ]
    seen: dict[str, object] = {}

    def fake_make_client(base_url, timeout):
        seen["base_url"] = base_url
        data = [SimpleNamespace(id=i) for i in ids]
        return SimpleNamespace(models=SimpleNamespace(list=lambda: SimpleNamespace(data=data)))

    monkeypatch.setattr(embed_mod, "_make_client", fake_make_client)

    got = list_embedding_models("http://localhost:9999/v1")
    assert got == [
        "text-embedding-embeddinggemma-300m",
        "text-embedding-nomic-embed-text-v1.5",
        "text-embedding-qwen3-embedding-4b",
    ]
    assert seen["base_url"] == "http://localhost:9999/v1"


class DeadClient:
    """Every request fails, as if LM Studio were not running."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.embeddings = SimpleNamespace(create=self._create)

    def _create(self, *, model: str, input: list[str], **kw):  # noqa: A002
        self.calls.append(list(input))
        raise RuntimeError("connection refused")


def test_backoff_is_exponential(tmp_path, monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(embed_mod, "_sleep", slept.append)

    e, _ = make_embedder(tmp_path, client=FakeClient(fail_first=2))
    try:
        e.embed(["cat"])
    finally:
        e.close()

    assert slept == [0.5, 1.0], "one sleep between attempts, doubling"


def test_solo_fallback_is_bounded(tmp_path):
    # A dead endpoint must not turn one batch into 3 + 8*3 requests.
    words = [f"w{i}" for i in range(8)]
    e, fake = make_embedder(tmp_path, batch_size=8, client=DeadClient())
    try:
        out = e.embed(words)
    finally:
        e.close()

    assert out == {}
    solo = [c[0] for c in fake.calls if len(c) == 1]
    assert solo == ["w0", "w0", "w0", "w1", "w1", "w1", "w2", "w2", "w2"]
    assert len(fake.calls) == 3 + 9
    assert rows(tmp_path) == []


def test_dead_endpoint_stops_after_two_empty_batches(tmp_path):
    words = [f"w{i}" for i in range(10)]
    e, fake = make_embedder(tmp_path, batch_size=2, client=DeadClient())
    try:
        out = e.embed(words)
    finally:
        e.close()

    assert out == {}
    asked = {w for call in fake.calls for w in call}
    assert asked == {"w0", "w1", "w2", "w3"}, "batches 3-5 must never be attempted"


def test_one_bad_word_does_not_trip_the_dead_batch_breaker(tmp_path):
    # batch_size=1 makes the poison word its own empty batch; the healthy words
    # after it must still be fetched.
    e, fake = make_embedder(tmp_path, batch_size=1, client=FakeClient(fail_words={"poison"}))
    try:
        out = e.embed(["poison", "cat", "dog"])
    finally:
        e.close()

    assert sorted(out) == ["cat", "dog"]
    assert rows(tmp_path) == [("m1", "cat", 4, 16), ("m1", "dog", 4, 16)]
