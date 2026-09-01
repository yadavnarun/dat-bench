"""Tests for datbench.score. No network: every embedder here is a local callable."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import pytest

from datbench.score import (
    COMMON_NOUNS,
    DEFAULT_CATEGORIES,
    BaselineStats,
    ScoreResult,
    category_floor,
    cosine_distance,
    dat_score,
    random_baseline,
    score_run,
)

# --- helpers ---------------------------------------------------------------

DIM = 8


def e(i: int, dim: int = DIM, sign: float = 1.0) -> list[float]:
    """Unit basis vector e_i (1-indexed), optionally negated."""
    v = [0.0] * dim
    v[i - 1] = sign
    return v


@dataclass(frozen=True)
class _Check:
    """Stands in for validate.WordCheck.

    Built locally so these tests exercise score.py alone: score_run reads only
    .valid and .clean, and the field names match the contract.
    """

    word: str
    clean: str
    valid: bool
    flags: tuple[str, ...]
    zipf: float | None


def check(clean: str, *, valid: bool = True) -> _Check:
    return _Check(
        word=clean,
        clean=clean,
        valid=valid,
        flags=() if valid else ("not_noun",),
        zipf=4.0,
    )


def orthonormal(words, dim: int = DIM) -> dict[str, list[float]]:
    return {w: e(i + 1, dim) for i, w in enumerate(words)}


def _unit(seed: str, dim: int) -> list[float]:
    # random.Random accepts a str seed and hashes it reproducibly across
    # processes, so these fixtures are stable without pinning a vector table.
    rng = random.Random(seed)
    v = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def random_embedder(dim: int = 16, calls: list | None = None):
    """Every word gets its own near-orthogonal unit vector: chance is ~1.0."""

    def embed(words):
        if calls is not None:
            calls.append(list(words))
        return {w: _unit(f"w::{w}", dim) for w in words}

    return embed


def clustered_embedder(dim: int = 24, spread: float = 0.15):
    """A fake embedder with real semantics: DEFAULT_CATEGORIES words cluster."""
    cat_of = {w: c for c, ws in DEFAULT_CATEGORIES.items() for w in ws}

    def embed(words):
        out = {}
        for w in words:
            cat = cat_of.get(w)
            if cat is None:
                out[w] = _unit(f"w::{w}", dim)
                continue
            base = _unit(f"cat::{cat}", dim)
            jitter = _unit(f"jit::{w}", dim)
            v = [b + spread * j for b, j in zip(base, jitter)]
            n = math.sqrt(sum(x * x for x in v))
            out[w] = [x / n for x in v]
        return out

    return embed


# --- cosine_distance -------------------------------------------------------


def test_cosine_distance_identical_is_zero():
    assert cosine_distance(e(1), e(1)) == pytest.approx(0.0)


def test_cosine_distance_orthogonal_is_one():
    assert cosine_distance(e(1), e(2)) == pytest.approx(1.0)


def test_cosine_distance_opposite_is_two():
    assert cosine_distance(e(1), e(1, sign=-1.0)) == pytest.approx(2.0)


def test_cosine_distance_ignores_magnitude():
    assert cosine_distance([3.0, 0.0], [0.0, 7.0]) == pytest.approx(1.0)
    assert cosine_distance([3.0, 0.0], [100.0, 0.0]) == pytest.approx(0.0)


def test_cosine_distance_hand_computed_45_degrees():
    assert cosine_distance([1.0, 1.0], [1.0, 0.0]) == pytest.approx(1.0 - 1 / math.sqrt(2))


def test_cosine_distance_identical_unnormalised_never_goes_negative():
    # Float error on 1 - sim can produce -2.2e-16, which would drag a mean below
    # the true floor of 0.
    v = [0.10345, -0.9214, 0.3311, 7.0, -0.0001]
    assert cosine_distance(v, list(v)) == 0.0


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ([0.0, 0.0], [1.0, 0.0]),
        ([1.0, 0.0], [0.0, 0.0]),
        ([0.0, 0.0], [0.0, 0.0]),
    ],
)
def test_cosine_distance_zero_norm_raises(a, b):
    # ValueError, not nan: a nan would silently poison the mean over 21 pairs.
    with pytest.raises(ValueError):
        cosine_distance(a, b)


def test_cosine_distance_dimension_mismatch_raises():
    with pytest.raises(ValueError):
        cosine_distance([1.0, 0.0], [1.0, 0.0, 0.0])


def test_cosine_distance_stays_in_range():
    words = [f"w{i}" for i in range(30)]
    vecs = random_embedder(dim=12)(words)
    for i, a in enumerate(words):
        for b in words[i + 1 :]:
            d = cosine_distance(vecs[a], vecs[b])
            assert 0.0 <= d <= 2.0


# --- dat_score -------------------------------------------------------------


def test_dat_score_all_orthogonal_is_one():
    words = [f"w{i}" for i in range(7)]
    assert dat_score(words, orthonormal(words), n_use=7) == pytest.approx(1.0)


def test_dat_score_all_identical_is_zero():
    words = [f"w{i}" for i in range(7)]
    vecs = {w: e(1) for w in words}
    assert dat_score(words, vecs, n_use=7) == pytest.approx(0.0)


def test_dat_score_opposite_pair_is_two():
    vecs = {"a": e(1), "b": e(1, sign=-1.0)}
    assert dat_score(["a", "b"], vecs, n_use=7) == pytest.approx(2.0)


def test_dat_score_averages_over_all_21_pairs():
    # w0 and w1 share a vector; the other five are orthogonal to everything.
    # 21 pairs, exactly one of which is 0 -> 20/21. A different denominator
    # (e.g. 7, or 28) cannot produce this number.
    words = ["w0", "w1", "w2", "w3", "w4", "w5", "w6"]
    vecs = {"w0": e(1), "w1": e(1)}
    vecs.update({w: e(i + 2) for i, w in enumerate(words[2:])})
    assert dat_score(words, vecs, n_use=7) == pytest.approx(20 / 21)


def test_dat_score_uses_only_the_first_n_use_words():
    words = [f"w{i}" for i in range(7)] + ["late"]
    vecs = orthonormal(words[:7])
    vecs["late"] = e(1, sign=-1.0)  # would pull the mean to 22/28 if counted
    assert dat_score(words, vecs, n_use=7) == pytest.approx(1.0)


def test_dat_score_skips_words_without_vectors():
    words = ["missing1", "a", "missing2", "b", "c"]
    vecs = {"a": e(1), "b": e(2), "c": e(3)}
    # 3 embedded words -> 3 pairs, all orthogonal.
    assert dat_score(words, vecs, n_use=7) == pytest.approx(1.0)


def test_n_use_budget_counts_only_words_that_have_vectors():
    # Gaps must not consume the budget: if they did, only w1..w5 would be used
    # and the answer would be 1.0 instead of 22/21.
    words = ["gapA", "w1", "gapB", "w2", "w3", "w4", "w5", "w6", "w7"]
    vecs = {f"w{i}": e(i) for i in range(1, 7)}
    vecs["w7"] = e(1, sign=-1.0)
    assert dat_score(words, vecs, n_use=7) == pytest.approx(22 / 21)


def test_dat_score_returns_none_with_fewer_than_two_vectors():
    assert dat_score(["a", "b", "c"], {"a": e(1)}, n_use=7) is None
    assert dat_score([], {}, n_use=7) is None


def test_dat_score_is_raw_mean_not_times_100():
    # The x100 convention belongs to GloVe-scale DAT scores; using it here would
    # falsely imply comparability with the published human norms.
    words = [f"w{i}" for i in range(7)]
    assert dat_score(words, orthonormal(words), n_use=7) == pytest.approx(1.0)
    assert dat_score(words, orthonormal(words), n_use=7) < 2.0


def test_dat_score_rejects_degenerate_n_use():
    with pytest.raises(ValueError):
        dat_score(["a", "b"], {"a": e(1), "b": e(2)}, n_use=1)


# --- score_run -------------------------------------------------------------


def valid_run(n: int = 7, dim: int = DIM):
    words = [f"w{i}" for i in range(n)]
    return [check(w) for w in words], orthonormal(words, dim)


def test_score_run_strict_happy_path():
    checks, vecs = valid_run(7)
    res = score_run(checks, vecs, policy="strict", n_use=7)
    assert isinstance(res, ScoreResult)
    assert res.scored is True
    assert res.reason is None
    assert res.score == pytest.approx(1.0)
    assert (res.n_candidates, res.n_valid, res.n_words_used) == (7, 7, 7)
    assert res.valid_rate == pytest.approx(1.0)


def test_score_run_strict_refuses_six_valid_words():
    checks, vecs = valid_run(6)
    res = score_run(checks, vecs, policy="strict", n_use=7)
    assert res.scored is False
    assert res.score is None
    assert res.n_valid == 6
    assert res.reason and "7" in res.reason


@pytest.mark.parametrize("policy", ["strict", "lenient"])
def test_refused_run_is_never_score_zero(policy):
    # The single easiest way to corrupt this benchmark's conclusions: a refusal
    # scored as 0.0 rewards invalidity with a floor instead of an exclusion.
    checks = [check(w, valid=False) for w in ("aa", "bb", "cc")]
    res = score_run(checks, {}, policy=policy, n_use=7, min_words=4)
    assert res.scored is False
    assert res.score is None
    assert res.score != 0.0
    assert res.reason


def test_score_run_lenient_scores_five_words():
    checks, vecs = valid_run(5)
    res = score_run(checks, vecs, policy="lenient", n_use=7, min_words=4)
    assert res.scored is True
    assert res.n_words_used == 5  # fewer pairs, hence a different variance
    assert res.score == pytest.approx(1.0)


def test_score_run_lenient_refuses_below_min_words():
    checks, vecs = valid_run(3)
    res = score_run(checks, vecs, policy="lenient", n_use=7, min_words=4)
    assert res.scored is False
    assert res.score is None


def test_score_run_lenient_caps_at_n_use():
    checks, vecs = valid_run(10, dim=10)
    res = score_run(checks, vecs, policy="lenient", n_use=7, min_words=4)
    assert res.n_valid == 10
    assert res.n_words_used == 7


def test_score_run_ignores_invalid_words_even_when_embedded():
    checks, vecs = valid_run(7)
    checks = list(checks) + [check("junk", valid=False)]
    vecs["junk"] = e(1, sign=-1.0)  # would move the score if it leaked in
    res = score_run(checks, vecs, policy="strict", n_use=7)
    assert res.score == pytest.approx(1.0)
    assert (res.n_candidates, res.n_valid) == (8, 7)
    assert res.valid_rate == pytest.approx(7 / 8)


def test_score_run_partial_embedding_coverage():
    checks, vecs = valid_run(7)
    for w in ("w5", "w6"):
        del vecs[w]
    strict = score_run(checks, vecs, policy="strict", n_use=7)
    assert strict.scored is False
    assert strict.score is None
    assert strict.n_valid == 7 and strict.n_words_used == 5

    lenient = score_run(checks, vecs, policy="lenient", n_use=7, min_words=4)
    assert lenient.scored is True
    assert lenient.n_words_used == 5


def test_score_run_valid_rate_guards_zero_candidates():
    res = score_run([], {}, policy="strict")
    assert res.valid_rate == 0.0
    assert res.n_candidates == 0
    assert res.scored is False
    assert res.score is None


def test_score_run_valid_rate_is_valid_over_candidates():
    checks = [check(f"w{i}") for i in range(7)] + [check("x", valid=False) for _ in range(3)]
    words = [f"w{i}" for i in range(7)]
    res = score_run(checks, orthonormal(words), policy="strict", n_use=7)
    assert res.valid_rate == pytest.approx(0.7)


def test_score_run_zero_norm_vector_refuses_instead_of_raising():
    checks, vecs = valid_run(7)
    vecs["w3"] = [0.0] * DIM
    res = score_run(checks, vecs, policy="strict", n_use=7)
    assert res.scored is False
    assert res.score is None
    assert res.reason and "embedding" in res.reason


def test_score_run_unknown_policy_raises():
    checks, vecs = valid_run(7)
    with pytest.raises(ValueError):
        score_run(checks, vecs, policy="generous")


# --- BaselineStats ---------------------------------------------------------


def stats(draws, **kw):
    xs = sorted(draws)
    defaults = dict(
        mean=sum(xs) / len(xs),
        sd=0.1,
        n=len(xs),
        k=7,
        p05=xs[0],
        p50=xs[len(xs) // 2],
        p95=xs[-1],
        seed=0,
        draws=tuple(draws),
    )
    return BaselineStats(**{**defaults, **kw})


def test_baseline_stats_constructs_positionally_in_contract_order():
    b = BaselineStats(0.38, 0.05, 1000, 7, 0.30, 0.38, 0.46, 0)
    assert (b.mean, b.sd, b.n, b.k, b.p05, b.p50, b.p95, b.seed) == (
        0.38, 0.05, 1000, 7, 0.30, 0.38, 0.46, 0,
    )


def test_percentile_of_is_fraction_of_draws_below():
    b = stats([i / 10 for i in range(10)])  # 0.0 .. 0.9
    assert b.percentile_of(0.55) == pytest.approx(0.6)
    assert b.percentile_of(-1.0) == pytest.approx(0.0)
    assert b.percentile_of(99.0) == pytest.approx(1.0)


def test_percentile_of_falls_back_to_normal_when_draws_absent():
    # out/baselines.json does not carry the draws; a reconstructed BaselineStats
    # must still answer rather than refuse.
    b = BaselineStats(0.40, 0.05, 1000, 7, 0.32, 0.40, 0.48, 0)
    assert b.percentile_of(0.40) == pytest.approx(0.5)
    assert b.percentile_of(0.45) == pytest.approx(0.8413, abs=1e-3)


def test_z_of_is_standard_score():
    b = stats([0.4, 0.5, 0.6], mean=0.5, sd=0.1)
    assert b.z_of(0.7) == pytest.approx(2.0)
    assert b.z_of(0.5) == pytest.approx(0.0)


def test_z_of_guards_zero_sd():
    b = stats([0.5, 0.5], mean=0.5, sd=0.0)
    assert math.isnan(b.z_of(0.9))


def test_percentile_of_guards_zero_sd_without_draws():
    b = BaselineStats(0.5, 0.0, 10, 7, 0.5, 0.5, 0.5, 0)
    assert math.isnan(b.percentile_of(0.9))


# --- random_baseline -------------------------------------------------------


VOCAB = [f"noun{i}" for i in range(40)]


def test_random_baseline_shape_and_bookkeeping():
    b = random_baseline(random_embedder(), VOCAB, n_draws=50, k=7, seed=3)
    assert (b.n, b.k, b.seed) == (50, 7, 3)
    assert len(b.draws) == 50
    assert b.p05 <= b.p50 <= b.p95
    assert b.sd > 0.0
    assert 0.0 <= b.mean <= 2.0


def test_random_baseline_is_reproducible_for_a_seed():
    a = random_baseline(random_embedder(), VOCAB, n_draws=30, k=7, seed=1)
    b = random_baseline(random_embedder(), VOCAB, n_draws=30, k=7, seed=1)
    c = random_baseline(random_embedder(), VOCAB, n_draws=30, k=7, seed=2)
    assert a.draws == b.draws
    assert a.draws != c.draws


def test_random_baseline_does_not_touch_global_random_state():
    random.seed(5)
    expected = [random.random() for _ in range(3)]

    random.seed(5)
    first = random_baseline(random_embedder(), VOCAB, n_draws=20, k=5, seed=9)
    assert [random.random() for _ in range(3)] == expected

    # And global state cannot influence the result either.
    random.seed(123456)
    second = random_baseline(random_embedder(), VOCAB, n_draws=20, k=5, seed=9)
    assert first.draws == second.draws


def test_random_baseline_embeds_the_vocab_once():
    calls: list[list[str]] = []
    random_baseline(random_embedder(calls=calls), VOCAB, n_draws=100, k=7, seed=0)
    assert len(calls) == 1  # 100 draws must not cost 100 round trips


def test_random_baseline_defaults_to_common_nouns():
    calls: list[list[str]] = []
    random_baseline(random_embedder(calls=calls), n_draws=5, k=7, seed=0)
    assert calls[0] == COMMON_NOUNS


def test_random_baseline_drops_words_the_embedder_cannot_represent():
    def partial(words):
        full = random_embedder()(words)
        out = {}
        for i, w in enumerate(words):
            if w.endswith("7"):
                continue  # missing entirely
            out[w] = [0.0] * 16 if i == 0 else full[w]  # first one is degenerate
        return out

    b = random_baseline(partial, VOCAB, n_draws=25, k=7, seed=0)
    assert b.n == 25  # no nan draws, no ValueError from the zero vector


def test_random_baseline_needs_enough_embeddable_words():
    with pytest.raises(ValueError):
        random_baseline(random_embedder(), ["a", "b", "c"], n_draws=10, k=7)


def test_random_baseline_rejects_degenerate_arguments():
    with pytest.raises(ValueError):
        random_baseline(random_embedder(), VOCAB, n_draws=0)
    with pytest.raises(ValueError):
        random_baseline(random_embedder(), VOCAB, k=1)
    with pytest.raises(ValueError):
        random_baseline(random_embedder(), [])


# --- category_floor --------------------------------------------------------


def test_category_floor_scores_each_category():
    cats = {"tight": ["a", "b", "c"], "spread": ["p", "q", "r"]}
    vecs = {"a": e(1), "b": e(1), "c": e(1), "p": e(1), "q": e(2), "r": e(3)}
    floors = category_floor(lambda words: {w: vecs[w] for w in words}, cats)
    assert floors == {"tight": pytest.approx(0.0), "spread": pytest.approx(1.0)}


def test_category_floor_uses_every_word_in_the_category():
    # 7 words, one duplicated pair -> 20/21, i.e. all 21 pairs were used.
    words = [f"w{i}" for i in range(7)]
    vecs = {"w0": e(1), "w1": e(1)}
    vecs.update({w: e(i + 2) for i, w in enumerate(words[2:])})
    floors = category_floor(lambda ws: {w: vecs[w] for w in ws}, {"c": words})
    assert floors["c"] == pytest.approx(20 / 21)


def test_category_floor_omits_categories_it_cannot_embed():
    cats = {"ok": ["a", "b"], "unknown": ["zz", "yy"]}
    vecs = {"a": e(1), "b": e(2)}
    floors = category_floor(lambda ws: {w: vecs[w] for w in ws if w in vecs}, cats)
    assert set(floors) == {"ok"}


def test_category_floor_defaults_to_default_categories():
    floors = category_floor(clustered_embedder())
    assert set(floors) == set(DEFAULT_CATEGORIES)


def test_category_floor_sits_below_chance():
    embed = clustered_embedder()
    baseline = random_baseline(embed, n_draws=200, k=7, seed=0)
    floors = category_floor(embed)
    for name, floor in floors.items():
        assert floor < baseline.mean, name
        assert baseline.percentile_of(floor) < 0.05, name


# --- shipped vocabularies --------------------------------------------------


def test_default_categories_are_seven_concrete_nouns_each():
    assert len(DEFAULT_CATEGORIES) >= 4
    for name, words in DEFAULT_CATEGORIES.items():
        assert len(words) == 7, name
        assert len(set(words)) == 7, name
        assert all(w.islower() and w.isalpha() for w in words), name


def test_common_nouns_is_a_usable_baseline_vocabulary():
    # Curated rather than /usr/share/dict: web2's archaic entries would put
    # "chance" somewhere no model actually operates.
    assert len(COMMON_NOUNS) >= 300
    assert len(set(COMMON_NOUNS)) == len(COMMON_NOUNS)
    assert all(w.islower() and w.isalpha() for w in COMMON_NOUNS)
