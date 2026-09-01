from __future__ import annotations

import math

import pytest

from datbench.analyze import (
    bootstrap_ci,
    cell_stats,
    distinct_pool,
    gaming_index,
    jaccard_self_overlap,
    rank_correlation,
)


def isnan(x: object) -> bool:
    return isinstance(x, float) and math.isnan(x)


# --- bootstrap_ci ---------------------------------------------------------------

def test_bootstrap_ci_too_few_points():
    assert all(isnan(v) for v in bootstrap_ci([]))
    assert all(isnan(v) for v in bootstrap_ci([0.5]))


def test_bootstrap_ci_is_reproducible_from_the_seed():
    xs = [0.31, 0.44, 0.52, 0.39, 0.47, 0.41, 0.36, 0.50]
    a = bootstrap_ci(xs, n_boot=2000, seed=7)
    b = bootstrap_ci(xs, n_boot=2000, seed=7)
    assert a == b
    # ...and a different seed moves it, i.e. the seed is actually wired in.
    assert bootstrap_ci(xs, n_boot=2000, seed=8) != a


def test_bootstrap_ci_brackets_the_sample_mean():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    lo, hi = bootstrap_ci(xs, n_boot=4000, seed=0)
    assert lo < 5.0 < hi
    assert 1.0 <= lo < hi <= 9.0


def test_bootstrap_ci_narrower_at_wider_alpha():
    xs = [0.2, 0.4, 0.6, 0.8, 1.0, 0.5, 0.3, 0.7]
    wide_lo, wide_hi = bootstrap_ci(xs, n_boot=4000, alpha=0.05, seed=1)
    tight_lo, tight_hi = bootstrap_ci(xs, n_boot=4000, alpha=0.5, seed=1)
    assert (tight_hi - tight_lo) < (wide_hi - wide_lo)


def test_bootstrap_ci_zero_variance_sample():
    lo, hi = bootstrap_ci([0.42, 0.42, 0.42], n_boot=100, seed=0)
    assert lo == hi == pytest.approx(0.42)


def test_bootstrap_ci_rejects_bad_params():
    with pytest.raises(ValueError):
        bootstrap_ci([1.0, 2.0], n_boot=0)
    with pytest.raises(ValueError):
        bootstrap_ci([1.0, 2.0], alpha=0.0)
    with pytest.raises(ValueError):
        bootstrap_ci([1.0, 2.0], alpha=1.0)


def test_bootstrap_ci_rejects_none():
    with pytest.raises(TypeError):
        bootstrap_ci([0.4, None, 0.5])


# --- cell_stats -----------------------------------------------------------------

def test_cell_stats_uses_sample_sd():
    stats = cell_stats([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    assert stats["n"] == 8
    assert stats["mean"] == pytest.approx(5.0)
    # sum of squared deviations = 32; ddof=1 -> 32/7, not the population 32/8=4.
    assert stats["sd"] == pytest.approx(math.sqrt(32.0 / 7.0))
    assert stats["sd"] != pytest.approx(2.0)
    assert stats["min"] == 2.0
    assert stats["max"] == 9.0
    assert stats["ci_lo"] < 5.0 < stats["ci_hi"]


def test_cell_stats_single_run_has_nan_sd_not_zero():
    stats = cell_stats([0.37])
    assert stats["n"] == 1
    assert stats["mean"] == pytest.approx(0.37)
    assert isnan(stats["sd"])  # sd=0.0 would read as perfect consistency
    assert isnan(stats["ci_lo"]) and isnan(stats["ci_hi"])
    assert stats["min"] == stats["max"] == pytest.approx(0.37)


def test_cell_stats_empty_is_n_zero_not_a_crash():
    stats = cell_stats([])
    assert stats["n"] == 0
    for key in ("mean", "sd", "ci_lo", "ci_hi", "min", "max"):
        assert isnan(stats[key]), key


def test_cell_stats_keys_are_exactly_the_contract():
    assert set(cell_stats([1.0, 2.0])) == {
        "n", "mean", "sd", "ci_lo", "ci_hi", "min", "max"
    }


def test_cell_stats_rejects_none():
    with pytest.raises(TypeError):
        cell_stats([0.4, 0.5, None])


def test_cell_stats_accepts_ints():
    assert cell_stats([1, 2, 3])["mean"] == pytest.approx(2.0)


# --- jaccard_self_overlap -------------------------------------------------------

def test_jaccard_identical_sets_is_total_collapse():
    words = ["cat", "thimble", "quark"]
    assert jaccard_self_overlap([words, list(words), list(words)]) == pytest.approx(1.0)


def test_jaccard_disjoint_sets_is_zero():
    assert jaccard_self_overlap([{"a", "b"}, {"c", "d"}]) == pytest.approx(0.0)


def test_jaccard_averages_over_all_pairs():
    # pairs: (1,2)=1/3, (1,3)=2/2=1, (2,3)=1/3 -> mean 5/9
    got = jaccard_self_overlap([{"a", "b"}, {"b", "c"}, {"a", "b"}])
    assert got == pytest.approx(5.0 / 9.0)


def test_jaccard_dedupes_within_a_replicate():
    assert jaccard_self_overlap([["a", "a", "b"], ["b", "a"]]) == pytest.approx(1.0)


def test_jaccard_needs_two_sets():
    assert isnan(jaccard_self_overlap([]))
    assert isnan(jaccard_self_overlap([{"a"}]))


def test_jaccard_two_empty_sets_is_undefined():
    assert isnan(jaccard_self_overlap([set(), set()]))


def test_jaccard_one_empty_replicate_against_a_real_one_is_zero():
    assert jaccard_self_overlap([set(), {"a", "b"}]) == pytest.approx(0.0)


# --- distinct_pool --------------------------------------------------------------

def test_distinct_pool_counts_and_ratio():
    got = distinct_pool([{"a", "b"}, {"b", "c"}])
    assert got == {"distinct": 3, "total": 4, "ratio": pytest.approx(0.75)}


def test_distinct_pool_total_is_the_sum_of_set_sizes():
    got = distinct_pool([["a", "a", "b"], ["b"]])
    assert got["total"] == 3  # 2 + 1, duplicates inside a replicate collapsed
    assert got["distinct"] == 2


def test_distinct_pool_guards_empty():
    got = distinct_pool([])
    assert got["distinct"] == 0 and got["total"] == 0
    assert isnan(got["ratio"])
    assert isnan(distinct_pool([set(), set()])["ratio"])


# --- rank_correlation -----------------------------------------------------------

def test_rank_correlation_perfect_agreement():
    scores = {
        "emb-a": {"m1": 0.1, "m2": 0.2, "m3": 0.3},
        "emb-b": {"m1": 0.5, "m2": 0.6, "m3": 0.9},
    }
    assert rank_correlation(scores) == {("emb-a", "emb-b"): pytest.approx(1.0)}


def test_rank_correlation_perfect_disagreement():
    scores = {
        "a": {"m1": 0.1, "m2": 0.2, "m3": 0.3},
        "b": {"m1": 0.9, "m2": 0.5, "m3": 0.1},
    }
    assert rank_correlation(scores)[("a", "b")] == pytest.approx(-1.0)


def test_rank_correlation_ties_use_average_ranks():
    # x ranks with ties averaged: [1, 2.5, 2.5, 4]; y ranks: [1, 2, 3, 4].
    # Pearson on those = 4.5 / sqrt(4.5 * 5) = sqrt(0.9) = 0.9486832980505138.
    # Ordinal (non-averaged) ranks would give exactly 1.0, so this pins tie handling.
    scores = {
        "a": {"m1": 1.0, "m2": 2.0, "m3": 2.0, "m4": 4.0},
        "b": {"m1": 10.0, "m2": 20.0, "m3": 30.0, "m4": 40.0},
    }
    got = rank_correlation(scores)[("a", "b")]
    assert got == pytest.approx(math.sqrt(0.9))
    assert got != pytest.approx(1.0)


def test_rank_correlation_ties_on_both_sides():
    # x ranks [1.5, 1.5, 3.5, 3.5], y ranks [1.5, 3.5, 1.5, 3.5]:
    # deviations (-1,-1,1,1) and (-1,1,-1,1) -> covariance 0.
    scores = {
        "a": {"m1": 0.1, "m2": 0.1, "m3": 0.2, "m4": 0.2},
        "b": {"m1": 0.5, "m2": 0.7, "m3": 0.5, "m4": 0.7},
    }
    assert rank_correlation(scores)[("a", "b")] == pytest.approx(0.0)


def test_rank_correlation_flat_series_is_undefined():
    scores = {
        "a": {"m1": 0.4, "m2": 0.4, "m3": 0.4},
        "b": {"m1": 0.1, "m2": 0.2, "m3": 0.3},
    }
    assert isnan(rank_correlation(scores)[("a", "b")])


def test_rank_correlation_only_shared_models():
    scores = {
        "a": {"m1": 0.1, "m2": 0.2, "m3": 0.3, "only_a": 99.0},
        "b": {"m1": 0.5, "m2": 0.6, "m3": 0.9, "only_b": -99.0},
    }
    assert rank_correlation(scores)[("a", "b")] == pytest.approx(1.0)


def test_rank_correlation_fewer_than_two_shared_models_is_nan():
    scores = {
        "a": {"m1": 0.1, "m2": 0.2},
        "b": {"m1": 0.5, "m9": 0.6},
    }
    assert isnan(rank_correlation(scores)[("a", "b")])
    assert isnan(rank_correlation({"a": {"m1": 0.1}, "b": {}})[("a", "b")])


def test_rank_correlation_pairs_are_sorted_and_unique():
    scores = {
        "z-emb": {"m1": 0.1, "m2": 0.2},
        "a-emb": {"m1": 0.3, "m2": 0.1},
        "m-emb": {"m1": 0.5, "m2": 0.9},
    }
    got = rank_correlation(scores)
    assert set(got) == {
        ("a-emb", "m-emb"), ("a-emb", "z-emb"), ("m-emb", "z-emb")
    }
    assert all(a < b for a, b in got)


def test_rank_correlation_fewer_than_two_embedders():
    assert rank_correlation({}) == {}
    assert rank_correlation({"solo": {"m1": 0.1, "m2": 0.2}}) == {}


def test_rank_correlation_rejects_none_scores():
    scores = {"a": {"m1": 0.1, "m2": None}, "b": {"m1": 0.5, "m2": 0.6}}
    with pytest.raises(TypeError):
        rank_correlation(scores)


# --- gaming_index ---------------------------------------------------------------

def test_gaming_index_multiplies():
    assert gaming_index(0.5, 0.8) == pytest.approx(0.40)
    assert gaming_index(0.5, 1.0) == pytest.approx(0.50)
    assert gaming_index(0.5, 0.0) == pytest.approx(0.0)


def test_gaming_index_penalises_the_rare_word_winner():
    # A model that wins on raw score but emits half-invalid words loses to a
    # slightly lower-scoring model that plays by the rules.
    assert gaming_index(0.60, 0.5) < gaming_index(0.50, 0.95)


def test_gaming_index_rejects_none():
    with pytest.raises(TypeError):
        gaming_index(None, 0.9)
    with pytest.raises(TypeError):
        gaming_index(0.5, None)
