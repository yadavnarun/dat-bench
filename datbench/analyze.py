"""Statistics over scored runs: spread, replicate self-overlap, embedder agreement.

Pure functions over plain lists/dicts. numpy only; scipy is deliberately not a
dependency, so Spearman is implemented here.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Collection, Mapping, Sequence

import numpy as np

__all__ = [
    "bootstrap_ci",
    "cell_stats",
    "jaccard_self_overlap",
    "distinct_pool",
    "rank_correlation",
    "gaming_index",
]

NAN = float("nan")

_NUMERIC = (int, float, np.integer, np.floating)


def _to_float(x: object, *, where: str) -> float:
    # A refused run carries score=None (CONTRACT §1). numpy would quietly turn that
    # into nan and hand back a plausible-looking number, so refuse it loudly here:
    # the caller filtered its rows wrong.
    if isinstance(x, bool) or not isinstance(x, _NUMERIC):
        raise TypeError(
            f"{where}: expected a float, got {x!r} ({type(x).__name__}). "
            "None means an unscored run leaked into the cell -- fix the caller, "
            "do not average it in."
        )
    return float(x)


def _floats(xs: Sequence[float], *, where: str) -> np.ndarray:
    return np.asarray([_to_float(x, where=where) for x in xs], dtype=float)


def bootstrap_ci(
    xs: Sequence[float], *, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean. len(xs) < 2 -> (nan, nan)."""
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    vals = _floats(xs, where="bootstrap_ci")
    if vals.size < 2:
        return (NAN, NAN)
    # default_rng(seed) rather than the legacy global state: the CI printed in a
    # report has to be reproducible from the seed alone.
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    means = vals[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return (float(lo), float(hi))


def cell_stats(scores: Sequence[float]) -> dict:
    """{"n","mean","sd","ci_lo","ci_hi","min","max"} for one cell's replicates."""
    vals = _floats(scores, where="cell_stats")
    n = int(vals.size)
    if n == 0:
        return {
            "n": 0,
            "mean": NAN,
            "sd": NAN,
            "ci_lo": NAN,
            "ci_hi": NAN,
            "min": NAN,
            "max": NAN,
        }
    # ddof=1, and nan at n=1: a single replicate has no measurable spread, and
    # sd=0.0 would be read as perfect run-to-run consistency.
    sd = float(vals.std(ddof=1)) if n > 1 else NAN
    ci_lo, ci_hi = bootstrap_ci(vals)
    return {
        "n": n,
        "mean": float(vals.mean()),
        "sd": sd,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "min": float(vals.min()),
        "max": float(vals.max()),
    }


def jaccard_self_overlap(word_sets: Sequence[Collection[str]]) -> float:
    """Mean pairwise Jaccard over all C(n,2) replicate pairs. <2 sets -> nan.

    1.0 means every replicate emitted the identical set (total mode collapse).
    A pair of two empty sets is 0/0, i.e. undefined, and is reported as nan rather
    than as 0.0 (no overlap) or 1.0 (identical) -- both of those would be a claim
    about a model that in fact produced nothing. An undefined pair makes the whole
    cell's overlap nan; dropping it would hide the empty replicates.
    """
    sets = [set(ws) for ws in word_sets]
    if len(sets) < 2:
        return NAN
    pairwise = []
    for a, b in itertools.combinations(sets, 2):
        union = len(a | b)
        pairwise.append(len(a & b) / union if union else NAN)
    return float(np.mean(pairwise))


def distinct_pool(word_sets: Sequence[Collection[str]]) -> dict:
    """{"distinct","total","ratio"} -- vocabulary breadth across a cell."""
    sets = [set(ws) for ws in word_sets]
    total = sum(len(s) for s in sets)
    distinct = len(set().union(*sets)) if sets else 0
    return {
        "distinct": distinct,
        "total": total,
        # No words at all: the ratio is undefined, not 1.0 ("all fresh").
        "ratio": distinct / total if total else NAN,
    }


def _average_ranks(vals: np.ndarray) -> np.ndarray:
    """1-based ranks, ties sharing the average of the positions they span."""
    order = np.argsort(vals, kind="stable")
    ordered = vals[order]
    ranks = np.empty(vals.size, dtype=float)
    i = 0
    while i < vals.size:
        j = i
        while j + 1 < vals.size and ordered[j + 1] == ordered[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    xd = x - x.mean()
    yd = y - y.mean()
    denom = math.sqrt(float(xd @ xd) * float(yd @ yd))
    if denom == 0.0:
        return NAN  # a flat series has no ranking to agree or disagree with
    return float((xd @ yd) / denom)


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    return _pearson(_average_ranks(x), _average_ranks(y))


def rank_correlation(
    scores_by_embedder: Mapping[str, Mapping[str, float]],
) -> dict[tuple[str, str], float]:
    """Spearman rho for every embedder pair, over the models both of them scored.

    Answers "is the leaderboard real, or an artifact of one embedder?". Keys are
    (embedder_a, embedder_b) with a < b, each pair once. A pair sharing fewer than
    2 models -> nan; correlating one point is meaningless.
    """
    embedders = sorted(scores_by_embedder)
    out: dict[tuple[str, str], float] = {}
    for a, b in itertools.combinations(embedders, 2):
        sa, sb = scores_by_embedder[a], scores_by_embedder[b]
        shared = sorted(set(sa) & set(sb))
        if len(shared) < 2:
            out[(a, b)] = NAN
            continue
        xs = [_to_float(sa[m], where=f"rank_correlation[{a}][{m}]") for m in shared]
        ys = [_to_float(sb[m], where=f"rank_correlation[{b}][{m}]") for m in shared]
        out[(a, b)] = _spearman(xs, ys)
    return out


def gaming_index(mean_score: float, valid_rate: float) -> float:
    """mean_score * valid_rate -- a score discounted by how much of the output was legal.

    Reported BESIDE the raw score, never instead of it: down-weighting by validity
    is a judgement call about how much an invalid word should cost, and the reader
    has to be able to see both numbers and disagree with ours.
    """
    return _to_float(mean_score, where="gaming_index(mean_score)") * _to_float(
        valid_rate, where="gaming_index(valid_rate)"
    )
