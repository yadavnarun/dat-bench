"""Render a scored benchmark run as Markdown / HTML.

This module is as much about honesty as presentation. Anything that would let a
reader mistake an arbitrary-scale local-embedding number for a published DAT
score, or a degraded run for a clean one, is a bug here.

Nothing is computed from raw runs: the whole report is driven by the `summary`
dict that `analyze`/`cli` produce, plus a `meta` dict of provenance. Every
optional key is read with `.get` and a fallback, so a partial run still renders.

SUMMARY SCHEMA
==============
`build_markdown(summary, meta)` / `build_html(summary, meta)` consume the dict
below. Every key is OPTIONAL unless marked REQUIRED-ish; a missing key degrades
one row/section into "not recorded" rather than raising. Floats may be `None` or
`nan` anywhere a statistic is undefined (both render as an em dash).

summary = {
  # ---- what this run was ------------------------------------------------
  "schema_version": 1,                    # int, bump if this shape changes
  "primary_embedder": "text-embedding-qwen3-embedding-4b",
                                          # str: the embedder the leaderboard is built from
  "primary_policy": "strict",             # str: "strict" | "lenient"
  "embedders": ["emb-a", "emb-b"],        # list[str], display order for the matrix
  "policies": ["strict", "lenient"],      # list[str], which policies were scored
  "n_replicates": 10,                     # int: replicates per cell (the "n=10" claim)
  "n_use": 7,                             # int: words scored per run (C(7,2)=21 pairs)
  "prompts": ["verbatim", "terse"],       # list[str], column/row order for grids
  "temperatures": [0.0, 0.7, 1.0],        # list[float | None]; None = model has no temperature

  # ---- which validity checks actually ran (validate.capabilities()) -----
  "capabilities": {"dictionary": True, "wordnet": False, "wordfreq": True},
                                          # dict[str, bool]. ABSENT => the report says so, loudly.

  # ---- chance baselines, one per embedder (out/baselines.json) ----------
  "baselines": {
    "emb-a": {
      "mean": 0.38, "sd": 0.05, "n": 1000, "k": 7,          # score.BaselineStats fields
      "p05": 0.30, "p50": 0.38, "p95": 0.46, "seed": 0,
      "categories": {"animals": 0.14, "tools": 0.19},        # optional: tight-set floors
      "paper_example": 0.41,                                  # optional: DAT paper's 7 words
    },
  },

  # ---- run accounting (runner.run_all + score stage) --------------------
  "run_counts": {"attempted": 480, "written": 478, "skipped": 0, "errors": 2,
                 "failed": 2, "refused": 31, "scored": 445},
                                          # dict[str, int]; all keys optional

  # ---- leaderboard: one entry per model, best-prompt cell, best first ---
  # Sorted best-first by the producer; the report re-sorts defensively on "mean".
  "leaderboard": [
    {
      "model_id": "gemma-4-e4b",                  # str  (REQUIRED-ish: the row label)
      "model_reported": ["google/gemma-4-e4b"],   # list[str] | str: what the API served
      "best_prompt": "maxcreative",               # str
      "best_temperature": 1.0,                    # float | None
      "n": 10,                                    # int: replicates in that cell
      "mean": 0.4123,                             # float: cell mean (raw distance, NOT x100)
      "sd": 0.021,                                # float
      "ci_lo": 0.399, "ci_hi": 0.425,             # float: 95% bootstrap CI of the mean
      "percentile_vs_chance": 0.93,               # float 0..1 vs this embedder's random
                                                  #   baseline; if absent, derived from
                                                  #   baselines[primary_embedder]
      "z_vs_chance": 1.61,                        # float; if absent, computed from the baseline
      "valid_rate": 0.90,                         # float in 0..1, mean over the model's runs
      "gaming_index": 0.371,                      # float = mean * valid_rate (analyze.gaming_index)
      "jaccard": 0.32,                            # float: replicate overlap (mode collapse)
      "distinct_pool": {"distinct": 54, "total": 90, "ratio": 0.60},   # analyze.distinct_pool
      "n_runs": 120, "n_scored": 111, "n_refused": 7, "n_failed": 2,   # int denominators
      "n_truncated": 0,                           # int: replies that hit max_tokens
                                                  #   (finish_reason="length") -- a harness
                                                  #   limit, not a model result
    },
  ],

  # ---- per-model detail, keyed by model_id -----------------------------
  "models": {
    "gemma-4-e4b": {
      "model_reported": ["google/gemma-4-e4b"],
      "n_runs": 120, "n_scored": 111, "n_refused": 7, "n_failed": 2,
      "n_truncated": 0,
      "valid_rate": 0.90,
      "truncated_by_prompt": [["cot", 9], ["verbatim", 2]],   # optional list[[prompt, count]]
      "notes": "local via LM Studio",       # optional, from ModelSpec.notes
      "refusal_reasons": [["only 5 valid words (strict needs 7)", 5],
                          ["fewer than 2 words have vectors", 2]],
                                            # optional list[[reason, count]] | dict[str, int]
      "grid": [                             # prompt x temperature cells, any order
        {"prompt_id": "verbatim", "temperature": 0.0,
         "n": 10, "mean": 0.401, "sd": 0.02, "ci_lo": 0.39, "ci_hi": 0.41,
         "valid_rate": 0.91, "jaccard": 0.30, "n_refused": 1, "n_failed": 0},
      ],
    },
  },

  # ---- embedder agreement (analyze.rank_correlation) -------------------
  # Accepted in three shapes, because the in-memory form has tuple keys and JSON
  # cannot: list[{"a","b","rho","n_models"}]  |  {("a","b"): rho}  |  {"a|b": rho}.
  "rank_correlation": [{"a": "emb-a", "b": "emb-b", "rho": 0.91, "n_models": 6}],

  # ---- optional extras --------------------------------------------------
  "flag_counts": {"rare": 12, "proper_noun": 3, "not_noun": 8},   # dict[str, int]
  "warnings": ["only 2 of 4 embedders reachable"],                # list[str]
}

meta = {
  "generated_at": "2026-08-31T12:00:00Z",   # str; defaults to now(UTC)
  "git_commit": "a1b2c3d",                  # str | None; probed from git only if the key is absent
  "git_dirty": False,                       # bool
  "tool_version": "0.1.0",
  "command": "python -m datbench report",
  "models_yaml": "models.yaml",
  "embed_base_url": "http://localhost:1234/v1",
  "out_dir": "out",
  "title": "dat-bench report",              # str, overrides the default H1
  "notes": "free text appended to provenance",
}
"""

from __future__ import annotations

import html as _htmllib
import math
import re
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

__all__ = ["build_markdown", "build_html"]

DASH = "\u2014"

# The published-norm numbers are quoted verbatim in the warning banner: a reader
# comparing 0.41 against 78 is the exact failure mode this report exists to stop.
HUMAN_NORM = "mean \u224878, range ~50\u201395"

_CAP_MEANING = {
    "dictionary": "`not_in_dict` \u2014 is it an English word at all",
    "wordnet": "`not_noun` and `proper_noun` \u2014 is it a noun, is it a name",
    "wordfreq": "`rare` \u2014 Zipf frequency vs the specialised-vocabulary threshold",
}

LIMITATIONS = [
    "**The `rare` flag is a weak proxy for \u201cspecialised vocabulary\u201d.** It is a Zipf "
    "frequency cut, and frequency is not technicality. *quark* scores Zipf 3.05 and "
    "*photon* 3.32, so both pass unflagged despite being physics jargon \u2014 while "
    "*thimble*, ordinary household vocabulary and the DAT paper's own example word, sits "
    "at 2.53 and clears the 2.5 threshold by 0.03. The cut is doing very little work at "
    "the boundary: technical words routinely slip through, and a slightly higher "
    "threshold would start rejecting legitimate common nouns. Read `valid_rate` and "
    "`gaming_index` as noisy indicators, not verdicts.",
    "**A deterministic cell has an effective sample size of 1, whatever `n` says.** "
    "At temperature 0, or under total mode collapse, every replicate returns the same "
    "words; the bootstrap then produces a zero-width interval that looks like extreme "
    "precision but rests on one observation. Such cells are marked "
    "`[degenerate: n_eff=1]` rather than given a CI. This interacts badly with "
    "best-cell selection: a deterministic cell has no sampling noise to pull its mean "
    "down, so the maximum over cells is drawn toward T=0 — check the Jaccard column "
    "before reading a best cell as capability. Compare models on a cell where they "
    "actually vary, and raise `n` only for cells whose Jaccard is below 1.",
    "**The local embedders compress the far end of the distance range.** Measured "
    "against GloVe 840B-300d on the DAT paper's own example pair: GloVe puts "
    "*cat*/*thimble* at 0.879, while `qwen3-embedding-4b` puts it at 0.396 — yet the two "
    "agree closely on the *near* pair *cat*/*dog* (0.198 vs 0.204). The compression is "
    "roughly 2.2x and it falls exactly where the DAT discriminates, so differences "
    "between two genuinely good models are squeezed into a narrower band here than on "
    "the published instrument. Practical consequence: resolving close models needs more "
    "replicates than it would with GloVe. If two CIs overlap, that is as likely to be "
    "the scale as the models. (Reproduce with `scripts/premise_check.py`.)",
    "**Temperature is not comparable across providers.** The same nominal `temperature` "
    "means different sampling behaviour on different stacks (different default top-p, "
    "different logit post-processing, some providers clamp or ignore it). Read the "
    "temperature grid within a model, never across models.",
    "__N_LIMITATION__",
    "**Scores are one embedder's opinion of semantic distance.** They inherit its biases "
    "(morphology, register, tokenisation). The cross-embedder agreement section is the "
    "only check on that; if it is weak, the ordering is not a fact about the models.",
    "**Refused runs are excluded, not scored zero.** That is the right call statistically "
    "\u2014 a floor of 0 would reward invalid output \u2014 but it means the leaderboard mean is "
    "conditional on producing scorable output. Always read it next to the refusal count.",
    "**A model can be prompted into a better number.** Only the best-prompt cell reaches "
    "the leaderboard, so the headline is an upper bound over the prompts tried, not a "
    "measure of typical behaviour.",
]


def _n_label(summary: Mapping[str, Any]) -> str:
    n = summary.get("n_replicates")
    return str(n) if isinstance(n, int) and n > 0 else "this n"


def _limitations(summary: Mapping[str, Any]) -> list[str]:
    """LIMITATIONS with the replicate-count bullet filled in from the actual run.

    Hardcoding "n=10" here would state a false denominator on every run that used a
    different n -- the exact kind of unearned precision this report exists to avoid.
    """
    n = summary.get("n_replicates")
    if isinstance(n, int) and n > 0:
        if n < 10:
            tail = (
                f"**n={n} replicates is a small sample and gives wide confidence intervals "
                f"on every cell.** At this n almost no prompt-to-prompt or "
                f"temperature-to-temperature difference here is resolvable; treat the grid "
                f"as indicative and raise n before drawing conclusions from it."
            )
        else:
            tail = (
                f"**n={n} replicates still gives wide confidence intervals on any single "
                f"cell.** Most prompt-to-prompt and temperature-to-temperature differences "
                f"here are inside the noise. If two CIs overlap, treat the cells as tied."
            )
    else:
        tail = (
            "**Replicate count unknown for this run**, so the confidence intervals cannot "
            "be interpreted. If two CIs overlap, treat the cells as tied."
        )
    return [tail if item == "__N_LIMITATION__" else item for item in LIMITATIONS]


# --------------------------------------------------------------------------- #
# scalars
# --------------------------------------------------------------------------- #
def _missing(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    return False


def _num(x: Any, nd: int = 3) -> str:
    if _missing(x):
        return DASH
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _int(x: Any) -> str:
    if _missing(x):
        return DASH
    try:
        return f"{int(x):d}"
    except (TypeError, ValueError):
        return str(x)


def _pct(x: Any, nd: int = 0) -> str:
    """0..1 fractions render as percentages; anything >1 is assumed already a percent."""
    if _missing(x):
        return DASH
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if -1.0 <= v <= 1.0:
        v *= 100.0
    return f"{v:.{nd}f}%"


def _temp(t: Any) -> str:
    # supports_temperature=False collapses to a single cell recorded as None
    # (CONTRACT §8). "n/a" says that; "0" would invent a setting we never sent.
    if t is None:
        return "n/a"
    try:
        return f"{float(t):g}"
    except (TypeError, ValueError):
        return str(t)


def _is_degenerate(entry: Mapping[str, Any]) -> bool:
    """Did every replicate in this cell return the same words?

    A deterministic cell (temperature 0, or a model in total mode collapse) repeats
    one answer n times. Bootstrapping that yields a zero-width interval, and
    printing it as a 95% CI claims precision from a single observation. Observed
    live: gemma-4-e4b at terse/T=0 gave Jaccard 1.00 and "0.401 [0.401, 0.401]"
    across 10 replicates -- effective n of 1, displayed as n=10.
    """
    lo, hi = entry.get("ci_lo"), entry.get("ci_hi")
    if not _missing(lo) and not _missing(hi):
        try:
            if float(hi) - float(lo) <= 1e-9:
                return True
        except (TypeError, ValueError):
            pass
    sd = entry.get("sd")
    if not _missing(sd):
        try:
            if float(sd) <= 1e-12:
                return True
        except (TypeError, ValueError):
            pass
    jac = entry.get("jaccard")
    if not _missing(jac):
        try:
            if float(jac) >= 0.999:
                return True
        except (TypeError, ValueError):
            pass
    return False


def _mean_ci(entry: Mapping[str, Any]) -> str:
    mean = _num(entry.get("mean"))
    lo, hi = entry.get("ci_lo"), entry.get("ci_hi")
    if _missing(lo) or _missing(hi):
        return f"{mean} [CI {DASH}]"
    if _is_degenerate(entry):
        # Flagged rather than silently printed: a zero-width interval is the one
        # case where the CI actively misleads.
        return f"{mean} [degenerate: n_eff=1]"
    return f"{mean} [{_num(lo)}, {_num(hi)}]"


def _reported(value: Any) -> str:
    if _missing(value):
        return DASH
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence):
        seen = [str(v) for v in value if v]
        return ", ".join(dict.fromkeys(seen)) or DASH
    return str(value)


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# --------------------------------------------------------------------------- #
# summary readers
# --------------------------------------------------------------------------- #
def _baseline(summary: Mapping[str, Any], embedder: Any) -> Mapping[str, Any]:
    baselines = summary.get("baselines") or {}
    if not isinstance(baselines, Mapping):
        return {}
    got = baselines.get(embedder) if embedder is not None else None
    if got is None and len(baselines) == 1:
        # Single-embedder run whose primary_embedder label drifted: better to use
        # the one baseline we have than to print "no baseline" beside a score.
        got = next(iter(baselines.values()))
    if isinstance(got, Mapping):
        # baselines.json nests the stats under "random" (CONTRACT §1); accept
        # both that and a flattened dict.
        inner = got.get("random")
        merged = dict(got)
        if isinstance(inner, Mapping):
            merged.update(inner)
        return merged
    return {}


def _chance(entry: Mapping[str, Any], base: Mapping[str, Any]) -> tuple[Any, Any]:
    """(percentile, z) vs chance, preferring what the producer recorded."""
    pct = entry.get("percentile_vs_chance")
    z = entry.get("z_vs_chance")
    mean, b_mean, b_sd = entry.get("mean"), base.get("mean"), base.get("sd")
    if _missing(pct) or _missing(z):
        if not _missing(mean) and not _missing(b_mean) and not _missing(b_sd):
            try:
                sd = float(b_sd)
                if sd > 0.0:
                    zz = (float(mean) - float(b_mean)) / sd
                    if _missing(z):
                        z = zz
                    if _missing(pct):
                        # Normal approximation: the empirical draws are not in
                        # summary.json, only mean/sd/percentiles are.
                        pct = _normal_cdf(zz)
            except (TypeError, ValueError):
                pass
    return pct, z


def _leaderboard(summary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = summary.get("leaderboard")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return []
    entries = [r for r in rows if isinstance(r, Mapping)]
    # Producer sorts best-first; re-sort so a hand-assembled or partially scored
    # summary still shows a ranking. Missing means sink to the bottom.
    return sorted(
        entries,
        key=lambda e: (
            _missing(e.get("mean")),
            -(float(e["mean"]) if not _missing(e.get("mean")) else 0.0),
            str(_model_id(e)),
        ),
    )


def _model_id(entry: Mapping[str, Any]) -> str:
    for key in ("model_id", "model", "id"):
        v = entry.get(key)
        if v:
            return str(v)
    return "?"


def _counts(*sources: Any) -> dict[str, Any]:
    """First non-missing value for each denominator key, across fallback sources."""
    keys = ("n_runs", "n_scored", "n_refused", "n_failed", "n_truncated", "valid_rate")
    out: dict[str, Any] = {}
    for key in keys:
        for src in sources:
            if isinstance(src, Mapping) and not _missing(src.get(key)):
                out[key] = src[key]
                break
    return out


def _rho_pairs(summary: Mapping[str, Any]) -> list[tuple[str, str, Any, Any]]:
    raw = summary.get("rank_correlation")
    pairs: list[tuple[str, str, Any, Any]] = []
    if isinstance(raw, Mapping):
        for key, rho in raw.items():
            a = b = None
            if isinstance(key, tuple | list) and len(key) == 2:
                a, b = str(key[0]), str(key[1])
            elif isinstance(key, str):
                for sep in ("::", "|", "__vs__"):
                    if sep in key:
                        a, b = key.split(sep, 1)
                        break
            if a is None or b is None:
                continue
            if isinstance(rho, Mapping):
                pairs.append((a, b, rho.get("rho"), rho.get("n_models")))
            else:
                pairs.append((a, b, rho, None))
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            a = item.get("a") or item.get("embedder_a") or item.get("x")
            b = item.get("b") or item.get("embedder_b") or item.get("y")
            rho = item.get("rho", item.get("spearman"))
            if a and b:
                pairs.append((str(a), str(b), rho, item.get("n_models")))
    return pairs


def _verdict(pairs: Sequence[tuple[str, str, Any, Any]]) -> str:
    rhos = [float(rho) for _a, _b, rho, _n in pairs if not _missing(rho)]
    if not rhos:
        return (
            "**Robustness to embedder choice: cannot be assessed.** Fewer than two "
            "embedders produced a comparable ranking, so nothing here rules out the "
            "ordering being an artifact of a single embedding model."
        )
    # With two models Spearman can only be +1 or -1, and with three it takes only a
    # handful of values -- so "rho = 1.00, the ranking is robust" is arithmetic, not
    # evidence. Observed live: 2 models gave rho = 1.00 across all 6 embedder pairs
    # while the two means differed by 0.005. Refuse the verdict rather than launder
    # a tautology as a robustness check.
    n_models = max(
        (int(n) for _a, _b, _r, n in pairs if isinstance(n, (int, float)) and n),
        default=0,
    )
    if n_models and n_models < 4:
        return (
            f"**Robustness to embedder choice: cannot be assessed — only "
            f"{n_models} model(s) ranked.** Spearman ρ over {n_models} items can "
            f"take only a few values (with 2 it is always ±1), so the "
            f"ρ = {_num(max(rhos), 2)} above is a property of the arithmetic, not "
            f"evidence that the embedders agree. Benchmark at least 4 models before "
            f"reading this section as a robustness check."
        )
    worst = min(rhos)
    if worst >= 0.9:
        head = "the ranking is robust"
        tail = (
            "every embedder pair agrees almost exactly, so the ordering is not an "
            "artifact of one embedder"
        )
    elif worst >= 0.7:
        head = "the ranking is broadly robust"
        tail = (
            "the top and bottom hold up, but adjacent models reorder between "
            "embedders \u2014 do not read small gaps as real"
        )
    elif worst >= 0.4:
        head = "the ranking is embedder-sensitive"
        tail = (
            "different embedders order these models materially differently; quote a "
            "ranking only together with the embedder it came from"
        )
    else:
        head = "the ranking is **not** robust"
        tail = (
            "embedders disagree about the ordering, which means this leaderboard is "
            "largely an artifact of the embedder chosen"
        )
    return (
        f"**Robustness to embedder choice: {head}** (worst pairwise Spearman "
        f"\u03c1 = {_num(worst, 2)} over {len(rhos)} pair(s)) \u2014 {tail}."
    )


def _git_commit() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    out = proc.stdout.strip()
    return out if proc.returncode == 0 and out else None


# --------------------------------------------------------------------------- #
# view model
# --------------------------------------------------------------------------- #
# Markdown and HTML render the SAME view model, so the two files cannot drift
# into stating different things -- which for the honesty sections is the point.
def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]], *, note: str = "") -> dict:
    return {
        "kind": "table",
        "headers": [str(h) for h in headers],
        "rows": [[str(c) for c in row] for row in rows],
        "note": note,
    }


def _para(text: str) -> dict:
    return {"kind": "para", "text": text}


def _bullets(items: Sequence[str]) -> dict:
    return {"kind": "bullets", "items": [str(i) for i in items]}


def _callout(title: str, lines: Sequence[str], *, tone: str = "warn") -> dict:
    return {"kind": "callout", "tone": tone, "title": title, "lines": [str(x) for x in lines]}


def _section(heading: str, blocks: Sequence[dict], *, level: int = 2) -> dict:
    return {"heading": heading, "level": level, "blocks": [b for b in blocks if b]}


def _scale_warning(summary: Mapping[str, Any]) -> dict:
    emb = summary.get("primary_embedder") or ", ".join(
        str(e) for e in (summary.get("embedders") or [])
    )
    emb_txt = f"`{emb}`" if emb else "local embedding models"
    return _callout(
        "Read this first: these are NOT DAT scores",
        [
            f"Scores here are mean pairwise cosine distances from **local LM Studio "
            f"embeddings** ({emb_txt}) \u2014 **not** the **GloVe 840B-300d** vectors used by "
            "the published Divergent Association Task.",
            f"They are therefore on an **arbitrary scale** and are **NOT comparable** to "
            f"the published DAT human norms ({HUMAN_NORM}). No \u00d7100 score is printed "
            "anywhere in this report, deliberately: a number near 78 here would be a "
            "coincidence, not a result.",
            "The interpretable number is the **percentile vs chance** beside each score, "
            "measured against that embedder's random-noun baseline (below). A raw mean "
            "distance on its own says nothing.",
            "Comparisons are valid **within this run only** \u2014 same embedder, same policy, "
            "same prompt set. Check the cross-embedder agreement section before quoting "
            "any ordering.",
        ],
    )


def _capabilities_section(summary: Mapping[str, Any]) -> dict:
    caps = summary.get("capabilities")
    blocks: list[dict] = []
    if not isinstance(caps, Mapping) or not caps:
        blocks.append(
            _callout(
                "Validity checks: NOT RECORDED",
                [
                    "This summary carries no `validate.capabilities()` block, so **which "
                    "validity checks actually ran is unknown**. Treat every `valid_rate`, "
                    "`gaming_index` and refusal count below as unverified: a run with "
                    "WordNet or wordfreq missing produces the same shaped numbers as a "
                    "clean one.",
                ],
            )
        )
        return _section("Which checks actually ran", blocks)

    rows = []
    missing = []
    for name, ok in caps.items():
        got = bool(ok)
        rows.append(
            [
                f"`{name}`",
                "yes" if got else "**NO**",
                _CAP_MEANING.get(str(name), "\u2014"),
            ]
        )
        if not got:
            missing.append(str(name))
    blocks.append(_table(["capability", "loaded", "checks it powers"], rows))
    if missing:
        blocks.append(
            _callout(
                "Degraded run: some validity checks did not run",
                [
                    "Missing: "
                    + ", ".join(f"`{m}`" for m in missing)
                    + ". The checks those power were **skipped, not passed** \u2014 words that "
                    "would have failed them were counted as valid.",
                    "So `valid_rate` is an **upper bound**, `gaming_index` is inflated, and "
                    "policy refusals are undercounted. This run is not equivalent to one "
                    "with all checks present and must not be compared to one.",
                ],
            )
        )
    else:
        blocks.append(
            _callout(
                "All validity checks ran",
                ["Dictionary, WordNet and wordfreq all loaded, so no DAT rule was skipped."],
                tone="ok",
            )
        )
    return _section("Which checks actually ran", blocks)


def _headline_section(summary: Mapping[str, Any]) -> dict | None:
    """State outright where the models sit against chance.

    Without this the reader is left to compare two raw distances and conclude the
    higher one is better. Observed live: the best of 24 cells scored 0.401 while the
    95th percentile of random seven-noun draws was 0.411 -- so the whole grid sat
    inside the chance distribution, and "gemma 0.401 beats lfm2.5 0.396" would have
    been reported as a finding. The comparison that matters is against chance, not
    between models.
    """
    emb = str(summary.get("primary_embedder") or "")
    base = summary.get("baselines")
    if not isinstance(base, Mapping):
        return None
    stats = base.get(emb)
    if isinstance(stats, Mapping) and isinstance(stats.get("random"), Mapping):
        stats = stats["random"]
    if not isinstance(stats, Mapping):
        return None
    p95, cmean = stats.get("p95"), stats.get("mean")
    if _missing(p95) or _missing(cmean):
        return None

    models = summary.get("models")
    cells: list[float] = []
    if isinstance(models, Mapping):
        for detail in models.values():
            if not isinstance(detail, Mapping):
                continue
            for cell in detail.get("grid") or []:
                if isinstance(cell, Mapping) and not _missing(cell.get("mean")):
                    try:
                        cells.append(float(cell["mean"]))
                    except (TypeError, ValueError):
                        pass
    if not cells:
        return None

    p95f, cmeanf = float(p95), float(cmean)
    above95 = [c for c in cells if c > p95f]
    abovemean = [c for c in cells if c > cmeanf]
    best = max(cells)

    lines = [
        f"Scored by `{emb}`, seven random common nouns average "
        f"**{_num(cmeanf)}** with a 95th percentile of **{_num(p95f)}**. "
        f"Of **{len(cells)}** prompt x temperature cells in this run, "
        f"**{len(abovemean)}** beat the chance *mean* and "
        f"**{len(above95)}** beat chance *p95*. Best cell: **{_num(best)}**.",
    ]
    if not above95:
        lines.append(
            "**No cell cleared the 95th percentile of chance.** On this scoring, no "
            "model here produced word sets reliably more spread out than seven nouns "
            "drawn at random — so differences *between* the models are not the "
            "story, and a leaderboard ordering among them should not be read as a "
            "capability ranking. They do sit well above the tight-category floors "
            "(see below), so this is not a failure to understand the task; it is a "
            "failure to beat random on it."
        )
    elif len(above95) < len(cells) / 2:
        lines.append(
            f"Only {len(above95)} of {len(cells)} cells cleared chance p95, so most "
            f"of this grid is indistinguishable from random word draws. Read the "
            f"percentile column before comparing models."
        )
    else:
        lines.append(
            f"{len(above95)} of {len(cells)} cells cleared chance p95, so the task "
            f"is discriminating here — model differences are worth reading, "
            f"subject to the CI caveats below."
        )
    return _section("Headline: how do these compare to chance?", [_bullets(lines)])


def _baseline_section(summary: Mapping[str, Any]) -> dict:
    baselines = summary.get("baselines")
    if not isinstance(baselines, Mapping) or not baselines:
        return _section(
            "Chance baselines",
            [
                _callout(
                    "No random-noun baseline recorded",
                    [
                        "Without a chance distribution, the scores below are **uninterpretable "
                        "numbers** \u2014 there is no way to tell a good score from the value you "
                        "get by naming seven nouns at random. Re-run the score stage with "
                        "`--baseline-draws` set.",
                    ],
                )
            ],
        )
    order = [str(e) for e in (summary.get("embedders") or [])] or sorted(
        str(k) for k in baselines
    )
    seen: list[str] = []
    for name in order:
        if name in {str(k) for k in baselines} and name not in seen:
            seen.append(name)
    for k in baselines:
        if str(k) not in seen:
            seen.append(str(k))

    rows = []
    extras: list[str] = []
    for name in seen:
        base = _baseline(summary, name)
        if not base:
            continue
        rows.append(
            [
                f"`{name}`",
                _int(base.get("k")),
                _int(base.get("n")),
                _num(base.get("mean")),
                _num(base.get("sd")),
                _num(base.get("p05")),
                _num(base.get("p50")),
                _num(base.get("p95")),
                _int(base.get("seed")),
            ]
        )
        cats = base.get("categories")
        if isinstance(cats, Mapping) and cats:
            floors = ", ".join(f"{k} {_num(v)}" for k, v in cats.items())
            extras.append(f"`{name}` tight-set floors: {floors}.")
        if not _missing(base.get("paper_example")):
            extras.append(
                f"`{name}` scores the DAT paper's own example words at "
                f"{_num(base.get('paper_example'))}."
            )
    blocks = [
        _para(
            "Seven **random common nouns**, scored by the same embedder, repeated `n` "
            "times. This is the zero point: a model that beats chance by little has not "
            "done the task, whatever its raw number looks like."
        ),
        _table(
            ["embedder", "k", "draws", "chance mean", "sd", "p05", "p50", "p95", "seed"],
            rows,
        ),
    ]
    if extras:
        blocks.append(_bullets(extras))
    blocks.append(
        _para(
            "Category floors are semantically tight sets (seven animals, seven tools) "
            "\u2014 an anchor for what a *low* score looks like on this embedder's scale."
        )
    )
    return _section("Chance baselines (per embedder)", blocks)


def _leaderboard_section(summary: Mapping[str, Any]) -> dict:
    entries = _leaderboard(summary)
    emb = summary.get("primary_embedder")
    policy = summary.get("primary_policy")
    models = summary.get("models") if isinstance(summary.get("models"), Mapping) else {}
    if not entries:
        return _section(
            "Leaderboard",
            [_para("No scored cells in this summary \u2014 nothing to rank.")],
        )
    base = _baseline(summary, emb)
    rows = []
    for i, e in enumerate(entries, start=1):
        mid = _model_id(e)
        pct, z = _chance(e, base)
        pool = e.get("distinct_pool") if isinstance(e.get("distinct_pool"), Mapping) else {}
        c = _counts(e, models.get(mid))
        rows.append(
            [
                _int(i) if not _missing(e.get("mean")) else DASH,
                f"`{mid}`",
                f"{e.get('best_prompt', DASH)} @ T={_temp(e.get('best_temperature'))}",
                _mean_ci(e),
                _pct(pct, 1),
                _num(z, 2),
                _pct(c.get("valid_rate")),
                _num(e.get("gaming_index")),
                _num(e.get("jaccard"), 2),
                f"{_int(pool.get('distinct'))}/{_int(pool.get('total'))}",
                f"{_int(c.get('n_scored'))} / {_int(c.get('n_refused'))}"
                f" / {_int(c.get('n_failed'))}",
            ]
        )
    note = (
        f"embedder `{emb or DASH}`, policy `{policy or DASH}`, "
        f"n={_int(summary.get('n_replicates'))} replicates per cell, "
        f"first {_int(summary.get('n_use'))} valid words scored. "
        "Score is a raw mean cosine distance on an arbitrary scale \u2014 read the "
        "percentile column, not the score."
    )
    blocks = [
        _table(
            [
                "#",
                "model",
                "best cell",
                "mean [95% CI]",
                "pct vs chance",
                "z",
                "valid rate",
                "gaming idx",
                "Jaccard",
                "distinct/total words",
                "scored/refused/failed",
            ],
            rows,
            note=note,
        ),
        _bullets(
            [
                "**mean [95% CI]** \u2014 the model's *best prompt x temperature* cell, "
                "percentile bootstrap of the mean. An upper bound over the prompts tried, "
                "not typical behaviour.",
                "**pct vs chance** \u2014 where that mean sits in this embedder's random-noun "
                "distribution. This is the only cross-embedder-meaningful number in the row.",
                "**valid rate** \u2014 share of emitted candidates that passed every DAT rule "
                "that actually ran (see capabilities above).",
                "**gaming idx** \u2014 mean x valid rate. Reported beside the score, never "
                "instead of it: a model that wins on rare or technical words falls here.",
                "**Jaccard** \u2014 mean replicate-to-replicate word overlap. 1.0 is total mode "
                "collapse (same ten words every run); near 0 is fresh output each time.",
                "**scored/refused/failed** \u2014 the denominators. Refused runs are *excluded*, "
                "never scored 0, so the mean is conditional on producing scorable output.",
            ]
        ),
    ]
    return _section("Leaderboard", blocks)


def _denominators_section(summary: Mapping[str, Any]) -> dict:
    models = summary.get("models") if isinstance(summary.get("models"), Mapping) else {}
    entries = _leaderboard(summary)
    ids = list(dict.fromkeys([_model_id(e) for e in entries] + [str(k) for k in models]))
    rows = []
    reasons: list[str] = []
    truncations: list[str] = []
    tot_runs = tot_failed = tot_refused = tot_scored = tot_trunc = 0
    for mid in ids:
        entry = next((e for e in entries if _model_id(e) == mid), {})
        c = _counts(models.get(mid), entry)
        rows.append(
            [
                f"`{mid}`",
                _int(c.get("n_runs")),
                _int(c.get("n_scored")),
                _int(c.get("n_refused")),
                _int(c.get("n_failed")),
                _int(c.get("n_truncated")),
                _pct(c.get("valid_rate")),
            ]
        )
        detail = models.get(mid)
        if isinstance(detail, Mapping):
            tb = detail.get("truncated_by_prompt")
            if isinstance(tb, (list, tuple)) and tb:
                pairs = []
                for item in tb:
                    if isinstance(item, (list, tuple)) and len(item) == 2 and item[1]:
                        pairs.append(f"`{item[0]}` ({_int(item[1])})")
                if pairs:
                    truncations.append(f"`{mid}`: " + ", ".join(pairs))
        for key, acc in (
            ("n_runs", "runs"),
            ("n_scored", "scored"),
            ("n_refused", "refused"),
            ("n_failed", "failed"),
            ("n_truncated", "truncated"),
        ):
            if not _missing(c.get(key)):
                try:
                    val = int(c[key])
                except (TypeError, ValueError):
                    continue
                if acc == "runs":
                    tot_runs += val
                elif acc == "scored":
                    tot_scored += val
                elif acc == "refused":
                    tot_refused += val
                elif acc == "truncated":
                    tot_trunc += val
                else:
                    tot_failed += val
        detail = models.get(mid) if isinstance(models.get(mid), Mapping) else {}
        raw = detail.get("refusal_reasons")
        items: list[tuple[str, Any]] = []
        if isinstance(raw, Mapping):
            items = [(str(k), v) for k, v in raw.items()]
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            for item in raw:
                if (
                    isinstance(item, Sequence)
                    and not isinstance(item, (str, bytes))
                    and len(item) >= 2
                ):
                    items.append((str(item[0]), item[1]))
        if items:
            top = ", ".join(f"{r} ({_int(n)})" for r, n in items[:4])
            reasons.append(f"`{mid}`: {top}")

    blocks: list[dict] = []
    if rows:
        blocks.append(
            _table(
                ["model", "runs", "scored", "policy-refused", "failed (API/error)",
                 "truncated (max_tokens)", "valid rate"],
                rows,
                note=(
                    f"totals across models: {tot_runs} runs, {tot_scored} scored, "
                    f"{tot_refused} policy-refused, {tot_failed} failed, "
                    f"{tot_trunc} truncated."
                ),
            )
        )
    rc = summary.get("run_counts")
    if isinstance(rc, Mapping) and rc:
        blocks.append(
            _bullets(
                [
                    "run stage: "
                    + ", ".join(f"**{k}** {_int(v)}" for k, v in rc.items())
                    + ".",
                ]
            )
        )
    if truncations:
        blocks.append(_para(
            "Truncated runs by prompt \u2014 these cells are measuring the token cap, "
            "not the model:"))
        blocks.append(_bullets(truncations))
    if reasons:
        blocks.append(_para("Most common refusal reasons:"))
        blocks.append(_bullets(reasons))
    flags = summary.get("flag_counts")
    if isinstance(flags, Mapping) and flags:
        ordered = sorted(flags.items(), key=lambda kv: (-(kv[1] or 0), str(kv[0])))
        blocks.append(
            _table(
                ["validity flag", "candidates flagged"],
                [[f"`{k}`", _int(v)] for k, v in ordered],
                note=(
                    "`rare` is a Zipf-frequency cut, not a technicality test "
                    "\u2014 see Known limitations."
                ),
            )
        )
    if not blocks:
        blocks.append(_para("No run accounting recorded in this summary."))
    else:
        blocks.insert(
            0,
            _para(
                "A leaderboard that hides its denominators is not a benchmark. "
                "**failed** = the API call errored (no words at all). **policy-refused** = "
                "words came back but too few were valid to score under this policy; those "
                "runs are excluded from every mean above, never scored 0. "
                "**truncated** = the reply hit `max_tokens` (`finish_reason=\"length\"`); "
                "that is a harness limit, not a model result \u2014 a reasoning model can "
                "spend the whole budget thinking and return an empty message. Truncation "
                "hits elaborate prompts first, so a non-zero count here can masquerade as "
                "prompt sensitivity. Raise `max_tokens` in models.yaml and re-run before "
                "reading anything into that model\u2019s numbers."
            ),
        )
    return _section("Denominators: failed and refused runs", blocks)


def _grid_blocks(summary: Mapping[str, Any], mid: str, detail: Mapping[str, Any]) -> list[dict]:
    grid = detail.get("grid")
    cells: list[Mapping[str, Any]] = []
    if isinstance(grid, Sequence) and not isinstance(grid, (str, bytes)):
        cells = [c for c in grid if isinstance(c, Mapping)]
    if not cells:
        return [_para("No prompt x temperature grid recorded for this model.")]

    prompt_order = [str(p) for p in (summary.get("prompts") or [])]
    for c in cells:
        p = str(c.get("prompt_id", "?"))
        if p not in prompt_order:
            prompt_order.append(p)
    prompt_order = [
        p for p in prompt_order if any(str(c.get("prompt_id", "?")) == p for c in cells)
    ]

    temp_order: list[Any] = []
    for t in summary.get("temperatures") or []:
        temp_order.append(t)
    for c in cells:
        t = c.get("temperature")
        if not any(_temp(t) == _temp(x) for x in temp_order):
            temp_order.append(t)
    temp_order = [
        t for t in temp_order if any(_temp(c.get("temperature")) == _temp(t) for c in cells)
    ]

    by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for c in cells:
        by_key[(str(c.get("prompt_id", "?")), _temp(c.get("temperature")))] = c

    rows = []
    for p in prompt_order:
        row = [f"`{p}`"]
        for t in temp_order:
            c = by_key.get((p, _temp(t)))
            if c is None:
                row.append(DASH)
                continue
            txt = _num(c.get("mean"))
            n = c.get("n")
            if not _missing(n):
                txt += f" (n={_int(n)})"
            ref = c.get("n_refused")
            if not _missing(ref):
                try:
                    if int(ref) > 0:
                        txt += f", {_int(ref)} refused"
                except (TypeError, ValueError):
                    pass
            row.append(txt)
        rows.append(row)

    def _spread(group_of) -> Any:
        buckets: dict[str, list[float]] = {}
        for c in cells:
            m = c.get("mean")
            if _missing(m):
                continue
            buckets.setdefault(group_of(c), []).append(float(m))
        means = [sum(v) / len(v) for v in buckets.values() if v]
        return max(means) - min(means) if len(means) > 1 else None

    p_spread = _spread(lambda c: str(c.get("prompt_id", "?")))
    t_spread = _spread(lambda c: _temp(c.get("temperature")))
    notes = []
    if p_spread is not None:
        notes.append(
            f"prompt sensitivity: best-minus-worst prompt mean = **{_num(p_spread)}**"
        )
    if t_spread is not None:
        notes.append(
            f"temperature sensitivity: best-minus-worst temperature mean = **{_num(t_spread)}**"
        )
    blocks = [
        _table(
            ["prompt \\ temperature"] + [f"T={_temp(t)}" for t in temp_order],
            rows,
            note="cell = mean score over that cell's replicates.",
        )
    ]
    if notes:
        blocks.append(
            _bullets(
                notes
                + [
                    f"At n={_n_label(summary)} per cell these spreads are usually inside "
                    "the noise; compare them against the CI width in the leaderboard "
                    "before calling a prompt or a temperature better.",
                ]
            )
        )
    reported = _reported(detail.get("model_reported"))
    if reported != DASH:
        blocks.append(_para(f"Served as: `{reported}`."))
    if detail.get("notes"):
        blocks.append(_para(f"Registry notes: {detail['notes']}."))
    return blocks


def _breakdown_sections(summary: Mapping[str, Any]) -> list[dict]:
    models = summary.get("models") if isinstance(summary.get("models"), Mapping) else {}
    entries = _leaderboard(summary)
    ids = [_model_id(e) for e in entries] + [
        str(k) for k in models if str(k) not in {_model_id(e) for e in entries}
    ]
    ids = list(dict.fromkeys(ids))
    if not models:
        return [
            _section(
                "Per-model breakdown",
                [
                    _para(
                        "No per-model detail recorded, so prompt and temperature "
                        "sensitivity cannot be shown."
                    )
                ],
            )
        ]
    out = [
        _section(
            "Per-model breakdown",
            [
                _para(
                    "Cell means for every prompt x temperature combination, so prompt "
                    "sensitivity and temperature sensitivity are visible instead of hidden "
                    "behind the best cell."
                )
            ],
        )
    ]
    for mid in ids:
        detail = models.get(mid)
        if not isinstance(detail, Mapping):
            continue
        out.append(_section(f"`{mid}`", _grid_blocks(summary, mid, detail), level=3))
    return out


def _agreement_section(summary: Mapping[str, Any]) -> dict:
    pairs = _rho_pairs(summary)
    blocks: list[dict] = []
    names = [str(e) for e in (summary.get("embedders") or [])]
    for a, b, _r, _n in pairs:
        for name in (a, b):
            if name not in names:
                names.append(name)
    lookup: dict[frozenset[str], Any] = {frozenset((a, b)): r for a, b, r, _n in pairs}
    if names and pairs:
        rows = []
        for a in names:
            row = [f"`{a}`"]
            for b in names:
                if a == b:
                    row.append("1.00")
                else:
                    row.append(_num(lookup.get(frozenset((a, b))), 2))
            rows.append(row)
        blocks.append(
            _table(
                ["Spearman \u03c1"] + [f"`{n}`" for n in names],
                rows,
                note=(
                    "rank correlation of the model ordering between two embedders, "
                    "over the models both scored."
                ),
            )
        )
    blocks.append(_para(_verdict(pairs)))
    if len(names) < 2:
        blocks.append(
            _para(
                "Only one embedder was used. There is no evidence here about whether the "
                "ranking survives a different embedding model \u2014 the single most likely "
                "way for this leaderboard to be wrong."
            )
        )
    return _section("Cross-embedder agreement", blocks)


def _provenance_section(summary: Mapping[str, Any], meta: Mapping[str, Any]) -> dict:
    embedders = [str(e) for e in (summary.get("embedders") or [])]
    if not embedders:
        base = summary.get("baselines")
        if isinstance(base, Mapping):
            embedders = [str(k) for k in base]
    entries = _leaderboard(summary)
    models_map = summary.get("models") if isinstance(summary.get("models"), Mapping) else {}
    served: list[str] = []
    for mid in dict.fromkeys([_model_id(e) for e in entries] + [str(k) for k in models_map]):
        entry = next((e for e in entries if _model_id(e) == mid), {})
        detail = models_map.get(mid) if isinstance(models_map.get(mid), Mapping) else {}
        rep = _reported(detail.get("model_reported") or entry.get("model_reported"))
        served.append(
            f"`{mid}` \u2192 `{rep}`"
            if rep != DASH
            else f"`{mid}` \u2192 {DASH} (not recorded)"
        )

    temps = ", ".join(_temp(t) for t in (summary.get("temperatures") or []))
    policies = ", ".join(f"`{p}`" for p in (summary.get("policies") or []))
    commit = meta.get("git_commit") if "git_commit" in meta else _git_commit()
    if meta.get("git_dirty"):
        commit = f"{commit or '?'} (working tree dirty)"

    items = [
        f"**generated**: {meta.get('generated_at') or _now()}",
        f"**dat-bench version**: {meta.get('tool_version') or 'unknown'}",
        f"**git commit**: {commit or 'not available'}",
        f"**summary schema**: v{_int(summary.get('schema_version'))}",
        f"**replicates (n)**: {_int(summary.get('n_replicates'))} per "
        "model x prompt x temperature cell",
        f"**words scored per run (n_use)**: {_int(summary.get('n_use'))}",
        f"**prompts**: {', '.join(f'`{p}`' for p in (summary.get('prompts') or [])) or DASH}",
        f"**temperatures**: {temps or DASH}",
        f"**policies scored**: {policies or DASH}",
        f"**embedders**: {', '.join(f'`{e}`' for e in embedders) or DASH}",
        f"**embedding endpoint**: {meta.get('embed_base_url') or 'not recorded'}",
        f"**model registry**: {meta.get('models_yaml') or 'not recorded'}",
        f"**command**: `{meta.get('command') or 'not recorded'}`",
    ]
    if meta.get("notes"):
        items.append(f"**notes**: {meta['notes']}")
    blocks = [_bullets(items)]
    if served:
        blocks.append(_para("Model ids actually served by the API (`response.model`):"))
        blocks.append(_bullets(served))
        blocks.append(
            _para(
                "A served id that differs from the requested one means the provider routed "
                "elsewhere; the row is labelled by what we asked for, so check this list "
                "before attributing a result to a model."
            )
        )
    warnings = summary.get("warnings")
    if isinstance(warnings, Sequence) and not isinstance(warnings, (str, bytes)) and warnings:
        blocks.append(_para("Warnings raised during this run:"))
        blocks.append(_bullets([str(w) for w in warnings]))
    return _section("Provenance", blocks)


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _build_view(summary: Mapping[str, Any] | None, meta: Mapping[str, Any] | None) -> dict:
    summary = summary if isinstance(summary, Mapping) else {}
    meta = meta if isinstance(meta, Mapping) else {}
    models = summary.get("models") if isinstance(summary.get("models"), Mapping) else {}
    n_models = len(_leaderboard(summary)) or len(models)
    embedders = summary.get("embedders") or []
    subtitle = (
        f"{n_models} model(s) \u00b7 {len(embedders) or '?'} embedder(s) \u00b7 "
        f"n={_int(summary.get('n_replicates'))} replicates per cell \u00b7 "
        f"generated {meta.get('generated_at') or _now()}"
    )
    sections = [
        _section("", [_scale_warning(summary)], level=0),
        _capabilities_section(summary),
        _headline_section(summary),
        _baseline_section(summary),
        _leaderboard_section(summary),
        _denominators_section(summary),
        *_breakdown_sections(summary),
        _agreement_section(summary),
        _provenance_section(summary, meta),
        _section("Known limitations", [_bullets(_limitations(summary))]),
    ]
    return {
        "title": str(meta.get("title") or "dat-bench: Divergent Association Task over N runs"),
        "subtitle": subtitle,
        # A section builder returns None when the summary lacks what it needs
        # (_headline_section without baselines, say). Drop those rather than render
        # an empty heading -- and never let one reach the renderer as None.
        "sections": [s for s in sections if s],
    }


# --------------------------------------------------------------------------- #
# markdown
# --------------------------------------------------------------------------- #
def _md_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _md_block(block: Mapping[str, Any]) -> list[str]:
    kind = block.get("kind")
    if kind == "para":
        return [str(block["text"]), ""]
    if kind == "bullets":
        return [f"- {item}" for item in block["items"]] + [""]
    if kind == "callout":
        tag = {"warn": "\u26a0\ufe0f", "ok": "\u2705"}.get(str(block.get("tone")), "\u2139\ufe0f")
        out = [f"> {tag} **{block['title']}**", ">"]
        for line in block["lines"]:
            out.append(f"> {line}")
            out.append(">")
        if out[-1] == ">":
            out.pop()
        return out + [""]
    if kind == "table":
        headers = block["headers"]
        out = [
            "| " + " | ".join(_md_cell(h) for h in headers) + " |",
            "|" + "|".join("---" for _ in headers) + "|",
        ]
        for row in block["rows"]:
            cells = [_md_cell(c) for c in row]
            cells += [""] * (len(headers) - len(cells))
            out.append("| " + " | ".join(cells[: len(headers)]) + " |")
        if block.get("note"):
            out += ["", f"*{block['note']}*"]
        return out + [""]
    return []


def build_markdown(summary: dict, meta: dict) -> str:
    """Render the full report as GitHub-flavoured Markdown."""
    view = _build_view(summary, meta)
    lines = [f"# {view['title']}", "", f"*{view['subtitle']}*", ""]
    for sec in view["sections"]:
        if sec["heading"]:
            lines.append(f"{'#' * max(1, int(sec['level']))} {sec['heading']}")
            lines.append("")
        for block in sec["blocks"]:
            lines.extend(_md_block(block))
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).rstrip() + "\n"


# --------------------------------------------------------------------------- #
# html
# --------------------------------------------------------------------------- #
_CSS = """
:root { color-scheme: light dark; }
body { margin: 0 auto; padding: 2rem 1.25rem 5rem; max-width: 68rem;
  font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  color: #1b1f24; background: #fbfbfa; }
h1 { font-size: 1.7rem; margin: 0 0 .25rem; letter-spacing: -.01em; }
h2 { font-size: 1.2rem; margin: 2.4rem 0 .6rem; padding-bottom: .3rem;
  border-bottom: 1px solid rgba(127,127,127,.28); }
h3 { font-size: 1rem; margin: 1.6rem 0 .4rem;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.subtitle { color: #5d646d; margin: 0 0 1.5rem; font-size: .9rem; }
p, li { max-width: 58rem; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .88em;
  background: rgba(127,127,127,.14); padding: .1em .32em; border-radius: 3px; }
.callout { border: 1px solid; border-left-width: 5px; border-radius: 6px;
  padding: .85rem 1rem; margin: 1.1rem 0; }
.callout .t { font-weight: 700; display: block; margin-bottom: .4rem; text-transform: none; }
.callout p { margin: .45rem 0; }
.callout.warn { border-color: #c2410c; background: rgba(234,88,12,.09); }
.callout.ok   { border-color: #15803d; background: rgba(22,163,74,.09); }
.callout.info { border-color: #1d4ed8; background: rgba(37,99,235,.09); }
.tw { overflow-x: auto; margin: .9rem 0; }
table { border-collapse: collapse; font-size: .87rem; min-width: 100%; }
th, td { padding: .38rem .6rem; text-align: left; white-space: nowrap;
  border-bottom: 1px solid rgba(127,127,127,.24); }
th { font-weight: 650; background: rgba(127,127,127,.1); position: sticky; top: 0; }
td:first-child, th:first-child { white-space: normal; }
tbody tr:hover { background: rgba(127,127,127,.07); }
.note { font-size: .82rem; color: #5d646d; margin: .25rem 0 0; }
@media (prefers-color-scheme: dark) {
  body { color: #e6e6e3; background: #16181c; }
  .subtitle, .note { color: #a0a6ae; }
  .callout.warn { border-color: #fb923c; background: rgba(251,146,60,.12); }
  .callout.ok   { border-color: #4ade80; background: rgba(74,222,128,.12); }
  .callout.info { border-color: #60a5fa; background: rgba(96,165,250,.12); }
}
"""


def _inline(text: str) -> str:
    """Escape, then honour the tiny inline subset the view model uses."""
    out = _htmllib.escape(str(text), quote=False)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"(?<![*\w])\*([^*]+)\*(?!\w)", r"<em>\1</em>", out)
    return out


def _html_block(block: Mapping[str, Any]) -> str:
    kind = block.get("kind")
    if kind == "para":
        return f"<p>{_inline(block['text'])}</p>"
    if kind == "bullets":
        items = "".join(f"<li>{_inline(i)}</li>" for i in block["items"])
        return f"<ul>{items}</ul>"
    if kind == "callout":
        tone = str(block.get("tone") or "info")
        tone = tone if tone in {"warn", "ok", "info"} else "info"
        body = "".join(f"<p>{_inline(line)}</p>" for line in block["lines"])
        return (
            f'<div class="callout {tone}"><span class="t">{_inline(block["title"])}</span>'
            f"{body}</div>"
        )
    if kind == "table":
        head = "".join(f"<th>{_inline(h)}</th>" for h in block["headers"])
        body = "".join(
            "<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in row) + "</tr>"
            for row in block["rows"]
        )
        note = f'<p class="note">{_inline(block["note"])}</p>' if block.get("note") else ""
        return (
            f'<div class="tw"><table><thead><tr>{head}</tr></thead>'
            f"<tbody>{body}</tbody></table>{note}</div>"
        )
    return ""


def _html_sections(view: Mapping[str, Any]) -> list[dict[str, str]]:
    """Pre-rendered, already-escaped fragments shared by both HTML paths."""
    out = []
    for sec in view["sections"]:
        level = max(1, min(6, int(sec["level"]) or 2)) if sec["heading"] else 0
        out.append(
            {
                "heading": _inline(sec["heading"]) if sec["heading"] else "",
                "level": str(level),
                "body": "\n".join(_html_block(b) for b in sec["blocks"]),
            }
        )
    return out


_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ title }}</title>
<style>{{ css }}</style>
</head><body>
<h1>{{ title }}</h1>
<p class="subtitle">{{ subtitle }}</p>
{% for sec in sections %}{% if sec.heading %}<h{{ sec.level }}>{{ sec.heading }}</h{{ sec.level }}>
{% endif %}{{ sec.body }}
{% endfor %}</body></html>
"""


def build_html(summary: dict, meta: dict) -> str:
    """Render the same report as a single self-contained HTML file.

    jinja2 when it is importable, a plain-string build otherwise -- the HTML
    report is not worth a hard dependency failure, and pyproject lists jinja2
    under an optional extra.
    """
    view = _build_view(summary, meta)
    ctx = {
        "title": _inline(view["title"]),
        "subtitle": _inline(view["subtitle"]),
        "css": _CSS,
        "sections": _html_sections(view),
    }
    try:
        import jinja2
    except ImportError:
        return _html_plain(ctx)
    # autoescape stays off: every fragment in ctx is already escaped by _inline,
    # so the jinja2 and fallback paths emit identical fragments and cannot drift.
    return jinja2.Template(_TEMPLATE, autoescape=False).render(**ctx)


def _html_plain(ctx: Mapping[str, Any]) -> str:
    parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{ctx['title']}</title>",
        f"<style>{ctx['css']}</style>",
        "</head><body>",
        f"<h1>{ctx['title']}</h1>",
        f"<p class=\"subtitle\">{ctx['subtitle']}</p>",
    ]
    for sec in ctx["sections"]:
        if sec["heading"]:
            parts.append(f"<h{sec['level']}>{sec['heading']}</h{sec['level']}>")
        parts.append(sec["body"])
    parts.append("</body></html>")
    return "\n".join(parts) + "\n"
