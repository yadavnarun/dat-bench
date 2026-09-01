"""Tests for datbench.report.

The point of this module is that the report cannot quietly lose its caveats, so
most of these tests are assertions about the honesty sections rather than about
formatting. No network, no filesystem: build_markdown/build_html are pure.
"""

from __future__ import annotations

import math
import re
import sys

import pytest

from datbench import report

NAN = float("nan")


@pytest.fixture
def meta() -> dict:
    return {
        "generated_at": "2026-08-31T12:00:00Z",
        "git_commit": "a1b2c3d",
        "git_dirty": False,
        "tool_version": "0.1.0",
        "command": "python -m datbench report --html",
        "models_yaml": "models.yaml",
        "embed_base_url": "http://localhost:1234/v1",
    }


@pytest.fixture
def summary() -> dict:
    """A realistic two-model, two-embedder, degraded-WordNet run."""
    return {
        "schema_version": 1,
        "primary_embedder": "text-embedding-qwen3-embedding-4b",
        "primary_policy": "strict",
        "embedders": [
            "text-embedding-qwen3-embedding-4b",
            "text-embedding-nomic-embed-text-v1.5",
        ],
        "policies": ["strict", "lenient"],
        "n_replicates": 10,
        "n_use": 7,
        "prompts": ["verbatim", "terse"],
        "temperatures": [0.0, 0.7],
        "capabilities": {"dictionary": True, "wordnet": False, "wordfreq": True},
        "baselines": {
            "text-embedding-qwen3-embedding-4b": {
                "random": {
                    "mean": 0.38,
                    "sd": 0.05,
                    "n": 1000,
                    "k": 7,
                    "p05": 0.30,
                    "p50": 0.38,
                    "p95": 0.46,
                    "seed": 0,
                },
                "categories": {"animals": 0.14, "tools": 0.19},
                "paper_example": 0.41,
            },
            "text-embedding-nomic-embed-text-v1.5": {
                "mean": 0.52,
                "sd": 0.04,
                "n": 1000,
                "k": 7,
                "p05": 0.45,
                "p50": 0.52,
                "p95": 0.59,
                "seed": 0,
            },
        },
        "run_counts": {"attempted": 80, "written": 80, "skipped": 0, "errors": 3},
        "leaderboard": [
            {
                "model_id": "gemma-4-e4b",
                "model_reported": ["google/gemma-4-e4b"],
                "best_prompt": "terse",
                "best_temperature": 0.7,
                "n": 10,
                "mean": 0.4812,
                "sd": 0.021,
                "ci_lo": 0.4690,
                "ci_hi": 0.4935,
                "percentile_vs_chance": 0.97,
                "z_vs_chance": 2.02,
                "valid_rate": 0.91,
                "gaming_index": 0.4379,
                "jaccard": 0.34,
                "distinct_pool": {"distinct": 54, "total": 90, "ratio": 0.6},
                "n_runs": 40,
                "n_scored": 34,
                "n_refused": 5,
                "n_failed": 1,
            },
            {
                "model_id": "lfm2.5-1.2b",
                "model_reported": ["liquid/lfm2.5-1.2b"],
                "best_prompt": "verbatim",
                "best_temperature": 0.0,
                "n": 10,
                "mean": 0.4013,
                "sd": 0.035,
                "ci_lo": 0.3810,
                "ci_hi": 0.4210,
                "percentile_vs_chance": 0.66,
                "z_vs_chance": 0.43,
                "valid_rate": 0.58,
                "gaming_index": 0.2328,
                "jaccard": 0.71,
                "distinct_pool": {"distinct": 21, "total": 88, "ratio": 0.239},
                "n_runs": 40,
                "n_scored": 18,
                "n_refused": 20,
                "n_failed": 2,
            },
        ],
        "models": {
            "gemma-4-e4b": {
                "model_reported": ["google/gemma-4-e4b"],
                "n_runs": 40,
                "n_scored": 34,
                "n_refused": 5,
                "n_failed": 1,
                "valid_rate": 0.91,
                "notes": "local via LM Studio",
                "refusal_reasons": [
                    ["only 5 valid words (strict needs 7)", 4],
                    ["empty response", 1],
                ],
                "grid": [
                    {"prompt_id": "verbatim", "temperature": 0.0, "n": 10, "mean": 0.4501,
                     "sd": 0.018, "ci_lo": 0.44, "ci_hi": 0.46, "valid_rate": 0.93,
                     "jaccard": 0.55, "n_refused": 0, "n_failed": 0},
                    {"prompt_id": "verbatim", "temperature": 0.7, "n": 9, "mean": 0.4602,
                     "sd": 0.02, "ci_lo": 0.45, "ci_hi": 0.47, "valid_rate": 0.90,
                     "jaccard": 0.41, "n_refused": 1, "n_failed": 0},
                    {"prompt_id": "terse", "temperature": 0.0, "n": 10, "mean": 0.4703,
                     "sd": 0.019, "ci_lo": 0.46, "ci_hi": 0.48, "valid_rate": 0.92,
                     "jaccard": 0.48, "n_refused": 0, "n_failed": 0},
                    {"prompt_id": "terse", "temperature": 0.7, "n": 10, "mean": 0.4812,
                     "sd": 0.021, "ci_lo": 0.469, "ci_hi": 0.4935, "valid_rate": 0.91,
                     "jaccard": 0.34, "n_refused": 4, "n_failed": 1},
                ],
            },
            "lfm2.5-1.2b": {
                "model_reported": ["liquid/lfm2.5-1.2b"],
                "n_runs": 40,
                "n_scored": 18,
                "n_refused": 20,
                "n_failed": 2,
                "valid_rate": 0.58,
                "notes": "local via LM Studio; small, expect low validity rate",
                "refusal_reasons": {"only 3 valid words (strict needs 7)": 15, "empty response": 5},
                "grid": [
                    {"prompt_id": "verbatim", "temperature": 0.0, "n": 8, "mean": 0.4013,
                     "sd": 0.035, "ci_lo": 0.381, "ci_hi": 0.421, "valid_rate": 0.60,
                     "jaccard": 0.71, "n_refused": 2, "n_failed": 0},
                    {"prompt_id": "terse", "temperature": 0.7, "n": 4, "mean": 0.3702,
                     "sd": 0.05, "ci_lo": 0.34, "ci_hi": 0.40, "valid_rate": 0.51,
                     "jaccard": 0.66, "n_refused": 6, "n_failed": 2},
                ],
            },
        },
        "rank_correlation": [
            {
                "a": "text-embedding-qwen3-embedding-4b",
                "b": "text-embedding-nomic-embed-text-v1.5",
                "rho": 0.93,
                "n_models": 2,
            }
        ],
        "flag_counts": {"rare": 12, "not_noun": 8, "proper_noun": 3},
        "warnings": ["2 of 4 embedders unreachable"],
    }


# ---------------------------------------------------------------- honesty ----
def test_markdown_states_arbitrary_scale_and_no_comparability(summary, meta):
    md = report.build_markdown(summary, meta)
    head = md[: md.index("## Which checks actually ran")]
    # All four claims must be above the first section, not buried in limitations.
    assert "arbitrary scale" in head
    assert "NOT comparable" in head
    assert "GloVe 840B-300d" in head
    assert "mean ≈78, range ~50–95" in head
    assert "local LM Studio" in head


def test_markdown_does_not_print_x100_scores(summary, meta):
    md = report.build_markdown(summary, meta)
    assert "0.481" in md  # the raw mean, as scored
    assert "48.12" not in md and "48.1" not in md  # never the GloVe-scale convention
    assert "×100" in md  # and it says so explicitly


def test_markdown_reports_capabilities_and_flags_degradation(summary, meta):
    md = report.build_markdown(summary, meta)
    assert "`dictionary`" in md and "`wordnet`" in md and "`wordfreq`" in md
    assert "**NO**" in md  # wordnet did not load
    assert "Degraded run" in md
    assert "upper bound" in md  # what a missing check does to valid_rate


def test_markdown_reports_all_capabilities_ran_when_clean(summary, meta):
    summary["capabilities"] = {"dictionary": True, "wordnet": True, "wordfreq": True}
    md = report.build_markdown(summary, meta)
    assert "All validity checks ran" in md
    assert "Degraded run" not in md


def test_missing_capabilities_is_loud_not_silent(summary, meta):
    del summary["capabilities"]
    md = report.build_markdown(summary, meta)
    assert "NOT RECORDED" in md
    assert "unverified" in md


def test_markdown_shows_failed_and_refused_denominators(summary, meta):
    md = report.build_markdown(summary, meta)
    section = md[md.index("## Denominators"): md.index("## Per-model breakdown")]
    assert "policy-refused" in section
    assert "failed (API/error)" in section
    # per-model counts, from the fixture
    assert "| 40 | 34 | 5 | 1 |" in section
    assert "| 40 | 18 | 20 | 2 |" in section
    assert "never scored 0" in section
    assert "only 5 valid words (strict needs 7) (4)" in section  # list-of-pairs form
    assert "only 3 valid words (strict needs 7) (15)" in section  # dict form


def test_known_limitations_names_the_three_mandatory_ones(summary, meta):
    md = report.build_markdown(summary, meta)
    lim = md[md.index("## Known limitations"):]
    assert "weak proxy" in lim
    assert "thimble" in lim and "2.53" in lim
    assert "quark" in lim and "3.05" in lim
    assert "Temperature is not comparable across providers" in lim
    assert f"n={summary['n_replicates']} replicates" in lim


def test_rare_limitation_does_not_claim_thimble_is_flagged(summary, meta):
    """thimble sits at Zipf 2.53 and the threshold is 2.5, so it is NOT flagged.

    An earlier draft asserted the opposite in prose while the code did the right
    thing -- a report that misdescribes its own validity rule is worse than one that
    omits the detail.
    """
    lim = report.build_markdown(summary, meta)[len("## Known limitations"):]
    rare = next(b for b in lim.split("\n- ") if "weak proxy" in b)
    assert "clears the 2.5 threshold" in rare
    assert "flagged as rare, though it is ordinary" not in rare


def test_replicate_count_in_limitations_tracks_the_actual_run(summary, meta):
    """The n in the limitations bullet must come from the run, not a literal.

    Hardcoding n=10 made every smaller run state a false denominator.
    """
    def limitations_of(n):
        s = dict(summary)
        s["n_replicates"] = n
        md = report.build_markdown(s, meta)
        return md[md.index("## Known limitations"):]

    small = limitations_of(3)
    assert "n=3 replicates is a small sample" in small
    # The stale literal must be gone from this section -- "(n=10)" still appears
    # legitimately in the per-cell grid, which counts cells, not replicates.
    assert "n=10" not in small

    assert "n=25 replicates" in limitations_of(25)
    assert "Replicate count unknown" in limitations_of(None)


# ------------------------------------------------------------ leaderboard ----
def test_leaderboard_rows_and_ordering(summary, meta):
    md = report.build_markdown(summary, meta)
    board = md[md.index("## Leaderboard"): md.index("## Denominators")]
    assert "`gemma-4-e4b`" in board and "`lfm2.5-1.2b`" in board
    assert board.index("`gemma-4-e4b`") < board.index("`lfm2.5-1.2b`")
    assert "terse @ T=0.7" in board
    assert "0.481 [0.469, 0.493]" in board
    assert "97.0%" in board  # percentile vs chance
    assert "91%" in board  # valid rate
    assert "0.438" in board  # gaming index
    assert "54/90" in board  # distinct/total word pool
    assert "34 / 5 / 1" in board  # scored / refused / failed


def test_percentile_is_derived_when_the_summary_omits_it(summary, meta):
    entry = summary["leaderboard"][0]
    del entry["percentile_vs_chance"]
    del entry["z_vs_chance"]
    entry["mean"] = 0.48  # baseline mean 0.38, sd 0.05 -> z = 2.0
    md = report.build_markdown(summary, meta)
    assert "97.7%" in md
    assert "2.00" in md


def test_baseline_section_reports_chance_per_embedder(summary, meta):
    md = report.build_markdown(summary, meta)
    sec = md[md.index("## Chance baselines"): md.index("## Leaderboard")]
    assert "0.380" in sec and "0.050" in sec  # qwen baseline, unnested from "random"
    assert "0.520" in sec  # nomic baseline, flat form
    assert "animals 0.140" in sec
    assert "0.410" in sec  # paper_example


def test_missing_baselines_says_scores_are_uninterpretable(summary, meta):
    del summary["baselines"]
    md = report.build_markdown(summary, meta)
    assert "No random-noun baseline recorded" in md
    assert "uninterpretable" in md


# ---------------------------------------------------------------- grids ------
def test_per_model_grid_shows_prompt_x_temperature(summary, meta):
    md = report.build_markdown(summary, meta)
    sec = md[md.index("### `gemma-4-e4b`"): md.index("### `lfm2.5-1.2b`")]
    assert "T=0" in sec and "T=0.7" in sec
    assert "`verbatim`" in sec and "`terse`" in sec
    assert "0.450 (n=10)" in sec
    assert "0.481 (n=10), 4 refused" in sec
    assert "prompt sensitivity" in sec and "temperature sensitivity" in sec
    assert "google/gemma-4-e4b" in sec


def test_grid_labels_a_none_temperature_as_not_applicable(summary, meta):
    summary["temperatures"] = [None]
    summary["models"]["gemma-4-e4b"]["grid"] = [
        {"prompt_id": "verbatim", "temperature": None, "n": 10, "mean": 0.44}
    ]
    md = report.build_markdown(summary, meta)
    assert "T=n/a" in md


# ----------------------------------------------------------- agreement -------
def test_spearman_matrix_and_robust_verdict(summary, meta):
    # n_models >= 4: below that the verdict is refused outright, since rho over a
    # handful of items is arithmetic rather than evidence.
    summary["rank_correlation"][0]["n_models"] = 8
    md = report.build_markdown(summary, meta)
    sec = md[md.index("## Cross-embedder agreement"): md.index("## Provenance")]
    assert "Spearman" in sec
    assert "0.93" in sec
    assert "ranking is robust" in sec


def test_weak_agreement_gets_a_blunt_verdict(summary, meta):
    summary["rank_correlation"][0]["rho"] = 0.11
    summary["rank_correlation"][0]["n_models"] = 8
    md = report.build_markdown(summary, meta)
    assert "**not** robust" in md
    assert "artifact of the embedder" in md


def test_tuple_keyed_rank_correlation_is_accepted(summary, meta):
    # analyze.rank_correlation returns dict[tuple[str, str], float] in memory.
    summary["rank_correlation"] = {("emb-a", "emb-b"): 0.82}
    summary["embedders"] = ["emb-a", "emb-b"]
    md = report.build_markdown(summary, meta)
    assert "0.82" in md
    assert "broadly robust" in md


def test_single_embedder_says_robustness_is_unknown(summary, meta):
    summary["embedders"] = ["only-one"]
    summary["rank_correlation"] = []
    md = report.build_markdown(summary, meta)
    assert "cannot be assessed" in md
    assert "Only one embedder was used" in md


# ----------------------------------------------------------- provenance ------
def test_provenance_block(summary, meta):
    md = report.build_markdown(summary, meta)
    sec = md[md.index("## Provenance"): md.index("## Known limitations")]
    assert "2026-08-31T12:00:00Z" in sec
    assert "a1b2c3d" in sec
    assert "text-embedding-qwen3-embedding-4b" in sec
    assert "n=10" not in sec or "10" in sec
    assert "`gemma-4-e4b` → `google/gemma-4-e4b`" in sec
    assert "`lfm2.5-1.2b` → `liquid/lfm2.5-1.2b`" in sec
    assert "2 of 4 embedders unreachable" in sec


def test_dirty_tree_is_marked(summary, meta):
    meta["git_dirty"] = True
    md = report.build_markdown(summary, meta)
    assert "working tree dirty" in md


# -------------------------------------------------------- missing keys -------
def test_empty_summary_still_renders(meta):
    md = report.build_markdown({}, meta)
    assert "arbitrary scale" in md
    assert "NOT RECORDED" in md  # capabilities
    assert "No random-noun baseline recorded" in md
    assert "nothing to rank" in md
    assert "## Known limitations" in md


def test_empty_summary_and_meta_still_renders():
    md = report.build_markdown({}, {})
    assert md.startswith("# dat-bench")
    assert "arbitrary scale" in md
    html = report.build_html({}, {})
    assert html.startswith("<!doctype html>")


def test_partial_summary_with_only_a_leaderboard_renders():
    summary = {
        "leaderboard": [
            {"model_id": "solo", "mean": 0.4},
            {"model_id": "nameless-cell"},  # no mean at all
        ]
    }
    md = report.build_markdown(summary, {"generated_at": "t"})
    assert "`solo`" in md
    assert "`nameless-cell`" in md
    # a model with no mean sinks below the ranked ones and gets no rank number
    assert md.index("`solo`") < md.index("`nameless-cell`")
    assert "No per-model detail recorded" in md


def test_nan_and_none_statistics_render_as_dashes(summary, meta):
    entry = summary["leaderboard"][0]
    entry.update({"sd": NAN, "ci_lo": NAN, "ci_hi": NAN, "jaccard": NAN, "valid_rate": None})
    summary["models"]["gemma-4-e4b"]["valid_rate"] = NAN
    summary["models"]["gemma-4-e4b"]["grid"][0]["mean"] = NAN
    md = report.build_markdown(summary, meta)
    # "provenance" contains the substring, so match nan only as a whole token.
    assert re.search(r"\bnan\b", md, re.IGNORECASE) is None
    assert f"[CI {report.DASH}]" in md


def test_unknown_shapes_do_not_raise(meta):
    # Deliberately wrong types in every tolerant slot.
    summary = {
        "leaderboard": {"not": "a list"},
        "models": ["not", "a", "mapping"],
        "baselines": 3,
        "capabilities": [],
        "rank_correlation": "nope",
        "run_counts": None,
        "flag_counts": (),
        "temperatures": None,
        "prompts": None,
    }
    md = report.build_markdown(summary, meta)
    assert "## Known limitations" in md
    assert report.build_html(summary, meta).endswith("</html>\n") or "</html>" in report.build_html(
        summary, meta
    )


# ------------------------------------------------------------------ html -----
def test_html_is_self_contained_and_theme_aware(summary, meta):
    html = report.build_html(summary, meta)
    assert "<style>" in html
    assert "prefers-color-scheme: dark" in html
    assert "http" not in html.split("<style>")[1].split("</style>")[0]  # no external assets
    assert "arbitrary scale" in html
    assert "GloVe 840B-300d" in html
    assert "<table>" in html
    assert "overflow-x: auto" in html  # wide tables scroll, page does not


def test_html_escapes_content_and_renders_inline_markup(summary, meta):
    summary["leaderboard"][0]["model_id"] = "<script>alert(1)</script>"
    summary["models"] = {}
    html = report.build_html(summary, meta)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
    assert "<strong>" in html and "<code>" in html


def test_html_falls_back_when_jinja2_is_absent(summary, meta, monkeypatch):
    # sys.modules[name] = None makes `import name` raise ImportError.
    monkeypatch.setitem(sys.modules, "jinja2", None)
    plain = report.build_html(summary, meta)
    assert plain.startswith("<!doctype html>")
    assert "arbitrary scale" in plain
    assert "GloVe 840B-300d" in plain
    assert "<table>" in plain

    monkeypatch.undo()
    templated = report.build_html(summary, meta)
    # Same facts either way: the two paths share every rendered fragment.
    for needle in ("Degraded run", "97.0%", "Spearman", "thimble", "<strong>"):
        assert needle in plain and needle in templated


def test_html_and_markdown_carry_the_same_headline_numbers(summary, meta):
    md = report.build_markdown(summary, meta)
    html = report.build_html(summary, meta)
    for needle in ("0.481", "97.0%", "0.438", "34 / 5 / 1", "0.93"):
        assert needle in md, needle
        assert needle in html, needle


def test_no_leftover_blank_run_or_trailing_whitespace(summary, meta):
    md = report.build_markdown(summary, meta)
    assert "\n\n\n" not in md
    assert md.endswith("\n") and not md.endswith("\n\n")
    assert not any(line != line.rstrip() for line in md.splitlines())


def test_percentile_helper_accepts_fractions_and_percents():
    assert report._pct(0.97, 1) == "97.0%"
    assert report._pct(97.0, 1) == "97.0%"
    assert report._pct(None) == report.DASH
    assert report._pct(NAN) == report.DASH
    assert math.isnan(float("nan"))  # guard against a fixture typo hiding a bug


def test_report_discloses_the_far_end_range_compression(summary, meta):
    """A report-only reader must learn the scale is compressed vs the real DAT.

    The baselines and category floors alone do not convey it: they say where chance
    sits, not that the usable band above chance is ~2.2x narrower than GloVe's, which
    is what determines how many replicates are needed to separate two good models.
    """
    lim = report.build_markdown(summary, meta)
    lim = lim[lim.index("## Known limitations"):]
    assert "compress" in lim.lower()
    assert "0.879" in lim and "0.396" in lim          # the measured contrast
    assert "0.198" in lim and "0.204" in lim          # near pairs agree
    assert "replicates" in lim


# ------------------------------------------------------------- truncation ----
# A reply that hit max_tokens (finish_reason="length") is a harness limit, not a
# capability result. Measured case: gemma-4-e4b at max_tokens=512 truncated 30/30
# verbatim and 29/30 cot runs while completing 30/30 terse, so its "best prompt"
# came out as terse purely because the other prompts never finished. Nothing in
# the report distinguished that from a model answering badly.

def _with_truncation(summary):
    s = dict(summary)
    s["models"] = {k: dict(v) for k, v in s["models"].items()}
    s["models"]["gemma-4-e4b"]["n_truncated"] = 30
    s["models"]["gemma-4-e4b"]["truncated_by_prompt"] = [["verbatim", 28], ["cot", 2]]
    s["leaderboard"] = [dict(e) for e in s["leaderboard"]]
    s["leaderboard"][0]["n_truncated"] = 30
    return s


def test_denominators_surface_truncated_runs(summary, meta):
    md = report.build_markdown(_with_truncation(summary), meta)
    section = md[md.index("## Denominators"):]
    assert "truncated" in section.lower()
    assert "max_tokens" in section
    # the per-prompt breakdown is what reveals it as an artifact rather than a finding
    assert "`verbatim` (28)" in section and "`cot` (2)" in section
    assert "masquerade as prompt sensitivity" in section


def test_truncation_is_not_conflated_with_a_model_failing(summary, meta):
    section = report.build_markdown(_with_truncation(summary), meta)
    section = section[section.index("## Denominators"):]
    assert "harness limit, not a model result" in section
    assert "Raise `max_tokens`" in section


def test_zero_truncation_does_not_add_a_breakdown(summary, meta):
    """A clean run must not grow a scary empty section."""
    md = report.build_markdown(summary, meta)
    section = md[md.index("## Denominators"):]
    assert "Truncated runs by prompt" not in section


# ------------------------------------------------------- degenerate cells ----
# A deterministic cell (T=0, or total mode collapse) repeats one answer n times.
# Bootstrapping that gives a zero-width interval, which reads as extreme precision
# but rests on a single observation. Observed live: gemma-4-e4b terse/T=0 produced
# Jaccard 1.00 and "0.401 [0.401, 0.401]" over 10 replicates.

@pytest.mark.parametrize(
    "overrides",
    [
        {"ci_lo": 0.401, "ci_hi": 0.401},   # zero-width interval
        {"sd": 0.0},                        # no spread
        {"jaccard": 1.0},                   # identical word sets every replicate
    ],
)
def test_degenerate_cell_is_flagged_instead_of_given_a_ci(summary, meta, overrides):
    s = dict(summary)
    s["leaderboard"] = [dict(e) for e in s["leaderboard"]]
    s["leaderboard"][0].update(overrides)
    board = report.build_markdown(s, meta)
    board = board[board.index("## Leaderboard"): board.index("## Denominators")]
    assert "degenerate: n_eff=1" in board


def test_a_genuinely_varying_cell_keeps_its_ci(summary, meta):
    board = report.build_markdown(summary, meta)
    board = board[board.index("## Leaderboard"): board.index("## Denominators")]
    assert "0.481 [0.469, 0.493]" in board
    assert "degenerate" not in board


def test_limitations_warn_that_best_cell_selection_favours_deterministic_cells(summary, meta):
    lim = report.build_markdown(summary, meta)
    lim = lim[lim.index("## Known limitations"):]
    assert "effective sample size of 1" in lim
    assert "Jaccard" in lim


# --------------------------------------------------- spearman with few models ----
def test_robustness_is_refused_when_too_few_models_were_ranked(summary, meta):
    """rho over 2 models is always +/-1, so it cannot evidence agreement."""
    s = dict(summary)
    s["rank_correlation"] = [
        {"a": "emb-a", "b": "emb-b", "rho": 1.0, "n_models": 2},
    ]
    md = report.build_markdown(s, meta)
    section = md[md.index("## Cross-embedder"):]
    assert "cannot be assessed" in section
    assert "only 2 model(s) ranked" in section
    assert "property of the arithmetic" in section
    assert "the ranking is robust" not in section


def test_robustness_verdict_is_given_once_enough_models_are_ranked(summary, meta):
    s = dict(summary)
    s["rank_correlation"] = [
        {"a": "emb-a", "b": "emb-b", "rho": 0.95, "n_models": 8},
    ]
    section = report.build_markdown(s, meta)
    section = section[section.index("## Cross-embedder"):]
    assert "the ranking is robust" in section
    assert "cannot be assessed" not in section


# --------------------------------------------------------------- headline ----
def test_headline_states_position_against_chance(summary, meta):
    md = report.build_markdown(summary, meta)
    sec = md[md.index("## Headline"):]
    sec = sec[: sec.index("## ", 3)] if "## " in sec[3:] else sec
    assert "random common nouns" in sec
    assert "95th percentile" in sec


def test_headline_says_so_when_nothing_beats_chance(summary, meta):
    """The real finding from the first full run: 0/24 cells cleared chance p95."""
    s = dict(summary)
    s["baselines"] = {
        s["primary_embedder"]: {"random": {"mean": 0.373, "sd": 0.023, "n": 1000,
                                          "k": 7, "p05": 0.34, "p50": 0.374,
                                          "p95": 0.4113, "seed": 0}},
    }
    s["models"] = {
        "m1": {"grid": [{"prompt_id": "terse", "temperature": 0.0, "mean": 0.401, "n": 10}]},
    }
    md = report.build_markdown(s, meta)
    assert "No cell cleared the 95th percentile of chance" in md
    assert "not the story" in md or "should not be read as a capability ranking" in md


def test_headline_is_omitted_when_baselines_are_missing(summary, meta):
    s = dict(summary)
    s.pop("baselines")
    md = report.build_markdown(s, meta)
    assert "## Headline" not in md          # no empty heading
    assert "## Leaderboard" in md           # rest of the report still renders
