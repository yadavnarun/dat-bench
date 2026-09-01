from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from datbench import cli
from datbench.providers import ChatResult
from datbench.runner import iter_rows

# Two disjoint answer sets, so the two models are genuinely different: `spread`
# gets orthogonal vectors (mean distance 1.0), `tight` gets near-identical ones
# (mean distance ~0.0025). Both fake embedders agree on that ordering, which is
# what makes the Spearman assertion meaningful.
SPREAD = ["cat", "thimble", "river", "guitar", "planet", "ladder", "onion", "candle"]
TIGHT = ["dog", "horse", "cow", "sheep", "goat", "pig", "hen", "duck"]
DIM = 16

YAML = """
defaults:
  max_tokens: 64
  max_concurrency: 1

models:
  - id: spread
    model: vendor/spread
    base_url: http://localhost:1234/v1
    api_key_env: null
    notes: local via LM Studio
  - id: tight
    model: vendor/tight
    base_url: http://localhost:1234/v1
    api_key_env: null
    supports_temperature: false
  - id: keyless
    model: vendor/keyless
    base_url: https://api.vendor.com/v1
    api_key_env: DATBENCH_TEST_MISSING_KEY
  - id: switched-off
    model: vendor/off
    base_url: https://api.vendor.com/v1
    api_key_env: null
    enabled: false

scoring:
  embedders: auto
  embed_base_url: http://localhost:1234/v1
  n_use: 4
  policies: [strict, lenient]
  min_words_lenient: 3
  rare_zipf_threshold: 2.5
  baseline_draws: 20
  baseline_seed: 0

run:
  n: 2
  temperatures: [0.0, 1.0]
  prompts: [terse, verbatim]
"""


def vector(word: str) -> list[float]:
    if word in TIGHT:
        v = [0.0] * DIM
        v[0] = 1.0
        v[1 + TIGHT.index(word)] = 0.05
        return v
    if word in SPREAD:
        v = [0.0] * DIM
        v[SPREAD.index(word)] = 1.0
        return v
    # Baseline vocabulary and category words: deterministic, never zero-norm.
    digest = hashlib.sha256(word.encode()).digest()
    return [(b / 255.0) - 0.4 for b in digest[:DIM]]


class FakeChat:
    def __init__(self, words: dict[str, list[str]]):
        self.words = words
        self.calls: list[dict] = []
        self.fail: set[str] = set()
        self.closed = False

    def complete(self, spec, prompt, *, temperature, max_tokens=None, timeout=120.0):
        self.calls.append(
            {
                "model_id": spec.id,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "prompt": prompt,
                "timeout": timeout,
            }
        )
        if spec.id in self.fail:
            return ChatResult("", "", "", {}, 3, error="http_500: upstream on fire")
        listed = "\n".join(f"{i}. {w}" for i, w in enumerate(self.words[spec.id], 1))
        return ChatResult(
            f"Sure! Here are the words:\n{listed}",
            f"{spec.model}-served",
            "stop",
            {"prompt_tokens": 11, "completion_tokens": 22},
            33,
        )

    def close(self):
        self.closed = True


class FakeEmbedder:
    def __init__(self, model: str):
        self.model = model
        self.calls: list[list[str]] = []
        self.closed = False

    def embed(self, words):
        self.calls.append(list(words))
        return {w: vector(w) for w in words}

    def close(self):
        self.closed = True


@dataclass
class Bench:
    root: Path
    client: FakeChat
    embedder_ids: list[str]
    made: list[FakeEmbedder] = field(default_factory=list)

    @property
    def out(self) -> Path:
        return self.root / "out"

    @property
    def runs(self) -> Path:
        return self.root / "runs" / "responses.jsonl"

    def factory(self, model, base_url=None, cache_path=None):
        made = FakeEmbedder(model)
        self.made.append(made)
        return made

    def cli(self, *argv: str) -> int:
        cmd, rest = argv[0], list(argv[1:])
        common = [
            "--config", str(self.root / "models.yaml"),
            "--out-dir", str(self.out),
            "--runs", str(self.runs),
        ]
        if cmd in ("run", "all"):
            common += ["--prompts-dir", str(self.root / "prompts")]
        if cmd in ("score", "all"):
            common += ["--cache", str(self.root / "cache.sqlite")]
        return cli.main(
            [cmd, *common, *rest],
            client=self.client,
            embedder_factory=self.factory,
            list_embedders=lambda url: list(self.embedder_ids),
        )

    def rows(self, name: str) -> list[dict]:
        return list(iter_rows(self.out / name))

    def json(self, name: str):
        return json.loads((self.out / name).read_text(encoding="utf-8"))


@pytest.fixture
def bench(tmp_path, monkeypatch) -> Bench:
    monkeypatch.delenv("DATBENCH_TEST_MISSING_KEY", raising=False)
    (tmp_path / "models.yaml").write_text(YAML, encoding="utf-8")
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "terse.txt").write_text("List 10 unrelated English nouns.", encoding="utf-8")
    (prompts / "verbatim.txt").write_text(
        "Please enter 10 words that are as different from each other as possible.",
        encoding="utf-8",
    )
    return Bench(
        root=tmp_path,
        client=FakeChat({"spread": list(SPREAD), "tight": list(TIGHT)}),
        embedder_ids=["emb-a", "emb-b"],
    )


# 2 models: spread = 2 prompts x 2 temps x n=2; tight has no temperature axis.
N_RUNS = 8 + 4


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
def test_models_lists_live_and_inert_with_exact_reasons(bench, capsys):
    assert bench.cli("models") == 0
    out = capsys.readouterr().out
    assert "LIVE (2)" in out and "INERT (2)" in out
    assert "spread" in out and "tight" in out
    assert "keyless" in out and "DATBENCH_TEST_MISSING_KEY is not set" in out
    assert "switched-off" in out and "disabled in models.yaml" in out
    assert "local via LM Studio" in out  # registry notes are shown
    assert not bench.client.calls  # listing never calls a provider


def test_models_probe_sends_one_cheap_request_per_live_entry(bench, capsys):
    assert bench.cli("models", "--probe") == 0
    out = capsys.readouterr().out
    assert len(bench.client.calls) == 2
    for call in bench.client.calls:
        # temperature omitted and one token: a reasoning model must not fail its
        # own probe, and a probe must not cost a real generation.
        assert call["temperature"] is None
        assert call["max_tokens"] == 1
    assert "served as vendor/spread-served" in out
    assert "2 of 2 live entry(ies) resolved." in out


def test_models_probe_reports_failures_and_still_exits_zero(bench, capsys):
    bench.client.fail = {"spread", "tight"}
    assert bench.cli("models", "--probe") == 0
    out = capsys.readouterr().out
    assert out.count("FAIL") == 2
    assert "upstream on fire" in out
    assert "0 of 2 live entry(ies) resolved." in out


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def test_run_issues_the_factorial_and_records_every_row(bench, capsys):
    assert bench.cli("run") == 0
    rows = list(iter_rows(bench.runs))
    assert len(rows) == N_RUNS
    assert len(bench.client.calls) == N_RUNS
    assert {r["model_id"] for r in rows} == {"spread", "tight"}
    # the temperature-less model is one cell recorded as null, not three
    tight = [r for r in rows if r["model_id"] == "tight"]
    assert len(tight) == 4
    assert {r["temperature"] for r in tight} == {None}
    assert {r["temperature"] for r in rows if r["model_id"] == "spread"} == {0.0, 1.0}
    assert all(r["model_reported"].endswith("-served") for r in rows)
    out = capsys.readouterr().out
    assert "mean" in out and "12 call(s)" in out


def test_run_resume_makes_zero_duplicate_api_calls(bench, capsys):
    assert bench.cli("run") == 0
    first = len(bench.client.calls)
    assert bench.cli("run") == 0
    assert len(bench.client.calls) == first
    out = capsys.readouterr().out
    assert f"skipped {N_RUNS}" in out
    assert "nothing to do" in out
    assert len(list(iter_rows(bench.runs))) == N_RUNS


def test_run_no_resume_reissues_every_call(bench):
    assert bench.cli("run", "--models", "spread", "--prompts", "terse", "--temps", "0", "--n", "1") == 0
    assert bench.cli("run", "--models", "spread", "--prompts", "terse", "--temps", "0", "--n", "1",
                     "--no-resume") == 0
    assert len(bench.client.calls) == 2
    assert len(list(iter_rows(bench.runs))) == 2


def test_run_dry_run_calls_nothing(bench, capsys):
    assert bench.cli("run", "--dry-run") == 0
    assert bench.client.calls == []
    assert not bench.runs.exists()
    assert "dry run" in capsys.readouterr().out


def test_run_flags_override_the_config(bench, capsys):
    assert bench.cli("run", "--n", "1", "--models", "spread", "--prompts", "terse",
                     "--temps", "0.5") == 0
    assert len(bench.client.calls) == 1
    assert bench.client.calls[0]["temperature"] == 0.5
    (row,) = list(iter_rows(bench.runs))
    assert row["prompt_id"] == "terse" and row["temperature"] == 0.5


def test_run_quiet_suppresses_per_call_lines(bench, capsys):
    assert bench.cli("run", "-q", "--n", "1", "--models", "spread", "--prompts", "terse",
                     "--temps", "0") == 0
    out = capsys.readouterr().out
    assert "[   1/1]" not in out
    assert "written 1" in out


def test_run_refuses_an_unknown_model_id(bench, capsys):
    assert bench.cli("run", "--models", "nope") == 1
    err = capsys.readouterr().err
    assert "unknown model id 'nope'" in err and "spread" in err
    assert bench.client.calls == []


def test_run_refuses_an_inert_model_with_the_reason(bench, capsys):
    assert bench.cli("run", "--models", "keyless") == 1
    err = capsys.readouterr().err
    assert "keyless" in err and "DATBENCH_TEST_MISSING_KEY is not set" in err


def test_run_refuses_an_unknown_prompt(bench, capsys):
    assert bench.cli("run", "--prompts", "nosuchprompt") == 1
    err = capsys.readouterr().err
    assert "nosuchprompt" in err and "available: terse, verbatim" in err


def test_run_reports_when_the_whole_registry_is_inert(bench, capsys):
    (bench.root / "models.yaml").write_text(
        YAML.replace("api_key_env: null\n    notes: local via LM Studio", "enabled: false")
        .replace("    supports_temperature: false", "    enabled: false"),
        encoding="utf-8",
    )
    assert bench.cli("run") == 1
    assert "no usable models" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# score
# --------------------------------------------------------------------------- #
def scored(bench) -> None:
    assert bench.cli("run", "-q") == 0
    assert bench.cli("score") == 0


def test_score_writes_the_four_artifacts_in_contract_shape(bench):
    scored(bench)
    words = bench.rows("words.jsonl")
    scores = bench.rows("scores.jsonl")
    assert len(words) == N_RUNS
    assert len(scores) == N_RUNS * 2 * 2  # runs x embedders x policies

    assert set(words[0]) == {"run_id", "candidates", "words"}
    assert set(words[0]["words"][0]) == {"word", "clean", "valid", "flags", "zipf"}
    assert set(scores[0]) == {
        "run_id", "embedder", "policy", "score", "n_candidates", "n_valid",
        "n_words_used", "valid_rate", "scored", "reason",
    }
    baselines = bench.json("baselines.json")
    assert set(baselines) == {"emb-a", "emb-b"}
    assert set(baselines["emb-a"]["random"]) == {
        "mean", "sd", "n", "k", "p05", "p50", "p95", "seed",
    }
    assert baselines["emb-a"]["random"]["k"] == 4  # scoring.n_use from the yaml
    assert baselines["emb-a"]["categories"]  # tight-set floors recorded


def test_score_embeds_the_union_of_valid_words_once_per_embedder(bench):
    scored(bench)
    assert [e.model for e in bench.made] == ["emb-a", "emb-b"]
    for made in bench.made:
        # call 1: every valid word from every run, deduplicated. Calls 2 and 3
        # are the random-noun baseline and the category floors.
        assert made.calls[0] == sorted(set(SPREAD) | set(TIGHT))
        assert len(made.calls) == 3
        assert made.closed


def test_score_run_ids_join_back_to_the_responses(bench):
    scored(bench)
    run_ids = {r["run_id"] for r in iter_rows(bench.runs)}
    assert {r["run_id"] for r in bench.rows("words.jsonl")} == run_ids
    assert {r["run_id"] for r in bench.rows("scores.jsonl")} == run_ids


def test_score_scores_the_spread_model_far_above_the_tight_one(bench):
    scored(bench)
    by_model = {}
    runs = {r["run_id"]: r for r in iter_rows(bench.runs)}
    for row in bench.rows("scores.jsonl"):
        if row["embedder"] == "emb-a" and row["policy"] == "strict":
            by_model.setdefault(runs[row["run_id"]]["model_id"], []).append(row["score"])
    assert all(abs(s - 1.0) < 1e-9 for s in by_model["spread"])
    assert all(s < 0.01 for s in by_model["tight"])


def test_a_refused_run_is_excluded_not_scored_zero(bench):
    bench.client.words["spread"] = ["cat", "river"]  # 2 valid words, strict needs 4
    scored(bench)
    rows = [r for r in bench.rows("scores.jsonl") if r["embedder"] == "emb-a"]
    runs = {r["run_id"]: r for r in iter_rows(bench.runs)}
    strict = [r for r in rows if r["policy"] == "strict" and runs[r["run_id"]]["model_id"] == "spread"]
    assert strict and all(r["scored"] is False for r in strict)
    assert all(r["score"] is None for r in strict)
    assert all("strict" in (r["reason"] or "") for r in strict)


def test_lenient_scores_what_strict_refuses(bench):
    bench.client.words["spread"] = ["cat", "river", "guitar"]  # 3: lenient min, strict needs 4
    scored(bench)
    runs = {r["run_id"]: r for r in iter_rows(bench.runs)}
    rows = [
        r for r in bench.rows("scores.jsonl")
        if r["embedder"] == "emb-a" and runs[r["run_id"]]["model_id"] == "spread"
    ]
    strict = [r for r in rows if r["policy"] == "strict"]
    lenient = [r for r in rows if r["policy"] == "lenient"]
    assert all(r["scored"] is False for r in strict)
    assert all(r["scored"] is True for r in lenient)
    assert all(r["n_words_used"] == 3 for r in lenient)


def test_score_ignores_failed_calls(bench, capsys):
    bench.client.fail = {"tight"}
    scored(bench)
    out = capsys.readouterr().out
    assert "4 failed" in out
    assert len(bench.rows("words.jsonl")) == N_RUNS - 4
    # no score row invents a result for a call that never returned text
    failed = {r["run_id"] for r in iter_rows(bench.runs) if r["error"]}
    assert failed and not (failed & {r["run_id"] for r in bench.rows("scores.jsonl")})
    assert bench.json("score_meta.json")["n_failed"] == 4


def test_score_flags_invalid_words(bench):
    bench.client.words["spread"] = ["cat", "42", "river", "guitar", "planet"]
    scored(bench)
    flags = [f for row in bench.rows("words.jsonl") for w in row["words"] for f in w["flags"]]
    assert "not_alpha" in flags
    row = next(r for r in bench.rows("scores.jsonl") if r["n_candidates"] == 5)
    assert row["n_valid"] == 4
    assert row["valid_rate"] == pytest.approx(0.8)


def test_score_records_the_capabilities_that_actually_judged_the_words(bench):
    scored(bench)
    meta = bench.json("score_meta.json")
    assert set(meta["capabilities"]) == {"dictionary", "wordnet", "wordfreq"}
    assert meta["n_use"] == 4 and meta["policies"] == ["strict", "lenient"]
    assert meta["embedders"] == ["emb-a", "emb-b"]
    assert meta["n_runs"] == N_RUNS


def test_score_needs_responses_first(bench, capsys):
    assert bench.cli("score") == 1
    assert "run `python -m datbench run` first" in capsys.readouterr().err


def test_score_auto_says_so_when_no_embedder_is_loaded(bench, capsys):
    bench.embedder_ids = []
    assert bench.cli("run", "-q") == 0
    assert bench.cli("score") == 1
    assert "is LM Studio running" in capsys.readouterr().err


def test_score_embedders_flag_overrides_auto(bench):
    assert bench.cli("run", "-q") == 0
    assert bench.cli("score", "--embedders", "only-this-one") == 0
    assert [e.model for e in bench.made] == ["only-this-one"]
    assert set(bench.json("baselines.json")) == {"only-this-one"}


def test_score_is_rerunnable_without_touching_the_api(bench):
    scored(bench)
    calls = len(bench.client.calls)
    before = {name: (bench.out / name).read_bytes() for name in ("words.jsonl", "scores.jsonl")}
    assert bench.cli("score") == 0
    assert len(bench.client.calls) == calls
    for name, blob in before.items():
        assert (bench.out / name).read_bytes() == blob


# --------------------------------------------------------------------------- #
# analyze
# --------------------------------------------------------------------------- #
def analyzed(bench) -> dict:
    scored(bench)
    assert bench.cli("analyze") == 0
    return bench.json("summary.json")


def test_analyze_writes_the_report_summary_schema(bench):
    summary = analyzed(bench)
    for key in (
        "schema_version", "primary_embedder", "primary_policy", "embedders", "policies",
        "n_replicates", "n_use", "prompts", "temperatures", "capabilities", "baselines",
        "run_counts", "leaderboard", "models", "rank_correlation", "flag_counts", "warnings",
    ):
        assert key in summary, key
    assert summary["schema_version"] == 1
    assert summary["primary_embedder"] == "emb-a"
    assert summary["primary_policy"] == "strict"
    assert summary["embedders"] == ["emb-a", "emb-b"]
    assert summary["n_replicates"] == 2
    assert summary["n_use"] == 4
    assert summary["prompts"] == ["terse", "verbatim"]
    assert summary["temperatures"] == [0.0, 1.0, None]
    assert summary["capabilities"] == bench.json("score_meta.json")["capabilities"]


def test_analyze_ranks_the_spread_model_first(bench):
    summary = analyzed(bench)
    board = summary["leaderboard"]
    assert [e["model_id"] for e in board] == ["spread", "tight"]
    top = board[0]
    assert top["mean"] == pytest.approx(1.0)
    assert top["valid_rate"] == pytest.approx(1.0)
    assert top["gaming_index"] == pytest.approx(top["mean"] * top["valid_rate"])
    assert top["n_runs"] == 8 and top["n_scored"] == 8 and top["n_refused"] == 0
    assert top["best_prompt"] in ("terse", "verbatim")
    assert top["ci_lo"] <= top["mean"] <= top["ci_hi"]
    assert top["model_reported"] == ["vendor/spread-served"]
    # the temperature-less model reports its single cell as null, not 0
    assert board[1]["best_temperature"] is None


def test_analyze_measures_mode_collapse_and_pool_breadth(bench):
    summary = analyzed(bench)
    top = summary["leaderboard"][0]
    # the fake repeats the identical word set every replicate: total collapse
    assert top["jaccard"] == pytest.approx(1.0)
    assert top["distinct_pool"] == {"distinct": 8, "total": 16, "ratio": 0.5}


def test_analyze_grid_covers_every_prompt_and_temperature_cell(bench):
    summary = analyzed(bench)
    spread = summary["models"]["spread"]["grid"]
    assert [(c["prompt_id"], c["temperature"]) for c in spread] == [
        ("terse", 0.0), ("terse", 1.0), ("verbatim", 0.0), ("verbatim", 1.0)
    ]
    assert all(c["n"] == 2 for c in spread)
    assert summary["models"]["tight"]["grid"][0]["temperature"] is None
    assert summary["models"]["spread"]["notes"] == "local via LM Studio"


def test_analyze_counts_failed_and_refused_runs_per_model(bench):
    bench.client.fail = {"tight"}
    bench.client.words["spread"] = ["cat", "river"]  # every spread run gets refused
    summary = analyzed(bench)
    detail = summary["models"]
    assert detail["tight"]["n_failed"] == 4
    assert detail["tight"]["n_runs"] == 4
    assert detail["spread"]["n_refused"] == 8
    assert detail["spread"]["n_scored"] == 0
    assert detail["spread"]["refusal_reasons"][0][1] == 8
    assert summary["run_counts"] == {"written": 12, "failed": 4, "scored": 0, "refused": 8}
    # a model with nothing scorable is still a row, with its denominators
    entry = next(e for e in summary["leaderboard"] if e["model_id"] == "spread")
    assert entry["mean"] is None
    assert any("no scorable runs" in w for w in summary["warnings"])


def test_analyze_reports_cross_embedder_agreement(bench):
    summary = analyzed(bench)
    (pair,) = summary["rank_correlation"]
    assert {pair["a"], pair["b"]} == {"emb-a", "emb-b"}
    assert pair["rho"] == pytest.approx(1.0)  # both embedders rank spread over tight
    assert pair["n_models"] == 2


def test_analyze_warns_when_only_one_embedder_scored(bench):
    assert bench.cli("run", "-q") == 0
    assert bench.cli("score", "--embedders", "solo") == 0
    assert bench.cli("analyze") == 0
    summary = bench.json("summary.json")
    assert summary["rank_correlation"] == []
    assert any("only one embedder" in w for w in summary["warnings"])


def test_analyze_percentile_and_z_come_from_the_embedder_baseline(bench):
    summary = analyzed(bench)
    top = summary["leaderboard"][0]
    base = summary["baselines"]["emb-a"]["random"]
    assert top["z_vs_chance"] == pytest.approx((top["mean"] - base["mean"]) / base["sd"])
    assert 0.0 <= top["percentile_vs_chance"] <= 1.0


def test_summary_json_is_strict_json(bench):
    analyzed(bench)
    text = (bench.out / "summary.json").read_text(encoding="utf-8")
    # nan/inf would make this file unreadable to every non-Python consumer
    assert "NaN" not in text and "Infinity" not in text


def test_summary_csv_has_a_row_per_cell_embedder_and_policy(bench):
    analyzed(bench)
    lines = (bench.out / "summary.csv").read_text(encoding="utf-8").strip().splitlines()
    header = lines[0].split(",")
    assert header[:6] == ["model_id", "model_reported", "prompt_id", "temperature",
                          "embedder", "policy"]
    # 6 cells (4 spread + 2 tight) x 2 embedders x 2 policies
    assert len(lines) - 1 == 6 * 2 * 2
    assert any(",n/a," in line for line in lines[1:])  # the temperature-less cell


def test_analyze_rejects_an_embedder_it_has_no_scores_for(bench, capsys):
    scored(bench)
    assert bench.cli("analyze", "--primary-embedder", "ghost") == 1
    assert "has no scores" in capsys.readouterr().err


def test_analyze_can_switch_the_primary_view(bench):
    scored(bench)
    assert bench.cli("analyze", "--primary-embedder", "emb-b",
                     "--primary-policy", "lenient") == 0
    summary = bench.json("summary.json")
    assert summary["primary_embedder"] == "emb-b"
    assert summary["primary_policy"] == "lenient"


def test_analyze_needs_scores_first(bench, capsys):
    assert bench.cli("run", "-q") == 0
    assert bench.cli("analyze") == 1
    assert "run `python -m datbench score` first" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def test_report_writes_markdown_and_html(bench):
    analyzed(bench)
    assert bench.cli("report") == 0
    md = (bench.out / "report.md").read_text(encoding="utf-8")
    html = (bench.out / "report.html").read_text(encoding="utf-8")
    # the four things CONTRACT section 10 requires to be unmissable
    assert "not comparable" in md and "GloVe" in md
    assert "emb-a" in md and "random" in md.lower()
    assert "wordnet" in md.lower()
    assert "refused" in md.lower()
    assert "spread" in md
    assert html.startswith("<!doctype html>")
    assert "<table" in html


def test_report_no_html_writes_only_markdown(bench):
    analyzed(bench)
    assert bench.cli("report", "--no-html") == 0
    assert (bench.out / "report.md").is_file()
    assert not (bench.out / "report.html").exists()


def test_report_html_flag_is_accepted(bench):
    analyzed(bench)
    assert bench.cli("report", "--html") == 0
    assert (bench.out / "report.html").is_file()


def test_report_needs_a_summary_first(bench, capsys):
    assert bench.cli("report") == 1
    assert "run `python -m datbench analyze` first" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# all
# --------------------------------------------------------------------------- #
def test_all_chains_every_stage(bench):
    assert bench.cli("all", "-q") == 0
    for name in ("words.jsonl", "scores.jsonl", "baselines.json", "score_meta.json",
                 "summary.json", "summary.csv", "report.md", "report.html"):
        assert (bench.out / name).is_file(), name
    assert len(bench.client.calls) == N_RUNS


def test_all_is_rerunnable_and_makes_no_new_calls(bench):
    assert bench.cli("all", "-q") == 0
    calls = len(bench.client.calls)
    summary = (bench.out / "summary.json").read_bytes()
    assert bench.cli("all", "-q") == 0
    assert len(bench.client.calls) == calls
    assert (bench.out / "summary.json").read_bytes() == summary


def test_all_stops_after_the_plan_on_dry_run(bench, capsys):
    assert bench.cli("all", "--dry-run") == 0
    assert bench.client.calls == []
    assert not (bench.out / "summary.json").exists()
    assert "stopping after the run plan" in capsys.readouterr().out


def test_run_fails_loudly_when_every_call_failed(bench, capsys):
    bench.client.fail = {"spread", "tight"}
    assert bench.cli("run", "-q") == 1
    err = capsys.readouterr().err
    assert "all 12 call(s) failed" in err
    # the rows are still on disk: a failed run is a record, not a gap
    assert len(list(iter_rows(bench.runs))) == N_RUNS


def test_all_stops_at_the_run_stage_when_every_call_failed(bench):
    bench.client.fail = {"spread", "tight"}
    assert bench.cli("all", "-q") == 1
    assert not (bench.out / "scores.jsonl").exists()


def test_an_interrupt_during_run_exits_130_and_stays_resumable(bench, capsys):
    class Interrupting(FakeChat):
        def complete(self, spec, prompt, **kw):
            if len(self.calls) >= 3:
                raise KeyboardInterrupt
            return super().complete(spec, prompt, **kw)

    bench.client = Interrupting({"spread": list(SPREAD), "tight": list(TIGHT)})
    assert bench.cli("run", "-q") == 130
    assert "re-run to resume" in capsys.readouterr().err
    assert len(list(iter_rows(bench.runs))) == 3

    bench.client = FakeChat({"spread": list(SPREAD), "tight": list(TIGHT)})
    assert bench.cli("run", "-q") == 0
    assert len(bench.client.calls) == N_RUNS - 3


def test_analyze_warns_when_responses_arrived_after_scoring(bench):
    assert bench.cli("run", "-q", "--models", "spread", "--prompts", "terse",
                     "--temps", "0", "--n", "1") == 0
    assert bench.cli("score") == 0
    assert bench.cli("run", "-q", "--models", "spread", "--prompts", "terse",
                     "--temps", "0", "--n", "2") == 0
    assert bench.cli("analyze") == 0
    warnings = bench.json("summary.json")["warnings"]
    assert any("no words/scores recorded" in w for w in warnings)
