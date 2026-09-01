"""`python -m datbench <cmd>` -- the only module that touches every stage.

CONTRACT.md section 9 is the authority for the command set. Every command is
idempotent: `run` skips run_ids already on disk, and `score`/`analyze`/`report`
rewrite their outputs from the inputs they read, so re-running a stage after a
crash or a config change is always safe and never costs an API call it does not
have to make.

Stage boundaries and the files that carry state between them:

    run      models.yaml + prompts/      -> runs/responses.jsonl   (append-only)
    score    runs/responses.jsonl        -> out/words.jsonl, out/scores.jsonl,
                                           out/baselines.json, out/score_meta.json
    analyze  those four                  -> out/summary.json, out/summary.csv
    report   out/summary.json            -> out/report.md, out/report.html

out/summary.json is written in report.py's SUMMARY SCHEMA shape; that docstring,
not this module, is the authority for it.

Injection points (all keyword-only, all defaulted) exist so the tests never open
a socket: main(argv, client=..., embedder_factory=..., list_embedders=...).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from datbench import __version__, analyze, report, runner
from datbench.embed import Embedder, list_embedding_models
from datbench.parse import parse_words
from datbench.providers import (
    ChatClient,
    ModelSpec,
    available_models,
    load_run_config,
    load_scoring_config,
)
from datbench.score import BaselineStats, category_floor, random_baseline, score_run
from datbench.validate import WordCheck, capabilities, validate_words

log = logging.getLogger("datbench")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "models.yaml"
DEFAULT_OUT = ROOT / "out"
DEFAULT_RUNS = ROOT / "runs" / "responses.jsonl"
DEFAULT_PROMPTS = ROOT / "prompts"
DEFAULT_CACHE = ROOT / "cache" / "embeddings.sqlite"

# The DAT asks for exactly ten words, so ten is what we read off a response; an
# eleventh token is scaffolding, not an answer.
WANT_WORDS = 10

SCHEMA_VERSION = 1

# One token, no temperature: enough to prove the model id resolves and the key
# works, cheap enough to run against every live entry.
PROBE_PROMPT = "Reply with the single word: ok"

WORDS_FILE = "words.jsonl"
SCORES_FILE = "scores.jsonl"
BASELINES_FILE = "baselines.json"
SCORE_META_FILE = "score_meta.json"
SUMMARY_JSON = "summary.json"
SUMMARY_CSV = "summary.csv"
REPORT_MD = "report.md"
REPORT_HTML = "report.html"


class CliError(Exception):
    """A user-fixable problem: bad flag, missing file, unusable config."""


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #
def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _split(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _str_list(text: str) -> list[str]:
    values = _split(text)
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated list, got nothing")
    return values


def _float_list(text: str) -> list[float]:
    out: list[float] = []
    for part in _split(text):
        try:
            out.append(float(part))
        except ValueError:
            raise argparse.ArgumentTypeError(f"{part!r} is not a number") from None
    if not out:
        raise argparse.ArgumentTypeError("expected a comma-separated list of numbers")
    return out


def _finite(x: Any) -> float | None:
    """A float that JSON can hold, or None. nan/inf mean 'undefined' here, and
    report.py already renders None as an em dash -- writing a bare NaN would
    produce a summary.json that strict JSON parsers reject."""
    if x is None:
        return None
    try:
        value = float(x)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, float):
        return _finite(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _write_json(path: Path, obj: Any) -> None:
    """Atomic replace: a re-run must never leave a half-written summary behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonable(obj), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")
            count += 1
    os.replace(tmp, path)
    return count


def _write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for row in rows:
            writer.writerow(["" if v is None else v for v in row])
    os.replace(tmp, path)


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CliError(f"{path} is not valid JSON: {exc}") from None


def _latest_rows(path: Path) -> list[dict]:
    """Rows keyed by run_id, newest wins, first-seen order preserved.

    responses.jsonl is append-only, so `run --no-resume` can leave two rows for
    one cell. The later row is the newer call; silently averaging both would
    double-count one cell.
    """
    out: dict[str, dict] = {}
    dropped = 0
    for row in runner.iter_rows(path):
        rid = row.get("run_id")
        if not isinstance(rid, str) or not rid:
            dropped += 1
            continue
        out[rid] = row
    if dropped:
        log.warning("%s: skipped %d row(s) with no run_id", path, dropped)
    return list(out.values())


def _temp_key(t: Any) -> str:
    return "none" if t is None else repr(float(t))


def _temp_label(t: Any) -> str:
    return "n/a" if t is None else f"{float(t):g}"


def _load_prompts(prompts_dir: Path, names: Sequence[str]) -> dict[str, str]:
    """Prompt text by stem, in the order asked for."""
    directory = Path(prompts_dir)
    have = sorted(p.stem for p in directory.glob("*.txt")) if directory.is_dir() else []
    out: dict[str, str] = {}
    for name in names:
        path = directory / f"{name}.txt"
        if not path.is_file():
            raise CliError(
                f"no prompt {name!r} in {directory} (available: {', '.join(have) or 'none'})"
            )
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise CliError(f"prompt file {path} is empty")
        out[name] = text
    if not out:
        raise CliError("no prompts selected")
    return out


def _mean(values: Sequence[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


@dataclass
class Deps:
    """Everything that would otherwise open a socket, injectable for tests."""

    client: Any = None
    embedder_factory: Callable[..., Any] | None = None
    list_embedders: Callable[[str], list[str]] | None = None
    _owned_client: bool = field(default=False, init=False)

    def chat_client(self) -> Any:
        if self.client is None:
            self.client = ChatClient()
            self._owned_client = True
        return self.client

    def embedder(self, model: str, base_url: str, cache_path: Path) -> Any:
        if self.embedder_factory is not None:
            return self.embedder_factory(model, base_url=base_url, cache_path=cache_path)
        return Embedder(model, base_url=base_url, cache_path=cache_path)

    def embedding_models(self, base_url: str) -> list[str]:
        if self.list_embedders is not None:
            return list(self.list_embedders(base_url))
        return list_embedding_models(base_url)

    def close(self) -> None:
        if self._owned_client and self.client is not None:
            self.client.close()
            self.client = None
            self._owned_client = False


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
def cmd_models(args: argparse.Namespace, deps: Deps) -> int:
    usable, skipped = available_models(args.config)
    print(f"registry: {args.config}")
    print()
    print(f"LIVE ({len(usable)})")
    if not usable:
        print("  (none)")
    for spec in usable:
        key = spec.api_key_env or "none"
        print(
            f"  {spec.id:<22} {spec.model:<32} {spec.base_url:<48} "
            f"key={key:<18} conc={spec.max_concurrency}"
        )
        if spec.notes:
            print(f"  {'':22} {spec.notes}")
    print()
    print(f"INERT ({len(skipped)})")
    if not skipped:
        print("  (none)")
    for model_id, reason in skipped:
        print(f"  {model_id:<22} {reason}")

    if not args.probe:
        print()
        print("re-run with --probe to send one 1-token request per live entry.")
        return 0

    if not usable:
        print()
        print("nothing to probe.")
        return 0

    print()
    print(f"probing {len(usable)} live entry(ies) with one {PROBE_PROMPT!r} request each")
    client = deps.chat_client()
    failures = 0
    for spec in usable:
        # temperature=None so providers omits the parameter: a reasoning model
        # that rejects temperature must not fail its own probe.
        result = client.complete(
            spec, PROBE_PROMPT, temperature=None, max_tokens=1, timeout=args.timeout
        )
        if result.error:
            failures += 1
            print(f"  FAIL {spec.id:<22} {result.error}")
        else:
            served = result.model_reported or "(not reported)"
            extra = f"  [{result.notes}]" if result.notes else ""
            print(f"  ok   {spec.id:<22} served as {served} ({result.latency_ms} ms){extra}")
    print()
    print(f"{len(usable) - failures} of {len(usable)} live entry(ies) resolved.")
    return 0


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def _select_specs(config: Path, wanted: Sequence[str] | None) -> list[ModelSpec]:
    usable, skipped = available_models(config)
    if wanted is None:
        if not usable:
            lines = "\n".join(f"  {mid}: {reason}" for mid, reason in skipped)
            raise CliError(
                "no usable models in "
                f"{config}. Every entry is inert:\n{lines or '  (registry is empty)'}"
            )
        return usable

    by_id = {spec.id: spec for spec in usable}
    inert = dict(skipped)
    chosen: list[ModelSpec] = []
    for model_id in wanted:
        if model_id in by_id:
            chosen.append(by_id[model_id])
        elif model_id in inert:
            # A typo'd or keyless id that silently ran nothing would look like a
            # model that scored nothing.
            raise CliError(f"model {model_id!r} is inert: {inert[model_id]}")
        else:
            known = ", ".join(sorted(list(by_id) + list(inert)))
            raise CliError(f"unknown model id {model_id!r}. Registry has: {known}")
    return chosen


def _progress_printer(quiet: bool) -> Callable[[dict], None] | None:
    if quiet:
        return None

    def show(event: dict) -> None:
        mark = "ok " if event["ok"] else "ERR"
        print(
            f"[{event['index']:>4}/{event['total']}] "
            f"{event['model_id']:<20} {event['cell']:<20} "
            f"rep {event['replicate']:<3} {mark} "
            f"{event['n_candidates']:>2}w  "
            f"mean {event['mean_candidates']:.1f}w  "
            f"{event['latency_ms']:>6}ms  errors {event['errors']}",
            flush=True,
        )
        if event["error"]:
            print(f"{'':>12} ! {event['error']}", flush=True)

    return show


def cmd_run(args: argparse.Namespace, deps: Deps) -> int:
    run_cfg = load_run_config(args.config)
    n = args.n if args.n is not None else int(run_cfg["n"])
    temps = args.temps if args.temps is not None else [float(t) for t in run_cfg["temperatures"]]
    prompt_names = args.prompts if args.prompts is not None else list(run_cfg["prompts"])

    specs = _select_specs(args.config, args.models)
    prompts = _load_prompts(args.prompts_dir, prompt_names)
    tasks = runner.build_tasks(specs, prompts, temps, n)

    fixed = [s.id for s in specs if not s.supports_temperature]
    print(
        f"{len(specs)} model(s) x {len(prompts)} prompt(s) x "
        f"{len(temps)} temperature(s) x n={n} -> {len(tasks)} call(s)"
    )
    print(f"  models      : {', '.join(s.id for s in specs)}")
    print(f"  prompts     : {', '.join(prompts)}")
    print(f"  temperatures: {', '.join(f'{t:g}' for t in temps)}")
    if fixed:
        print(f"  no temperature support (single cell, recorded as null): {', '.join(fixed)}")
    print(f"  responses   : {args.runs}")

    if args.dry_run:
        already = runner.existing_run_ids(args.runs)
        todo = [t for t in tasks if t.run_id not in already]
        print(f"  dry run: {len(todo)} call(s) would be made, {len(tasks) - len(todo)} skipped")
        for task in todo[:20]:
            print(f"    {task.run_id}  {task.spec.id:<20} {task.cell:<20} rep {task.replicate}")
        if len(todo) > 20:
            print(f"    ... and {len(todo) - 20} more")
        return 0

    counts = runner.run_all(
        tasks,
        deps.chat_client(),
        args.runs,
        resume=not args.no_resume,
        progress=_progress_printer(args.quiet),
        timeout=args.timeout,
    )
    print()
    print(
        f"run: attempted {counts['attempted']}, written {counts['written']}, "
        f"skipped {counts['skipped']} (already recorded), errors {counts['errors']}"
    )
    if counts["written"] == 0 and counts["skipped"]:
        print("nothing to do: every cell was already in the responses file.")
    if counts["written"] and counts["errors"] == counts["written"]:
        # Every single call failed. Continuing to score would report a dead
        # endpoint or a wrong model id as a model that produced no valid words.
        raise CliError(
            f"all {counts['errors']} call(s) failed -- see the errors above. "
            "Check the endpoint and the model id (`python -m datbench models --probe`)."
        )
    return 0


# --------------------------------------------------------------------------- #
# score
# --------------------------------------------------------------------------- #
def _resolve_embedders(args: argparse.Namespace, cfg: dict, deps: Deps) -> list[str]:
    requested = args.embedders if args.embedders is not None else cfg["embedders"]
    if isinstance(requested, str):
        requested = _split(requested) or ["auto"]
    requested = [str(r) for r in requested]
    if len(requested) == 1 and requested[0] == "auto":
        found = deps.embedding_models(args.embed_base_url or cfg["embed_base_url"])
        if not found:
            raise CliError(
                "no embedding models found at "
                f"{args.embed_base_url or cfg['embed_base_url']}: is LM Studio running "
                "with an embedding model loaded? (pass --embedders id1,id2 to name them)"
            )
        return found
    return list(dict.fromkeys(requested))


def _words_row(run_id: str, candidates: Sequence[str], checks: Sequence[WordCheck]) -> dict:
    return {
        "run_id": run_id,
        "candidates": list(candidates),
        "words": [
            {
                "word": c.word,
                "clean": c.clean,
                "valid": c.valid,
                "flags": list(c.flags),
                "zipf": _finite(c.zipf),
            }
            for c in checks
        ],
    }


def cmd_score(args: argparse.Namespace, deps: Deps) -> int:
    cfg = load_scoring_config(args.config)
    n_use = args.n_use if args.n_use is not None else int(cfg["n_use"])
    min_words = (
        args.min_words if args.min_words is not None else int(cfg["min_words_lenient"])
    )
    rare_threshold = (
        args.rare_zipf if args.rare_zipf is not None else float(cfg["rare_zipf_threshold"])
    )
    draws = args.baseline_draws if args.baseline_draws is not None else int(cfg["baseline_draws"])
    seed = int(cfg.get("baseline_seed", 0))
    policies = args.policies if args.policies is not None else [str(p) for p in cfg["policies"]]
    base_url = args.embed_base_url or str(cfg["embed_base_url"])
    out_dir = Path(args.out_dir)

    rows = _latest_rows(args.runs)
    if not rows:
        raise CliError(
            f"no responses in {args.runs}: run `python -m datbench run` first"
        )

    # Parse and validate ONCE per run: the checks are embedder-independent, and
    # re-validating per embedder would multiply the only slow local step.
    checks_by_run: dict[str, list[WordCheck]] = {}
    words_rows: list[dict] = []
    n_failed = 0
    for row in rows:
        run_id = str(row["run_id"])
        if row.get("error"):
            # A failed call produced no words at all. Scoring it would report a
            # provider outage as a model emitting invalid words.
            n_failed += 1
            continue
        candidates = parse_words(row.get("response_text") or "", want=WANT_WORDS)
        checks = validate_words(candidates, rare_zipf_threshold=rare_threshold)
        checks_by_run[run_id] = checks
        words_rows.append(_words_row(run_id, candidates, checks))

    # The whole reason embeddings are cheap here: every valid word from every run
    # goes to each embedder in one deduplicated call, not once per run.
    union = sorted({c.clean for checks in checks_by_run.values() for c in checks if c.valid})
    print(
        f"{len(rows)} run(s): {len(checks_by_run)} with a response, {n_failed} failed. "
        f"{len(union)} distinct valid word(s) to embed."
    )

    embedders = _resolve_embedders(args, cfg, deps)
    print(f"embedders: {', '.join(embedders)}")

    score_rows: list[dict] = []
    baselines: dict[str, dict] = {}
    warnings: list[str] = []
    used: list[str] = []

    for name in embedders:
        embedder = deps.embedder(name, base_url, Path(args.cache))
        try:
            vecs = embedder.embed(union) if union else {}
            missing = len(union) - len(vecs)
            if union and not vecs:
                warnings.append(f"embedder {name!r} returned no vectors at all; skipped")
                print(f"  {name}: no vectors returned -- skipping this embedder")
                continue
            if missing:
                warnings.append(f"embedder {name!r} could not embed {missing} word(s)")

            baseline_block: dict[str, Any] = {}
            try:
                stats = random_baseline(embedder.embed, n_draws=draws, k=n_use, seed=seed)
                baseline_block["random"] = {
                    "mean": stats.mean, "sd": stats.sd, "n": stats.n, "k": stats.k,
                    "p05": stats.p05, "p50": stats.p50, "p95": stats.p95,
                    "seed": stats.seed,
                    # `draws` is deliberately not serialised: BaselineStats
                    # reconstructs without it and falls back to a normal
                    # approximation for percentile_of.
                }
            except Exception as exc:  # noqa: BLE001 -- a missing anchor is not a dead stage
                warnings.append(f"random baseline failed for {name!r}: {exc}")
                print(f"  {name}: random baseline failed: {exc}")
            try:
                baseline_block["categories"] = category_floor(embedder.embed)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"category floor failed for {name!r}: {exc}")
            if baseline_block:
                baselines[name] = baseline_block

            counts = Counter()
            for run_id, checks in checks_by_run.items():
                for policy in policies:
                    result = score_run(
                        checks, vecs, policy=policy, n_use=n_use, min_words=min_words
                    )
                    counts[(policy, result.scored)] += 1
                    score_rows.append(
                        {
                            "run_id": run_id,
                            "embedder": name,
                            "policy": policy,
                            "score": _finite(result.score),
                            "n_candidates": result.n_candidates,
                            "n_valid": result.n_valid,
                            "n_words_used": result.n_words_used,
                            "valid_rate": result.valid_rate,
                            "scored": result.scored,
                            "reason": result.reason,
                        }
                    )
            used.append(name)
            summary = ", ".join(
                f"{p}: {counts[(p, True)]} scored / {counts[(p, False)]} refused"
                for p in policies
            )
            print(f"  {name}: {summary}")
        finally:
            embedder.close()

    if not used:
        raise CliError(
            "no embedder produced vectors; nothing scored. "
            f"Check that an embedding model is loaded at {base_url}."
        )

    n_words = _write_jsonl(out_dir / WORDS_FILE, words_rows)
    n_scores = _write_jsonl(out_dir / SCORES_FILE, score_rows)
    _write_json(out_dir / BASELINES_FILE, baselines)
    # Provenance for analyze/report: the capabilities recorded here are the ones
    # that actually judged these words, which is not necessarily what is
    # installed when the report is rendered.
    _write_json(
        out_dir / SCORE_META_FILE,
        {
            "generated_at": _now(),
            "tool_version": __version__,
            # The embedders that actually produced scores, in request order --
            # analyze reads this as the display order, and naming one that was
            # skipped would put an empty column in the report.
            "embedders": used,
            "requested_embedders": embedders,
            "policies": policies,
            "n_use": n_use,
            "min_words_lenient": min_words,
            "rare_zipf_threshold": rare_threshold,
            "want": WANT_WORDS,
            "baseline_draws": draws,
            "baseline_seed": seed,
            "embed_base_url": base_url,
            "capabilities": capabilities(),
            "n_runs": len(rows),
            "n_failed": n_failed,
            "n_distinct_words": len(union),
            "warnings": warnings,
        },
    )
    print()
    print(f"wrote {out_dir / WORDS_FILE} ({n_words} rows)")
    print(f"wrote {out_dir / SCORES_FILE} ({n_scores} rows)")
    print(f"wrote {out_dir / BASELINES_FILE} ({len(baselines)} embedder(s))")
    caps = capabilities()
    off = [k for k, v in caps.items() if not v]
    if off:
        print(f"WARNING: validity checks unavailable: {', '.join(off)} -- see the report.")
    return 0


# --------------------------------------------------------------------------- #
# analyze
# --------------------------------------------------------------------------- #
def _ordered(preferred: Sequence[str], observed: Sequence[str]) -> list[str]:
    """Preferred order, filtered to what was actually observed, extras appended."""
    out = [p for p in preferred if p in observed]
    out += [o for o in observed if o not in out]
    return out


def _baseline_stats(baselines: Any, embedder: str) -> BaselineStats | None:
    block = baselines.get(embedder) if isinstance(baselines, dict) else None
    if not isinstance(block, dict):
        return None
    inner = block.get("random")
    stats = inner if isinstance(inner, dict) else block
    try:
        # draws are not in baselines.json by design, so percentile_of falls back
        # to the normal approximation -- see score.BaselineStats.
        return BaselineStats(
            mean=float(stats["mean"]),
            sd=float(stats["sd"]),
            n=int(stats["n"]),
            k=int(stats["k"]),
            p05=float(stats["p05"]),
            p50=float(stats["p50"]),
            p95=float(stats["p95"]),
            seed=int(stats.get("seed", 0)),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _sorted_cells(
    cellmap: dict, prompts_order: Sequence[str], temps_order: Sequence[Any]
) -> list[tuple[tuple[str, str], dict]]:
    p_rank = {p: i for i, p in enumerate(prompts_order)}
    t_rank = {_temp_key(t): i for i, t in enumerate(temps_order)}
    return sorted(
        cellmap.items(),
        key=lambda kv: (p_rank.get(kv[0][0], 1 << 30), t_rank.get(kv[0][1], 1 << 30), kv[0]),
    )


def _round(x: Any, nd: int = 6) -> float | None:
    value = _finite(x)
    return None if value is None else round(value, nd)


def _headline_embedder(embedders: list[str], configured: object) -> str:
    """Pick the embedder whose numbers head the leaderboard.

    Falling back to embedders[0] means the headline is chosen by the alphabet, which
    on this machine hands it to embeddinggemma-300m (15.8% separation between chance
    and a tight-category floor) over qwen3-embedding-4b (29.4%) -- i.e. the weaker
    discriminator wins on sort order. models.yaml records the measured choice; see
    scripts/premise_check.py. Substring match, so the config can name the family
    without pinning the provider's full id string.
    """
    if isinstance(configured, str) and configured.strip():
        want = configured.strip().lower()
        exact = [e for e in embedders if e.lower() == want]
        if exact:
            return exact[0]
        partial = [e for e in embedders if want in e.lower()]
        if partial:
            return sorted(partial, key=len)[0]
        # Naming an embedder that was not scored is a config error worth surfacing,
        # but not worth failing the whole analyze stage over.
        print(
            f"warning: headline_embedder {configured!r} matched none of the scored "
            f"embedders ({', '.join(embedders)}); falling back to {embedders[0]}",
            file=sys.stderr,
        )
    return embedders[0]


def cmd_analyze(args: argparse.Namespace, deps: Deps) -> int:
    out_dir = Path(args.out_dir)
    rows = _latest_rows(args.runs)
    if not rows:
        raise CliError(f"no responses in {args.runs}: run `python -m datbench run` first")
    score_rows = list(runner.iter_rows(out_dir / SCORES_FILE))
    if not score_rows:
        raise CliError(
            f"no scores in {out_dir / SCORES_FILE}: run `python -m datbench score` first"
        )
    words_rows = {
        str(r["run_id"]): r for r in runner.iter_rows(out_dir / WORDS_FILE) if r.get("run_id")
    }
    baselines = _read_json(out_dir / BASELINES_FILE) or {}
    score_meta = _read_json(out_dir / SCORE_META_FILE) or {}
    run_cfg = load_run_config(args.config)
    scoring_cfg = load_scoring_config(args.config)

    # ---- per-run facts ---------------------------------------------------
    run_info: dict[str, dict] = {}
    for row in rows:
        run_info[str(row["run_id"])] = {
            "model_id": str(row.get("model_id") or "?"),
            "prompt_id": str(row.get("prompt_id") or "?"),
            "temperature": row.get("temperature"),
            "replicate": row.get("replicate") or 0,
            "error": row.get("error"),
            "model_reported": str(row.get("model_reported") or ""),
            # finish_reason == "length" means the harness cut the model off, not that
            # the model answered badly. Without carrying it here the two are
            # indistinguishable downstream, and a too-low max_tokens silently reads
            # as a capability result.
            "finish_reason": str(row.get("finish_reason") or ""),
        }

    valid_words: dict[str, set[str]] = {}
    run_valid_rate: dict[str, float] = {}
    flag_counts: Counter[str] = Counter()
    for rid, wrow in words_rows.items():
        words = wrow.get("words") or []
        valid = [w for w in words if w.get("valid")]
        valid_words[rid] = {str(w.get("clean")) for w in valid}
        # A response with no candidates at all is a real 0%, not a missing value:
        # the model was asked for ten words and produced nothing usable.
        run_valid_rate[rid] = (len(valid) / len(words)) if words else 0.0
        for w in words:
            for flag in w.get("flags") or []:
                flag_counts[str(flag)] += 1

    scores_idx: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for srow in score_rows:
        rid, emb, pol = srow.get("run_id"), srow.get("embedder"), srow.get("policy")
        if rid and emb and pol:
            scores_idx[(str(emb), str(pol))][str(rid)] = srow

    # ---- axes ------------------------------------------------------------
    embedders = _ordered(
        [str(e) for e in (score_meta.get("embedders") or [])],
        sorted({key[0] for key in scores_idx}),
    )
    policies = _ordered(
        [str(p) for p in (score_meta.get("policies") or ["strict", "lenient"])],
        sorted({key[1] for key in scores_idx}),
    )
    if not embedders or not policies:
        raise CliError(
            f"{out_dir / SCORES_FILE} has no row carrying both an embedder and a policy; "
            "re-run `python -m datbench score`"
        )
    primary_embedder = args.primary_embedder or _headline_embedder(
        embedders, scoring_cfg.get("headline_embedder")
    )
    if primary_embedder not in embedders:
        raise CliError(
            f"embedder {primary_embedder!r} has no scores. Scored: {', '.join(embedders)}"
        )
    primary_policy = args.primary_policy or ("strict" if "strict" in policies else policies[0])
    if primary_policy not in policies:
        raise CliError(f"policy {primary_policy!r} has no scores. Scored: {', '.join(policies)}")

    prompts_order = _ordered(
        [str(p) for p in run_cfg["prompts"]],
        list(dict.fromkeys(info["prompt_id"] for info in run_info.values())),
    )
    temp_values: dict[str, Any] = {}
    for info in run_info.values():
        temp_values.setdefault(_temp_key(info["temperature"]), info["temperature"])
    cfg_temp_keys = [_temp_key(float(t)) for t in run_cfg["temperatures"]]
    temps_order = [temp_values[k] for k in cfg_temp_keys if k in temp_values]
    temps_order += [v for k, v in temp_values.items() if k not in set(cfg_temp_keys)]

    model_order = list(dict.fromkeys(info["model_id"] for info in run_info.values()))
    runs_by_model: dict[str, list[str]] = defaultdict(list)
    cells: dict[str, dict[tuple[str, str], dict]] = {mid: {} for mid in model_order}
    for rid, info in run_info.items():
        runs_by_model[info["model_id"]].append(rid)
        key = (info["prompt_id"], _temp_key(info["temperature"]))
        cell = cells[info["model_id"]].setdefault(
            key,
            {"prompt_id": info["prompt_id"], "temperature": info["temperature"], "run_ids": []},
        )
        cell["run_ids"].append(rid)
    for cellmap in cells.values():
        for cell in cellmap.values():
            cell["run_ids"].sort(key=lambda r: (run_info[r]["replicate"], r))

    # ---- cell statistics -------------------------------------------------
    def cell_shape(cell: dict) -> dict:
        rids = cell["run_ids"]
        answered = [r for r in rids if r in words_rows]
        sets = [valid_words[r] for r in answered]
        return {
            "n_runs": len(rids),
            "n_failed": sum(1 for r in rids if run_info[r]["error"]),
            "valid_rate": _mean([run_valid_rate[r] for r in answered]),
            "jaccard": analyze.jaccard_self_overlap(sets),
            "distinct_pool": analyze.distinct_pool(sets),
        }

    def cell_numbers(cell: dict, embedder: str, policy: str) -> tuple[dict, int, list[dict]]:
        idx = scores_idx.get((embedder, policy), {})
        got = [idx[r] for r in cell["run_ids"] if r in idx]
        values = [
            float(row["score"])
            for row in got
            if row.get("scored") and _finite(row.get("score")) is not None
        ]
        refused = [row for row in got if not row.get("scored")]
        return analyze.cell_stats(values), len(values), refused

    shapes = {
        (mid, key): cell_shape(cell)
        for mid, cellmap in cells.items()
        for key, cell in cellmap.items()
    }

    spec_notes: dict[str, str] = {}
    try:
        usable, _skipped = available_models(args.config)
        spec_notes = {spec.id: spec.notes for spec in usable if spec.notes}
    except Exception as exc:  # noqa: BLE001 -- notes are cosmetic, the summary is not
        log.debug("could not read registry notes: %s", exc)

    # ---- leaderboard + per-model detail ---------------------------------
    baseline_of = {emb: _baseline_stats(baselines, emb) for emb in embedders}
    leaderboard: list[dict] = []
    models_detail: dict[str, dict] = {}
    warnings: list[str] = [str(w) for w in (score_meta.get("warnings") or [])]
    stale = [
        rid for rid, info in run_info.items() if not info["error"] and rid not in words_rows
    ]
    if stale:
        # Responses arrived after the last score stage: their cells would be
        # silently under-counted rather than visibly missing.
        warnings.append(
            f"{len(stale)} response(s) have no words/scores recorded; "
            "re-run `python -m datbench score`"
        )

    for mid in model_order:
        mruns = runs_by_model[mid]
        n_runs = len(mruns)
        n_failed = sum(1 for r in mruns if run_info[r]["error"])
        model_valid_rate = _mean([run_valid_rate[r] for r in mruns if r in words_rows])
        primary = scores_idx.get((primary_embedder, primary_policy), {})
        got = [primary[r] for r in mruns if r in primary]
        n_scored = sum(1 for g in got if g.get("scored"))
        n_refused = sum(1 for g in got if not g.get("scored"))
        reasons = Counter(
            str(g.get("reason") or "unspecified") for g in got if not g.get("scored")
        )

        grid: list[dict] = []
        best: dict | None = None
        for key, cell in _sorted_cells(cells[mid], prompts_order, temps_order):
            shape = shapes[(mid, key)]
            stats, n_cell_scored, refused = cell_numbers(cell, primary_embedder, primary_policy)
            entry = {
                "prompt_id": cell["prompt_id"],
                "temperature": cell["temperature"],
                "n": n_cell_scored,
                "mean": _finite(stats["mean"]),
                "sd": _finite(stats["sd"]),
                "ci_lo": _finite(stats["ci_lo"]),
                "ci_hi": _finite(stats["ci_hi"]),
                "min": _finite(stats["min"]),
                "max": _finite(stats["max"]),
                "valid_rate": _finite(shape["valid_rate"]),
                "jaccard": _finite(shape["jaccard"]),
                "distinct_pool": _jsonable(shape["distinct_pool"]),
                "n_runs": shape["n_runs"],
                "n_refused": len(refused),
                "n_failed": shape["n_failed"],
            }
            grid.append(entry)
            if entry["mean"] is not None and (best is None or entry["mean"] > best["mean"]):
                best = entry

        # A truncated run is a harness limit (max_tokens too low for this model's
        # reasoning), not a capability result. Counted separately so it cannot be
        # read as the model failing the task -- and broken out per prompt, because
        # truncation hits elaborate prompts first and would otherwise masquerade as
        # a genuine prompt-sensitivity finding.
        truncated = [r for r in mruns if run_info[r]["finish_reason"] == "length"]
        trunc_by_prompt = Counter(run_info[r]["prompt_id"] for r in truncated)
        counts = {
            "n_runs": n_runs,
            "n_scored": n_scored,
            "n_refused": n_refused,
            "n_failed": n_failed,
            "n_truncated": len(truncated),
        }
        reported = sorted({run_info[r]["model_reported"] for r in mruns if run_info[r]["model_reported"]})
        models_detail[mid] = {
            "model_reported": reported,
            **counts,
            "valid_rate": _finite(model_valid_rate),
            "notes": spec_notes.get(mid, ""),
            "refusal_reasons": [[reason, count] for reason, count in reasons.most_common(6)],
            "truncated_by_prompt": [[p, c] for p, c in trunc_by_prompt.most_common()],
            "grid": grid,
        }

        entry = {"model_id": mid, "model_reported": reported, **counts,
                 "valid_rate": _finite(model_valid_rate)}
        if best is None:
            # Still a row: a model that never produced a scorable answer is a
            # result, and dropping it would hide the denominator.
            entry["mean"] = None
            warnings.append(f"{mid}: no scorable runs under {primary_policy}/{primary_embedder}")
        else:
            entry.update(
                best_prompt=best["prompt_id"],
                best_temperature=best["temperature"],
                n=best["n"],
                mean=best["mean"],
                sd=best["sd"],
                ci_lo=best["ci_lo"],
                ci_hi=best["ci_hi"],
                jaccard=best["jaccard"],
                distinct_pool=best["distinct_pool"],
            )
            if model_valid_rate is not None:
                entry["gaming_index"] = _round(
                    analyze.gaming_index(best["mean"], model_valid_rate)
                )
            stats = baseline_of.get(primary_embedder)
            if stats is not None:
                entry["percentile_vs_chance"] = _finite(stats.percentile_of(best["mean"]))
                entry["z_vs_chance"] = _finite(stats.z_of(best["mean"]))
        leaderboard.append(entry)

    leaderboard.sort(key=lambda e: (e.get("mean") is None, -(e.get("mean") or 0.0), e["model_id"]))

    # ---- cross-embedder agreement ---------------------------------------
    # Per model, the mean over ALL its scored runs -- not the best cell. Best-cell
    # means would let each embedder pick a different prompt, so a disagreement
    # about prompts would masquerade as a disagreement about models.
    by_embedder: dict[str, dict[str, float]] = {}
    for emb in embedders:
        idx = scores_idx.get((emb, primary_policy), {})
        per_model: dict[str, float] = {}
        for mid in model_order:
            values = [
                float(idx[r]["score"])
                for r in runs_by_model[mid]
                if r in idx and idx[r].get("scored") and _finite(idx[r].get("score")) is not None
            ]
            if values:
                per_model[mid] = sum(values) / len(values)
        if per_model:
            by_embedder[emb] = per_model

    rank_correlation: list[dict] = []
    if len(by_embedder) >= 2:
        for (a, b), rho in sorted(analyze.rank_correlation(by_embedder).items()):
            rank_correlation.append(
                {
                    "a": a,
                    "b": b,
                    "rho": _round(rho),
                    "n_models": len(set(by_embedder[a]) & set(by_embedder[b])),
                }
            )
    else:
        warnings.append(
            "only one embedder produced scores: the ranking's robustness to embedder "
            "choice cannot be assessed"
        )

    caps = score_meta.get("capabilities")
    if not isinstance(caps, dict) or not caps:
        caps = capabilities()
        warnings.append(
            "validity capabilities were not recorded by the score stage; the report "
            "shows this machine's current capabilities instead"
        )
    off = sorted(k for k, v in caps.items() if not v)
    if off:
        warnings.append(f"validity checks unavailable during scoring: {', '.join(off)}")

    primary = scores_idx.get((primary_embedder, primary_policy), {})
    run_counts = {
        "written": len(rows),
        "failed": sum(1 for info in run_info.values() if info["error"]),
        "scored": sum(1 for v in primary.values() if v.get("scored")),
        "refused": sum(1 for v in primary.values() if not v.get("scored")),
    }

    # How much of the grid each scorer places above its OWN chance band. Reporting
    # only the primary embedder's count invites quoting the most generous of the
    # four as though it were the result: measured here they span 12/180 to 72/180,
    # and two scorers place no model at all above their p95 -- i.e. they have
    # almost no headroom, so a strong correlation on them describes models
    # climbing up to chance rather than past it.
    chance_coverage = {}
    for emb in embedders:
        base = baseline_of.get(emb)
        if not base:
            continue
        cell_means, model_means = [], defaultdict(list)
        # `cells` is the internal shape carrying run_ids; models_detail["grid"] is
        # the already-serialised output and has no run_ids to score from.
        for mid, cellmap in cells.items():
            for _key, cell in cellmap.items():
                stats, _n, _ref = cell_numbers(cell, emb, primary_policy)
                mu = stats.get("mean")
                if mu is None or (isinstance(mu, float) and math.isnan(mu)):
                    continue
                cell_means.append(mu)
                model_means[mid].append(mu)
        if not cell_means:
            continue
        p95 = base.p95
        cmean = base.mean
        per_model = {m: sum(v) / len(v) for m, v in model_means.items()}
        chance_coverage[emb] = {
            "chance_mean": _round(cmean),
            "chance_p95": _round(p95),
            "cells_total": len(cell_means),
            "cells_above_mean": sum(1 for v in cell_means if v > cmean),
            "cells_above_p95": sum(1 for v in cell_means if v > p95),
            "models_total": len(per_model),
            "models_above_p95": sum(1 for v in per_model.values() if v > p95),
            "max_z": _round(base.z_of(max(cell_means))),
        }

    summary = {
        "schema_version": SCHEMA_VERSION,
        "primary_embedder": primary_embedder,
        "primary_policy": primary_policy,
        "embedders": embedders,
        "policies": policies,
        "n_replicates": max((int(info["replicate"] or 0) for info in run_info.values()), default=0)
        or int(run_cfg["n"]),
        "n_use": int(score_meta.get("n_use") or 7),
        "prompts": prompts_order,
        "temperatures": temps_order,
        "capabilities": caps,
        "baselines": baselines,
        "run_counts": run_counts,
        "leaderboard": leaderboard,
        "models": models_detail,
        "rank_correlation": rank_correlation,
        "chance_coverage": chance_coverage,
        "flag_counts": dict(flag_counts.most_common()),
        "warnings": list(dict.fromkeys(warnings)),
    }
    _write_json(out_dir / SUMMARY_JSON, summary)

    # ---- long-format csv: one row per (model, cell, embedder, policy) ----
    header = [
        "model_id", "model_reported", "prompt_id", "temperature", "embedder", "policy",
        "n_scored", "mean", "sd", "ci_lo", "ci_hi", "min", "max",
        "valid_rate", "gaming_index", "jaccard", "distinct", "total", "distinct_ratio",
        "n_runs", "n_refused", "n_failed",
    ]
    csv_rows: list[list[Any]] = []
    for emb in embedders:
        for policy in policies:
            for mid in model_order:
                reported = "|".join(models_detail[mid]["model_reported"])
                for key, cell in _sorted_cells(cells[mid], prompts_order, temps_order):
                    shape = shapes[(mid, key)]
                    stats, n_cell_scored, refused = cell_numbers(cell, emb, policy)
                    mean = _finite(stats["mean"])
                    rate = _finite(shape["valid_rate"])
                    pool = shape["distinct_pool"]
                    csv_rows.append(
                        [
                            mid, reported, cell["prompt_id"], _temp_label(cell["temperature"]),
                            emb, policy, n_cell_scored,
                            _round(mean), _round(stats["sd"]), _round(stats["ci_lo"]),
                            _round(stats["ci_hi"]), _round(stats["min"]), _round(stats["max"]),
                            _round(rate),
                            _round(analyze.gaming_index(mean, rate))
                            if (mean is not None and rate is not None) else None,
                            _round(shape["jaccard"]),
                            pool["distinct"], pool["total"], _round(pool["ratio"]),
                            shape["n_runs"], len(refused), shape["n_failed"],
                        ]
                    )
    _write_csv(out_dir / SUMMARY_CSV, header, csv_rows)

    print(f"wrote {out_dir / SUMMARY_JSON}")
    print(f"wrote {out_dir / SUMMARY_CSV} ({len(csv_rows)} rows)")
    print(
        f"primary: {primary_embedder} / {primary_policy} | "
        f"{len(model_order)} model(s), {len(embedders)} embedder(s), "
        f"{run_counts['scored']} scored, {run_counts['refused']} refused, "
        f"{run_counts['failed']} failed"
    )
    for entry in leaderboard[:10]:
        mean = entry.get("mean")
        print(
            f"  {entry['model_id']:<22} "
            f"{'--' if mean is None else f'{mean:.4f}'}  "
            f"{entry.get('best_prompt', '-'):<12} T={_temp_label(entry.get('best_temperature'))}"
        )
    for warning in summary["warnings"]:
        print(f"  ! {warning}")
    return 0


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def cmd_report(args: argparse.Namespace, deps: Deps) -> int:
    out_dir = Path(args.out_dir)
    summary = _read_json(out_dir / SUMMARY_JSON)
    if not isinstance(summary, dict) or not summary:
        raise CliError(
            f"no summary at {out_dir / SUMMARY_JSON}: run `python -m datbench analyze` first"
        )
    score_meta = _read_json(out_dir / SCORE_META_FILE) or {}
    meta = {
        "generated_at": _now(),
        "tool_version": __version__,
        # 'git_commit' is deliberately absent: report.py probes git itself when
        # the key is missing, and a wrong commit is worse than none.
        "command": getattr(args, "invocation", "python -m datbench report"),
        "models_yaml": str(args.config),
        "embed_base_url": score_meta.get("embed_base_url") or "",
        "out_dir": str(out_dir),
    }
    if args.title:
        meta["title"] = args.title

    md_path = out_dir / REPORT_MD
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(report.build_markdown(summary, meta), encoding="utf-8")
    print(f"wrote {md_path}")
    if not args.no_html:
        html_path = out_dir / REPORT_HTML
        html_path.write_text(report.build_html(summary, meta), encoding="utf-8")
        print(f"wrote {html_path}")
    return 0


# --------------------------------------------------------------------------- #
# all
# --------------------------------------------------------------------------- #
def cmd_all(args: argparse.Namespace, deps: Deps) -> int:
    for name, func in (
        ("run", cmd_run),
        ("score", cmd_score),
        ("analyze", cmd_analyze),
        ("report", cmd_report),
    ):
        print(f"\n===== {name} =====")
        code = func(args, deps)
        if code != 0:
            return code
        if name == "run" and args.dry_run:
            # Nothing was called, so there is nothing new to score. Stopping
            # here beats failing the next stage on an empty responses file.
            print("dry run: stopping after the run plan.")
            return 0
    return 0


# --------------------------------------------------------------------------- #
# argparse
# --------------------------------------------------------------------------- #
class _Formatter(argparse.RawDescriptionHelpFormatter):
    def __init__(self, prog: str) -> None:
        super().__init__(prog, max_help_position=32, width=100)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, metavar="PATH",
        help=f"model registry + run/scoring defaults (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT, metavar="DIR",
        help=f"where words/scores/summary/report land (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--runs", type=Path, default=DEFAULT_RUNS, metavar="PATH",
        help=f"append-only responses file (default: {DEFAULT_RUNS})",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="-v for INFO logging, -vv for DEBUG",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="suppress per-call progress lines (errors still print)",
    )


def _add_run_flags(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("run")
    group.add_argument(
        "--n", type=int, default=None, metavar="N",
        help="replicates per (model x prompt x temperature) cell (default: models.yaml run.n)",
    )
    group.add_argument(
        "--models", type=_str_list, default=None, metavar="a,b",
        help="only these model ids (default: every live entry in the registry)",
    )
    group.add_argument(
        "--prompts", type=_str_list, default=None, metavar="a,b",
        help="prompt stems from the prompts dir (default: models.yaml run.prompts)",
    )
    group.add_argument(
        "--temps", type=_float_list, default=None, metavar="0,0.7,1",
        help="temperatures to sweep (default: models.yaml run.temperatures). A model with "
             "supports_temperature=false collapses to one cell recorded as null",
    )
    group.add_argument(
        "--prompts-dir", type=Path, default=DEFAULT_PROMPTS, metavar="DIR",
        help=f"directory of <stem>.txt prompt files (default: {DEFAULT_PROMPTS})",
    )
    group.add_argument(
        "--no-resume", action="store_true",
        help="re-issue calls whose run_id is already recorded (default: skip them)",
    )
    group.add_argument(
        "--timeout", type=float, default=120.0, metavar="SEC",
        help="per-request timeout in seconds (default: 120)",
    )
    group.add_argument(
        "--dry-run", action="store_true",
        help="print the plan and the run_ids that would be called, then stop",
    )


def _add_score_flags(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("score")
    group.add_argument(
        "--embedders", type=_str_list, default=None, metavar="auto|a,b",
        help="'auto' asks the endpoint for every model id containing 'embed' "
             "(default: models.yaml scoring.embedders)",
    )
    group.add_argument(
        "--policies", type=_str_list, default=None, metavar="strict,lenient",
        help="scoring policies to apply (default: models.yaml scoring.policies)",
    )
    group.add_argument(
        "--baseline-draws", type=int, default=None, metavar="N",
        help="random-noun draws per embedder for the chance baseline "
             "(default: models.yaml scoring.baseline_draws)",
    )
    group.add_argument(
        "--n-use", type=int, default=None, metavar="N",
        help="valid words scored per run (default: models.yaml scoring.n_use)",
    )
    group.add_argument(
        "--min-words", type=int, default=None, metavar="N",
        help="minimum valid words the lenient policy accepts "
             "(default: models.yaml scoring.min_words_lenient)",
    )
    group.add_argument(
        "--rare-zipf", type=float, default=None, metavar="Z",
        help="Zipf frequency below which a word is flagged 'rare' "
             "(default: models.yaml scoring.rare_zipf_threshold)",
    )
    group.add_argument(
        "--embed-base-url", default=None, metavar="URL",
        help="embedding endpoint (default: models.yaml scoring.embed_base_url)",
    )
    group.add_argument(
        "--cache", type=Path, default=DEFAULT_CACHE, metavar="PATH",
        help=f"sqlite vector cache, keyed on (model, word) (default: {DEFAULT_CACHE})",
    )


def _add_analyze_flags(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("analyze")
    group.add_argument(
        "--primary-embedder", default=None, metavar="ID",
        help="embedder the leaderboard is built from (default: the first one scored)",
    )
    group.add_argument(
        "--primary-policy", default=None, metavar="strict|lenient",
        help="policy the leaderboard is built from (default: strict when present)",
    )


def _add_report_flags(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("report")
    group.add_argument(
        "--html", action="store_true",
        help="write report.html as well as report.md (this is the default; the flag "
             "exists so the documented invocation works)",
    )
    group.add_argument(
        "--no-html", action="store_true",
        help="write only report.md",
    )
    group.add_argument(
        "--title", default=None, metavar="TEXT",
        help="override the report's H1",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m datbench",
        formatter_class=_Formatter,
        description=(
            "Benchmark LLMs on the Divergent Association Task: N runs per "
            "(model x prompt x temperature) cell, scored with local embeddings.\n\n"
            "Scores are mean pairwise cosine distance from LOCAL embeddings, not GloVe "
            "840B-300d, so they are on an arbitrary scale and are NOT comparable to the "
            "published DAT human norms. Read every score against the per-embedder "
            "random-noun baseline the score stage computes."
        ),
        epilog=(
            "typical use:\n"
            "  python -m datbench models --probe        # what is live, and does the id resolve\n"
            "  python -m datbench all                   # run -> score -> analyze -> report\n"
            "  python -m datbench run --n 3 --models gemma-4-e4b --temps 0,1\n"
            "  python -m datbench score --embedders auto --policies strict,lenient\n"
            "\nevery command is idempotent: run skips run_ids already recorded, and the "
            "later stages rewrite their outputs from the files they read."
        ),
    )
    parser.add_argument("--version", action="version", version=f"dat-bench {__version__}")
    subs = parser.add_subparsers(dest="cmd", required=True, metavar="<command>")

    p_models = subs.add_parser(
        "models", formatter_class=_Formatter,
        help="list every registry entry, live or inert, with the reason",
        description="List every models.yaml entry and, for the inert ones, exactly why.",
    )
    _add_common(p_models)
    p_models.add_argument(
        "--probe", action="store_true",
        help="send one 1-token request per live entry to check the model id resolves",
    )
    p_models.add_argument(
        "--timeout", type=float, default=30.0, metavar="SEC",
        help="per-probe timeout in seconds (default: 30)",
    )
    p_models.set_defaults(func=cmd_models)

    p_run = subs.add_parser(
        "run", formatter_class=_Formatter,
        help="issue the factorial of LLM calls (resumable)",
        description=(
            "Issue one call per (model, prompt, temperature, replicate) and append each "
            "result to the responses file as it lands.\n\n"
            "Resumable and idempotent: a run_id already in the file is skipped, so an "
            "interrupted factorial costs nothing to restart. Rows are flushed one at a "
            "time, so a kill -9 loses at most the call in flight."
        ),
    )
    _add_common(p_run)
    _add_run_flags(p_run)
    p_run.set_defaults(func=cmd_run)

    p_score = subs.add_parser(
        "score", formatter_class=_Formatter,
        help="parse + validate + embed + score the recorded responses",
        description=(
            "Read the responses file, extract candidate words, apply the DAT rules, embed "
            "the union of every valid word ONCE per embedder, and score every run under "
            "every policy. Writes words.jsonl, scores.jsonl, baselines.json and "
            "score_meta.json. No LLM calls, so re-scoring with a different embedder or "
            "policy is free."
        ),
    )
    _add_common(p_score)
    _add_score_flags(p_score)
    p_score.set_defaults(func=cmd_score)

    p_analyze = subs.add_parser(
        "analyze", formatter_class=_Formatter,
        help="write out/summary.json + out/summary.csv",
        description=(
            "Aggregate the scored runs into per-cell statistics, bootstrap CIs, "
            "mode-collapse Jaccard, distinct word pools, cross-embedder Spearman and the "
            "leaderboard. summary.json is written in report.py's SUMMARY SCHEMA shape."
        ),
    )
    _add_common(p_analyze)
    _add_analyze_flags(p_analyze)
    p_analyze.set_defaults(func=cmd_analyze)

    p_report = subs.add_parser(
        "report", formatter_class=_Formatter,
        help="render out/report.md and out/report.html",
        description="Render the summary as Markdown and (unless --no-html) HTML.",
    )
    _add_common(p_report)
    _add_report_flags(p_report)
    p_report.set_defaults(func=cmd_report)

    p_all = subs.add_parser(
        "all", formatter_class=_Formatter,
        help="run -> score -> analyze -> report",
        description=(
            "Chain every stage, stopping at the first failure. Accepts the flags of all "
            "four stages."
        ),
    )
    _add_common(p_all)
    _add_run_flags(p_all)
    _add_score_flags(p_all)
    _add_analyze_flags(p_all)
    _add_report_flags(p_all)
    p_all.set_defaults(func=cmd_all)

    return parser


def _setup_logging(verbose: int, quiet: bool) -> None:
    if verbose >= 2:
        level = logging.DEBUG
    elif verbose == 1:
        level = logging.INFO
    elif quiet:
        level = logging.ERROR
    else:
        level = logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def main(
    argv: Sequence[str] | None = None,
    *,
    client: Any = None,
    embedder_factory: Callable[..., Any] | None = None,
    list_embedders: Callable[[str], list[str]] | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    _setup_logging(args.verbose, args.quiet)
    args.invocation = "python -m datbench " + " ".join(
        list(argv) if argv is not None else sys.argv[1:]
    )
    deps = Deps(client=client, embedder_factory=embedder_factory, list_embedders=list_embedders)
    try:
        return int(args.func(args, deps))
    except CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        # Every stage flushes as it goes, so an interrupt is resumable, not a loss.
        print("\ninterrupted; re-run to resume", file=sys.stderr)
        return 130
    except (OSError, ValueError) as exc:
        # Malformed yaml, an unreadable prompt file, a bad flag combination: the
        # user's problem to fix, and a traceback would bury the message. -vv
        # still shows it.
        log.debug("stage failed", exc_info=True)
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        deps.close()
