"""The factorial loop: build every call, skip what is already on disk, append what lands.

CONTRACT.md section 8 is the authority for RunTask, build_tasks, existing_run_ids
and run_all.

Two properties matter more than anything else here:

* `run_id` is a pure function of the cell coordinates, so a re-run recognises
  work already done and costs zero API calls (CONTRACT section 1).
* every row is written and flushed the moment it lands, so a kill -9 loses at
  most the single call that was in flight.

Additions beyond the contract's signature, all with defaults so a contract-shaped
call is unaffected:

    run_id_for(...)        the id function itself, so cli/tests can join without
                           constructing a RunTask
    iter_rows(path)        tolerant reader for runs/responses.jsonl (cli needs it,
                           and it must forgive the truncated last line that a
                           kill -9 leaves behind)
    run_all(..., timeout=) per-request timeout, passed through to the client
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from datbench.parse import parse_words
from datbench.providers import ChatResult, ModelSpec

log = logging.getLogger(__name__)

__all__ = [
    "RunTask",
    "build_tasks",
    "existing_run_ids",
    "iter_rows",
    "run_all",
    "run_id_for",
]

# Replicates are numbered from 1: the number appears in progress output and in
# every row, and "replicate 0 of 10" reads like a bug report.
FIRST_REPLICATE = 1

_ERROR_MAX_CHARS = 180


def run_id_for(
    model_id: str, prompt_id: str, temperature: float | None, replicate: int
) -> str:
    """sha1 of the cell coordinates, first 16 hex chars (CONTRACT section 1).

    `temperature!r` is part of the payload verbatim, which is why build_tasks
    coerces temperatures to float: `0` and `0.0` name the same cell but would
    otherwise hash differently, and the second run would redo the first one's work.
    """
    payload = f"{model_id}|{prompt_id}|{temperature!r}|{replicate}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class RunTask:
    spec: ModelSpec
    prompt_id: str
    prompt_text: str
    temperature: float | None
    replicate: int

    @property
    def run_id(self) -> str:
        return run_id_for(self.spec.id, self.prompt_id, self.temperature, self.replicate)

    @property
    def cell(self) -> str:
        """Human label for progress output: `verbatim@0.7`, or `verbatim@n/a`."""
        t = "n/a" if self.temperature is None else f"{self.temperature:g}"
        return f"{self.prompt_id}@{t}"


def _normalise_temps(temperatures: Sequence[float]) -> list[float]:
    out: list[float] = []
    for t in temperatures:
        if t is None:
            raise ValueError(
                "temperatures must be numbers; None is how a spec with "
                "supports_temperature=False is recorded, not an input"
            )
        value = float(t)
        if value not in out:  # 0 and 0.0 are one cell, and one run_id
            out.append(value)
    return out


def build_tasks(
    specs: Sequence[ModelSpec],
    prompts: Mapping[str, str],
    temperatures: Sequence[float],
    n: int,
) -> list[RunTask]:
    """Full factorial, ordered model -> prompt -> temperature -> replicate.

    A spec with supports_temperature=False collapses to ONE cell recorded as
    temperature=None. Three identical cells would triple that model's cost and
    then present the same distribution three times as if it were evidence about
    temperature.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not prompts:
        raise ValueError("no prompts: nothing to ask")
    temps = _normalise_temps(temperatures)

    tasks: list[RunTask] = []
    for spec in specs:
        spec_temps: list[float | None]
        if spec.supports_temperature:
            if not temps:
                raise ValueError("no temperatures: nothing to sweep")
            spec_temps = list(temps)
        else:
            spec_temps = [None]
        for prompt_id, prompt_text in prompts.items():
            for temperature in spec_temps:
                for replicate in range(FIRST_REPLICATE, FIRST_REPLICATE + n):
                    tasks.append(
                        RunTask(
                            spec=spec,
                            prompt_id=prompt_id,
                            prompt_text=prompt_text,
                            temperature=temperature,
                            replicate=replicate,
                        )
                    )
    return tasks


def iter_rows(path: Path) -> Iterator[dict]:
    """Yield the JSON objects in a .jsonl file, skipping what will not parse.

    A truncated final line is the normal residue of a kill -9 during a write, so
    it is a warning and a skip, never a crash: the alternative is that one bad
    byte makes an otherwise complete factorial unreadable.
    """
    p = Path(path)
    if not p.is_file():
        return
    with p.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                log.warning("%s:%d is not JSON (%s); skipping", p, lineno, exc)
                continue
            if isinstance(row, dict):
                yield row
            else:
                log.warning("%s:%d is not an object; skipping", p, lineno)


def existing_run_ids(path: Path) -> set[str]:
    """Every run_id already recorded. Missing file -> empty set, not an error."""
    out: set[str] = set()
    for row in iter_rows(path):
        rid = row.get("run_id")
        if isinstance(rid, str) and rid:
            out.add(rid)
    return out


def completed_run_ids(path: Path) -> set[str]:
    """run_ids whose most recent row SUCCEEDED -- the correct basis for resume.

    Resuming on existing_run_ids() instead would skip a cell whose only row carries
    an error, so a transient 429 or a timeout would exclude that cell from the
    benchmark permanently and no amount of re-running could recover it. The failure
    is silent: the run reports "skipped", the cell simply has less data, and the
    mean is quietly computed over the replicates that happened to succeed.

    An empty response_text counts as not-done for the same reason. A reasoning model
    that exhausts max_tokens before emitting anything returns finish_reason="length"
    with an empty message and error=None; treating that as complete would freeze a
    harness misconfiguration into the dataset (measured: gemma-4-e4b at
    max_tokens=512 returned empty on 30/30 verbatim runs). Retrying costs one call
    and, once max_tokens is raised, actually succeeds. A truncated reply that DID
    return words is kept -- the words are real.

    Newest-wins per run_id, matching cli._latest_rows, so a retry appended after a
    failed row supersedes it.
    """
    latest: dict[str, dict] = {}
    for row in iter_rows(path):
        rid = row.get("run_id")
        if isinstance(rid, str) and rid:
            latest[rid] = row
    done = set()
    for rid, row in latest.items():
        if row.get("error"):
            continue
        text = row.get("response_text")
        if not isinstance(text, str) or not text.strip():
            continue
        done.add(rid)
    return done


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _row(task: RunTask, result: ChatResult) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_id": task.run_id,
        "model_id": task.spec.id,
        # Provenance: what the API said it served, never our own label.
        "model_reported": result.model_reported,
        "prompt_id": task.prompt_id,
        "temperature": task.temperature,
        "replicate": task.replicate,
        "response_text": result.text,
        "finish_reason": result.finish_reason,
        "usage": dict(result.usage or {}),
        "latency_ms": result.latency_ms,
        "ts": _now(),
        "error": result.error,
    }
    # Only when there is something to say, so a clean row is exactly the shape
    # CONTRACT section 1 documents.
    if getattr(result, "notes", ""):
        row["notes"] = result.notes
    return row


def _call(client: Any, task: RunTask, timeout: float) -> ChatResult:
    try:
        return client.complete(
            task.spec,
            task.prompt_text,
            temperature=task.temperature,
            timeout=timeout,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:  # noqa: BLE001 -- a failed run is a row, not a gap
        # providers.complete promises never to raise for an API failure. If it
        # does anyway, that is a bug there; losing the cell would hide it.
        text = f"{type(exc).__name__}: {exc}"
        log.error("%s: client raised: %s", task.spec.id, text)
        return ChatResult("", "", "", {}, 0, error=text[:_ERROR_MAX_CHARS])


def _n_candidates(text: str) -> int:
    try:
        return len(parse_words(text))
    except Exception:  # noqa: BLE001 -- progress cosmetics must never kill a run
        log.debug("parse failed while counting candidates", exc_info=True)
        return 0


def run_all(
    tasks: Sequence[RunTask],
    client: Any,
    out_path: Path,
    *,
    resume: bool = True,
    progress: Callable[[dict], None] | None = None,
    timeout: float = 120.0,
) -> dict:
    """Execute `tasks`, appending one row per call to `out_path`.

    Returns {"attempted","written","skipped","errors"} where attempted is the
    number of tasks handed in (so attempted == written + skipped for a run that
    completes) and errors counts rows carrying an error string.

    Concurrency is per model and capped at spec.max_concurrency: LM Studio
    entries are 1 because the same box serves the embedding models used for
    scoring. Groups run one after another, so two local models never contend.
    """
    tasks = list(tasks)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts = {"attempted": len(tasks), "written": 0, "skipped": 0, "errors": 0}

    # Only SUCCESSFUL rows count as done: an errored row must be retried, or one
    # transient failure would exclude that cell from the benchmark forever.
    done: set[str] = completed_run_ids(out_path) if resume else set()
    pending: list[RunTask] = []
    for task in tasks:
        rid = task.run_id
        if rid in done:
            counts["skipped"] += 1
            continue
        # Also guards against a duplicate inside `tasks` itself, which would
        # otherwise write the same run_id twice in one pass.
        done.add(rid)
        pending.append(task)

    total = len(pending)
    if not pending:
        return counts

    groups: dict[str, list[RunTask]] = {}
    for task in pending:
        groups.setdefault(task.spec.id, []).append(task)

    lock = threading.Lock()
    state = {"index": 0, "candidates": 0, "ok": 0, "latency": 0}

    with out_path.open("a", encoding="utf-8") as fh:

        def sink(task: RunTask, result: ChatResult) -> None:
            row = _row(task, result)
            n_cand = 0 if result.error else _n_candidates(result.text)
            with lock:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                # flush per row, not per batch: a kill -9 then loses at most the
                # call in flight. fsync is deliberately not called -- flushed
                # bytes already survive process death, and only a machine crash
                # needs the extra syscall per row.
                fh.flush()
                counts["written"] += 1
                if result.error:
                    counts["errors"] += 1
                else:
                    state["ok"] += 1
                    state["candidates"] += n_cand
                state["index"] += 1
                state["latency"] += result.latency_ms
                if progress is not None:
                    progress(_event(task, result, row, counts, state, total, n_cand))

        try:
            for model_id, group in groups.items():
                workers = max(1, group[0].spec.max_concurrency)
                if workers == 1:
                    for task in group:
                        sink(task, _call(client, task, timeout))
                    continue
                log.info("%s: %d call(s) at concurrency %d", model_id, len(group), workers)
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = {pool.submit(_call, client, t, timeout): t for t in group}
                    try:
                        for future in as_completed(futures):
                            task = futures[future]
                            try:
                                sink(task, future.result())
                            except (KeyboardInterrupt, SystemExit):
                                raise
                            except BaseException as exc:  # noqa: BLE001
                                log.error("%s: %s", task.run_id, exc)
                                sink(
                                    task,
                                    ChatResult("", "", "", {}, 0, error=str(exc)[:_ERROR_MAX_CHARS]),
                                )
                    except KeyboardInterrupt:
                        # Without cancel_futures the pool's own shutdown would
                        # still work through every queued call, so Ctrl-C would
                        # not actually stop the run.
                        pool.shutdown(wait=False, cancel_futures=True)
                        raise
        except KeyboardInterrupt:
            # Every row so far is flushed, so the next invocation resumes from
            # here. Re-raised rather than swallowed: a partial factorial must not
            # look like a completed one to the caller's exit code.
            log.warning(
                "interrupted after %d of %d call(s); re-run to resume",
                counts["written"],
                total,
            )
            raise

    return counts


def _event(
    task: RunTask,
    result: ChatResult,
    row: Mapping[str, Any],
    counts: Mapping[str, int],
    state: Mapping[str, int],
    total: int,
    n_candidates: int,
) -> dict[str, Any]:
    """One progress record. `mean_candidates` is the running mean of candidate
    words per SUCCESSFUL response -- the cheapest live signal that a model is
    answering the task at all rather than refusing or rambling."""
    ok = int(state["ok"])
    index = int(state["index"])
    return {
        "run_id": row["run_id"],
        "model_id": task.spec.id,
        "prompt_id": task.prompt_id,
        "temperature": task.temperature,
        "replicate": task.replicate,
        "cell": task.cell,
        "index": index,
        "total": total,
        "ok": result.error is None,
        "error": result.error,
        "latency_ms": result.latency_ms,
        "n_candidates": n_candidates,
        "mean_candidates": (state["candidates"] / ok) if ok else 0.0,
        "mean_latency_ms": (state["latency"] / index) if index else 0.0,
        "errors": int(counts["errors"]),
    }
