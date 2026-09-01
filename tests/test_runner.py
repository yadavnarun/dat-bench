from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

import pytest

from datbench.providers import ChatResult, ModelSpec
from datbench.runner import (
    RunTask,
    build_tasks,
    completed_run_ids,
    existing_run_ids,
    iter_rows,
    run_all,
    run_id_for,
)

PROMPTS = {"terse": "list ten nouns", "verbatim": "please enter ten words"}
TEMPS = [0.0, 0.7, 1.0]


def spec(model_id: str = "local", **kw) -> ModelSpec:
    fields = dict(
        id=model_id,
        model=f"vendor/{model_id}",
        base_url="http://localhost:1234/v1",
        api_key_env=None,
    )
    fields.update(kw)
    return ModelSpec(**fields)


class FakeClient:
    """Records every call. Never touches a socket."""

    def __init__(self, *, text: str = "1. cat\n2. thimble\n3. river", fail=None, notes: str = ""):
        self.calls: list[dict] = []
        self.text = text
        self.fail = fail or (lambda s, p, t: False)
        self.notes = notes
        self._lock = threading.Lock()

    def complete(self, spec, prompt, *, temperature, max_tokens=None, timeout=120.0):
        with self._lock:
            self.calls.append(
                {
                    "model_id": spec.id,
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "timeout": timeout,
                }
            )
        if self.fail(spec, prompt, temperature):
            return ChatResult("", "", "", {}, 4, error="http_500: boom")
        return ChatResult(
            self.text,
            f"{spec.model}-served",
            "stop",
            {"prompt_tokens": 11, "completion_tokens": 22},
            33,
            notes=self.notes,
        )


def rows_of(path: Path) -> list[dict]:
    return list(iter_rows(path))


# --------------------------------------------------------------------------- #
# run_id
# --------------------------------------------------------------------------- #
def test_run_id_matches_the_contract_formula():
    expected = hashlib.sha1(b"gemma-4-e4b|verbatim|0.7|3").hexdigest()[:16]
    assert run_id_for("gemma-4-e4b", "verbatim", 0.7, 3) == expected
    assert len(expected) == 16


def test_run_id_is_deterministic_and_coordinate_sensitive():
    base = run_id_for("m", "p", 0.7, 1)
    assert base == run_id_for("m", "p", 0.7, 1)
    assert base != run_id_for("m2", "p", 0.7, 1)
    assert base != run_id_for("m", "p2", 0.7, 1)
    assert base != run_id_for("m", "p", 1.0, 1)
    assert base != run_id_for("m", "p", 0.7, 2)


def test_run_id_of_a_temperatureless_cell_uses_none():
    assert run_id_for("m", "p", None, 1) == hashlib.sha1(b"m|p|None|1").hexdigest()[:16]


def test_run_task_run_id_matches_the_helper():
    task = RunTask(spec("m"), "verbatim", "text", 0.7, 3)
    assert task.run_id == run_id_for("m", "verbatim", 0.7, 3)
    assert task.cell == "verbatim@0.7"
    assert RunTask(spec("m"), "verbatim", "t", None, 1).cell == "verbatim@n/a"


# --------------------------------------------------------------------------- #
# build_tasks
# --------------------------------------------------------------------------- #
def test_build_tasks_is_the_full_factorial():
    tasks = build_tasks([spec("a"), spec("b")], PROMPTS, TEMPS, 2)
    assert len(tasks) == 2 * 2 * 3 * 2
    assert len({t.run_id for t in tasks}) == len(tasks)


def test_build_tasks_orders_model_then_prompt_then_temperature_then_replicate():
    tasks = build_tasks([spec("a"), spec("b")], PROMPTS, [0.0, 1.0], 2)
    coords = [(t.spec.id, t.prompt_id, t.temperature, t.replicate) for t in tasks[:5]]
    assert coords == [
        ("a", "terse", 0.0, 1),
        ("a", "terse", 0.0, 2),
        ("a", "terse", 1.0, 1),
        ("a", "terse", 1.0, 2),
        ("a", "verbatim", 0.0, 1),
    ]
    # every task of model a precedes every task of model b
    ids = [t.spec.id for t in tasks]
    assert ids == sorted(ids, key=lambda x: 0 if x == "a" else 1)


def test_fixed_temperature_spec_collapses_to_one_cell_recorded_as_none():
    fixed = spec("fixed", supports_temperature=False)
    tasks = build_tasks([fixed], PROMPTS, TEMPS, 4)
    # one cell per prompt, not one per prompt x temperature
    assert len(tasks) == len(PROMPTS) * 4
    assert {t.temperature for t in tasks} == {None}
    assert len({t.run_id for t in tasks}) == len(tasks)


def test_fixed_temperature_spec_is_unaffected_by_the_temperature_list():
    fixed = spec("fixed", supports_temperature=False)
    one = build_tasks([fixed], PROMPTS, [0.0], 1)
    many = build_tasks([fixed], PROMPTS, [0.0, 0.5, 0.9, 1.0], 1)
    assert [t.run_id for t in one] == [t.run_id for t in many]


def test_mixed_specs_keep_their_own_temperature_axes():
    tasks = build_tasks(
        [spec("a"), spec("fixed", supports_temperature=False)], PROMPTS, TEMPS, 1
    )
    assert sum(1 for t in tasks if t.spec.id == "a") == 6
    assert sum(1 for t in tasks if t.spec.id == "fixed") == 2


def test_equal_temperatures_collapse_to_one_cell():
    # 0 and 0.0 name the same cell; two cells would be two identical run_ids.
    tasks = build_tasks([spec("a")], {"terse": "t"}, [0, 0.0, 0.7], 1)
    assert [t.temperature for t in tasks] == [0.0, 0.7]


def test_int_and_float_temperatures_hash_identically():
    a = build_tasks([spec("a")], {"terse": "t"}, [0], 1)
    b = build_tasks([spec("a")], {"terse": "t"}, [0.0], 1)
    assert [t.run_id for t in a] == [t.run_id for t in b]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n": 0},
        {"prompts": {}},
        {"temperatures": []},
        {"temperatures": [None]},
    ],
)
def test_build_tasks_rejects_impossible_plans(kwargs):
    call = {"specs": [spec("a")], "prompts": PROMPTS, "temperatures": TEMPS, "n": 1}
    call.update(kwargs)
    with pytest.raises(ValueError):
        build_tasks(call["specs"], call["prompts"], call["temperatures"], call["n"])


# --------------------------------------------------------------------------- #
# existing_run_ids / iter_rows
# --------------------------------------------------------------------------- #
def test_existing_run_ids_of_a_missing_file_is_empty(tmp_path):
    assert existing_run_ids(tmp_path / "nope.jsonl") == set()


def test_existing_run_ids_survives_a_truncated_tail(tmp_path):
    path = tmp_path / "responses.jsonl"
    path.write_text(
        json.dumps({"run_id": "aaa"}) + "\n"
        + "\n"
        + json.dumps({"run_id": "bbb"}) + "\n"
        + '{"run_id": "ccc", "response_te',  # exactly what a kill -9 leaves
        encoding="utf-8",
    )
    assert existing_run_ids(path) == {"aaa", "bbb"}


def test_iter_rows_skips_non_objects(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text('[1,2]\n{"run_id": "ok"}\n', encoding="utf-8")
    assert [r["run_id"] for r in iter_rows(path)] == ["ok"]


# --------------------------------------------------------------------------- #
# run_all
# --------------------------------------------------------------------------- #
def test_run_all_writes_the_contract_row_shape(tmp_path):
    tasks = build_tasks([spec("a")], {"terse": "list ten nouns"}, [0.7], 1)
    client = FakeClient()
    counts = run_all(tasks, client, tmp_path / "runs" / "responses.jsonl")

    assert counts == {"attempted": 1, "written": 1, "skipped": 0, "errors": 0}
    (row,) = rows_of(tmp_path / "runs" / "responses.jsonl")
    assert set(row) == {
        "run_id", "model_id", "model_reported", "prompt_id", "temperature",
        "replicate", "response_text", "finish_reason", "usage", "latency_ms",
        "ts", "error",
    }
    assert row["run_id"] == tasks[0].run_id
    assert row["model_id"] == "a"
    # provenance: what the API said it served, not our label
    assert row["model_reported"] == "vendor/a-served"
    assert row["temperature"] == 0.7
    assert row["replicate"] == 1
    assert row["error"] is None
    assert row["usage"] == {"prompt_tokens": 11, "completion_tokens": 22}
    assert row["ts"].endswith("Z")


def test_run_all_passes_temperature_and_prompt_through(tmp_path):
    tasks = build_tasks([spec("a")], {"terse": "list ten nouns"}, [0.0], 1)
    client = FakeClient()
    run_all(tasks, client, tmp_path / "r.jsonl", timeout=7.5)
    assert client.calls == [
        {
            "model_id": "a",
            "prompt": "list ten nouns",
            "temperature": 0.0,
            "max_tokens": None,
            "timeout": 7.5,
        }
    ]


def test_fixed_temperature_cell_is_recorded_as_null(tmp_path):
    tasks = build_tasks([spec("f", supports_temperature=False)], {"terse": "t"}, TEMPS, 1)
    client = FakeClient()
    run_all(tasks, client, tmp_path / "r.jsonl")
    (row,) = rows_of(tmp_path / "r.jsonl")
    assert row["temperature"] is None
    assert client.calls[0]["temperature"] is None


def test_resume_makes_zero_duplicate_api_calls(tmp_path):
    path = tmp_path / "r.jsonl"
    tasks = build_tasks([spec("a")], PROMPTS, [0.0, 1.0], 3)
    client = FakeClient()

    first = run_all(tasks, client, path)
    assert first["written"] == len(tasks)
    after_first = len(client.calls)

    second = run_all(tasks, client, path)
    assert second == {
        "attempted": len(tasks),
        "written": 0,
        "skipped": len(tasks),
        "errors": 0,
    }
    assert len(client.calls) == after_first  # the whole point of resume
    assert len(rows_of(path)) == len(tasks)


def test_resume_reruns_only_the_missing_cells(tmp_path):
    path = tmp_path / "r.jsonl"
    tasks = build_tasks([spec("a")], PROMPTS, [0.0], 2)
    client = FakeClient()
    run_all(tasks[:1], client, path)
    counts = run_all(tasks, client, path)
    assert counts["skipped"] == 1
    assert counts["written"] == len(tasks) - 1
    assert len(client.calls) == len(tasks)
    assert len({r["run_id"] for r in rows_of(path)}) == len(tasks)


def test_no_resume_reissues_every_call(tmp_path):
    path = tmp_path / "r.jsonl"
    tasks = build_tasks([spec("a")], {"terse": "t"}, [0.0], 2)
    client = FakeClient()
    run_all(tasks, client, path)
    counts = run_all(tasks, client, path, resume=False)
    assert counts["written"] == 2
    assert counts["skipped"] == 0
    assert len(client.calls) == 4
    assert len(rows_of(path)) == 4  # append-only: the newer row is the later one


def test_a_duplicate_task_is_written_once(tmp_path):
    tasks = build_tasks([spec("a")], {"terse": "t"}, [0.0], 1)
    client = FakeClient()
    counts = run_all(tasks + tasks, client, tmp_path / "r.jsonl")
    assert counts == {"attempted": 2, "written": 1, "skipped": 1, "errors": 0}
    assert len(client.calls) == 1


def test_every_row_is_flushed_before_the_next_call(tmp_path):
    """kill -9 must lose at most one call, so the row must be on disk already."""
    path = tmp_path / "r.jsonl"
    tasks = build_tasks([spec("a")], {"terse": "t"}, [0.0], 4)
    seen: list[tuple[int, int]] = []

    def progress(event):
        seen.append((event["index"], len(rows_of(path))))

    run_all(tasks, FakeClient(), path, progress=progress)
    # every event saw exactly as many rows on disk as calls completed
    assert seen == [(1, 1), (2, 2), (3, 3), (4, 4)]


def test_a_failed_call_is_a_row_not_a_gap(tmp_path):
    tasks = build_tasks([spec("a")], PROMPTS, [0.0], 1)
    client = FakeClient(fail=lambda s, p, t: p.startswith("please"))
    counts = run_all(tasks, client, tmp_path / "r.jsonl")

    assert counts == {"attempted": 2, "written": 2, "skipped": 0, "errors": 1}
    rows = {r["prompt_id"]: r for r in rows_of(tmp_path / "r.jsonl")}
    assert rows["verbatim"]["error"] == "http_500: boom"
    assert rows["verbatim"]["response_text"] == ""
    assert rows["verbatim"]["run_id"]  # still joinable
    assert rows["terse"]["error"] is None


def test_a_failed_call_is_retried_on_the_next_run(tmp_path):
    """Resume skips only SUCCESSFUL rows, so a failed cell is re-attempted.

    The alternative -- skipping any recorded run_id, errored or not -- means one
    transient 429 or timeout drops that cell from the benchmark permanently, and
    silently: the run reports it as "skipped" and the cell's mean is quietly taken
    over however many replicates happened to succeed. The cost of retrying is one
    cheap re-failing call per invocation for a genuinely dead model (providers.py
    does not retry 400/401/404 internally), and that failure stays visible in the
    report's failed-run count. Silent bias is the worse trade.
    """
    path = tmp_path / "r.jsonl"
    tasks = build_tasks([spec("a")], {"terse": "t"}, [0.0], 1)
    run_all(tasks, FakeClient(fail=lambda *_: True), path)
    counts = run_all(tasks, FakeClient(), path)
    assert counts["skipped"] == 0, "an errored cell must not be skipped"
    assert counts["written"] == 1
    # Now that it has succeeded, resume leaves it alone.
    assert run_all(tasks, FakeClient(), path)["skipped"] == 1


def test_a_client_that_raises_still_produces_a_row(tmp_path):
    class Exploding:
        def complete(self, *a, **kw):
            raise RuntimeError("kaboom")

    tasks = build_tasks([spec("a")], {"terse": "t"}, [0.0], 1)
    counts = run_all(tasks, Exploding(), tmp_path / "r.jsonl")
    assert counts["errors"] == 1
    (row,) = rows_of(tmp_path / "r.jsonl")
    assert "kaboom" in row["error"]


def test_notes_are_recorded_only_when_present(tmp_path):
    tasks = build_tasks([spec("a")], {"terse": "t"}, [0.0], 1)
    run_all(tasks, FakeClient(), tmp_path / "clean.jsonl")
    run_all(tasks, FakeClient(notes="provider rejected temperature"), tmp_path / "noted.jsonl")
    assert "notes" not in rows_of(tmp_path / "clean.jsonl")[0]
    assert rows_of(tmp_path / "noted.jsonl")[0]["notes"] == "provider rejected temperature"


def test_progress_reports_cell_position_and_running_mean(tmp_path):
    tasks = build_tasks([spec("a")], {"terse": "t"}, [0.0], 2)
    events: list[dict] = []
    run_all(tasks, FakeClient(), tmp_path / "r.jsonl", progress=events.append)

    assert [e["index"] for e in events] == [1, 2]
    assert all(e["total"] == 2 for e in events)
    assert all(e["model_id"] == "a" and e["cell"] == "terse@0" for e in events)
    # three candidates per response, so the running mean is 3.0 throughout
    assert [e["mean_candidates"] for e in events] == [3.0, 3.0]
    assert [e["ok"] for e in events] == [True, True]
    assert events[-1]["errors"] == 0
    assert events[-1]["mean_latency_ms"] == 33.0


def test_progress_running_mean_ignores_failed_calls(tmp_path):
    tasks = build_tasks([spec("a")], PROMPTS, [0.0], 1)
    events: list[dict] = []
    run_all(
        tasks,
        FakeClient(fail=lambda s, p, t: p.startswith("please")),
        tmp_path / "r.jsonl",
        progress=events.append,
    )
    assert events[-1]["errors"] == 1
    # the failed call contributes no candidates and does not drag the mean to 1.5
    assert events[-1]["mean_candidates"] == 3.0


def test_run_all_groups_calls_by_model(tmp_path):
    specs = [spec("a", max_concurrency=2), spec("b", max_concurrency=2)]
    tasks = build_tasks(specs, {"terse": "t"}, [0.0], 3)
    run_all(tasks, FakeClient(), tmp_path / "r.jsonl")
    order = [r["model_id"] for r in rows_of(tmp_path / "r.jsonl")]
    assert order == ["a"] * 3 + ["b"] * 3  # no interleaving between models


class ConcurrencyProbe:
    """Blocks every call until `want` of them overlap, so a sequential runner
    fails the assertion instead of hanging the suite."""

    def __init__(self, want: int) -> None:
        self.want = want
        self.lock = threading.Lock()
        self.inflight = 0
        self.peak = 0
        self.reached = threading.Event()

    def complete(self, spec, prompt, *, temperature, max_tokens=None, timeout=120.0):
        with self.lock:
            self.inflight += 1
            self.peak = max(self.peak, self.inflight)
            if self.inflight >= self.want:
                self.reached.set()
        self.reached.wait(timeout=1.0)
        with self.lock:
            self.inflight -= 1
        return ChatResult("1. cat\n2. dog", "m", "stop", {}, 1)


def test_max_concurrency_one_means_strictly_sequential(tmp_path):
    tasks = build_tasks([spec("a", max_concurrency=1)], {"terse": "t"}, [0.0], 3)
    probe = ConcurrencyProbe(want=1)
    run_all(tasks, probe, tmp_path / "r.jsonl")
    assert probe.peak == 1  # LM Studio entries must never overlap


def test_max_concurrency_four_runs_four_at_a_time(tmp_path):
    tasks = build_tasks([spec("cloud", max_concurrency=4)], {"terse": "t"}, [0.0], 8)
    probe = ConcurrencyProbe(want=4)
    counts = run_all(tasks, probe, tmp_path / "r.jsonl")
    assert probe.peak == 4
    assert counts["written"] == 8
    assert len(rows_of(tmp_path / "r.jsonl")) == 8


def test_run_all_creates_the_parent_directory(tmp_path):
    tasks = build_tasks([spec("a")], {"terse": "t"}, [0.0], 1)
    path = tmp_path / "deep" / "nested" / "responses.jsonl"
    run_all(tasks, FakeClient(), path)
    assert path.is_file()


def test_run_all_with_nothing_to_do_touches_no_client(tmp_path):
    counts = run_all([], FakeClient(), tmp_path / "r.jsonl")
    assert counts == {"attempted": 0, "written": 0, "skipped": 0, "errors": 0}


def test_an_interrupt_keeps_the_rows_written_so_far(tmp_path):
    """Ctrl-C must be resumable: the flushed rows stay, the exception propagates."""
    path = tmp_path / "r.jsonl"
    tasks = build_tasks([spec("a")], {"terse": "t"}, [0.0], 5)

    class Interrupting(FakeClient):
        def complete(self, spec, prompt, *, temperature, max_tokens=None, timeout=120.0):
            if len(self.calls) >= 2:
                raise KeyboardInterrupt
            return super().complete(
                spec, prompt, temperature=temperature, max_tokens=max_tokens, timeout=timeout
            )

    with pytest.raises(KeyboardInterrupt):
        run_all(tasks, Interrupting(), path)
    assert len(rows_of(path)) == 2

    counts = run_all(tasks, FakeClient(), path)
    assert counts["skipped"] == 2
    assert counts["written"] == 3


def test_an_interrupt_cancels_the_queued_concurrent_calls(tmp_path):
    """Ctrl-C arrives in the main thread, which is where run_all consumes results.

    Without cancel_futures the pool's shutdown would still work through every
    queued call, so the interrupt would not actually stop the run.
    """
    tasks = build_tasks([spec("cloud", max_concurrency=2)], {"terse": "t"}, [0.0], 30)

    class Slow(FakeClient):
        def complete(self, spec, prompt, **kw):
            time.sleep(0.02)
            return super().complete(spec, prompt, **kw)

    def progress(event):
        if event["index"] >= 2:
            raise KeyboardInterrupt

    client = Slow()
    with pytest.raises(KeyboardInterrupt):
        run_all(tasks, client, tmp_path / "r.jsonl", progress=progress)
    assert len(client.calls) < len(tasks)
    # the rows that did land are intact and resumable
    assert len(rows_of(tmp_path / "r.jsonl")) >= 2


# ------------------------------------------------- resume retries failures ----
# Regression: run_all resumed on existing_run_ids(), which includes rows carrying
# an error. A cell that hit a transient 429 or timeout was therefore skipped on
# every subsequent run and excluded from the benchmark permanently -- silently,
# since the run just reported it as "skipped".

def test_completed_run_ids_excludes_errored_rows(tmp_path):
    p = tmp_path / "responses.jsonl"
    p.write_text(
        json.dumps({"run_id": "ok1", "error": None, "response_text": "cat dog"}) + "\n"
        + json.dumps({"run_id": "bad1", "error": "429 rate limited", "response_text": ""}) + "\n"
        + json.dumps({"run_id": "bad2", "error": "timeout", "response_text": ""}) + "\n",
        encoding="utf-8",
    )
    assert existing_run_ids(p) == {"ok1", "bad1", "bad2"}
    assert completed_run_ids(p) == {"ok1"}


def test_completed_run_ids_is_newest_wins(tmp_path):
    """A retry appended after a failure supersedes it, and vice versa."""
    p = tmp_path / "responses.jsonl"
    p.write_text(
        json.dumps({"run_id": "a", "error": "boom", "response_text": ""}) + "\n"
        + json.dumps({"run_id": "a", "error": None, "response_text": "cat dog"}) + "\n"
        + json.dumps({"run_id": "b", "error": None, "response_text": "cat dog"}) + "\n"
        + json.dumps({"run_id": "b", "error": "boom", "response_text": ""}) + "\n",
        encoding="utf-8",
    )
    assert completed_run_ids(p) == {"a"}


def test_resume_retries_an_errored_cell_and_leaves_successes_alone(tmp_path):
    """The end-to-end behaviour: rerunning must re-issue only the failed call."""
    ms = spec()
    tasks = build_tasks([ms], {"verbatim": "p"}, [0.7], 2)
    out = tmp_path / "responses.jsonl"

    # First pass: replicate 1 succeeds, replicate 2 errors.
    calls: list[str] = []

    class FirstPass:
        def complete(self, spec, prompt, *, temperature, max_tokens=None, timeout=120.0):
            calls.append(prompt)
            if len(calls) == 1:
                return ChatResult("cat dog", spec.model, "stop", {}, 1)
            return ChatResult("", spec.model, "error", {}, 1, error="429 rate limited")

    first = run_all(tasks, FirstPass(), out)
    assert first["written"] == 2 and first["errors"] == 1

    # Second pass: only the errored cell should be re-issued.
    retried: list[str] = []

    class SecondPass:
        def complete(self, spec, prompt, *, temperature, max_tokens=None, timeout=120.0):
            retried.append(prompt)
            return ChatResult("tree stone", spec.model, "stop", {}, 1)

    second = run_all(tasks, SecondPass(), out)
    assert len(retried) == 1, "the failed cell was not retried"
    assert second["skipped"] == 1 and second["written"] == 1
    assert completed_run_ids(out) == {t.run_id for t in tasks}


def test_completed_run_ids_treats_an_empty_reply_as_not_done(tmp_path):
    """An empty message is not an answer, whatever finish_reason says.

    A reasoning model that spends max_tokens thinking returns
    finish_reason="length" with empty text and error=None. Counting that as done
    freezes a harness misconfiguration into the dataset.
    """
    p = tmp_path / "responses.jsonl"
    p.write_text(
        json.dumps({"run_id": "good", "error": None, "response_text": "cat dog"}) + "\n"
        + json.dumps({"run_id": "empty", "error": None, "response_text": "",
                      "finish_reason": "length"}) + "\n"
        + json.dumps({"run_id": "blank", "error": None, "response_text": "   \n"}) + "\n"
        + json.dumps({"run_id": "nokey", "error": None}) + "\n",
        encoding="utf-8",
    )
    assert completed_run_ids(p) == {"good"}


def test_a_truncated_reply_that_returned_words_is_kept(tmp_path):
    """Truncation mid-explanation still yielded real words; do not re-bill it."""
    p = tmp_path / "responses.jsonl"
    p.write_text(
        json.dumps({"run_id": "t", "error": None, "finish_reason": "length",
                    "response_text": "1. cat\n2. thimble\n3. river\nThese were chosen bec"})
        + "\n",
        encoding="utf-8",
    )
    assert completed_run_ids(p) == {"t"}
