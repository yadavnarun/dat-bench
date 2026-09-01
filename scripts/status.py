"""Live status of an in-flight `datbench run`.

Counts DISTINCT run_ids, newest-row-wins. Counting raw rows instead overstates
progress, because responses.jsonl is append-only: a retried cell leaves its old
row in place, so one cell can hold several rows.

Usage: .venv/bin/python scripts/status.py [--watch]
"""
from __future__ import annotations

import collections
import json
import os
import subprocess
import sys
import time

RESP = "runs/responses.jsonl"
CONFIG = "models.yaml"


def plan():
    """Expected shape of the run, derived from models.yaml -- never hardcoded.

    An earlier version pinned the model list and cell count to the two models
    that existed at the time, so once the roster grew it reported percentages
    over 100 and silently omitted every new model from the table.
    """
    try:
        import yaml
        cfg = yaml.safe_load(open(CONFIG, encoding="utf-8")) or {}
    except Exception:
        return None
    run = dict(cfg.get("run") or {})
    prompts = [str(x) for x in run.get("prompts") or []]
    temps = list(run.get("temperatures") or [])
    n = int(run.get("n") or 0)
    models, fixed = [], set()
    for m in cfg.get("models") or []:
        if not m.get("enabled", True):
            continue
        env = m.get("api_key_env")
        if env and not os.environ.get(str(env)):
            # Match the runner: an entry whose key is unset is not part of the plan.
            if not _dotenv_has(str(env)):
                continue
        mid = str(m.get("id"))
        models.append(mid)
        if m.get("supports_temperature") is False:
            fixed.add(mid)
    return {"models": models, "prompts": prompts, "temps": temps, "n": n, "fixed": fixed}


def _dotenv_has(name):
    try:
        for line in open(".env", encoding="utf-8"):
            if line.startswith(name + "=") and line.split("=", 1)[1].strip():
                return True
    except OSError:
        pass
    return False


def latest_rows(path):
    """Newest row per run_id -- matches cli._latest_rows and completed_run_ids."""
    out = {}
    try:
        fh = open(path, encoding="utf-8")
    except FileNotFoundError:
        return out
    with fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a kill -9 can leave a partial tail line
            rid = row.get("run_id")
            if isinstance(rid, str) and rid:
                out[rid] = row
    return out


def is_done(row):
    if row.get("error"):
        return False
    text = row.get("response_text")
    return isinstance(text, str) and bool(text.strip())


def running():
    """PIDs of real run processes, excluding anything merely mentioning the name.

    Matching on the command line loosely also matches a watcher whose own command
    contains "datbench run" -- which is how a self-matching wait loop never exits.
    """
    try:
        out = subprocess.run(["ps", "-eo", "pid,command"], capture_output=True,
                             text=True, timeout=10).stdout
    except Exception:
        return []
    pids = []
    for line in out.splitlines()[1:]:
        pid, _, cmd = line.strip().partition(" ")
        if "-m datbench run" in cmd and "ps -eo" not in cmd:
            pids.append(pid)
    return pids


def render():
    rows = latest_rows(RESP)
    done = {rid: r for rid, r in rows.items() if is_done(r)}
    P = plan()
    if not P or not P["models"]:
        print(f"  {len(done)} complete of {len(rows)} rows (models.yaml unreadable)")
        return len(done), len(rows), bool(running())
    PROMPTS, TEMPS, N = P["prompts"], P["temps"], P["n"]
    MODELS = P["models"]
    # A fixed-temperature model contributes one cell per prompt, not one per
    # prompt x temperature -- counting it as three overstates the denominator.
    per_model = {m: len(PROMPTS) * (1 if m in P["fixed"] else max(1, len(TEMPS))) * N
                 for m in MODELS}
    total = sum(per_model.values())

    per = collections.Counter()
    fr = collections.Counter()
    trunc = collections.Counter()
    lat = collections.defaultdict(list)
    for r in done.values():
        per[(r.get("model_id"), r.get("prompt_id"))] += 1
        fr[r.get("finish_reason")] += 1
        if r.get("finish_reason") == "length":
            trunc[(r.get("model_id"), r.get("prompt_id"))] += 1
        ms = r.get("latency_ms")
        if isinstance(ms, (int, float)):
            lat[r.get("model_id")].append(ms)

    pids = running()
    print(f"complete {len(done)}/{total}  ({100 * len(done) / total:.0f}%)"
          f"   rows on disk {len(rows)}   retry-pending {len(rows) - len(done)}"
          f"   procs {len(pids) or 'idle'}")
    print(f"finish reasons: {dict(fr)}")
    for m in MODELS:
        cell = (1 if m in P["fixed"] else max(1, len(TEMPS))) * N
        bits = []
        for p in PROMPTS:
            n = per[(m, p)]
            mark = "*" if trunc[(m, p)] else ""
            flag = "" if n == cell else ("." if n == 0 else "~")
            bits.append(f"{p}={n:2d}/{cell}{mark}{flag}")
        med = ""
        if lat[m]:
            srt = sorted(lat[m])
            med = f"  median {srt[len(srt) // 2] / 1000:.1f}s"
        done_all = all(per[(m, p)] >= cell for p in PROMPTS)
        print(f"  {'OK ' if done_all else '   '}{m:18} " + "  ".join(bits) + med)
    if any(trunc.values()):
        print("  * = contains truncated (max_tokens) runs -- harness limit, not a result")
    return len(done), total, bool(pids)


def main():
    watch = "--watch" in sys.argv
    while True:
        if watch:
            os.system("clear")
        print(time.strftime("%H:%M:%S"), "dat-bench run status")
        done, total, alive = render()
        if not watch:
            return
        if done >= total or not alive:
            print("\nrun finished" if done >= total else "\nno run process active")
            return
        time.sleep(20)


if __name__ == "__main__":
    main()
