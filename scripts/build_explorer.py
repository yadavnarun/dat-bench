"""Build out/explorer.html -- the interactive run explorer.

Bundles every run, every word, every score and the raw model replies into one
self-contained page. The whole dataset is ~300KB, so nothing is sampled or
truncated: the point of this page is that you can read what each model actually
said on each attempt, not a summary of it.

Reads:  runs/responses.jsonl, out/words.jsonl, out/scores.jsonl,
        out/baselines.json, out/summary.json, prompts/*.txt
Writes: out/explorer.html   (template: scripts/explorer_template.html)

Usage: .venv/bin/python scripts/build_explorer.py
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "scripts" / "explorer_template.html"
OUT = REPO / "out" / "explorer.html"
PLACEHOLDER = "/*__DATA__*/"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a killed writer can leave a partial tail line
            if isinstance(row, dict):
                rows.append(row)
    return rows


def read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def latest_by_run_id(rows: list[dict]) -> dict[str, dict]:
    """responses.jsonl is append-only; a retried cell leaves its old row behind."""
    out: dict[str, dict] = {}
    for row in rows:
        rid = row.get("run_id")
        if isinstance(rid, str) and rid:
            out[rid] = row
    return out


def percentile_of(value, baseline) -> float | None:
    """Where `value` sits in the chance distribution, from the stored quantiles.

    Interpolates across p05/p50/p95 rather than assuming normality -- the random
    draw distribution is not quite symmetric and a z-score would misreport the
    tails, which is exactly where the interesting answers live.
    """
    if value is None or not isinstance(baseline, dict):
        return None
    pts = []
    for key, q in (("p05", 0.05), ("p50", 0.50), ("p95", 0.95)):
        v = baseline.get(key)
        if isinstance(v, (int, float)):
            pts.append((float(v), q))
    if len(pts) < 2:
        return None
    pts.sort()
    if value <= pts[0][0]:
        return pts[0][1]
    if value >= pts[-1][0]:
        return pts[-1][1]
    for (x0, q0), (x1, q1) in zip(pts, pts[1:]):
        if x0 <= value <= x1:
            if x1 == x0:
                return q1
            return q0 + (q1 - q0) * (value - x0) / (x1 - x0)
    return None


def main() -> int:
    responses = latest_by_run_id(read_jsonl(REPO / "runs" / "responses.jsonl"))
    words = {r["run_id"]: r for r in read_jsonl(REPO / "out" / "words.jsonl")
             if r.get("run_id")}
    baselines = read_json(REPO / "out" / "baselines.json") or {}
    summary = read_json(REPO / "out" / "summary.json") or {}

    if not responses:
        raise SystemExit("no runs found -- run `python -m datbench run` first")

    # scores: run_id -> embedder -> policy -> score
    scores: dict[str, dict[str, dict[str, object]]] = collections.defaultdict(
        lambda: collections.defaultdict(dict))
    for row in read_jsonl(REPO / "out" / "scores.jsonl"):
        rid, emb, pol = row.get("run_id"), row.get("embedder"), row.get("policy")
        if rid and emb and pol:
            scores[rid][emb][pol] = {
                "score": row.get("score"),
                "scored": bool(row.get("scored")),
                "reason": row.get("reason"),
                "n_words_used": row.get("n_words_used"),
            }

    primary = summary.get("primary_embedder") or (
        sorted(baselines)[0] if baselines else "")
    primary_policy = summary.get("primary_policy") or "strict"

    pbase = baselines.get(primary) or {}
    if isinstance(pbase.get("random"), dict):
        pbase = pbase["random"]

    # Where each model ran. Colour in the page encodes this two-level split
    # (local vs hosted), not model identity: 13 models cannot be told apart by
    # hue, and local-vs-cloud is the comparison a reader actually wants.
    providers: dict[str, str] = {}
    try:
        import yaml
        cfg = yaml.safe_load((REPO / "models.yaml").read_text(encoding="utf-8")) or {}
        for m in cfg.get("models") or []:
            url = str(m.get("base_url") or "")
            if "localhost" in url or "127.0.0.1" in url:
                providers[str(m.get("id"))] = "local"
            elif "openai.com" in url:
                providers[str(m.get("id"))] = "OpenAI"
            elif "anthropic" in url:
                providers[str(m.get("id"))] = "Anthropic"
            elif "googleapis" in url:
                providers[str(m.get("id"))] = "Google"
            else:
                providers[str(m.get("id"))] = "other"
    except Exception:
        pass

    prompts = {}
    for p in sorted((REPO / "prompts").glob("*.txt")):
        prompts[p.stem] = p.read_text(encoding="utf-8").strip()

    runs = []
    for rid, resp in responses.items():
        wrow = words.get(rid, {})
        wlist = wrow.get("words") or []
        sc = scores.get(rid, {})
        pscore = (sc.get(primary, {}).get(primary_policy) or {}).get("score")
        runs.append({
            "id": rid,
            "model": resp.get("model_id"),
            "provider": providers.get(str(resp.get("model_id")), "other"),
            "served": resp.get("model_reported"),
            "prompt": resp.get("prompt_id"),
            "temp": resp.get("temperature"),
            "rep": resp.get("replicate"),
            "raw": resp.get("response_text") or "",
            "finish": resp.get("finish_reason"),
            "ms": resp.get("latency_ms"),
            "tokens": (resp.get("usage") or {}).get("completion_tokens"),
            "error": resp.get("error"),
            "candidates": wrow.get("candidates") or [],
            "words": [
                {
                    "w": w.get("word"),
                    "c": w.get("clean"),
                    "ok": bool(w.get("valid")),
                    "f": w.get("flags") or [],
                    "z": w.get("zipf"),
                }
                for w in wlist
            ],
            "score": pscore,
            "pct": percentile_of(pscore, pbase),
            "scores": {e: {p: v for p, v in pol.items()} for e, pol in sc.items()},
        })

    runs.sort(key=lambda r: (str(r["model"]), str(r["prompt"]),
                            -1 if r["temp"] is None else r["temp"], r["rep"] or 0))

    # Vocabulary: how often each model reaches for a word. This is what makes mode
    # collapse legible -- a model with 10 distinct words across 100 slots is not
    # doing the task, however good its score looks.
    vocab: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    vocab_valid: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in runs:
        for w in r["words"]:
            key = (w["c"] or w["w"] or "").lower()
            if not key:
                continue
            vocab[r["model"]][key] += 1
            if w["ok"]:
                vocab_valid[r["model"]][key] += 1

    models = sorted({str(r["model"]) for r in runs})

    # How many DIFFERENT models reach for each word. Per-model vocabulary shows
    # repetition within a model; this shows convergence BETWEEN models, which no
    # per-model view can see -- and it is the more surprising number.
    breadth: collections.Counter = collections.Counter()
    totals: collections.Counter = collections.Counter()
    for m in models:
        for w, n in vocab_valid[m].items():
            breadth[w] += 1
            totals[w] += n
    shared = sorted(breadth.items(), key=lambda kv: (-kv[1], -totals[kv[0]]))[:24]
    convergence = {
        "n_models": len(models),
        "distinct_total": len(breadth),
        "only_one_model": sum(1 for _w, n in breadth.items() if n == 1),
        "top": [[w, n, totals[w]] for w, n in shared],
    }
    vocab_out = {
        m: {
            "total_slots": sum(vocab[m].values()),
            "distinct": len(vocab[m]),
            "top": vocab[m].most_common(60),
        }
        for m in models
    }

    # Per-embedder ranking of models, by each model's best cell under that
    # embedder. This is the run's most important result: if the ranks disagree,
    # the leaderboard is a property of the scoring model, not of the models.
    per_emb_best: dict[str, dict[str, float]] = collections.defaultdict(dict)
    cellacc: dict[tuple, list] = collections.defaultdict(list)
    for row in read_jsonl(REPO / "out" / "scores.jsonl"):
        if row.get("policy") != primary_policy or not row.get("scored"):
            continue
        resp = responses.get(row.get("run_id"))
        if not resp or row.get("score") is None:
            continue
        cellacc[(row["embedder"], resp["model_id"], resp["prompt_id"],
                 str(resp["temperature"]))].append(float(row["score"]))
    for (emb, mid, _p, _t), vals in cellacc.items():
        mean = sum(vals) / len(vals)
        if mean > per_emb_best[emb].get(mid, float("-inf")):
            per_emb_best[emb][mid] = mean

    ranking = {}
    for emb, by_model in per_emb_best.items():
        order = sorted(by_model, key=lambda m: -by_model[m])
        ranking[emb] = {
            "rank": {m: i + 1 for i, m in enumerate(order)},
            "best": {m: round(v, 4) for m, v in by_model.items()},
        }

    def spearman(a: dict, b: dict):
        shared = sorted(set(a) & set(b))
        if len(shared) < 3:
            return None
        def ranks(d):
            vals = sorted(((d[k], k) for k in shared))
            out, i = {}, 0
            while i < len(vals):
                j = i
                while j + 1 < len(vals) and vals[j + 1][0] == vals[i][0]:
                    j += 1
                avg = (i + j) / 2 + 1          # average ranks for ties
                for k in range(i, j + 1):
                    out[vals[k][1]] = avg
                i = j + 1
            return out
        ra, rb = ranks(a), ranks(b)
        n = len(shared)
        ma = sum(ra.values()) / n
        mb = sum(rb.values()) / n
        num = sum((ra[k] - ma) * (rb[k] - mb) for k in shared)
        da = sum((ra[k] - ma) ** 2 for k in shared) ** 0.5
        db = sum((rb[k] - mb) ** 2 for k in shared) ** 0.5
        return None if da == 0 or db == 0 else round(num / (da * db), 3)

    embs_sorted = sorted(per_emb_best)
    rho_pairs = []
    for i, a in enumerate(embs_sorted):
        for b in embs_sorted[i + 1:]:
            r = spearman(per_emb_best[a], per_emb_best[b])
            if r is not None:
                rho_pairs.append({"a": a, "b": b, "rho": r})

    # Flatten the per-cell grid so the page does not have to re-derive it.
    cells = []
    for mid, detail in (summary.get("models") or {}).items():
        for c in detail.get("grid") or []:
            mean = c.get("mean")
            sd = c.get("sd")
            jac = c.get("jaccard")
            degenerate = bool(
                (isinstance(sd, (int, float)) and sd <= 1e-12)
                or (isinstance(jac, (int, float)) and jac >= 0.999)
            )
            cells.append({
                "model": mid,
                "provider": providers.get(mid, "other"),
                "prompt": c.get("prompt_id"),
                "temp": c.get("temperature"),
                "n": c.get("n"),
                "mean": mean,
                "sd": sd,
                "ci": [c.get("ci_lo"), c.get("ci_hi")],
                "jaccard": jac,
                "valid_rate": c.get("valid_rate"),
                "refused": c.get("n_refused"),
                "degenerate": degenerate,
                "pct": percentile_of(mean, pbase),
            })

    data = {
        "generated": summary.get("generated_at"),
        "primary_embedder": primary,
        "primary_policy": primary_policy,
        "embedders": summary.get("embedders") or sorted(baselines),
        "n_replicates": summary.get("n_replicates"),
        "n_use": summary.get("n_use"),
        "capabilities": summary.get("capabilities") or {},
        "baselines": baselines,
        "chance": pbase,
        "leaderboard": summary.get("leaderboard") or [],
        "models_detail": summary.get("models") or {},
        "cells": cells,
        "ranking": ranking,
        "rho_pairs": rho_pairs,
        "prompts": prompts,
        "providers": providers,
        "runs": runs,
        "vocab": vocab_out,
        "convergence": convergence,
        "flag_counts": summary.get("flag_counts") or {},
    }

    if not TEMPLATE.exists():
        raise SystemExit(f"missing template: {TEMPLATE}")
    html = TEMPLATE.read_text(encoding="utf-8")
    if PLACEHOLDER not in html:
        raise SystemExit(f"template has no {PLACEHOLDER} placeholder")

    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    # </script> inside embedded JSON would close the host script tag early.
    blob = blob.replace("</", "<\\/")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html.replace(PLACEHOLDER, blob), encoding="utf-8")

    kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT}  ({kb:.0f} KB)")
    print(f"  {len(runs)} runs, {sum(len(r['words']) for r in runs)} words, "
          f"{len(cells)} cells, {len(data['embedders'])} embedders")
    for m in models:
        v = vocab_out[m]
        print(f"  {m:14} {v['distinct']:4d} distinct words across {v['total_slots']:4d} slots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
