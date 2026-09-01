# dat-bench design

**Date:** 2026-08-31
**Status:** approved, implemented

Benchmark LLMs on the Divergent Association Task (DAT) over N runs, scored with local
embeddings.

---

## 1. What the task is

The DAT, from [datcreativity.com](https://www.datcreativity.com/task) (Olson & Martin,
University of Toronto Mississauga; published in *PNAS*), asks a participant for

> 10 words that are as different from each other as possible, in all meanings and uses of
> the words

under five constraints: single words, English, nouns only, no proper nouns, no specialised
vocabulary, thought of unaided.

The published scoring takes the **first 7 valid words**, computes the mean pairwise cosine
distance between their **GloVe 840B-300d** vectors, and multiplies by 100. Human scores
run roughly 50–95 with a mean near 78.

## 2. What this project measures — and what it does not

It measures **the mean pairwise semantic distance between the nouns a model emits under a
given instruction, at a given temperature, as judged by a specific local embedding
model.**

Three limits are load-bearing and must appear in every report:

1. **Not the published score.** We use local LM Studio embeddings, not GloVe 840B-300d.
   Raw values are on an arbitrary scale and are **not comparable** to the human norms. The
   `×100` convention is deliberately not applied, because printing a number near 78 would
   invite exactly that comparison.
2. **Not a creativity measure.** The DAT's validity as a creativity instrument comes from
   its correlation with human creativity measures in human populations. That validation
   does not transfer to language models. We measure semantic spread, not creativity.
3. **Validity is part of the result.** A model can inflate its score with rare or
   technical words — which *breaks the task's rules*. Score without validity rate is
   meaningless here, so they are always reported together.

## 3. Decisions

| Decision | Choice | Why |
|---|---|---|
| Scorer | Local LM Studio embeddings only | User's call, made against a stated alternative. Avoids a ~2GB GloVe download; costs comparability with human norms. |
| Embedders | **All four** local models | Words are cached and a full factorial yields only a few hundred unique words, so embedders 2–4 are nearly free — and cross-embedder Spearman is the only way to tell a real ranking from an artifact of one embedder. |
| Scale anchoring | Random-noun baseline + category floor, **per embedder** | This is what rescues an arbitrary-scale number. A score is reported as a percentile against that embedder's own chance distribution, which *is* comparable across embedders even though raw scales are not. |
| Run design | Full factorial: model × 4 prompts × 3 temps × N=10 | Separates capability from prompt luck; exposes temperature sensitivity and mode collapse. A single fixed prompt cannot distinguish a weak model from a badly-prompted one. |
| Provider layer | One OpenAI-compatible client, `base_url` per model | OpenAI, Anthropic, Gemini, DeepSeek, Groq, Together, OpenRouter and LM Studio all serve `/v1/chat/completions`. No litellm, no LangChain. Adding a model is a 4-line YAML edit. |
| Pipeline shape | Two-phase: `run` → JSONL, then pure `score`/`analyze` over it | Re-scoring with a different embedder or validity policy costs zero LLM calls, and a crash resumes. For a benchmark re-run whenever a model ships, this is the only shape that doesn't punish iteration. |
| Key handling | `models.yaml` ships pre-populated, entries inert until their env var exists | Adding a key is the only step to activate a provider. `datbench models` states exactly why each inert entry is inert. |

### Rejected

- **GloVe 840B-300d** — would give published-norm comparability; rejected by the user over
  the ~2GB download. This is the single biggest limitation of the project and is disclosed
  in the report rather than papered over.
- **Single embedder** — cheaper, but one embedder's quirks could drive the entire
  leaderboard with nothing to detect it.
- **Single-pass monolith** — ~40% less code, but every policy tweak re-bills the whole
  factorial and a crash loses the run.
- **scipy** — needed only for Spearman, which is ~15 lines with average-rank tie handling.

## 4. Architecture

```
prompts/*.txt   4 instruction variants
models.yaml     registry + scoring/run defaults
      |
      v
runner.py  --> runs/responses.jsonl      (raw LLM responses; append-only, resumable)
      |
      v
parse.py  -> validate.py  -> out/words.jsonl
      |
      v
embed.py (LM Studio + sqlite cache) -> score.py -> out/scores.jsonl
                                                   out/baselines.json
      |
      v
analyze.py -> out/summary.json/.csv -> report.py -> out/report.md/.html
```

`run_id = sha1(model_id|prompt_id|temperature|replicate)[:16]` joins all three JSONL
files and is what makes resume idempotent.

Module boundaries and exact signatures live in [`CONTRACT.md`](../../../CONTRACT.md),
which is the interface authority — the modules were implemented in parallel against it.

## 5. Metrics

**Per cell** (model × prompt × temperature, N replicates): mean, sample sd, 95% percentile
bootstrap CI, validity rate, mean pairwise Jaccard across replicates (mode collapse),
distinct-word pool.

**Per model:** best-prompt mean with CI, percentile vs that embedder's chance baseline,
prompt spread (robustness), temperature sensitivity, `gaming_index = score × valid_rate`,
counts of failed and policy-refused runs.

**Across embedders:** Spearman ρ on the model ranking for every embedder pair.

### Scoring policies

- `strict` — needs ≥7 valid words, else the run is **refused**.
- `lenient` — needs ≥4; records how many were used, since fewer words means fewer pairs
  and different variance.

A refused run is **never** scored 0. Zero is a floor that rewards invalidity with a
number; exclusion plus a visible denominator is the honest treatment.

## 6. Embedder calibration (measured 2026-08-31)

Before trusting local embeddings, the premise was checked directly: does a semantically
*tight* set score below random nouns? Method: 30 draws of 7 random common nouns vs. three
tight categories (animals, tools, colours), all four embedders.

**The premise holds.** Every embedder placed every tight category strictly below its
weakest random draw, and all four reproduced the paper's own contrast,
`d(cat,dog) < d(cat,thimble)`.

| Embedder | dim | random baseline (mean) | worst floor | separation | d(cat,dog) | d(cat,thimble) |
|---|---|---|---|---|---|---|
| GloVe 840B-300d *(reference, not used)* | 300 | — | — | — | 0.198 | **0.879** |
| `qwen3-embedding-4b` | 2560 | 0.410 | 0.289 | **29.4%** | 0.204 | 0.396 |
| `qwen3-embedding-0.6b` | 1024 | 0.392 | 0.288 | 26.6% | 0.136 | 0.324 |
| `embeddinggemma-300m` | 768 | 0.430 | 0.362 | 15.8% | 0.244 | 0.377 |
| `nomic-embed-text-v1.5` | 768 | 0.521 | 0.467 | 10.3% | 0.364 | 0.573 |

Two consequences that shape how results must be read:

1. **Range compression.** The local models track GloVe closely on *near* pairs
   (`qwen3-4b` gives 0.204 vs GloVe's 0.198 for cat/dog) but compress the far end by
   roughly 2.2× (0.396 vs 0.879 for cat/thimble). The far end is exactly where the DAT
   discriminates between good responses. So score *differences* between strong models are
   compressed relative to the published instrument, and resolving them needs more
   replicates than N=10 to clear the CI.
2. **Embedders are not interchangeable.** Separation ranges from 29.4% (`qwen3-4b`) down
   to 10.3% (`nomic`), and raw baselines span 0.39–0.52. `qwen3-embedding-4b` is the
   strongest discriminator and is the sensible headline scorer; `nomic` is the weakest and
   a ranking that depends on it should be treated with suspicion. This is precisely what
   the cross-embedder Spearman matrix is there to expose.

Reproduce with `scripts/premise_check.py`.

## 7. Known limitations

- Arbitrary scale; no comparability to published human norms (§2).
- The `rare` flag is a **weak** proxy for "specialised vocabulary": Zipf rates `thimble`
  2.53 (correctly valid — it is the paper's own example) but `quark` 3.05, so genuinely
  technical words slip through. Noun and proper-noun checks are solid; this one is not.
- Temperature is not comparable across providers, and some reasoning models ignore or
  reject it entirely.
- N=10 gives wide per-cell CIs. Cell-to-cell differences need care; the best-of-12-cells
  "best prompt" figure is a maximum over correlated estimates and is labelled as such.
- Validity checks degrade if WordNet or wordfreq is absent. `validate.capabilities()` is
  printed in every report so a degraded run cannot be mistaken for a clean one.
