# dat-bench: Divergent Association Task over N runs

*21 model(s) · 4 embedder(s) · n=10 replicates per cell · generated 2026-09-01T08:17:04Z*

> ⚠️ **Read this first: these are NOT DAT scores**
>
> Scores here are mean pairwise cosine distances from **local LM Studio embeddings** (`text-embedding-qwen3-embedding-4b`) — **not** the **GloVe 840B-300d** vectors used by the published Divergent Association Task.
>
> They are therefore on an **arbitrary scale** and are **NOT comparable** to the published DAT human norms (mean ≈78, range ~50–95). No ×100 score is printed anywhere in this report, deliberately: a number near 78 here would be a coincidence, not a result.
>
> The interpretable number is the **percentile vs chance** beside each score, measured against that embedder's random-noun baseline (below). A raw mean distance on its own says nothing.
>
> Comparisons are valid **within this run only** — same embedder, same policy, same prompt set. Check the cross-embedder agreement section before quoting any ordering.

## Which checks actually ran

| capability | loaded | checks it powers |
|---|---|---|
| `dictionary` | yes | `not_in_dict` — is it an English word at all |
| `wordnet` | yes | `not_noun` and `proper_noun` — is it a noun, is it a name |
| `wordfreq` | yes | `rare` — Zipf frequency vs the specialised-vocabulary threshold |

> ✅ **All validity checks ran**
>
> Dictionary, WordNet and wordfreq all loaded, so no DAT rule was skipped.

## Headline: how do these compare to chance?

- Scored by `text-embedding-qwen3-embedding-4b`, seven random common nouns average **0.373** with a 95th percentile of **0.411**. Of **180** prompt x temperature cells in this run, **172** beat the chance *mean* and **72** beat chance *p95*. Best cell: **0.463**.
- Only 72 of 180 cells cleared chance p95, so most of this grid is indistinguishable from random word draws. Read the percentile column before comparing models.

## Chance baselines (per embedder)

Seven **random common nouns**, scored by the same embedder, repeated `n` times. This is the zero point: a model that beats chance by little has not done the task, whatever its raw number looks like.

| embedder | k | draws | chance mean | sd | p05 | p50 | p95 | seed |
|---|---|---|---|---|---|---|---|---|
| `text-embedding-embeddinggemma-300m` | 7 | 1000 | 0.412 | 0.023 | 0.373 | 0.412 | 0.451 | 0 |
| `text-embedding-nomic-embed-text-v1.5` | 7 | 1000 | 0.523 | 0.017 | 0.494 | 0.523 | 0.550 | 0 |
| `text-embedding-qwen3-embedding-0.6b` | 7 | 1000 | 0.367 | 0.025 | 0.326 | 0.367 | 0.406 | 0 |
| `text-embedding-qwen3-embedding-4b` | 7 | 1000 | 0.373 | 0.023 | 0.335 | 0.374 | 0.411 | 0 |

- `text-embedding-embeddinggemma-300m` tight-set floors: animals 0.278, tools 0.237, colours 0.362, furniture 0.338.
- `text-embedding-nomic-embed-text-v1.5` tight-set floors: animals 0.340, tools 0.467, colours 0.444, furniture 0.406.
- `text-embedding-qwen3-embedding-0.6b` tight-set floors: animals 0.200, tools 0.288, colours 0.132, furniture 0.243.
- `text-embedding-qwen3-embedding-4b` tight-set floors: animals 0.212, tools 0.289, colours 0.215, furniture 0.287.

Category floors are semantically tight sets (seven animals, seven tools) — an anchor for what a *low* score looks like on this embedder's scale.

## Leaderboard

| # | model | best cell | mean [95% CI] | pct vs chance | z | valid rate | gaming idx | Jaccard | distinct/total words | scored/refused/failed |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `gpt-5.1` | maxcreative @ T=0 | 0.463 [0.452, 0.473] | 100.0% | 3.87 | 99% | 0.459 | 0.49 | 21/100 | 120 / 0 / 0 |
| 2 | `gpt-4o-mini` | cot @ T=0 | 0.443 [degenerate: n_eff=1] | 99.9% | 3.02 | 100% | 0.441 | 0.80 | 13/100 | 120 / 0 / 0 |
| 3 | `gpt-5.6-luna` | cot @ T=n/a | 0.441 [0.428, 0.455] | 99.8% | 2.95 | 99% | 0.437 | 0.22 | 46/97 | 40 / 0 / 0 |
| 4 | `gpt-5.6-sol` | maxcreative @ T=n/a | 0.440 [0.423, 0.455] | 99.8% | 2.90 | 99% | 0.436 | 0.10 | 61/100 | 40 / 0 / 0 |
| 5 | `gpt-4o` | maxcreative @ T=0 | 0.435 [degenerate: n_eff=1] | 99.6% | 2.67 | 98% | 0.427 | 1.00 | 10/100 | 120 / 0 / 0 |
| 6 | `gpt-5.6-terra` | cot @ T=n/a | 0.430 [0.423, 0.437] | 99.3% | 2.46 | 99% | 0.425 | 0.21 | 47/97 | 40 / 0 / 0 |
| 7 | `gpt-5.4-mini` | terse @ T=0 | 0.429 [degenerate: n_eff=1] | 99.2% | 2.42 | 100% | 0.429 | 1.00 | 10/100 | 120 / 0 / 0 |
| 8 | `gpt-4.1-mini` | maxcreative @ T=1 | 0.426 [0.422, 0.433] | 98.9% | 2.30 | 100% | 0.424 | 0.43 | 32/100 | 120 / 0 / 0 |
| 9 | `gpt-5.4` | maxcreative @ T=0.7 | 0.426 [0.416, 0.435] | 98.9% | 2.29 | 100% | 0.424 | 0.41 | 26/100 | 120 / 0 / 0 |
| 10 | `gpt-5-search-api` | maxcreative @ T=n/a | 0.425 [0.417, 0.433] | 98.7% | 2.23 | 100% | 0.425 | 0.28 | 40/100 | 40 / 0 / 0 |
| 11 | `gpt-3.5-turbo` | cot @ T=1 | 0.425 [0.416, 0.432] | 98.7% | 2.23 | 100% | 0.423 | 0.23 | 45/97 | 120 / 0 / 0 |
| 12 | `gpt-5.4-nano` | terse @ T=0 | 0.423 [degenerate: n_eff=1] | 98.5% | 2.16 | 99% | 0.420 | 0.81 | 12/100 | 120 / 0 / 0 |
| 13 | `gpt-5.5` | maxcreative @ T=n/a | 0.422 [0.407, 0.440] | 98.3% | 2.13 | 99% | 0.417 | 0.06 | 66/99 | 40 / 0 / 0 |
| 14 | `gpt-4.1` | maxcreative @ T=0.7 | 0.421 [0.414, 0.430] | 98.1% | 2.07 | 100% | 0.420 | 0.20 | 39/100 | 120 / 0 / 0 |
| 15 | `gpt-5.2` | terse @ T=1 | 0.418 [0.415, 0.422] | 97.5% | 1.96 | 98% | 0.410 | 0.57 | 24/100 | 120 / 0 / 0 |
| 16 | `o4-mini` | maxcreative @ T=n/a | 0.417 [0.409, 0.426] | 97.2% | 1.92 | 100% | 0.416 | 0.24 | 38/100 | 40 / 0 / 0 |
| 17 | `gpt-5` | verbatim @ T=n/a | 0.413 [0.406, 0.421] | 96.0% | 1.75 | 98% | 0.405 | 0.11 | 53/99 | 40 / 0 / 0 |
| 18 | `gpt-5-nano` | terse @ T=n/a | 0.404 [0.397, 0.413] | 91.3% | 1.36 | 97% | 0.392 | 0.27 | 41/100 | 39 / 1 / 0 |
| 19 | `gpt-5-mini` | verbatim @ T=n/a | 0.404 [0.396, 0.413] | 90.9% | 1.33 | 100% | 0.403 | 0.19 | 46/100 | 40 / 0 / 0 |
| 20 | `gemma-4-e4b` | terse @ T=0 | 0.401 [degenerate: n_eff=1] | 88.9% | 1.22 | 100% | 0.401 | 1.00 | 10/100 | 119 / 1 / 0 |
| 21 | `lfm2.5-1.2b` | terse @ T=0 | 0.396 [degenerate: n_eff=1] | 84.4% | 1.01 | 94% | 0.373 | 0.88 | 9/80 | 120 / 0 / 0 |

*embedder `text-embedding-qwen3-embedding-4b`, policy `strict`, n=10 replicates per cell, first 7 valid words scored. Score is a raw mean cosine distance on an arbitrary scale — read the percentile column, not the score.*

- **mean [95% CI]** — the model's *best prompt x temperature* cell, percentile bootstrap of the mean. An upper bound over the prompts tried, not typical behaviour.
- **pct vs chance** — where that mean sits in this embedder's random-noun distribution. This is the only cross-embedder-meaningful number in the row.
- **valid rate** — share of emitted candidates that passed every DAT rule that actually ran (see capabilities above).
- **gaming idx** — mean x valid rate. Reported beside the score, never instead of it: a model that wins on rare or technical words falls here.
- **Jaccard** — mean replicate-to-replicate word overlap. 1.0 is total mode collapse (same ten words every run); near 0 is fresh output each time.
- **scored/refused/failed** — the denominators. Refused runs are *excluded*, never scored 0, so the mean is conditional on producing scorable output.

## Denominators: failed and refused runs

A leaderboard that hides its denominators is not a benchmark. **failed** = the API call errored (no words at all). **policy-refused** = words came back but too few were valid to score under this policy; those runs are excluded from every mean above, never scored 0. **truncated** = the reply hit `max_tokens` (`finish_reason="length"`); that is a harness limit, not a model result — a reasoning model can spend the whole budget thinking and return an empty message. Truncation hits elaborate prompts first, so a non-zero count here can masquerade as prompt sensitivity. Raise `max_tokens` in models.yaml and re-run before reading anything into that model’s numbers.

| model | runs | scored | policy-refused | failed (API/error) | truncated (max_tokens) | valid rate |
|---|---|---|---|---|---|---|
| `gpt-5.1` | 120 | 120 | 0 | 0 | 0 | 99% |
| `gpt-4o-mini` | 120 | 120 | 0 | 0 | 0 | 100% |
| `gpt-5.6-luna` | 40 | 40 | 0 | 0 | 0 | 99% |
| `gpt-5.6-sol` | 40 | 40 | 0 | 0 | 0 | 99% |
| `gpt-4o` | 120 | 120 | 0 | 0 | 0 | 98% |
| `gpt-5.6-terra` | 40 | 40 | 0 | 0 | 0 | 99% |
| `gpt-5.4-mini` | 120 | 120 | 0 | 0 | 0 | 100% |
| `gpt-4.1-mini` | 120 | 120 | 0 | 0 | 0 | 100% |
| `gpt-5.4` | 120 | 120 | 0 | 0 | 0 | 100% |
| `gpt-5-search-api` | 40 | 40 | 0 | 0 | 0 | 100% |
| `gpt-3.5-turbo` | 120 | 120 | 0 | 0 | 0 | 100% |
| `gpt-5.4-nano` | 120 | 120 | 0 | 0 | 0 | 99% |
| `gpt-5.5` | 40 | 40 | 0 | 0 | 0 | 99% |
| `gpt-4.1` | 120 | 120 | 0 | 0 | 0 | 100% |
| `gpt-5.2` | 120 | 120 | 0 | 0 | 0 | 98% |
| `o4-mini` | 40 | 40 | 0 | 0 | 0 | 100% |
| `gpt-5` | 40 | 40 | 0 | 0 | 0 | 98% |
| `gpt-5-nano` | 40 | 39 | 1 | 0 | 0 | 97% |
| `gpt-5-mini` | 40 | 40 | 0 | 0 | 0 | 100% |
| `gemma-4-e4b` | 120 | 119 | 1 | 0 | 1 | 100% |
| `lfm2.5-1.2b` | 120 | 120 | 0 | 0 | 0 | 94% |

*totals across models: 1800 runs, 1798 scored, 2 policy-refused, 0 failed, 1 truncated.*

- run stage: **written** 1800, **failed** 0, **scored** 1798, **refused** 2.

Truncated runs by prompt — these cells are measuring the token cap, not the model:

- `gemma-4-e4b`: `verbatim` (1)

Most common refusal reasons:

- `gpt-5-nano`: policy 'strict' needs >= 7 valid words, got 0 (1)
- `gemma-4-e4b`: policy 'strict' needs >= 7 valid words, got 2 (1)

| validity flag | candidates flagged |
|---|---|
| `proper_noun` | 72 |
| `rare` | 42 |
| `multiword` | 37 |
| `not_in_dict` | 24 |
| `not_noun` | 24 |
| `duplicate` | 20 |

*`rare` is a Zipf-frequency cut, not a technicality test — see Known limitations.*

## Per-model breakdown

Cell means for every prompt x temperature combination, so prompt sensitivity and temperature sensitivity are visible instead of hidden behind the best cell.

### `gpt-5.1`

| prompt \ temperature | T=0 | T=0.7 | T=1 |
|---|---|---|---|
| `verbatim` | 0.398 (n=10) | 0.387 (n=10) | 0.403 (n=10) |
| `terse` | 0.421 (n=10) | 0.431 (n=10) | 0.420 (n=10) |
| `cot` | 0.418 (n=10) | 0.429 (n=10) | 0.436 (n=10) |
| `maxcreative` | 0.463 (n=10) | 0.450 (n=10) | 0.451 (n=10) |

*cell = mean score over that cell's replicates.*

- prompt sensitivity: best-minus-worst prompt mean = **0.059**
- temperature sensitivity: best-minus-worst temperature mean = **0.003**
- At n=10 per cell these spreads are usually inside the noise; compare them against the CI width in the leaderboard before calling a prompt or a temperature better.

Served as: `gpt-5.1-2025-11-13`.

### `gpt-4o-mini`

| prompt \ temperature | T=0 | T=0.7 | T=1 |
|---|---|---|---|
| `verbatim` | 0.410 (n=10) | 0.420 (n=10) | 0.418 (n=10) |
| `terse` | 0.433 (n=10) | 0.426 (n=10) | 0.422 (n=10) |
| `cot` | 0.443 (n=10) | 0.440 (n=10) | 0.426 (n=10) |
| `maxcreative` | 0.428 (n=10) | 0.434 (n=10) | 0.432 (n=10) |

*cell = mean score over that cell's replicates.*

- prompt sensitivity: best-minus-worst prompt mean = **0.020**
- temperature sensitivity: best-minus-worst temperature mean = **0.006**
- At n=10 per cell these spreads are usually inside the noise; compare them against the CI width in the leaderboard before calling a prompt or a temperature better.

Served as: `gpt-4o-mini-2024-07-18`.

### `gpt-5.6-luna`

| prompt \ temperature | T=n/a |
|---|---|
| `verbatim` | 0.440 (n=10) |
| `terse` | 0.403 (n=10) |
| `cot` | 0.441 (n=10) |
| `maxcreative` | 0.430 (n=10) |

*cell = mean score over that cell's replicates.*

- prompt sensitivity: best-minus-worst prompt mean = **0.038**
- At n=10 per cell these spreads are usually inside the noise; compare them against the CI width in the leaderboard before calling a prompt or a temperature better.

Served as: `gpt-5.6-luna`.

Registry notes: ~155 reasoning tokens per answer.

### `gpt-5.6-sol`

| prompt \ temperature | T=n/a |
|---|---|
| `verbatim` | 0.434 (n=10) |
| `terse` | 0.376 (n=10) |
| `cot` | 0.423 (n=10) |
| `maxcreative` | 0.440 (n=10) |

*cell = mean score over that cell's replicates.*

- prompt sensitivity: best-minus-worst prompt mean = **0.064**
- At n=10 per cell these spreads are usually inside the noise; compare them against the CI width in the leaderboard before calling a prompt or a temperature better.

Served as: `gpt-5.6-sol`.

Registry notes: reasoning model; ~240 reasoning tokens per answer.

### `gpt-4o`

| prompt \ temperature | T=0 | T=0.7 | T=1 |
|---|---|---|---|
| `verbatim` | 0.384 (n=10) | 0.391 (n=10) | 0.400 (n=10) |
| `terse` | 0.412 (n=10) | 0.415 (n=10) | 0.409 (n=10) |
| `cot` | 0.406 (n=10) | 0.416 (n=10) | 0.421 (n=10) |
| `maxcreative` | 0.435 (n=10) | 0.418 (n=10) | 0.421 (n=10) |

*cell = mean score over that cell's replicates.*

- prompt sensitivity: best-minus-worst prompt mean = **0.033**
- temperature sensitivity: best-minus-worst temperature mean = **0.003**
- At n=10 per cell these spreads are usually inside the noise; compare them against the CI width in the leaderboard before calling a prompt or a temperature better.

Served as: `gpt-4o-2024-08-06`.

### `gpt-5.6-terra`

| prompt \ temperature | T=n/a |
|---|---|
| `verbatim` | 0.392 (n=10) |
| `terse` | 0.395 (n=10) |
| `cot` | 0.430 (n=10) |
| `maxcreative` | 0.427 (n=10) |

*cell = mean score over that cell's replicates.*

- prompt sensitivity: best-minus-worst prompt mean = **0.038**
- At n=10 per cell these spreads are usually inside the noise; compare them against the CI width in the leaderboard before calling a prompt or a temperature better.

Served as: `gpt-5.6-terra`.

Registry notes: ~93 reasoning tokens per answer.

### `gpt-5.4-mini`

| prompt \ temperature | T=0 | T=0.7 | T=1 |
|---|---|---|---|
| `verbatim` | 0.412 (n=10) | 0.398 (n=10) | 0.398 (n=10) |
| `terse` | 0.429 (n=10) | 0.416 (n=10) | 0.399 (n=10) |
| `cot` | 0.387 (n=10) | 0.388 (n=10) | 0.401 (n=10) |
| `maxcreative` | 0.420 (n=10) | 0.423 (n=10) | 0.411 (n=10) |

*cell = mean score over that cell's replicates.*

- prompt sensitivity: best-minus-worst prompt mean = **0.026**
- temperature sensitivity: best-minus-worst temperature mean = **0.009**
- At n=10 per cell these spreads are usually inside the noise; compare them against the CI width in the leaderboard before calling a prompt or a temperature better.

Served as: `gpt-5.4-mini-2026-03-17`.

### `gpt-4.1-mini`

| prompt \ temperature | T=0 | T=0.7 | T=1 |
|---|---|---|---|
| `verbatim` | 0.411 (n=10) | 0.414 (n=10) | 0.412 (n=10) |
| `terse` | 0.405 (n=10) | 0.408 (n=10) | 0.403 (n=10) |
| `cot` | 0.408 (n=10) | 0.409 (n=10) | 0.412 (n=10) |
| `maxcreative` | 0.422 (n=10) | 0.421 (n=10) | 0.426 (n=10) |

*cell = mean score over that cell's replicates.*

- prompt sensitivity: best-minus-worst prompt mean = **0.018**
- temperature sensitivity: best-minus-worst temperature mean = **0.002**
- At n=10 per cell these spreads are usually inside the noise; compare them against the CI width in the leaderboard before calling a prompt or a temperature better.

Served as: `gpt-4.1-mini-2025-04-14`.

### `gpt-5.4`

| prompt \ temperature | T=0 | T=0.7 | T=1 |
|---|---|---|---|
| `verbatim` | 0.379 (n=10) | 0.386 (n=10) | 0.385 (n=10) |
| `terse` | 0.417 (n=10) | 0.403 (n=10) | 0.405 (n=10) |
| `cot` | 0.406 (n=10) | 0.413 (n=10) | 0.405 (n=10) |
| `maxcreative` | 0.401 (n=10) | 0.426 (n=10) | 0.426 (n=10) |

*cell = mean score over that cell's replicates.*

- prompt sensitivity: best-minus-worst prompt mean = **0.034**
- temperature sensitivity: best-minus-worst temperature mean = **0.006**
- At n=10 per cell these spreads are usually inside the noise; compare them against the CI width in the leaderboard before calling a prompt or a temperature better.

Served as: `gpt-5.4-2026-03-05`.

### `gpt-5-search-api`

| prompt \ temperature | T=n/a |
|---|---|
| `verbatim` | 0.403 (n=10) |
| `terse` | 0.396 (n=10) |
| `cot` | 0.421 (n=10) |
| `maxcreative` | 0.425 (n=10) |

*cell = mean score over that cell's replicates.*

- prompt sensitivity: best-minus-worst prompt mean = **0.029**
- At n=10 per cell these spreads are usually inside the noise; compare them against the CI width in the leaderboard before calling a prompt or a temperature better.

Served as: `gpt-5-search-api-2025-10-14`.

Registry notes: search-tuned variant, not a general chat model.

### `gpt-3.5-turbo`

| prompt \ temperature | T=0 | T=0.7 | T=1 |
|---|---|---|---|
| `verbatim` | 0.417 (n=10) | 0.413 (n=10) | 0.409 (n=10) |
| `terse` | 0.359 (n=10) | 0.374 (n=10) | 0.381 (n=10) |
| `cot` | 0.419 (n=10) | 0.413 (n=10) | 0.425 (n=10) |
| `maxcreative` | 0.396 (n=10) | 0.401 (n=10) | 0.394 (n=10) |

*cell = mean score over that cell's replicates.*

- prompt sensitivity: best-minus-worst prompt mean = **0.048**
- temperature sensitivity: best-minus-worst temperature mean = **0.004**
- At n=10 per cell these spreads are usually inside the noise; compare them against the CI width in the leaderboard before calling a prompt or a temperature better.

Served as: `gpt-3.5-turbo-0125`.

### `gpt-5.4-nano`

| prompt \ temperature | T=0 | T=0.7 | T=1 |
|---|---|---|---|
| `verbatim` | 0.403 (n=10) | 0.384 (n=10) | 0.383 (n=10) |
| `terse` | 0.423 (n=10) | 0.415 (n=10) | 0.411 (n=10) |
| `cot` | 0.363 (n=10) | 0.377 (n=10) | 0.393 (n=10) |
| `maxcreative` | 0.396 (n=10) | 0.401 (n=10) | 0.381 (n=10) |

*cell = mean score over that cell's replicates.*

- prompt sensitivity: best-minus-worst prompt mean = **0.039**
- temperature sensitivity: best-minus-worst temperature mean = **0.004**
- At n=10 per cell these spreads are usually inside the noise; compare them against the CI width in the leaderboard before calling a prompt or a temperature better.

Served as: `gpt-5.4-nano-2026-03-17`.

### `gpt-5.5`

| prompt \ temperature | T=n/a |
|---|---|
| `verbatim` | 0.418 (n=10) |
| `terse` | 0.413 (n=10) |
| `cot` | 0.408 (n=10) |
| `maxcreative` | 0.422 (n=10) |

*cell = mean score over that cell's replicates.*

- prompt sensitivity: best-minus-worst prompt mean = **0.014**
- At n=10 per cell these spreads are usually inside the noise; compare them against the CI width in the leaderboard before calling a prompt or a temperature better.

Served as: `gpt-5.5-2026-04-23`.

Registry notes: reasoning model; ~512 reasoning tokens per answer.

### `gpt-4.1`

| prompt \ temperature | T=0 | T=0.7 | T=1 |
|---|---|---|---|
| `verbatim` | 0.380 (n=10) | 0.394 (n=10) | 0.402 (n=10) |
| `terse` | 0.405 (n=10) | 0.407 (n=10) | 0.403 (n=10) |
| `cot` | 0.412 (n=10) | 0.403 (n=10) | 0.404 (n=10) |
| `maxcreative` | 0.407 (n=10) | 0.421 (n=10) | 0.421 (n=10) |

*cell = mean score over that cell's replicates.*

- prompt sensitivity: best-minus-worst prompt mean = **0.024**
- temperature sensitivity: best-minus-worst temperature mean = **0.006**
- At n=10 per cell these spreads are usually inside the noise; compare them against the CI width in the leaderboard before calling a prompt or a temperature better.

Served as: `gpt-4.1-2025-04-14`.

### `gpt-5.2`

| prompt \ temperature | T=0 | T=0.7 | T=1 |
|---|---|---|---|
| `verbatim` | 0.400 (n=10) | 0.393 (n=10) | 0.405 (n=10) |
| `terse` | 0.417 (n=10) | 0.408 (n=10) | 0.418 (n=10) |
| `cot` | 0.404 (n=10) | 0.407 (n=10) | 0.393 (n=10) |
| `maxcreative` | 0.418 (n=10) | 0.413 (n=10) | 0.407 (n=10) |

*cell = mean score over that cell's replicates.*

- prompt sensitivity: best-minus-worst prompt mean = **0.015**
- temperature sensitivity: best-minus-worst temperature mean = **0.005**
- At n=10 per cell these spreads are usually inside the noise; compare them against the CI width in the leaderboard before calling a prompt or a temperature better.

Served as: `gpt-5.2-2025-12-11`.

### `o4-mini`

| prompt \ temperature | T=n/a |
|---|---|
| `verbatim` | 0.410 (n=10) |
| `terse` | 0.395 (n=10) |
| `cot` | 0.404 (n=10) |
| `maxcreative` | 0.417 (n=10) |

*cell = mean score over that cell's replicates.*

- prompt sensitivity: best-minus-worst prompt mean = **0.022**
- At n=10 per cell these spreads are usually inside the noise; compare them against the CI width in the leaderboard before calling a prompt or a temperature better.

Served as: `o4-mini-2025-04-16`.

Registry notes: reasoning model; the most verbose here, needs a much larger token ceiling.

### `gpt-5`

| prompt \ temperature | T=n/a |
|---|---|
| `verbatim` | 0.413 (n=10) |
| `terse` | 0.393 (n=10) |
| `cot` | 0.410 (n=10) |
| `maxcreative` | 0.405 (n=10) |

*cell = mean score over that cell's replicates.*

- prompt sensitivity: best-minus-worst prompt mean = **0.020**
- At n=10 per cell these spreads are usually inside the noise; compare them against the CI width in the leaderboard before calling a prompt or a temperature better.

Served as: `gpt-5-2025-08-07`.

Registry notes: heaviest reasoner in the roster, ~3136 reasoning tokens per answer.

### `gpt-5-nano`

| prompt \ temperature | T=n/a |
|---|---|
| `verbatim` | 0.393 (n=9), 1 refused |
| `terse` | 0.404 (n=10) |
| `cot` | 0.391 (n=10) |
| `maxcreative` | 0.404 (n=10) |

*cell = mean score over that cell's replicates.*

- prompt sensitivity: best-minus-worst prompt mean = **0.013**
- At n=10 per cell these spreads are usually inside the noise; compare them against the CI width in the leaderboard before calling a prompt or a temperature better.

Served as: `gpt-5-nano-2025-08-07`.

Registry notes: ~1728 reasoning tokens per answer -- the nano thinks more than the mini.

### `gpt-5-mini`

| prompt \ temperature | T=n/a |
|---|---|
| `verbatim` | 0.404 (n=10) |
| `terse` | 0.402 (n=10) |
| `cot` | 0.400 (n=10) |
| `maxcreative` | 0.397 (n=10) |

*cell = mean score over that cell's replicates.*

- prompt sensitivity: best-minus-worst prompt mean = **0.007**
- At n=10 per cell these spreads are usually inside the noise; compare them against the CI width in the leaderboard before calling a prompt or a temperature better.

Served as: `gpt-5-mini-2025-08-07`.

Registry notes: ~960 reasoning tokens per answer.

### `gemma-4-e4b`

| prompt \ temperature | T=0 | T=0.7 | T=1 |
|---|---|---|---|
| `verbatim` | 0.365 (n=10) | 0.376 (n=9), 1 refused | 0.379 (n=10) |
| `terse` | 0.401 (n=10) | 0.395 (n=10) | 0.390 (n=10) |
| `cot` | 0.364 (n=10) | 0.374 (n=10) | 0.363 (n=10) |
| `maxcreative` | 0.381 (n=10) | 0.378 (n=10) | 0.380 (n=10) |

*cell = mean score over that cell's replicates.*

- prompt sensitivity: best-minus-worst prompt mean = **0.028**
- temperature sensitivity: best-minus-worst temperature mean = **0.003**
- At n=10 per cell these spreads are usually inside the noise; compare them against the CI width in the leaderboard before calling a prompt or a temperature better.

Served as: `google/gemma-4-e4b`.

Registry notes: local via LM Studio; reasoning model, ~1000 completion tokens per answer.

### `lfm2.5-1.2b`

| prompt \ temperature | T=0 | T=0.7 | T=1 |
|---|---|---|---|
| `verbatim` | 0.390 (n=10) | 0.392 (n=10) | 0.384 (n=10) |
| `terse` | 0.396 (n=10) | 0.388 (n=10) | 0.385 (n=10) |
| `cot` | 0.349 (n=10) | 0.349 (n=10) | 0.349 (n=10) |
| `maxcreative` | 0.382 (n=10) | 0.382 (n=10) | 0.382 (n=10) |

*cell = mean score over that cell's replicates.*

- prompt sensitivity: best-minus-worst prompt mean = **0.041**
- temperature sensitivity: best-minus-worst temperature mean = **0.004**
- At n=10 per cell these spreads are usually inside the noise; compare them against the CI width in the leaderboard before calling a prompt or a temperature better.

Served as: `liquid/lfm2.5-1.2b`.

Registry notes: local via LM Studio; small and fast (~0.2s/call), expect low validity rate.

## Cross-embedder agreement

| Spearman ρ | `text-embedding-embeddinggemma-300m` | `text-embedding-nomic-embed-text-v1.5` | `text-embedding-qwen3-embedding-0.6b` | `text-embedding-qwen3-embedding-4b` |
|---|---|---|---|---|
| `text-embedding-embeddinggemma-300m` | 1.00 | 0.85 | 0.04 | 0.49 |
| `text-embedding-nomic-embed-text-v1.5` | 0.85 | 1.00 | 0.15 | 0.60 |
| `text-embedding-qwen3-embedding-0.6b` | 0.04 | 0.15 | 1.00 | 0.76 |
| `text-embedding-qwen3-embedding-4b` | 0.49 | 0.60 | 0.76 | 1.00 |

*rank correlation of the model ordering between two embedders, over the models both scored.*

**Robustness to embedder choice: the ranking is **not** robust** (worst pairwise Spearman ρ = 0.04 over 6 pair(s)) — embedders disagree about the ordering, which means this leaderboard is largely an artifact of the embedder chosen.

## Provenance

- **generated**: 2026-09-01T08:17:04Z
- **dat-bench version**: 0.1.0
- **git commit**: not available
- **summary schema**: v1
- **replicates (n)**: 10 per model x prompt x temperature cell
- **words scored per run (n_use)**: 7
- **prompts**: `verbatim`, `terse`, `cot`, `maxcreative`
- **temperatures**: 0, 0.7, 1, n/a
- **policies scored**: `strict`, `lenient`
- **embedders**: `text-embedding-embeddinggemma-300m`, `text-embedding-nomic-embed-text-v1.5`, `text-embedding-qwen3-embedding-0.6b`, `text-embedding-qwen3-embedding-4b`
- **embedding endpoint**: http://localhost:1234/v1
- **model registry**: /Users/narunyadav/sp/dat-bench/models.yaml
- **command**: `python -m datbench report --html`

Model ids actually served by the API (`response.model`):

- `gpt-5.1` → `gpt-5.1-2025-11-13`
- `gpt-4o-mini` → `gpt-4o-mini-2024-07-18`
- `gpt-5.6-luna` → `gpt-5.6-luna`
- `gpt-5.6-sol` → `gpt-5.6-sol`
- `gpt-4o` → `gpt-4o-2024-08-06`
- `gpt-5.6-terra` → `gpt-5.6-terra`
- `gpt-5.4-mini` → `gpt-5.4-mini-2026-03-17`
- `gpt-4.1-mini` → `gpt-4.1-mini-2025-04-14`
- `gpt-5.4` → `gpt-5.4-2026-03-05`
- `gpt-5-search-api` → `gpt-5-search-api-2025-10-14`
- `gpt-3.5-turbo` → `gpt-3.5-turbo-0125`
- `gpt-5.4-nano` → `gpt-5.4-nano-2026-03-17`
- `gpt-5.5` → `gpt-5.5-2026-04-23`
- `gpt-4.1` → `gpt-4.1-2025-04-14`
- `gpt-5.2` → `gpt-5.2-2025-12-11`
- `o4-mini` → `o4-mini-2025-04-16`
- `gpt-5` → `gpt-5-2025-08-07`
- `gpt-5-nano` → `gpt-5-nano-2025-08-07`
- `gpt-5-mini` → `gpt-5-mini-2025-08-07`
- `gemma-4-e4b` → `google/gemma-4-e4b`
- `lfm2.5-1.2b` → `liquid/lfm2.5-1.2b`

A served id that differs from the requested one means the provider routed elsewhere; the row is labelled by what we asked for, so check this list before attributing a result to a model.

## Known limitations

- **The `rare` flag is a weak proxy for “specialised vocabulary”.** It is a Zipf frequency cut, and frequency is not technicality. *quark* scores Zipf 3.05 and *photon* 3.32, so both pass unflagged despite being physics jargon — while *thimble*, ordinary household vocabulary and the DAT paper's own example word, sits at 2.53 and clears the 2.5 threshold by 0.03. The cut is doing very little work at the boundary: technical words routinely slip through, and a slightly higher threshold would start rejecting legitimate common nouns. Read `valid_rate` and `gaming_index` as noisy indicators, not verdicts.
- **A deterministic cell has an effective sample size of 1, whatever `n` says.** At temperature 0, or under total mode collapse, every replicate returns the same words; the bootstrap then produces a zero-width interval that looks like extreme precision but rests on one observation. Such cells are marked `[degenerate: n_eff=1]` rather than given a CI. This interacts badly with best-cell selection: a deterministic cell has no sampling noise to pull its mean down, so the maximum over cells is drawn toward T=0 — check the Jaccard column before reading a best cell as capability. Compare models on a cell where they actually vary, and raise `n` only for cells whose Jaccard is below 1.
- **The local embedders compress the far end of the distance range.** Measured against GloVe 840B-300d on the DAT paper's own example pair: GloVe puts *cat*/*thimble* at 0.879, while `qwen3-embedding-4b` puts it at 0.396 — yet the two agree closely on the *near* pair *cat*/*dog* (0.198 vs 0.204). The compression is roughly 2.2x and it falls exactly where the DAT discriminates, so differences between two genuinely good models are squeezed into a narrower band here than on the published instrument. Practical consequence: resolving close models needs more replicates than it would with GloVe. If two CIs overlap, that is as likely to be the scale as the models. (Reproduce with `scripts/premise_check.py`.)
- **Temperature is not comparable across providers.** The same nominal `temperature` means different sampling behaviour on different stacks (different default top-p, different logit post-processing, some providers clamp or ignore it). Read the temperature grid within a model, never across models.
- **n=10 replicates still gives wide confidence intervals on any single cell.** Most prompt-to-prompt and temperature-to-temperature differences here are inside the noise. If two CIs overlap, treat the cells as tied.
- **Scores are one embedder's opinion of semantic distance.** They inherit its biases (morphology, register, tokenisation). The cross-embedder agreement section is the only check on that; if it is weak, the ordering is not a fact about the models.
- **Refused runs are excluded, not scored zero.** That is the right call statistically — a floor of 0 would reward invalid output — but it means the leaderboard mean is conditional on producing scorable output. Always read it next to the refusal count.
- **A model can be prompted into a better number.** Only the best-prompt cell reaches the leaderboard, so the headline is an upper bound over the prompts tried, not a measure of typical behaviour.
