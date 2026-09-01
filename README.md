# dat-bench

**Do language models get more creative as they get better? 21 of them took the same
10-word creativity test 1800 times. The answer is not the one the leaderboard gives.**

[![tests](https://github.com/yadavnarun/dat-bench/actions/workflows/tests.yml/badge.svg)](https://github.com/yadavnarun/dat-bench/actions/workflows/tests.yml)
[![explore the data](https://img.shields.io/badge/explore-live%20data-3a4fa0)](https://yadavnarun.github.io/dat-bench/)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

### → **[Explore all 1800 answers](https://yadavnarun.github.io/dat-bench/)** ←

[![the explorer](docs/img/explorer.png)](https://yadavnarun.github.io/dat-bench/)

---

## The test

The [Divergent Association Task](https://www.datcreativity.com/task) is a real
psychology instrument (Olson & Martin, *PNAS*). You name **ten words as different from
each other as possible**; a computer measures how far apart their meanings are. Naming
*thimble* and *galaxy* takes more reaching than naming *cat* and *dog*, and in humans
that reaching correlates with other creativity measures.

21 models each did it **1800 times**: 4 prompt variants × up to 3 temperatures × 10
repeats. Every answer, every word, every score is in this repo.

## Three findings

### 1. The leaderboard is mostly an artifact of the scoring model

Four embedding models scored the same 1800 answers and ranked all 21 contestants. They
disagree almost completely — the worst-agreeing pair correlates at **Spearman ρ = +0.04**.

| model | gemma-300m | nomic | qwen3-0.6b | qwen3-4b | rank swing |
|---|---|---|---|---|---|
| `lfm2.5-1.2b` | **1st** | 21st | 21st | 21st | **±20** |
| `gpt-4o-mini` | 21st | 16th | **1st** | 2nd | **±20** |
| `gemma-4-e4b` | 10th | 13th | 3rd | 20th | ±17 |
| `gpt-5.6-terra` | 3rd | 6th | 19th | 6th | ±16 |
| … | | | | | |
| `gpt-5.1` | 2nd | 1st | 4th | 1st | ±3 |
| `gpt-5.4-mini` | 8th | 9th | 7th | 7th | ±2 |

A 1.2B model running on a laptop places **1st** under one scorer and **21st** under the
other three. Only **5 of 21 models** hold their position within 3 places across all four.

So the honest reading is narrow but real: `gpt-5.1` and `gpt-5.6-sol` are stably near the
top whichever scorer you use, and `gpt-5-mini` stably near the bottom. **Everything
between them is noise**, and a single-scorer leaderboard would present it as a ranking.

### 2. Most answers are no better than random words

Seven common nouns pulled from a dictionary at random score **0.373** on average, and
land below **0.411** nineteen times out of twenty. Only **72 of 180** prompt ×
temperature cells beat that 95th percentile. The models clearly understand the task —
they score far above what naming seven animals gets you — but on most settings they do
not beat chance at it.

This is why the harness computes a **random-noun baseline per scorer** and reports every
score as a percentile against it. Without that reference, "0.44 beats 0.42" reads like a
result.

### 3. Independent models converge on the same words

Across all 1800 runs there are only **797 distinct words**. `chair` and `pillow` appear
in **all 21 models**; `mountain`, `apple`, `river`, `thunder`, `cloud`, `justice` and
`hammer` in 20 of 21 — despite every model being told to *"think of the words on your
own."* Meanwhile **46%** of the vocabulary comes from a single model.

Repetition within a model is just as stark: `lfm2.5-1.2b` used **45 different words to
fill 1194 slots**, and `gpt-5.4` used 97 across 1200 — fewer than `gpt-4o`'s 198, despite
being newer. None of this shows up in the score.

[![run cards](docs/img/runs.png)](https://yadavnarun.github.io/dat-bench/)

*Nine consecutive attempts by the same model at temperature 0, returning an identical
answer each time. Ten "measurements" carrying the information of one.*

## What the numbers are not

Scoring uses **local embedding models via LM Studio**, not the GloVe 840B-300d vectors
the published test uses. That is a deliberate trade (no 2GB download) and it costs
comparability:

- Scores are on an **arbitrary scale** and are **not comparable** to the published human
  norms (mean ≈78). No number here is multiplied by 100, so nothing looks like one.
- The local scorers agree with GloVe on *near* word pairs (`cat`/`dog`: 0.204 vs 0.198)
  but **compress the far end ~2.2×** (`cat`/`thimble`: 0.396 vs 0.879) — exactly where
  this test discriminates. Run `scripts/premise_check.py` to reproduce.
- **This is not a creativity measure.** The DAT's validity comes from correlating with
  human creativity *in humans*. That does not transfer to language models.

Every one of these appears in the generated report, not just here.

## Run it

```bash
git clone https://github.com/yadavnarun/dat-bench && cd dat-bench
scripts/make_report.sh --open
```

One command, cold start: creates the venv, installs deps, fetches WordNet, checks
[LM Studio](https://lmstudio.ai) is up with an embedding model loaded, runs the pipeline,
and opens the explorer. Or stage by stage — each is idempotent:

```bash
python -m datbench models          # which models are live, and why the rest aren't
python -m datbench run --n 10      # the factorial (resumable)
python -m datbench score           # parse -> validate -> embed -> score
python -m datbench analyze
python -m datbench report --html
```

**Requires** [LM Studio](https://lmstudio.ai) on `:1234` with an embedding model loaded.
API keys only for whichever cloud models you want; local models need none.

### Adding a model is four lines

Every provider here speaks the OpenAI wire format, so the only per-provider facts are the
URL and the key's env var:

```yaml
  - id: my-model
    model: vendor/model-name
    base_url: https://api.vendor.com/v1
    api_key_env: VENDOR_API_KEY
```

An entry whose key is unset stays **inert** — skipped with a reason naming the variable,
never aborting the run. So cloud entries ship pre-populated and activate the moment you
add a key.

## How it works

```
prompts/*.txt ─┐
models.yaml ───┴─> runner ──> runs/responses.jsonl        (append-only, resumable)
                                     │
                     parse ──> validate ──> out/words.jsonl
                                     │
              embed (LM Studio + sqlite cache) ──> score ──> out/scores.jsonl
                                     │                       out/baselines.json
                              analyze ──> report ──> docs/report.html
                                     └──> build_explorer ──> docs/index.html
```

Two phases on purpose: `run` collects raw replies, then `score`/`analyze` are pure
functions over them. So you can **re-score with a different embedder or validity policy
without re-paying a single API call**, and an interrupted run resumes for free.

`run_id = sha1(model|prompt|temperature|replicate)[:16]` joins every artifact and makes
resume idempotent.

| | |
|---|---|
| `datbench/providers.py` | one OpenAI-compatible client, every provider |
| `datbench/runner.py` | factorial loop, resumable, per-model concurrency |
| `datbench/parse.py` | model reply → candidate words |
| `datbench/validate.py` | the task's five rules |
| `datbench/embed.py` | LM Studio embeddings + sqlite cache |
| `datbench/score.py` | the score, plus chance baseline and category floors |
| `datbench/analyze.py` | bootstrap CIs, Jaccard, Spearman (no scipy) |
| `datbench/report.py` | markdown + HTML |
| [`CONTRACT.md`](CONTRACT.md) | the module interface authority — read before editing |
| [`docs/design.md`](docs/design.md) | design rationale and rejected alternatives |

**453 tests, all network-free.** Every provider and embedder call is faked, so CI
exercises the same paths a real run does without spending anything.

## Design decisions worth stealing

Most of these exist because the first version of the benchmark quietly lied.

- **A refused run is never scored 0.** Zero is a floor that rewards invalid output.
  Refused runs are excluded and counted visibly instead.
- **Validity rate is reported beside every score.** A model can inflate its score with
  rare or technical words, which breaks the task's own rules. `gaming_index =
  score × valid_rate` sits next to the raw number, never instead of it.
- **Deterministic cells are flagged, not given a confidence interval.** At temperature 0
  ten identical replies produce a zero-width CI that looks like extreme precision but
  rests on one observation. Those read `[degenerate: n_eff=1]`.
- **Truncation is counted separately from failure.** A reasoning model that spends its
  whole token budget thinking returns an *empty* message with no error — indistinguishable
  from a model that answered badly. `gpt-5` needs 3169 output tokens; at a 3072 cap it
  returns nothing, on every run.
- **Robustness verdicts are refused when the sample can't support them.** Spearman ρ over
  two models is always ±1, so the report says "cannot be assessed" rather than laundering
  a tautology as a robustness check.
- **`validate.capabilities()` is printed in every report** — which rule checks actually
  loaded, so a degraded run can't be mistaken for a clean one.

## The data

| file | what |
|---|---|
| `data/responses.jsonl` | every raw model reply — the part that cost real API calls. A committed snapshot of `runs/responses.jsonl`, which is the working file the pipeline appends to |
| `data/summary.json` / `.csv` | per-model and per-cell aggregates |
| `data/baselines.json` | chance distribution + category floors, per scorer |

`out/words.jsonl` and `out/scores.jsonl` are derived and regenerate from the responses
with `python -m datbench score`, so they are not committed. `responses.jsonl` is
append-only, so it holds a few superseded rows from retried calls — the pipeline resolves
each `run_id` to its newest row, and so should you.

## Limitations

Beyond the scale caveats above: the `rare` flag is a weak proxy for "specialised
vocabulary" (Zipf rates *thimble* 2.53 — correctly valid, it is the paper's own example —
but *quark* 3.05, so technical words slip through); temperature is not comparable across
providers, and several reasoning models reject it outright; n=10 gives wide per-cell CIs,
and the headline best-cell figure is a maximum over correlated estimates, labelled as such
in the report.

---

Task from [datcreativity.com](https://www.datcreativity.com/task) (Olson & Martin,
University of Toronto Mississauga). This repo is an independent benchmark and is not
affiliated with the authors of the task.
