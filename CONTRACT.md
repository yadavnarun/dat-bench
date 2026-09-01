# dat-bench module contract

**This file is the interface authority.** Every module is implemented against the
signatures below. If an implementation needs to deviate, the deviation is a bug in
this file — fix the file first, then the code.

Python 3.12. No module imports another module's *internals*; only the names listed
here. `runner.py`, `cli.py` and `report.py` are the only modules allowed to import
several others.

---

## 0. Vocabulary

- **candidate** — a token pulled out of an LLM response by `parse.py`. Not yet judged.
- **check** — a candidate after `validate.py` has applied the DAT rules.
- **valid word** — a check with `valid=True`.
- **cell** — one (model_id, prompt_id, temperature) combination. A cell holds N replicates.
- **run** — one LLM call = one row in `runs/responses.jsonl`.
- **embedder** — one LM Studio embedding model id.
- **policy** — how many valid words a run must have to be scorable. See §5.

---

## 1. On-disk artifacts

All JSONL: one JSON object per line, UTF-8, append-only, newest-last.
`run_id` is the join key across all three files.

### `runs/responses.jsonl` — written by `runner.py`

```json
{
  "run_id": "3f9a1c...",
  "model_id": "gemma-4-e4b",
  "model_reported": "google/gemma-4-e4b",
  "prompt_id": "verbatim",
  "temperature": 0.7,
  "replicate": 3,
  "response_text": "1. cat\n2. thimble\n...",
  "finish_reason": "stop",
  "usage": {"prompt_tokens": 118, "completion_tokens": 44},
  "latency_ms": 812,
  "ts": "2026-08-31T12:00:00Z",
  "error": null
}
```

- `run_id` = `sha1(f"{model_id}|{prompt_id}|{temperature!r}|{replicate}")`, first 16 hex
  chars. **Deterministic** — this is what makes runs resumable and idempotent.
- `model_reported` is whatever the API said it actually served (`response.model`), which
  may differ from `model_id`. Provenance: never overwrite it with `model_id`.
- On failure: `response_text=""`, `error` = a short string, other fields still present.
  A failed run is a row, not a gap. Never silently drop it.

### `out/words.jsonl` — written by `cli.py score` stage

```json
{
  "run_id": "3f9a1c...",
  "candidates": ["cat", "thimble", "New York", "quark"],
  "words": [
    {"word": "cat", "clean": "cat", "valid": true, "flags": [], "zipf": 5.42},
    {"word": "New York", "clean": "new york", "valid": false,
     "flags": ["multiword", "proper_noun"], "zipf": null}
  ]
}
```

### `out/scores.jsonl` — written by `cli.py score` stage

One row per (run_id, embedder, policy).

```json
{
  "run_id": "3f9a1c...",
  "embedder": "text-embedding-qwen3-embedding-4b",
  "policy": "strict",
  "score": 0.4123,
  "n_candidates": 10,
  "n_valid": 9,
  "n_words_used": 7,
  "valid_rate": 0.9,
  "scored": true,
  "reason": null
}
```

- `scored=false` + a `reason` string when the policy refused the run. `score` is then
  `null`. **A refused run is never treated as score 0** — that would reward invalidity
  with a floor instead of an exclusion.

### `out/baselines.json` — written by `cli.py score` stage

```json
{
  "text-embedding-qwen3-embedding-4b": {
    "random": {"mean": 0.38, "sd": 0.05, "n": 1000, "k": 7,
               "p05": 0.30, "p50": 0.38, "p95": 0.46, "seed": 0},
    "categories": {"animals": 0.14, "tools": 0.19, "colours": 0.11},
    "paper_example": 0.41
  }
}
```

---

## 2. `providers.py`

```python
@dataclass(frozen=True)
class ModelSpec:
    id: str                      # our stable label, used in output
    model: str                   # the string the API wants
    base_url: str                # e.g. "http://localhost:1234/v1"
    api_key_env: str | None      # None => no auth (LM Studio)
    enabled: bool = True
    max_tokens: int = 512
    supports_temperature: bool = True
    max_concurrency: int = 1
    notes: str = ""

@dataclass(frozen=True)
class ChatResult:
    text: str
    model_reported: str
    finish_reason: str
    usage: dict[str, int]
    latency_ms: int
    error: str | None = None

def load_models(path: Path) -> list[ModelSpec]: ...
    # Parses models.yaml. Skips entries with enabled=false. An entry whose
    # api_key_env names an unset/empty var is dropped with a log line naming the
    # var -- that is the "activates when you add a key" behaviour.

def available_models(path: Path) -> tuple[list[ModelSpec], list[tuple[str, str]]]: ...
    # -> (usable specs, [(model_id, reason_skipped)]). cli uses this to print
    # exactly why a model is inert.

class ChatClient:
    def complete(self, spec: ModelSpec, prompt: str, *,
                 temperature: float | None, max_tokens: int | None = None,
                 timeout: float = 120.0) -> ChatResult: ...
        # Never raises for API/network failure -- returns ChatResult(error=...).
        # Retries 3x with exponential backoff on 429/5xx/timeout only.
        # If spec.supports_temperature is False, omits the temperature param
        # entirely (do not send temperature=None).
```

Uses the `openai` SDK pointed at `spec.base_url`. One client, every provider —
OpenAI, OpenRouter, DeepSeek, Groq, Together, Anthropic and Gemini all serve
`/v1/chat/completions`. No litellm, no LangChain.

---

## 3. `parse.py`

```python
def parse_words(text: str, *, want: int = 10) -> list[str]: ...
```

Extracts candidate words from a raw response, **order preserved**, at most `want`.
Deduplication is NOT done here — a repeated word is a real signal and `validate.py`
flags it. Must handle, in this priority order:

1. A JSON array or a JSON object with a `words` key, anywhere in the text
   (including inside a ```json fence).
2. A numbered list: `1. cat`, `1) cat`, `1 - cat`.
3. A bulleted list: `- cat`, `* cat`, `• cat`.
4. Newline-separated bare words.
5. Comma- or semicolon-separated on one line.

Strips markdown emphasis (`**cat**`), surrounding quotes, trailing punctuation, and
leading/trailing whitespace. Drops obvious preamble lines ("Here are 10 words:",
"Sure!"). A line containing a colon keeps only the part after the final colon when
the part before it is ≤4 words. Never returns empty strings.

---

## 4. `validate.py`

```python
@dataclass(frozen=True)
class WordCheck:
    word: str                    # as the model emitted it
    clean: str                   # lowercased, stripped, internal spaces -> "-"
    valid: bool
    flags: tuple[str, ...]       # sorted, from FLAGS
    zipf: float | None

FLAGS = ("empty", "multiword", "not_alpha", "not_in_dict",
         "not_noun", "proper_noun", "rare", "duplicate")

def validate_words(words: Sequence[str], *,
                   rare_zipf_threshold: float = 2.5) -> list[WordCheck]: ...

def capabilities() -> dict[str, bool]: ...
    # {"dictionary": True, "wordnet": False, "wordfreq": True} -- what actually
    # loaded. report.py prints this so a degraded run is never mistaken for a
    # clean one.
```

Rules, mapped to the DAT's five instructions:

| DAT rule | flag | how |
|---|---|---|
| single word | `multiword` | whitespace after cleaning; a hyphenated compound is allowed (the reference implementation maps `cul de sac` → `cul-de-sac`) |
| English word | `not_in_dict` | WordNet lemmas, else `/usr/share/dict/words` |
| noun | `not_noun` | `wordnet.synsets(w, pos=NOUN)` non-empty |
| no proper noun | `proper_noun` | present in dict only capitalised, or in `/usr/share/dict/propernames` |
| no specialised vocab | `rare` | `wordfreq.zipf_frequency(w,'en') < rare_zipf_threshold` |
| — | `duplicate` | a `clean` value already seen earlier in this list |
| — | `not_alpha` | contains chars outside `[a-z-]` after cleaning |

`valid = not flags`. **Graceful degradation is mandatory:** if WordNet is absent, skip
`not_noun`/`proper_noun` and report it via `capabilities()` — never fabricate a pass.
Missing `wordfreq` skips `rare`. A missing dictionary skips `not_in_dict`.

---

## 5. `score.py`

```python
def cosine_distance(a: Sequence[float], b: Sequence[float]) -> float: ...
    # 1 - cos_sim, range 0..2. Zero-norm vector -> raise ValueError.

def dat_score(words: Sequence[str], vecs: Mapping[str, Sequence[float]], *,
              n_use: int = 7) -> float | None: ...
    # Mean pairwise cosine distance over the FIRST n_use words. C(7,2)=21 pairs.
    # Returns None if fewer than 2 words have vectors.
    # NOTE: returns the raw mean distance, NOT x100. The x100 convention belongs
    # to GloVe-scale scores; we are on local embeddings and must not imply
    # comparability with published human norms.

@dataclass(frozen=True)
class ScoreResult:
    score: float | None
    n_candidates: int
    n_valid: int
    n_words_used: int
    valid_rate: float
    scored: bool
    reason: str | None

def score_run(checks: Sequence[WordCheck], vecs: Mapping[str, Sequence[float]], *,
              policy: str = "strict", n_use: int = 7,
              min_words: int = 4) -> ScoreResult: ...
    # policy="strict":  needs >= n_use valid words, else scored=False.
    # policy="lenient": needs >= min_words valid words; scores on however many
    #                   are available (n_words_used records it, because fewer
    #                   words means fewer pairs and a different variance).

@dataclass(frozen=True)
class BaselineStats:
    mean: float; sd: float; n: int; k: int
    p05: float; p50: float; p95: float; seed: int
    def percentile_of(self, score: float) -> float: ...
    def z_of(self, score: float) -> float: ...

def random_baseline(embed_fn, vocab: Sequence[str], *, n_draws: int = 1000,
                    k: int = 7, seed: int = 0) -> BaselineStats: ...
    # Chance distribution for this embedder: draw k random common nouns, score,
    # repeat. This is what makes an arbitrary-scale number interpretable.

def category_floor(embed_fn, categories: Mapping[str, Sequence[str]]) -> dict[str, float]: ...
    # Semantically-tight sets (7 animals, 7 tools) -> a low anchor.
```

`embed_fn: Callable[[Sequence[str]], dict[str, list[float]]]` — inject the embedder so
`score.py` is testable with a fixture and needs no network.

---

## 6. `embed.py`

```python
class Embedder:
    def __init__(self, model: str, *, base_url: str = "http://localhost:1234/v1",
                 cache_path: Path = Path("cache/embeddings.sqlite"),
                 batch_size: int = 32, timeout: float = 120.0): ...
    model: str
    def embed(self, words: Sequence[str]) -> dict[str, list[float]]: ...
        # Cache-first, deduped, order-independent. Only uncached words hit the
        # network. STRICTLY SEQUENTIAL: one request at a time, batched. The local
        # box is shared with the chat models -- do not parallelise it.
    def close(self) -> None: ...

def list_embedding_models(base_url: str = "http://localhost:1234/v1") -> list[str]: ...
    # GET /models, filtered to ids containing "embed".
```

Cache schema — key on the embedder too, or vectors from different models collide:

```sql
CREATE TABLE IF NOT EXISTS emb (
  model TEXT NOT NULL, word TEXT NOT NULL,
  dim INTEGER NOT NULL, vec BLOB NOT NULL,   -- float32 little-endian
  ts TEXT NOT NULL, PRIMARY KEY (model, word)
);
```

---

## 7. `analyze.py`

```python
def bootstrap_ci(xs: Sequence[float], *, n_boot: int = 10000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float]: ...
    # Percentile bootstrap of the mean. len(xs)<2 -> (nan, nan).

def cell_stats(scores: Sequence[float]) -> dict: ...
    # {"n","mean","sd","ci_lo","ci_hi","min","max"}. sd is sample sd (ddof=1).

def jaccard_self_overlap(word_sets: Sequence[Collection[str]]) -> float: ...
    # Mean pairwise Jaccard across replicates of one cell. 1.0 = the model emits
    # an identical set every run (total mode collapse); ~0 = fully fresh. <2 sets
    # -> nan.

def distinct_pool(word_sets: Sequence[Collection[str]]) -> dict: ...
    # {"distinct","total","ratio"} -- vocabulary breadth across a cell.

def rank_correlation(scores_by_embedder: Mapping[str, Mapping[str, float]]
                     ) -> dict[tuple[str, str], float]: ...
    # Spearman rho between every pair of embedders over the shared model ranking.
    # Answers: is the leaderboard real, or an artifact of one embedder?
    # Implement Spearman directly (rank + Pearson); do not add a scipy dependency.

def gaming_index(mean_score: float, valid_rate: float) -> float: ...
    # score * valid_rate. A model that wins on rare/technical words ranks lower
    # here than on raw score. Reported beside, never instead of, the raw number.
```

Pure functions over plain lists/dicts. numpy is fine; scipy is not a dependency.

---

## 8. `runner.py`

```python
@dataclass(frozen=True)
class RunTask:
    spec: ModelSpec; prompt_id: str; prompt_text: str
    temperature: float; replicate: int
    @property
    def run_id(self) -> str: ...

def build_tasks(specs, prompts: Mapping[str, str], temperatures: Sequence[float],
                n: int) -> list[RunTask]: ...
    # Full factorial. A spec with supports_temperature=False collapses to a
    # single temperature cell (recorded as temperature=None), not 3 duplicates.

def existing_run_ids(path: Path) -> set[str]: ...
    # Every recorded run_id, errored or not.

def completed_run_ids(path: Path) -> set[str]: ...
    # run_ids whose MOST RECENT row succeeded. This -- not existing_run_ids -- is
    # the basis for resume: skipping any recorded run_id would drop a cell that hit
    # a transient 429 or timeout from the benchmark permanently, and silently (the
    # run reports "skipped" and the cell's mean is quietly taken over whichever
    # replicates happened to succeed). Newest-wins per run_id, matching
    # cli._latest_rows, so a retry appended after a failed row supersedes it.

def run_all(tasks, client, out_path: Path, *, resume: bool = True,
            progress=None) -> dict: ...
    # Skips tasks whose run_id already has a SUCCESSFUL row when resume=True; an
    # errored cell is re-attempted. Appends and flushes after every row, so a
    # kill -9 loses at most one call.
    # Groups by model and honours spec.max_concurrency (LM Studio entries are 1).
    # -> {"attempted","written","skipped","errors"}
```

---

## 9. `cli.py`

`python -m datbench <cmd>`, argparse:

- `models` — list every entry, usable or not, with the reason it is inert.
- `run [--n 10] [--models a,b] [--prompts ...] [--temps 0,0.7,1] [--no-resume]`
- `score [--embedders auto|a,b] [--policies strict,lenient] [--baseline-draws 1000]`
- `analyze` — writes `out/summary.json` + `out/summary.csv`
- `report [--html]` — writes `out/report.md`, `out/report.html`
- `all` — run → score → analyze → report

Every command is re-runnable and idempotent.

---

## 10. `report.py`

```python
def build_markdown(summary: dict, meta: dict) -> str: ...
def build_html(summary: dict, meta: dict) -> str: ...
```

The report MUST state, near the top and unmissably:

1. Scores come from **local LM Studio embeddings, not GloVe 840B-300d**, and are
   therefore **on an arbitrary scale and not comparable** to the published DAT
   human norms (mean ≈78, range ~50–95).
2. The random-noun baseline for each embedder, so a raw number can be read as a
   percentile against chance.
3. `validate.capabilities()` — which checks actually ran.
4. Counts of failed and policy-refused runs. A leaderboard that hides its
   denominators is not a benchmark.
