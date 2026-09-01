#!/usr/bin/env bash
#
# One command, cold start to HTML report.
#
#   scripts/make_report.sh                  # full pipeline, then print the paths
#   scripts/make_report.sh --open           # ...and open the HTML report
#   scripts/make_report.sh --n 3            # fewer replicates (quick smoke)
#   scripts/make_report.sh --models lfm2.5-1.2b --n 5
#   scripts/make_report.sh --report-only    # skip the LLM calls, just re-render
#
# Any other flag is forwarded to `datbench all` verbatim, so
#   scripts/make_report.sh --primary-embedder text-embedding-nomic-embed-text-v1.5
# works. See `python -m datbench all --help`.
#
# Everything is idempotent: `run` skips cells already recorded, so re-running after
# an interrupt costs only the calls that did not complete.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

VENV="$REPO/.venv"
PY="$VENV/bin/python"
EMBED_URL="${DATBENCH_EMBED_URL:-http://localhost:1234/v1}"

OPEN_AFTER=0
REPORT_ONLY=0
FORWARD=()
for arg in "$@"; do
  case "$arg" in
    --open)        OPEN_AFTER=1 ;;
    --report-only) REPORT_ONLY=1 ;;
    *)             FORWARD+=("$arg") ;;
  esac
done

say() { printf '\033[1m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- environment
if [[ ! -x "$PY" ]]; then
  say "creating venv"
  command -v uv >/dev/null 2>&1 \
    && uv venv --python 3.12 \
    || python3 -m venv .venv
fi

# Cheap import probe rather than an unconditional install: a no-op reinstall on
# every report costs several seconds and a network round trip.
if ! "$PY" -c "import openai, yaml, numpy, nltk, wordfreq, jinja2" >/dev/null 2>&1; then
  say "installing dependencies"
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$PY" -q \
      "openai>=1.40" "pyyaml>=6.0" "numpy>=1.24" \
      "nltk>=3.8" "wordfreq>=3.0" "jinja2>=3.1" "pytest>=8.0"
  else
    "$PY" -m pip install -q --upgrade pip
    "$PY" -m pip install -q \
      "openai>=1.40" "pyyaml>=6.0" "numpy>=1.24" \
      "nltk>=3.8" "wordfreq>=3.0" "jinja2>=3.1" "pytest>=8.0"
  fi
fi

# WordNet powers the noun and proper-noun checks. Without it those rules are
# silently skipped -- the report says so via capabilities(), but a validity rate
# measured with half the rules off is not the number anyone wants.
if ! "$PY" -c "from nltk.corpus import wordnet as wn; wn.synsets('cat')" >/dev/null 2>&1; then
  say "downloading WordNet corpus"
  "$PY" -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"
fi

# ---------------------------------------------------------------- preflight
if [[ "$REPORT_ONLY" -eq 0 ]]; then
  say "checking LM Studio at $EMBED_URL"
  models_json="$(curl -sf --max-time 10 "$EMBED_URL/models" || true)"
  [[ -n "$models_json" ]] || die "LM Studio is not answering at $EMBED_URL.
  Start it and load at least one embedding model, or set DATBENCH_EMBED_URL.
  Scoring is local, so there is no fallback."

  n_embed="$("$PY" - "$models_json" <<'EOF'
import json, sys
data = json.loads(sys.argv[1]).get("data", [])
print(sum(1 for m in data if "embed" in str(m.get("id", ""))))
EOF
)"
  [[ "$n_embed" -gt 0 ]] || die "LM Studio is up but has no embedding model loaded.
  Load one (e.g. text-embedding-qwen3-embedding-4b) and re-run."
  say "$n_embed embedding model(s) available"

  say "live models in the registry"
  "$PY" -m datbench models | sed 's/^/    /'
fi

# ---------------------------------------------------------------- pipeline
if [[ "$REPORT_ONLY" -eq 1 ]]; then
  say "re-rendering report only (no LLM calls)"
  "$PY" -m datbench analyze "${FORWARD[@]+"${FORWARD[@]}"}"
  "$PY" -m datbench report --html "${FORWARD[@]+"${FORWARD[@]}"}"
else
  say "running pipeline: run -> score -> analyze -> report"
  "$PY" -m datbench all --html "${FORWARD[@]+"${FORWARD[@]}"}"
fi

say "building the interactive explorer"
"$PY" "$REPO/scripts/build_explorer.py"

# ---------------------------------------------------------------- output
MD="$REPO/out/report.md"
HTML="$REPO/out/report.html"
EXPLORER="$REPO/out/explorer.html"
echo
say "done"
printf '    explorer : %s   <- start here: every answer, browsable\n' "$EXPLORER"
printf '    report   : %s\n' "$HTML"
printf '    markdown : %s\n' "$MD"
printf '    data     : %s\n' "$REPO/out/summary.json  (+ summary.csv, scores.jsonl, words.jsonl)"

# The headline is the one thing worth reading without opening the file: it says
# whether anything actually beat chance.
if [[ -f "$MD" ]]; then
  echo
  sed -n '/## Headline/,/^## /p' "$MD" | sed '$d' | sed 's/^/    /'
fi

if [[ "$OPEN_AFTER" -eq 1 ]]; then
  # The explorer is the one a person wants; the markdown report is the audit trail.
  TARGET="$EXPLORER"; [[ -f "$TARGET" ]] || TARGET="$HTML"
  case "$(uname -s)" in
    Darwin) open "$TARGET" ;;
    Linux)  xdg-open "$TARGET" >/dev/null 2>&1 || say "open $TARGET manually" ;;
    *)      say "open $TARGET manually" ;;
  esac
fi
