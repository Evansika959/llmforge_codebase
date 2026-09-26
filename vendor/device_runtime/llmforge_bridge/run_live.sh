#!/usr/bin/env bash
# Live inference with the C engine on the bundled LLMForge model.
# Tokens stream to your terminal as they are generated.
#
# Usage:
#   llmforge_bridge/run_live.sh                 # interactive: type prompts, Ctrl-D or "quit" to exit
#   llmforge_bridge/run_live.sh "Once upon a"   # one-shot: generate from this prompt
#
# Tunables (env vars):
#   MODEL   model .bin        (default: bundled smollm2_135M Q8)
#   TOK     tokenizer .bin    (default: bundled gpt2 tokenizer)
#   ENGINE  runq_gelu|run_gelu (default: runq_gelu, the Q8 engine)
#   TEMP    temperature       (default: 0.8;  0 = deterministic/greedy)
#   TOPP    top-p             (default: 0.9)
#   STEPS   max tokens        (default: 256)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-models/smollm2_135M/smollm2_135M.q8.bin}"
TOK="${TOK:-models/smollm2_135M/tokenizer_gpt2.bin}"
ENGINE="${ENGINE:-runq_gelu}"
TEMP="${TEMP:-0.8}"
TOPP="${TOPP:-0.9}"
STEPS="${STEPS:-256}"

# build the engine if missing
if [ ! -x "./$ENGINE" ]; then
  echo "building engine ($ENGINE)..." >&2
  make rungelu >/dev/null
fi

# sanity: files present
[ -f "$MODEL" ] || { echo "model not found: $MODEL" >&2; exit 1; }
[ -f "$TOK" ]   || { echo "tokenizer not found: $TOK" >&2; exit 1; }

# LFS-pointer guard: a bundled model that wasn't pulled is a ~130-byte text pointer
if head -c 64 "$MODEL" | grep -q "git-lfs"; then
  echo "ERROR: $MODEL is a Git LFS pointer, not real weights. Run: git lfs pull" >&2
  exit 1
fi

gen() {  # $1 = prompt  (the engine echoes the full prompt, then streams the continuation)
  ./"$ENGINE" "$MODEL" -g "$TOK" -i "$1" -t "$TEMP" -p "$TOPP" -n "$STEPS" 2>/dev/null
}

if [ "$#" -gt 0 ]; then
  # one-shot mode
  gen "$*"
else
  # interactive REPL
  echo "live inference — model: $MODEL  engine: $ENGINE  (temp=$TEMP top-p=$TOPP)" >&2
  echo "type a prompt and press Enter; 'quit' or Ctrl-D to exit." >&2
  while true; do
    printf '\nprompt> ' >&2
    if ! IFS= read -r line; then echo >&2; break; fi   # Ctrl-D
    [ -z "$line" ] && continue
    [ "$line" = "quit" ] && break
    gen "$line"
    echo
  done
fi
