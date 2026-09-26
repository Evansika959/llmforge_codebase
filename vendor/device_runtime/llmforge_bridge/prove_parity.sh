#!/usr/bin/env bash
# Prove the C engine reproduces the original PyTorch model bit-for-bit.
#
# End-to-end: export checkpoint -> build C engine -> run BOTH on the same prompt
# (PyTorch reference vs C) -> diff token-for-token -> decode to text.
#
# Usage:
#   llmforge_bridge/prove_parity.sh [CKPT_DIR] [PROMPT] [N_STEPS]
# Defaults: smollm2_135M checkpoint, "The quick brown fox", 24 steps.
set -euo pipefail

CKPT="${1:-$HOME/Evo_GPT_checkpoints_backup/smollm2_135M}"
PROMPT="${2:-The quick brown fox}"
N="${3:-24}"

# repo root = parent of this script's dir
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BIN=smollm2_parity_v0.bin
IDS=/tmp/parity_ids.txt
REF=/tmp/parity_ref.txt
COUT=/tmp/parity_c.txt

echo "== 1. export checkpoint -> llama2.c v0 (fp32) =="
python3 llmforge_bridge/export_llmforge.py "$CKPT" "$BIN" --version 0 >/dev/null
echo "   wrote $BIN"

echo "== 2. build C engine (GELU-gated) =="
make rungelu >/dev/null
echo "   built run_gelu"

echo "== 3. tokenize prompt with tiktoken gpt2 =="
python3 -c "import tiktoken,sys; print(*tiktoken.get_encoding('gpt2').encode(sys.argv[1]))" "$PROMPT" > "$IDS"
echo "   prompt='$PROMPT'  ids=$(cat "$IDS")"

echo "== 4. PyTorch reference (fp32 CPU, greedy) =="
python3 llmforge_bridge/ref_dump.py "$CKPT" "$IDS" "$N" 2>/dev/null > "$REF"

echo "== 5. C engine (run_gelu, greedy -t 0) =="
./run_gelu "$BIN" -I "$IDS" -t 0 -n "$N" 2>/dev/null > "$COUT"

echo "== 6. compare token-for-token =="
echo "   PyTorch: $(tr '\n' ' ' < "$REF")"
echo "   C:       $(tr '\n' ' ' < "$COUT")"
if diff -q "$REF" "$COUT" >/dev/null; then
  echo "   RESULT: IDENTICAL — $(wc -l < "$COUT" | tr -d ' ')/$N tokens match (C == PyTorch)"
else
  echo "   RESULT: MISMATCH"; diff "$REF" "$COUT"; exit 1
fi

echo "== 7. decode to text (tiktoken) =="
python3 -c "import tiktoken,sys; e=tiktoken.get_encoding('gpt2'); ids=[int(x) for x in open(sys.argv[1]).read().split()]; print('   '+sys.argv[2]+e.decode(ids))" "$COUT" "$PROMPT"

echo "== done. (artifact: $BIN — gitignored) =="
