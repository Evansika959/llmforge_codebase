#!/usr/bin/env bash
# Pack the code, math and retrieval buckets with the Qwen3 tokenizer, for the Qwen3-0.6B recipe.
#
# The Qwen3 scales share one tokenizer and vocabulary, so one pack serves all of them. The recipe's web
# bucket is fineweb10bt, packed from the full sample-10BT parquet as docs/supernet.md describes. The
# caps match the original packs: about 300M tokens of code and of math, and all of HotpotQA, which
# comes to about 130M tokens.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PACK_TOKENIZER=${PACK_TOKENIZER:-Qwen/Qwen3-0.6B-Base}
export TOKENIZERS_PARALLELISM=true
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"

pack () {  # name  jsonl  cap
  local out=$1 src=$2 cap=$3
  if [ -f "${LLMFORGE_DATA:-data}/packed/${out}_tokens.npy" ] && [ -f "${LLMFORGE_DATA:-data}/packed/${out}_seglens.jsonl" ]; then
    echo "[skip] $out already packed"; return
  fi
  echo "=== packing $out from $src (cap $cap) ==="
  python -m llmforge.supernet.data.pack_stream --jsonl "${LLMFORGE_DATA:-data}/raw/${src}.jsonl" \
         --out "$out" --max-tokens "$cap" || echo "[FAIL] $out"
}

pack code code       300000000
pack math openmath   300000000
pack rag  rag_hotpot 0
echo "=== done ==="; ls -la ${LLMFORGE_DATA:-data}/packed/{code,math,rag}_*
