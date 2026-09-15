#!/usr/bin/env bash
# Repack the four raw buckets with SmolLM2's tokenizer (49,152 vocab), prefix sl_.
#
# A tokenizer change forces a repack: token IDs are not portable, and the packed .npy holds IDs.
# Qwen3's 151,936-token vocabulary and SmolLM2's 49,152 also give different tokens-per-byte, so
# the same raw text yields a different number of sequences -- there is no way to reuse the pack.
#
# Caps are ~4x the expected training budget (~500M tokens for a 135M uptrain), so ablations can
# run without repeating data. Buckets are packed one at a time because the rust tokenizer already
# saturates every core.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PACK_TOKENIZER=HuggingFaceTB/SmolLM2-135M
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

pack sl_fineweb fineweb_edu  800000000
pack sl_code    code         500000000
pack sl_math    openmath     400000000
pack sl_rag     rag_hotpot   250000000
echo "=== done ==="; ls -la ${LLMFORGE_DATA:-data}/packed/sl_*
