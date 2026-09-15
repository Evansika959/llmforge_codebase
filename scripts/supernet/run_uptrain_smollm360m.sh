#!/usr/bin/env bash
# SmolLM2-360M A+B supernet. Covers 144.7M-361.8M under the 40%-100% usable-range rule, which
# butts against the 135M supernet's 53.8M-134.5M with only a 10.2M gap -- two base models give
# continuous coverage of 50M-360M, and no single one can, because embeddings are unsliceable.
#
# Data needs no repack: SmolLM2-135M and -360M share one tokenizer, verified token-identical,
# vocab 49,152, so the sl_* buckets serve both.
#
# The learning rate is swept, not inherited. 1.7B's optimum did not transfer to 135M -- 3e-5 was an
# order of magnitude too small there -- so there is no reason 135M's transfers here either. The
# 135M sweep also never bracketed its optimum: it kept improving to the top of the grid at 8e-3.
set -eo pipefail
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"

MODEL=${MODEL:-smollm2-360m}
STEPS=${STEPS:-2500}
LR=${LR:-1e-3}
BATCH=${BATCH:-4}
ACCUM=${ACCUM:-4}
MIX=${MIX:-sl_fineweb=0.40,sl_code=0.25,sl_math=0.20,sl_rag=0.15}
TAG=${TAG:-lr${LR}}
CKPT_DIR=${CKPT_DIR:-runs/supernet/sl360_${TAG}}
mkdir -p "$CKPT_DIR" logs/supernet
LOG="logs/supernet/sl360_${TAG}.log"
echo "=== SmolLM2-360M A+B | $TAG | $((STEPS*BATCH*ACCUM*4096/1000000))M tokens ==="
python -m llmforge.supernet.train.uptrain \
  --model "$MODEL" --knobs ab --steps "$STEPS" --batch "$BATCH" --accum "$ACCUM" --chunked \
  --lr "$LR" --K 2 --warmup 200 --anneal-frac 0.1 --teacher base --min-weight 1.0 --mix "$MIX" \
  --eval-every 250 --eval-batch 2 --ckpt-every 2500 --ckpt-dir "$CKPT_DIR" --seed 0 "$@" 2>&1 | tee "$LOG"
