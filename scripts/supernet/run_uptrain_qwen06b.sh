#!/usr/bin/env bash
# Qwen3-0.6B A+B supernet. Fills 251M-596M, which no SmolLM2 base reaches.
#
# Coverage across the four bases, under the 40%-100% usable-range rule:
#   SmolLM2-135M   54M - 134M
#   SmolLM2-360M  145M - 362M
#   Qwen3-0.6B    251M - 596M   <- this run
#   Qwen3-1.7B    (trained)
# 0.6B overlaps 360M over 251-362M, which is useful rather than wasteful: an overlap is the only
# place two independently trained supernets can be checked against each other on the same
# architectures.
#
# The 40% floor costs nothing here. Qwen3's 151,936-token vocabulary is 155.6M of embeddings = 26%
# of the model and cannot be sliced, so the grid's own minimum is already 42.1% and all 32 uniform
# shapes are inside the usable range.
#
# Data is the QWEN-tokenizer packs (fineweb10bt/code/math/rag), NOT the sl_* buckets, which hold
# SmolLM2 token ids. Token ids are not portable across tokenizers.
#
# The learning rate is swept, never inherited. Every scale so far has had a different optimum:
# 1.7B wanted ~2e-4, 135M wanted >=2e-3, 360M's guard-passing optimum was 1e-3.
set -eo pipefail
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"

MODEL=${MODEL:-qwen3-0.6b}
STEPS=${STEPS:-2500}
LR=${LR:-5e-4}
BATCH=${BATCH:-2}
ACCUM=${ACCUM:-8}
MIX=${MIX:-fineweb10bt=0.40,code=0.25,math=0.20,rag=0.15}
TAG=${TAG:-lr${LR}}
CKPT_DIR=${CKPT_DIR:-runs/supernet/qw06_${TAG}}
mkdir -p "$CKPT_DIR" logs/supernet
LOG="logs/supernet/qw06_${TAG}.log"
echo "=== Qwen3-0.6B A+B | $TAG | $((STEPS*BATCH*ACCUM*4096/1000000))M tokens ==="
python -m llmforge.supernet.train.uptrain \
  --model "$MODEL" --knobs ab --steps "$STEPS" --batch "$BATCH" --accum "$ACCUM" --chunked \
  --lr "$LR" --K 2 --warmup 200 --anneal-frac 0.1 --teacher base --min-weight 1.0 --mix "$MIX" \
  --eval-every 250 --eval-batch 2 --ckpt-every 2500 --ckpt-dir "$CKPT_DIR" --seed 0 "$@" 2>&1 | tee "$LOG"
