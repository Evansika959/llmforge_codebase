#!/usr/bin/env bash
# Stage A for Qwen3-4B-Base: elastic per-layer d_qk / d_v on a published checkpoint.
#
# 4B has no prior supernet to warm-start from, so this trains stage A cold -- the required
# first step under any reading of the plan. Stages B/C/D (n_kv, n_h, d_mlp, gates) warm-start
# from whatever this produces.
#
# Memory/throughput, measured on this H100 80GB (3-step smoke, --chunked throughout):
#   batch 1 accum 16 --grad-ckpt   42.0 s/step   43.2 GiB
#   batch 2 accum  8 --grad-ckpt   38.1 s/step   48.1 GiB   <- chosen
#   batch 4 accum  4 --grad-ckpt   38.1 s/step   57.9 GiB
#   batch 8 accum  2 --grad-ckpt   39.3 s/step   77.6 GiB
#   batch 2 accum  8 (no grad-ckpt)          OOM
# Throughput plateaus at ~38 s/step, so batch 2 buys the same speed as batch 4 with 10 GiB
# more headroom for the full-logits eval path. --grad-ckpt is mandatory, not optional.
# accum=8 keeps 65,536 tokens/step, identical to the 0.6B and 1.7B runs, so step counts
# stay comparable across scales.
#
# Budget: 8000 steps = 524M tokens at ~38.1 s/step = ~85 h (3.5 days). The 1.7B run reached 99% of its MIN-config gain by
# 4,250 steps (279M tokens) and 100% by ~10,250, with the final ~7,000 steps yielding
# 0.001 nats -- so this is deliberately not a 16k-step run. Watch [eval] and extend if the
# curve is still moving.
set -eo pipefail

cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"

MODEL=${MODEL:-qwen3-4b}
STEPS=${STEPS:-8000}
MIX=${MIX:-fineweb10bt=1.0}
CKPT_DIR=${CKPT_DIR:-runs/supernet/${MODEL}_stageA}
SEED=${SEED:-0}

mkdir -p "$CKPT_DIR" logs/supernet
LOG="logs/supernet/uptrain_${MODEL}_stageA.log"

echo "=== supernet stage A: $MODEL ==="
echo "steps=$STEPS mix=$MIX ckpt=$CKPT_DIR seed=$SEED"
echo "log -> $LOG"

python -m llmforge.supernet.train.uptrain \
  --model "$MODEL" \
  --steps "$STEPS" \
  --batch 2 --accum 8 \
  --chunked --grad-ckpt \
  --lr 3e-5 --K 2 --warmup 200 --anneal-frac 0.1 \
  --teacher base \
  --mix "$MIX" \
  --eval-every 250 --eval-batch 1 \
  --ckpt-every 1000 \
  --ckpt-dir "$CKPT_DIR" \
  --seed "$SEED" 2>&1 | tee "$LOG"
