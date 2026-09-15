#!/usr/bin/env bash
# Combined A+B supernet run: d_qk, d_v, n_h and d_mlp elastic in a single training.
#
# All four are the same class of operation -- each selects a subset of the live weights and the
# survivors keep their function -- so they train together rather than in stages. n_kv is NOT here:
# merging asks surviving weights to compute something new, needs its own mechanism (rotation
# alignment), and carries an unresolved design choice, so it gets a separate stage and its own
# ablation.
#
# Measured on this H100 80GB (3-step smoke, qwen3-4b, --chunked --grad-ckpt):
#   batch 2 accum 8   33-36 s/step   48.1 GiB peak
# Slightly faster than stage A because sampled configs are narrower. --grad-ckpt is mandatory.
#
# Budget: 8000 steps = 524M tokens at ~35 s/step = ~78 h (3.3 days). The 1.7B two-knob run reached
# 99% of its MIN-config gain by 279M tokens; four knobs plausibly need more, so watch [eval] and
# extend rather than assuming this is enough.
#
# MIN is the TRUE corner of all four knobs (not a raised floor), because the sandwich rule only
# bounds the interior if MIN is the actual minimum. It is correspondingly hard -- the smoke test
# shows MIN loss ~32 against FULL ~2.3. If that destabilises training, lower --min-weight rather
# than moving the floor, so the change stays visible.
set -eo pipefail

cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"

MODEL=${MODEL:-qwen3-4b}
STEPS=${STEPS:-8000}
MIX=${MIX:-fineweb10bt=1.0}
MINW=${MINW:-1.0}
CKPT_DIR=${CKPT_DIR:-runs/supernet/${MODEL}_ab}
SEED=${SEED:-0}

mkdir -p "$CKPT_DIR" logs/supernet
LOG="logs/supernet/uptrain_${MODEL}_ab.log"

echo "=== supernet A+B: $MODEL ==="
echo "steps=$STEPS mix=$MIX min-weight=$MINW ckpt=$CKPT_DIR seed=$SEED"
echo "log -> $LOG"

python -m llmforge.supernet.train.uptrain \
  --model "$MODEL" \
  --knobs ab \
  --steps "$STEPS" \
  --batch 2 --accum 8 \
  --chunked --grad-ckpt \
  --lr 3e-5 --K 2 --warmup 200 --anneal-frac 0.1 \
  --teacher base \
  --min-weight "$MINW" \
  --mix "$MIX" \
  --eval-every 250 --eval-batch 1 \
  --ckpt-every 1000 \
  --ckpt-dir "$CKPT_DIR" \
  --seed "$SEED" 2>&1 | tee "$LOG"
