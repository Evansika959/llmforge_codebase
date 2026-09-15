#!/usr/bin/env bash
# Step 1 of the post-DSE plan: does a math/code data mix stop the slices losing capability?
#
# The original 4B supernet trained on FineWeb-Edu alone and its slices collapsed on mathematics:
# 49.5% at full width, 10.0% at 75% size, 0% below that, against 56.0% for the published base.
# Uptraining cost 6.5 points; slicing cost 39.5 more. The leading explanation is that math was
# never exercised -- neither in the data nor in the teacher's targets -- so the shared weights had
# no reason to preserve it.
#
# This is a CONTROLLED rerun. Everything is held identical to the original: same base checkpoint
# (no warm start), same knobs, same sampler, same batch/accum, same LR and teacher. Two things
# change, and both are stated here so the comparison stays readable:
#
#   1. MIX. fineweb10bt 0.50 / math 0.30 / code 0.20 instead of fineweb10bt 1.00. All three are
#      already packed at seqlen 4096 with the same Qwen tokenizer (verified: max id 151643 against
#      a 151669 vocab, and the math shard decodes to LaTeX worked solutions, 118/120 sampled
#      sequences carrying markup -- competition-style, not elementary word problems).
#      At 2500 steps the math bucket supplies ~49M of its 300M tokens, so nothing repeats.
#
#   2. STEPS. 2500, not 8000. The original run's own trajectory says 94.8% of the total gain
#      landed by step 1000 and 98.2% by step 2000, so 8000 was mostly wasted. The schedule is
#      SIZED to 2500 -- warmup and anneal fit inside it -- rather than truncated at it.
#
# Read the step-count change carefully when comparing: it is a real difference from the original,
# justified by that run's convergence data, not a free variable. runs/supernet/qwen3-4b_ab/step2000 and
# step3000 exist if a step-matched control is wanted.
#
# ~30.5 s/step measured on the original run => 2500 steps ~= 21 h.
set -eo pipefail

cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"

MODEL=${MODEL:-qwen3-4b}
STEPS=${STEPS:-2500}
MIX=${MIX:-fineweb10bt=0.5,math=0.3,code=0.2}
MINW=${MINW:-1.0}
CKPT_DIR=${CKPT_DIR:-runs/supernet/${MODEL}_ab_mathmix}
SEED=${SEED:-0}

mkdir -p "$CKPT_DIR" logs/supernet
LOG="logs/supernet/uptrain_${MODEL}_ab_mathmix.log"

echo "=== supernet A+B, math/code mix: $MODEL ==="
echo "steps=$STEPS (~$((STEPS * 65536 / 1000000))M tokens) mix=$MIX"
echo "ckpt=$CKPT_DIR seed=$SEED  |  ~30.5 s/step => ~$((STEPS * 305 / 36000)) h"
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
  --ckpt-every 500 \
  --ckpt-dir "$CKPT_DIR" \
  --seed "$SEED" 2>&1 | tee "$LOG"
