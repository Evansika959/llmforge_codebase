#!/usr/bin/env bash
# Qwen3-1.7B A+B supernet, trained from the PUBLISHED base checkpoint -- no warm start from any
# earlier supernet. This is the clean run for the report; the 4B run was the methodology test.
#
# Separate from run_uptrain_ab.sh because three settings genuinely differ at this scale, and
# burying them in environment overrides makes the run hard to reproduce from the repo alone.
#
# 1. NO --grad-ckpt. Mandatory at 4B (48.1 GiB peak with it), unnecessary here. Measured on this
#    H100 80GB, batch 2 accum 8, 12-step smoke including an eval:
#      with    --grad-ckpt   18.7 s/step   26.0 GiB peak
#      without --grad-ckpt   14.5 s/step   49.4 GiB peak   <- 22% faster, 30 GiB headroom
#    Peak reaches 49.4 GiB by step 3 and does not move through step 12 or across evals. The
#    sandwich always includes the FULL config, which is the memory high-water mark, so this is
#    the worst case rather than a lucky sample. batch 4 accum 4 without --grad-ckpt OOMs.
#
# 2. batch 2 accum 8, NOT a wider split. Effective batch stays 16 sequences x 4096 tokens, which
#    matches the 4B run so the two are comparable. Wider splits were measured and buy nothing:
#    batch 4 accum 4 and batch 8 accum 2 both ran 17.9 s/step with --grad-ckpt (vs 18.7 at 2/8),
#    i.e. this is compute-bound, and batch 8 spends 18 GiB more for no gain.
#
# 3. STALE -- SEE docs/supernet.md BEFORE USING THIS RECIPE. The step-count argument below is sound only
#    at --lr 3e-5, and that learning rate is roughly an order of magnitude too small: --clip 1.0
#    against gradient norms near 200 makes every update lr times a unit direction, so the step size
#    IS the learning rate. At lr 2e-4 the two hardest probes fall 2.5 nats further in 3500 steps
#    than this recipe reaches in 5000. The flat tail below is a model that cannot move, not one
#    that has converged.
#
#    5000 steps, not 8000. The 4B A+B run's second half was nearly wasted: steps 4000-8000 took
#    34 h and moved the hardest probe 0.017 nats, with mlp-lo ending marginally worse. 5000 steps
#    = 328M tokens clears both known convergence marks -- the 4B four-knob run was within 0.017
#    nats of final by 262M, and the 1.7B two-knob run reached 99% of its MIN gain by 279M -- with
#    margin for four knobs being harder than two.
#    The schedule is SIZED to 5000, not truncated at it. The 4B's anneal did real work (full
#    recovered 2.475 -> 2.465 as the LR decayed), so the anneal must fit inside the budget.
#
# Space: 28 genes (4 knobs x 7 groups of 4 layers). n_h grid is only {8, 16} here since n_q=16,
# against {8, 16, 32} at 4B -- the head knob is coarser at this scale, by geometry not by choice.
#
# CAVEAT worth recording before the run, not after. The NSGA-II search over the finished 4B
# supernet found that per-layer heterogeneity does NOT beat a uniform grid, because sandwich
# training compresses per-layer sensitivity 6x (CV 2.580 -> 0.429). This recipe samples per-layer
# widths uniformly at random, so it should flatten the 1.7B the same way and its search will
# likely also come back near-uniform. That is worth having as a second data point at another
# scale, but if the goal is a supernet where per-layer search finds something, the sampling
# distribution is the thing to change -- not the search.
set -eo pipefail

cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"

MODEL=${MODEL:-qwen3-1.7b}
STEPS=${STEPS:-5000}
MIX=${MIX:-fineweb10bt=1.0}
MINW=${MINW:-1.0}
CKPT_DIR=${CKPT_DIR:-runs/supernet/${MODEL}_ab}
SEED=${SEED:-0}

mkdir -p "$CKPT_DIR" logs/supernet
LOG="logs/supernet/uptrain_${MODEL}_ab.log"

echo "=== supernet A+B from published base: $MODEL ==="
echo "steps=$STEPS (~$((STEPS * 65536 / 1000000))M tokens) mix=$MIX min-weight=$MINW"
echo "ckpt=$CKPT_DIR seed=$SEED  |  ~14.5 s/step => ~$((STEPS * 145 / 36000)) h"
echo "log -> $LOG"

python -m llmforge.supernet.train.uptrain \
  --model "$MODEL" \
  --knobs ab \
  --steps "$STEPS" \
  --batch 2 --accum 8 \
  --chunked \
  --lr 3e-5 --K 2 --warmup 200 --anneal-frac 0.1 \
  --teacher base \
  --min-weight "$MINW" \
  --mix "$MIX" \
  --eval-every 250 --eval-batch 1 \
  --ckpt-every 500 \
  --ckpt-dir "$CKPT_DIR" \
  --seed "$SEED" 2>&1 | tee "$LOG"
