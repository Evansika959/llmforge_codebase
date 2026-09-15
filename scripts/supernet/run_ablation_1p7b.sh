#!/usr/bin/env bash
# One arm of the sampling-recipe ablation. All arms run 3500 steps, because the baseline's own
# trajectory is flat to the third decimal on every probe from 3500 to 5000 -- six GPU-hours that
# moved nothing. The LR schedule is sized to the budget, so shortening it keeps the anneal shape.
#
# What this exists to test. The baseline's training splits into three phases: everything improves
# for 500 steps, then from 500 to 3500 it REDISTRIBUTES -- het1 -3.89, het2 -1.93, min -0.80, while
# full +0.03, qk-lo +0.04 and mlp-lo +0.09 get WORSE. That is the sandwich objective lifting the
# weak corner at the expense of the strong one, and it is the mechanism Ning et al. (NeurIPS 2021)
# blame for rank correlation degrading as subnets-per-step rises. The arms below vary the two knobs
# that control that pressure, plus the learning rate, since converging by step 500 also admits the
# reading that the step size is too small to leave the basin it started in.
#
#   scripts/supernet/run_ablation_1p7b.sh <arm> <python>
set -eo pipefail
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false PYTHONPATH="$(pwd)/src"

ARM=$1; PY=${2:-python}
K=2; LR=3e-5; MINW=1.0; MIX=fineweb10bt=1.0; CLIP=1.0
case "$ARM" in
  K1)      K=1 ;;                 # fewer random draws per step -- Ning et al.'s S=1 end
  K4)      K=4 ;;                 # more draws -- the other end of the same dose-response
  min0.3)  MINW=0.3 ;;            # less weight on the narrowest corner
  lr6e-5)  LR=6e-5 ;;             # is the fast plateau a step-size artefact?
  lr1e-4)  LR=1e-4 ;;             # --clip 1.0 against gn~200 means the update is lr x unit
  lr2e-4)  LR=2e-4 ;;             # direction, so the step size IS the learning rate. Bracket it.
  lr4e-4)  LR=4e-4 ;;             # lr2e-4 is flat from 3000 to 3500, so its own optimum is found;
                                  # this asks whether a larger one exists before divergence.
  # The optimiser is AdamW, whose update lr*m/(sqrt(v)+eps) is scale-invariant in the gradient, so
  # clipping cannot act as a step size -- lr alone sets that. The first attempt at these arms
  # lowered lr to "compensate" for a larger clip and merely re-ran the learning-rate sweep 10x and
  # 100x too small. Learning rate is now held at the 2e-4 optimum and clip is the only variable, so
  # what is measured is the DIRECTION change on steps whose gradient norm exceeds the threshold --
  # with a median norm near 150, clip 1.0 redirects essentially every step and clip 100 almost none.
  clip10)  LR=2e-4; CLIP=10 ;;
  clip100) LR=2e-4; CLIP=100 ;;
  *) echo "unknown arm: $ARM" >&2; exit 1 ;;
esac
OUT=runs/supernet/abl_${ARM}
mkdir -p "$OUT" logs/supernet
echo "=== arm $ARM: K=$K lr=$LR clip=$CLIP min-weight=$MINW mix=$MIX steps=3500 -> $OUT ==="
$PY -m llmforge.supernet.train.uptrain \
  --model qwen3-1.7b --knobs ab --steps 3500 --batch 2 --accum 8 --chunked \
  --lr "$LR" --clip "$CLIP" --K "$K" --warmup 200 --anneal-frac 0.1 --teacher base \
  --min-weight "$MINW" --mix "$MIX" \
  --eval-every 250 --eval-batch 1 --ckpt-every 500 \
  --ckpt-dir "$OUT" --seed 0 2>&1 | tee "logs/supernet/abl_${ARM}.log"
