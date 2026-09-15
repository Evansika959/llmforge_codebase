#!/usr/bin/env bash
# Blocked supernet: train over the SAME space the search explores.
#
# The defect this fixes. The trainer sampled every layer independently (144 dims, 1.6e82
# configurations) while NSGA-II searched 4-layer groups (36 dims, 3.5e20). Capacity was therefore
# spent on a space 1e61 times larger than anything the search could reach, and no sampled
# configuration was ever visited twice -- 16,000 draws into 3.5e20, so the supernet was asked to
# learn a 144-dimensional function from single samples.
#
# The fix uses MEASURED blocks, not a fixed stride. experiments/supernet_fidelity/layer_sensitivity_top1.py shrinks
# one layer at a time and records top-1 disagreement with the full model -- temperature-invariant,
# and the quantity that compounds over a generated sequence, unlike cross-entropy. Attention and
# MLP get separate partitions because their depth profiles genuinely differ:
#   attn [0] [1-11] [12-21] [22-24] [25-35]   peaks at layer 0 and the 22-24 band
#   mlp  [0] [1-3]  [4-33]  [34-35]           peaks at the last two layers
# Genome: 19 genes, ~1e11 configurations -- still a large design space, 1e71 fewer dimensions.
#
# ONE variable changes versus runs/supernet/qwen3-4b_ab. Same base checkpoint, knobs, batch, LR, teacher,
# data and step count, so a difference in the result is attributable to granularity alone. The
# question this run answers: is the capability collapse caused by capacity dilution?
#
# Decision rule, fixed in advance: w75 currently scores 10.0% on Minerva MATH with 69.4% top-1
# agreement. Above ~25% means dilution was the main cause and this line continues. Still near 10%
# means dilution is not the cause, and the next levers are on-policy distillation (which targets
# the compounding mechanism directly) and dedicated per-config recovery.
set -eo pipefail
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"

MODEL=${MODEL:-qwen3-4b}
STEPS=${STEPS:-2500}
MIX=${MIX:-fineweb10bt=1.0}
CKPT_DIR=${CKPT_DIR:-runs/supernet/${MODEL}_ab_blocks}
mkdir -p "$CKPT_DIR" logs/supernet
LOG="logs/supernet/uptrain_${MODEL}_ab_blocks.log"
echo "=== blocked A+B supernet: $MODEL, $STEPS steps ==="
python -m llmforge.supernet.train.uptrain \
  --model "$MODEL" --knobs ab --blocks \
  --steps "$STEPS" --batch 2 --accum 8 --chunked --grad-ckpt \
  --lr 3e-5 --K 2 --warmup 200 --anneal-frac 0.1 --teacher base \
  --min-weight 1.0 --mix "$MIX" \
  --eval-every 250 --eval-batch 1 --ckpt-every 500 \
  --ckpt-dir "$CKPT_DIR" --seed 0 2>&1 | tee "$LOG"
