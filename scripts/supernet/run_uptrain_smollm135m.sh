#!/usr/bin/env bash
# SmolLM2-135M A+B supernet, from the published base. This is the 50-150M-band predictor.
#
# Why this scale exists: Qwen3 cannot reach the band. Its 151,936-token vocabulary is 156M of
# embeddings at hidden 1024 -- above the top of the band -- and embeddings are not sliceable.
# SmolLM2-135M's 49,152 vocabulary is 28M, so the elastic span measures 51.5M-134.5M end to end.
#
# What is carried over from the 1.7B ablation, so it is not re-litigated here:
#   lr      the dominant factor by far. 3e-5 was ~10x too small at 1.7B; 2e-4 took 2.5 nats off
#           the two hardest probes in 70% of the steps. lr for THIS scale is swept, not assumed --
#           a 135M model pretrained at 3e-3 does not inherit a 1.7B model's optimum.
#   K       sandwich extras. K=1 won 1 probe of 8, K=4 fewer. Left at 2.
#   min-weight  0.3 won zero probes and moved `full` by +0.001. Left at 1.0.
#   clip    irrelevant once lr is right: 0.001-0.043 spread across clip 1.0/10/100. Left at 1.0.
#           Under AdamW the update is scale-invariant in the gradient, so clip is not a step size.
#
# Data: the 40/25/20/15 mix agreed for supernet uptraining, repacked with SmolLM2's tokenizer
# (scripts/supernet/pack_smollm.sh). One honest substitution: the 15% slot is meant to be long documents
# (>=8K) because half the search space is KV cache, which only matters at long context. No long-doc
# bucket is packed yet, so `rag` stands in. That is a known gap, not a silent choice.
set -eo pipefail

cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
# Exported so the run works from a fresh clone without installing the package.
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"

MODEL=${MODEL:-smollm2-135m}
STEPS=${STEPS:-4500}
LR=${LR:-5e-4}
BATCH=${BATCH:-8}
ACCUM=${ACCUM:-2}
MIX=${MIX:-sl_fineweb=0.40,sl_code=0.25,sl_math=0.20,sl_rag=0.15}
TAG=${TAG:-lr${LR}}
CKPT_DIR=${CKPT_DIR:-runs/supernet/sl135_${TAG}}
SEED=${SEED:-0}

mkdir -p "$CKPT_DIR" logs/supernet
LOG="logs/supernet/sl135_${TAG}.log"
TOK_PER_STEP=$((BATCH * ACCUM * 4096))

echo "=== SmolLM2-135M A+B supernet | $TAG ==="
echo "steps=$STEPS x ${TOK_PER_STEP} tok = ~$((STEPS * TOK_PER_STEP / 1000000))M tokens"
echo "lr=$LR mix=$MIX ckpt=$CKPT_DIR"
echo "log -> $LOG"

python -m llmforge.supernet.train.uptrain \
  --model "$MODEL" \
  --knobs ab \
  --steps "$STEPS" \
  --batch "$BATCH" --accum "$ACCUM" \
  --chunked \
  --lr "$LR" --K 2 --warmup 200 --anneal-frac 0.1 \
  --teacher base \
  --min-weight 1.0 \
  --mix "$MIX" \
  --eval-every 250 --eval-batch 2 \
  --ckpt-every 500 \
  --ckpt-dir "$CKPT_DIR" \
  --seed "$SEED" "$@" 2>&1 | tee "$LOG"
