#!/usr/bin/env bash
# gt_driver.sh <python> <model> <steps> <init> <outdir> <arch>...
PY=$1; MODEL=$2; STEPS=$3; INIT=$4; OUT=$5; shift 5
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH="$(pwd)/src" TOKENIZERS_PARALLELISM=false
for A in "$@"; do
  if [ -f $OUT/$A/arch.json ]; then echo "### $A already done, skip"; continue; fi
  echo "### [$(date +%H:%M)] START $A"
  $PY experiments/supernet_fidelity/groundtruth.py --model $MODEL --arch $A --steps $STEPS \
      --init "$INIT" --out $OUT --batch 1 --accum 16 2>&1 | grep -viE "^ *$|warning" | tail -8
  echo "### [$(date +%H:%M)] DONE $A"
done
echo "### ALL DONE"
