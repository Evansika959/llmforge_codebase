#!/usr/bin/env bash
# Benchmark each supernet recipe on downstream tasks, not just held-out NLL.
#
# NLL and capability come apart on this project: slices that looked merely degraded on NLL had
# already fallen to chance on MATH, so a recipe that improves every NLL probe has not yet been
# shown to preserve anything a deployment cares about. The learning-rate arm wins 8/8 probes; this
# asks whether that survives contact with a benchmark.
#
# Scope is four configs, not the full 23. HellaSwag on the complete 10,042-item set takes minutes
# per config and the within-band signal is ~1.4 points against a 1.3-point detection threshold, so
# truncating it would measure noise -- the saving has to come from running fewer configs instead.
#   full   drift at full width, the price of elasticity
#   L1_b   the best L1 architecture
#   L3_b   mid-range
#   L4_b   the tightest band, where the predictor is at chance and capability collapse lives
set -eo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}" TOKENIZERS_PARALLELISM=false
PY=${PY:-python}
ONLY=${ONLY:-full,L1_b,L3_b,L4_b}
TASKS=${TASKS:-hellaswag}
for spec in "$@"; do
  name=${spec%%=*}; ckpt=${spec#*=}
  [ -d "$ckpt" ] || { echo "!! $name: missing $ckpt"; continue; }
  echo "=== $name  ($ckpt) ==="
  $PY experiments/supernet_fidelity/score_slices.py --model qwen3-1.7b --ckpt "$ckpt" \
      --tasks "$TASKS" --only "$ONLY" --out "runs/supernet/bench_${name}.json" 2>&1 \
    | grep -viE "^ *$|Token indices|Passed an already"
done
