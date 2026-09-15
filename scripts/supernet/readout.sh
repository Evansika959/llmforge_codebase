#!/usr/bin/env bash
# The full readout for one supernet version: does a better-trained supernet rank differently?
#
# The conclusion that ranking is insensitive to supernet training was drawn on a checkpoint we now
# know was undertrained -- the learning-rate arm beats its final 5000-step probes at step 1750. So
# the question is open again, and it is cheap to reask: the 22 ground-truth architectures already
# exist, so a version only needs its slices rescored.
#
# Three numbers, in order of what they can resolve:
#   spread        sd of slice NLL across the 22 architectures. This is the score concentration Ning
#                 et al. (NeurIPS 2021) blame for rank correlation falling, measured directly and
#                 precisely -- if a recipe compresses the scores, it shows up here first.
#   within-band   concordant pairs against both ground-truth arms. 20 within-band pairs cannot
#                 resolve the ~0.035 tau effect the literature reports, so this is a coarse check.
#   head-tied     the same restricted to pairs with equal head count, where two free heuristics
#                 are controlled. This is where the supernet has to earn its keep.
#
#   scripts/supernet/readout.sh <name> <ckpt> [python]
set -eo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}" TOKENIZERS_PARALLELISM=false
NAME=$1; CKPT=$2; PY=${3:-python}
S=runs/supernet/slice_nll_${NAME}.json

[ -d "$CKPT" ] || { echo "!! $NAME: missing $CKPT"; exit 1; }
echo "=== readout: $NAME  ($CKPT) ==="
[ -f "$S" ] || $PY experiments/supernet_fidelity/slice_nll.py --model qwen3-1.7b --ckpt "$CKPT" \
    --shapes 8 --n 96 --out "$S" 2>&1 | grep -viE "^ *$|warn|it/s"

for arm in gt17_sn gt17_base; do
  echo "--- vs $arm ---"
  $PY experiments/supernet_fidelity/rank_fidelity.py --nll --slices "$S" --truth "runs/supernet/${arm}_nll.json" 2>&1 \
    | sed -n '/WITHIN COST/,$p' | grep -viE "^ *$"
done

$PY - "$S" <<'PY'
import json, sys, itertools, numpy as np
from llmforge.supernet.config import SPECS
from llmforge.supernet.eval.archsets import arch_set
A = arch_set(SPECS["qwen3-1.7b"], k=8)
S = json.load(open(sys.argv[1]))
T = json.load(open("runs/supernet/gt17_sn_nll.json"))
ks = sorted(k for k in A if k in T and k in S)
v = np.array([S[k]["nll"] for k in ks])
print(f"--- score spread (the concentration mechanism) ---")
print(f"  n={len(ks)}  mean {v.mean():.3f}  sd {v.std(ddof=1):.3f}  range {v.min():.3f}-{v.max():.3f}")
for L in "1234":
    w = np.array([S[k]["nll"] for k in ks if k[1] == L])
    print(f"  L{L}  sd {w.std(ddof=1):.4f}  range {w.min():.3f}-{w.max():.3f}")
# head-tied pairs, the only regime where the free heuristics are controlled
pr = [(a, b) for a, b in itertools.combinations(ks, 2)
      if a[1] == b[1] and A[a]['n_h'] == A[b]['n_h']]
c = sum(1 for a, b in pr if (S[a]["nll"] < S[b]["nll"]) == (T[a]["nll"] < T[b]["nll"]))
print(f"--- head-tied within-band: {c}/{len(pr)} concordant  tau-a {(2*c-len(pr))/len(pr):+.3f}")
PY
