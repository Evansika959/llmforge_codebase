# Layerwise predictor refit

Offline XGBoost baseline/refit for the aligned heterogeneous measurement protocol.
Does not change the C kernel, collection sources, live queue or AL checkpoints.

Modules: `data.py` freezes completed rows and checks matching device/kernel/workload
contracts and grouped splits; `features.py` defines architecture-only features and
inference; `run.py` trains and saves models; `report.py` scores held-out predictions.

```bash
python -m scripts.prediction_research.layerwise_refit.run \
  --output scripts/prediction/outputs/NEW_UNIQUE_REFIT_DIRECTORY
python -m unittest scripts.prediction_research.layerwise_refit.test_features
```

Inputs: completed accepted measurements from the fixed-KV 500, variable-KV 500,
and both new 500-point parts. Each SQLite read uses a read transaction; the combined
snapshot records what was available during the read, not a globally atomic time
across databases. The serialized JSON and its hash freeze the training input.
Legacy homogeneous/AL measurements are deliberately not mixed with this protocol.

Features include total parameters, Q8 bytes, FP32 KV capacity, global group size,
depth/width, layer-shape transitions, and per-layer shape/GQA/matrix/memory
statistics (sum/mean/std/min/max/adjacent change and four ordered quarter means).
Quarter summaries are order-sensitive but not a lossless sequence encoding.
No temperature, voltage, battery, timing measurements, campaign IDs or pattern
labels are used as model inputs.

Models: three independent log-target XGBoost regressors, depth 3, up to 1500 trees,
learning rate .03, min child weight 3, lambda 5, subsample .85, column sample .9,
hist trees, two CPU threads, patience 50. Seeds 42/123/2026. Both baseline and refit
use the **same original 100 validation rows** for stopping. New validation rows
remain held out, unused for stopping or model selection. Registry test assignments
are unchanged, with permutation groups kept together. No test-based model choice.
Missing/nonpositive targets are excluded per target, preserving valid timing data;
baseline-drift warnings are retained, not silently filtered.

The initial1000 baseline is fitted anew on original heterogeneous training rows;
it is not the old homogeneous predictor. Compare baseline/refit on identical test
cohorts. Snapshot from 2026-09-20 contains 1354 rows: 1083 train / 134 validation /
137 test. Two training energy labels are missing, leaving 1081 energy training rows.
Three-seed mean overall test MAPE: throughput 25.92 -> 25.53%, TTFT 22.92 -> 21.68%,
dynamic energy 13.52 -> 13.05%. Ranking is mixed; this is not proof of statistical
significance or active-learning superiority. New-cohort test has only 37 rows.

Artifacts: `dataset_snapshot.json`, `manifest.json`, `metrics.json`,
`predictions.npz`, and six `.joblib` files (baseline/refit, three seeds).
Each checkpoint stores three target models, feature names, counts, protocol and
snapshot hash. Checkpoint reload predictions are checked exactly during training.
No automatic deployment or checkpoint replacement occurs.

```python
import joblib
from scripts.prediction_research.layerwise_refit.features import predict
bundle = joblib.load('scripts/prediction/outputs/layerwise_refit_20260920/all_current_seed42.joblib')
# Each architecture is the ordered layerwise dictionary from architectures.jsonl.
values = predict(bundle, architectures)
# Columns: decode tok/s, TTFT ms, dynamic mJ/output token.
```

Dynamic energy integrates **prefill + decode**, subtracts the pre-idle baseline,
and divides by 32 output tokens. It is not decode-only energy. Use this checkpoint
only for the recorded hardware, workload, skeleton and operator search space.
