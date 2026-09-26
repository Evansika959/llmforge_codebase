# Layerwise XGBoost / Transformer comparison

Latest: [frozen 2,000-architecture release](../../../diliverable/layerwise_2000_20260921/README.md),
with 1600/200/200 grouped splits, all nine XGBoost/Transformer checkpoints and a
validation-selected final model. Runtime output:
`scripts/prediction/outputs/layerwise_compare_2000_20260921/`.
Its three-seed mean test MAPE is 26.06/21.95/11.57% (XGBoost),
26.46/22.63/12.13% (Transformer 1x32), and 26.52/22.73/11.78% (Transformer 2x64).
The final XGBoost seed-2026 model has single-checkpoint MAPE 25.91/22.29/11.61%.
See that release for complete metrics and limitations; all paired difference CIs
include zero. `release.py` separates read-only dataset backup from model packaging.

**Historical experiment documented below:** `scripts/prediction/outputs/layerwise_compare_1745_20260921/`.
The actual snapshot UTC timestamp is in `manifest.json`; the directory is an experiment ID.

At the status check, the additional 1,000-point sweep was paused at **745/1000**
(part1 500, part2 245). Its runner had exited with incomplete display restoration;
the watch was connected. This comparison does not restart hardware collection.
The two earlier 500-point campaigns are complete: **1,745 architectures total**.

## Reproduce

From the repository root, use the existing `nanollmforge` environment:

```bash
python -m scripts.prediction_research.layerwise_compare.run \
  --output scripts/prediction/outputs/NEW_UNIQUE_COMPARISON_DIRECTORY
python -m unittest scripts.prediction_research.layerwise_compare.test_training \
  scripts.prediction_research.layerwise_refit.test_features
```

`run` snapshots the data available at invocation. To replay this exact dataset,
add `--snapshot scripts/prediction/outputs/layerwise_compare_1745_20260921/dataset_snapshot.json`
and use the recorded seeds/config/dependency versions. The original source is
archived in `source_snapshot/`; live registries can grow. The output path must
not exist. Runtime outputs are Git-ignored; source code and this report are outside
the output folders. No existing model checkpoint is overwritten or deployed.

Modules: `models.py` defines the Transformer/token groups; `training.py` contains
fitting and inference; `report.py` contains evaluation/bootstrap; `run.py` freezes
data and orchestrates the comparison. Shared architecture features and registry
readers live in `../layerwise_refit/`. The hardware kernel and measurement sources
were not edited.

## Dataset and evaluation

- Four source registries: fixed-KV 500, variable-KV 500, v3 part1 500 and part2 245.
- Matching device, kernel, wrapper, workload and measurement semantics are checked.
- SQLite reads select completed, accepted measurements only. Database targets are
  checked against saved result artifacts. Snapshot records artifact/contract hashes.
- Preserve original registry splits: **1391 training / 170 validation / 184 test**.
  Full layer-multiset permutation groups stay within a split across all sources.
- There are **2 missing energy labels in training**: 1389 energy-training labels;
  all 170 validation and 184 test rows have all three targets. Neural loss masks
  missing energy while retaining these rows' timing labels. No energy imputation.
- All recorded baseline-drift warnings remain included. No residual-based cleaning.
- All models use the same **110 architecture-only features**, including global
  size/depth/group, memory demand, per-layer shape/GQA statistics and quarter means.
  Initial temperature, battery, voltage, measured timing and cohort IDs are excluded.
- Seeds **42, 123, 2026**; each seed is evaluated separately and scores averaged.
  Test labels are never passed to fitting/early stopping. All 170 validation rows
  are used here, unlike the earlier 1354-row refit which stopped on only the original
  100 validation rows. These runs are therefore not a controlled learning curve.
- Targets: decode throughput tok/s, TTFT ms for **49 actual prompt tokens**, and
  baseline-subtracted prefill+decode energy mJ / **32 output tokens**. There are
  31 decode forwards. Energy is not decode-only energy.

## Model recipes

XGBoost: three independent natural-log target regressors, max depth 3, up to 1500
trees, learning rate .03, min child weight 3, lambda 5, subsample .85, column sample
.9, hist trees, two CPU threads. Each target stops on validation log-RMSE after
50 non-improving rounds. Actual best iterations are stored in each checkpoint.

Transformers: the same 110 features grouped into **11 tokens**, plus CLS and learned
positional embeddings. One token contains global descriptors; ten tokens contain
the per-layer statistic families. Configuration A: **1 layer, width 32, 12,995
parameters**. Configuration B: **2 layers, width 64, 75,843 parameters**. Both use
4 heads, FFN width 2x embedding width, GELU, dropout .1, pre-norm blocks, and a
LayerNorm + linear three-target head. These are tabular feature tokens, not one
token per physical model layer; quarter statistics are not lossless sequences.

Neural preprocessing: log1p features and standardized natural-log targets, fitted
only on training rows. Masked Smooth L1 averages each target before averaging
targets. AdamW lr .001 / weight decay .01, cosine schedule up to 300 epochs, batch
128, clip norm 1, patience 45, checkpoint selected by mean validation MAPE.
Different objectives/stopping metrics are part of these fixed model recipes;
this is not an architecture-only causal ablation or exhaustive tuning study.

## Results on the identical 184 test architectures

Values are three-seed means. MAPE is a percentage; rank correlations are unitless.

| Model | Metric | MAPE | Spearman | Kendall tau-b | Pairwise accuracy | Recall@32 |
|---|---|---:|---:|---:|---:|---:|
| XGBoost | Throughput | 23.46% | 0.8327 | 0.6484 | 82.42% | 67.71% |
| XGBoost | TTFT | 20.41% | 0.8959 | 0.7224 | 86.12% | 76.04% |
| XGBoost | Dynamic energy/token | 11.74% | 0.9513 | 0.8129 | 90.65% | 81.25% |
| Transformer 1x32 | Throughput | 24.18% | 0.8212 | 0.6406 | 82.03% | 63.54% |
| Transformer 1x32 | TTFT | 20.41% | 0.8944 | 0.7175 | 85.88% | 73.96% |
| Transformer 1x32 | Dynamic energy/token | 12.31% | 0.9497 | 0.8073 | 90.36% | 77.08% |
| Transformer 2x64 | Throughput | 25.81% | 0.8139 | 0.6282 | 81.41% | 59.38% |
| Transformer 2x64 | TTFT | 21.28% | 0.8845 | 0.7033 | 85.16% | 69.79% |
| Transformer 2x64 | Dynamic energy/token | 12.47% | 0.9485 | 0.8066 | 90.33% | 79.17% |

XGBoost is the current engineering recommendation: lowest aggregate error, highest
Spearman on each metric, and faster fitting in this run (~0.61s for three targets
per seed, versus ~5.8s / ~11.9s for the small / larger Transformers; these times
are descriptive and not a controlled speed benchmark). TTFT error ties the small
Transformer when rounded. Validation chose **2x64** among the two Transformers;
1x32 performed better on this test set, but must not be retroactively presented
as a validation-selected winner. The best XGBoost seed by mean validation MAPE is 42.

Paired group bootstrap (3000 draws, averaging paired errors across training seeds):
small Transformer minus XGBoost MAPE deltas are +0.72, +0.00, +0.57 percentage
points, with 95% intervals [-0.61,2.02], [-1.35,1.23], [-0.33,1.41]. Thus its small
disadvantages are not resolved by this uncertainty analysis. For 2x64 throughput,
the difference is +2.35 points [0.77,3.86]. Bootstrap conditions on fitted models;
it does not account for uncertainty from collecting a different training dataset.

Dataset composition matters: XGBoost test MAPE on the original 100 holdouts is
33.09/25.73/12.55%, versus 11.99/14.08/10.78% on the 84 new holdouts. These are
different architectures/sessions, not evidence of a causal improvement by batch.
The combined scores should not hide this difference. Both older test results and
the current comparison are exploratory test reuse, not a fresh confirmatory test.

Full `metrics.json` also includes per-seed and family/cohort breakdowns, NDCG@32,
selection regret and Top-32 regret. Pairwise accuracy ignores exact truth ties and
credits prediction ties by half. Recall@32 is top-32 set overlap. NDCG uses linear
normalized throughput or inverse-cost gain, with log2 position discount. Regret
is measured against the actual best in the finite test set; lower is better.

## Artifacts and inference

The output contains the frozen dataset, manifest with dependencies/source hashes,
nine `.joblib` checkpoints, six neural histories, test predictions, full metrics,
checkpoint hashes, source copies, and a Markdown results table. All nine saved
checkpoints were reloaded and compared against their original validation outputs.

```python
import joblib
from scripts.prediction_research.layerwise_refit.features import matrix
from scripts.prediction_research.layerwise_compare.training import predict
pack = joblib.load('scripts/prediction/outputs/layerwise_compare_1745_20260921/XGBoost_seed42.joblib')
x, _ = matrix(architectures, pack['features'])
metrics = predict(pack, x)
# Columns: tok/s, TTFT ms, dynamic mJ/output token.
```

Each architecture is an ordered layerwise dictionary from `architectures.jsonl`.
These checkpoints use the recorded watch/workload protocol; they do not implement
the old homogeneous AL bundle interface. All prior models remain available.
