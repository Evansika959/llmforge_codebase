# Minimal-feature XGBoost ablation

```bash
python -m scripts.prediction_research.minimal_features \
  --workspace scripts/prediction/outputs/active_learning_watch5_dynamic_under45 \
  --output scripts/prediction/outputs/minimal_features_al50_with_controls
```

Choose a fresh output directory when rerunning. This reads the latest committed
dataset and checkpoint; the current completed experiment used round 50 with 1,962
training rows and 150 fixed validation rows. Incomplete round 51 is excluded.
The original dataset, checkpoint, contracts and device remain unchanged.

Five feature sets (params; params+KV; params+group; params+KV+group; physics32) use
identical depth-3 XGBoost log-target settings, validation early stopping, and seeds
42/123/2026. There is no per-variant hyperparameter tuning or test-set scoring.
KV means FP32 bytes per context token across all layers, not allocator capacity.

At round 50, three-feature throughput/TTFT/dynamic-energy validation MAPEs are
21.15/21.12/40.54%, versus full-feature 22.08/22.33/44.10%. Params+group alone gives
21.66/22.65/44.35%. This is exploratory evidence on a repeatedly reused validation
set, also used for stopping; it is not proof of prospective improvement or a causal
quantization-group intervention. Simplification may reduce overfitting, but that
explanation has not been established. No production model was replaced.

Outputs include the frozen dataset snapshot, manifest, fifteen diagnostic
checkpoints, validation predictions and report. All checkpoints are round-trip
verified, and full-feature seed 42 must reproduce the committed checkpoint.
For diagnostic inference, load a trusted checkpoint, transform configurations with
its `profile`, slice `feature_indices`, and call `inference.trees.tree_predict` on
its `estimator`. These checkpoints are not `predict_bundle` production bundles.
