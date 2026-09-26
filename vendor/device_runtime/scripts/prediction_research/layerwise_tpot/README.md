# Layerwise TPOT refit

This isolated offline experiment predicts `tpot_ms`, `ttft_ms`, and
`dynamic_energy_per_token_mj`, in that order. All targets are costs (lower is better).
It does not change historical throughput checkpoints, hardware collection, or active-learning state.

- `data.py`: verify frozen evidence hashes, derive TPOT from raw decode timing,
  cross-check reciprocal throughput, and reject split leakage/invalid labels.
- `run.py`: preserve the 110 architecture-only features and existing grouped splits;
  fit three independent log-space XGBoost regressors for each of three seeds;
  choose the final seed using validation only and export checkpoints/provenance.
- `report.py`: cost-aware error/ranking metrics, per-group-size reports, and documentation.
- `test_tpot.py`: denominator, invalid-label, non-mutation, and ranking-direction tests.

Shared model/training implementations remain in `layerwise_compare/training.py` and
`prediction/models/trees.py`; shared feature construction is in `layerwise_refit/features.py`.
Do not use the historical comparison report's default first-target direction for TPOT.

```bash
python -m unittest scripts.prediction_research.layerwise_tpot.test_tpot -v
python -m scripts.prediction_research.layerwise_tpot.run \
  --snapshot diliverable/layerwise_2000_gs64_refresh_20260922/dataset_snapshot.json \
  --output scripts/prediction/outputs/layerwise_tpot_reproduction
```

Use the `nanollmforge` environment. The output directory must not already exist.
The input data-only release and old model releases remain immutable.

The [2026-09-22 model release](../../../diliverable/layerwise_tpot_2000_20260922/README.md)
contains the results, validation-selected model, reproducible labels, and source hashes.
TPOT excludes prefill and divides decode duration by 31 forwards; dynamic energy still
includes prefill and decode and divides by 32 output tokens. Changing to TPOT does not
remove measurement noise, and TPOT MAPE must not be presented as the old throughput MAPE.
