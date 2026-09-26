# Compact surrogate research

This isolated package preserves frozen production/AL source contracts. It reuses
the historical 1,564-architecture JSON snapshot and exact 1100/150/314 split.
It does not access ADB, append measurements or deploy models.

Definitions are in `models.py`, optimization/inference in `training.py`, experiment
orchestration in `run.py`, and evaluation/figures in `report.py`.

```bash
python -m unittest scripts.prediction_research.compact_surrogates.test_compact -v
MPLCONFIGDIR=/private/tmp/nanollmforge-matplotlib python -u -m scripts.prediction_research.compact_surrogates.run \
  --output scripts/prediction/outputs/compact_surrogates_1564
```

Outputs are exclusive and Git-ignored. Four capacity candidates per neural family
use three seeds each. Models/scalers/profile and configuration are saved together;
`training.predict(pack, pack['profile'].transform(configs))` returns positive
throughput, TTFT in ms and dynamic energy in mJ/output-token. Only trusted local
joblib files should be loaded. Predictions are architecture-only. Config selection
is sealed before test evaluation; the reused test set still makes this exploratory.
Training is Smooth L1 on standardized logs with AdamW, cosine learning-rate decay,
and validation-MAPE early stopping. Thus comparisons with the historic large
Transformer are recipe comparisons, not a capacity-only controlled ablation.
