# Pixel Watch 5 predictor with dynamic energy

This directory holds verified copies of existing data and checkpoints for the on-device target. Producing it involved no retraining, no rewrite of the original experiment and no hardware run. The method is described in `docs/hw_device.md`.

Snapshot date: **2026-09-14**. The rounds and metrics below describe this snapshot and do not follow later experiments. Module documentation is in the [prediction README](../../../src/llmforge/hw/device/prediction/README.md), and the operating steps are in the [active-learning guide](../../../src/llmforge/hw/device/prediction/active_learning/README.md).

## Contents

| File | Purpose |
|---|---|
| `data/dataset_latest.json` | Latest complete valid data: dynamic round 9 with 1,552 train, 150 validation and 314 test observations, 2,016 in total |
| `models/predictor_best.joblib` | Best checkpoint on validation within the current dynamic measurement stage: the round 1 XGBoost model trained on 1,472 observations |
| `data/dataset_best_model.json` | Dataset version that matches the best checkpoint's fingerprint exactly: 1,472 train, 150 validation and 314 test observations |
| `models/predictor_latest.joblib` | Latest checkpoint from round 9, matching `dataset_latest.json` exactly, for further training |
| `evaluation/model_selection.json` | Revalidation results and the selection rule for all 10 checkpoints: the initial model and rounds 1 to 9 |
| `manifest.json` | File SHA-256 digests, original locations, dataset fingerprints, sample counts, protocol and Python package versions |
| `provenance/` | Original run settings, state, code fingerprints, the energy-label transition record and the measurement CSV of every completed round in this stage |

"Best" refers only to the **current real-watch dynamic stage**. It is chosen by the arithmetic mean of the three targets' MAPE on the same fixed 150-row validation set, and it is not a global winner across all historical experiments. The test set was not used for model selection and was not evaluated again. Choosing a checkpoint by repeated use of the validation set does not amount to independent final test performance.

Fixed-validation results of the best checkpoint:

| Target | MAPE | MAE | R² |
|---|---:|---:|---:|
| Decode throughput | 22.024% | 5.490 tokens/s | 0.5857 |
| TTFT | 22.384% | 197.712 ms | 0.7337 |
| Dynamic energy | 43.558% | 5.940 mJ/output token | 0.8423 |

The mean MAPE over the three targets is **29.322%**. The latest checkpoint is not the best one, since its round 9 energy MAPE is 44.100%. The best checkpoint is not presented as a model trained on all of the latest data.

## Data and protocol

- Inputs are architecture fields only. Temperature, battery level and baseline power are not predictor inputs.
- Energy is baseline-subtracted: `max(active_power_w - baseline_power_w, 0) * duration_s * 1000 / 32`. It includes the prefill window, so it is not pure decode energy.
- Workload: a nominal 48-token prompt that tokenizes to 49 tokens, 32 output tokens and 31 decode forward passes.
- The historical protocol object keeps its 40 °C setting. Later measurements used a strict admission rule of below 45 °C before inference, recorded in `provenance/settings.json` and in each observation's measurement context. This is not a guarantee on the peak temperature during inference.
- `AL_56edd3ffbca9b338` has zero energy after baseline subtraction and cannot be used for log-space training. Its original record stays in the quarantine list of `provenance/energy_transition.json`. It was neither deleted nor clamped to a fake positive value, and it is not counted among the 2,016 valid observations.
- The interrupted round 10 contributed no complete new measurement to this dataset. Full raw traces, the candidate pool and the interrupted state remain in the original acquisition workspace. This directory is not a backup of that workspace.

## Using the model

Run with the repository environment active. The bundles need the `llmforge.hw.device.prediction` package and are not standalone applications. Exact package versions are listed in `manifest.json`.

```python
from llmforge.hw.device.prediction.data.dataset import MeasurementDataset
from llmforge.hw.device.prediction.models.serialization import load_bundle
from llmforge.hw.device.prediction.inference.predictor import predict_bundle

dataset = MeasurementDataset.load('assets/device/pixel_watch5/data/dataset_latest.json')
model = load_bundle('assets/device/pixel_watch5/models/predictor_best.joblib')
configs = [dataset.observations[0]['architecture']]  # replace with the architectures to predict
predictions = predict_bundle(model, configs)
print(model['targets'])  # decode_tok_s, ttft_ms, dynamic_energy_per_token_mj
print(predictions)
```

The CLI reads a CSV with all architecture columns and writes to an output file that must not exist yet:

```bash
python -m llmforge.hw.device.prediction predict \
  --model assets/device/pixel_watch5/models/predictor_best.joblib \
  --configs YOUR_ARCHITECTURES.csv \
  --output runs/device/delivery_predictions.csv
```

Load only trusted joblib files, because pickle and joblib deserialization can execute code.

## Continuing experiments and exporting again

To continue the original experiment after charging, use the original acquisition workspace, for example `runs/device/active_learning_watch5_dynamic_under45`. Do not use this directory as a resumable active-learning workspace. A new workspace needs a matching checkpoint and dataset: the best model with `dataset_best_model.json`, or the latest model with `dataset_latest.json`.

To export again, choose a new destination. The exporter refuses to overwrite an existing delivery, never accesses the watch and never trains a model:

```bash
python -m llmforge.hw.device.package_deliverable \
  --source runs/device/active_learning_watch5_dynamic_under45 \
  --destination runs/device/deliverable_next
```

## Release copy

This copy differs from the original export in the ways listed here. Predictions of both checkpoints are unchanged, and `tests/device/test_assets_integrity.py` checks each point.

- Absolute locations on the acquisition machine were removed from provenance paths. Paths are relative to the directory that held the acquisition workspaces.
- The device identity record is withheld.
- Dataset fingerprints and their parent pointers were recomputed along the full round chain, together with the fingerprints in `evaluation/model_selection.json`, `provenance/state.json` and `manifest.json` and the `dataset_sha256` field of both checkpoints. File digests in `manifest.json` and `provenance/contract.json` were updated to match.
- Digests of files that are not in this directory refer to the original files.
