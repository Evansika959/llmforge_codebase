# Hardware performance prediction

A reusable prediction package, separate from device measurement code in `src/llmforge/hw/device/measurement/sweep/`.

Active learning now defaults to baseline-subtracted **dynamic energy**. The [active-learning guide](active_learning/README.md) documents the audited `active switch-energy` transition, resumable `active run` command, and target-specific `active monitor` accuracy reports. Existing gross experiments are preserved rather than overwritten.

```text
src/llmforge/hw/device/prediction/
├── data/          Dataset versions, hardware CSV ingestion, frozen legacy snapshots
├── features/      Architecture-only analytic and hardware-aware features
├── models/        Neural definitions, XGBoost specification, checkpoint compatibility
├── training/      Shared optimizers, ensemble fitting, round-based training pipeline
├── inference/     Shared prediction API and CSV prediction commands
├── evaluation/    Metrics and historical-result audits
├── reporting/     Reports and plots; no model fitting
├── experiments/   Historical comparisons and fixed-cohort research experiments
├── active_learning/ Accuracy-oriented acquisition, resumable hardware rounds, replay
├── tests/         Checkpoint regression, ingestion, split and leakage tests
└── (outputs)     Run outputs are written to runs/device/ and ignored by Git
```

Core modules do not import experiment scripts. Define a model in `models/`, fit it in `training/`, and expose its prediction in `inference/`. Shared architecture transformations belong in `features/`. Historical ablations retain their experiment-specific settings rather than silently changing old results.

## Curated dataset and model delivery

The [shipped predictor assets](../../../../../assets/device/pixel_watch5/README.md) are the curated export. The **2026-09-14 snapshot** contains:

| Artifact under `assets/device/pixel_watch5/` | Meaning |
|---|---|
| `data/dataset_latest.json` | Dynamic round 9: 1,552 training + 150 validation + 314 test = 2,016 valid observations |
| `models/predictor_best.joblib` | Dynamic round 1: lowest mean of the three fixed-validation MAPEs among the stage's initial and completed checkpoints |
| `data/dataset_best_model.json` | Exact dataset matching that best checkpoint: 1,472 training + the same 464 holdout observations |
| `models/predictor_latest.joblib` | Dynamic round 9 checkpoint, matching `dataset_latest.json` |

Best energy validation MAPE is **43.558%**; three-target mean MAPE is **29.322%**. This is the best within the current dynamic stage, not across incompatible historical experiments, and not an independent test-set claim. The best checkpoint was **not** trained on the latest 1,552-row training set. For checkpoint-based initialization, always use its matching dataset; for prediction, either checkpoint accepts architecture configurations alone.

The export includes selection results, source/protocol metadata, package versions, SHA-256 checks and the quarantined zero-energy record. See its README for inference examples and scope. Original raw traces and interrupted experiment state remain in the local workspace; this export is not a resumable AL workspace or a full raw-data backup.

To create a later snapshot without accessing hardware or retraining, run from the repository root and choose a new destination:

```bash
python -m llmforge.hw.device.package_deliverable \
  --source runs/device/active_learning_watch5_dynamic_under45 \
  --destination runs/device/deliverable_next
```

The exporter verifies committed history, requires the source workspace lock, reevaluates fixed validation only, and refuses to overwrite an existing destination. It copies artifacts; it does not move or delete experiment data.

Run outputs under `runs/device/` are ignored by Git. A fresh checkout therefore does not supply historical acquisition workspaces or a resumable experiment workspace; use the curated delivery for prediction, and retain/back up the original workspace separately for resume or historical reproductions.

## Commands

Run with the repository environment active:

```bash
python -m llmforge.hw.device.prediction --help
python -m llmforge.hw.device.prediction train --help
python -m llmforge.hw.device.prediction predict --help
python -m llmforge.hw.device.prediction dataset --help
python -m llmforge.hw.device.prediction active --help
```

The former flat `python src/llmforge/hw/device/prediction/<name>.py` commands are replaced by `python -m llmforge.hw.device.prediction <name-with-hyphens>`. For example:

```bash
python -m llmforge.hw.device.prediction compare-surrogates \
  --output runs/device/surrogate_comparison_rerun

python -m llmforge.hw.device.prediction train-proposed-physics-surrogate \
  --output runs/device/proposed_physics_surrogate_rerun

python -m llmforge.hw.device.prediction evaluate-gross-energy \
  --output runs/device/gross_energy_comparison_1564_rerun
```

These commands reproduce the historical experiments. They are separate from the generic round-based `train` command below.

## Continual measurement workflow

The new round pipeline has no hard-coded sample count. It supports XGBoost and the grouped Transformer, with architecture-only physics32 inputs and log targets. Each round fits from scratch using all training observations in that dataset version. Stacked-ensemble research remains available through the historical experiment command.

### 1. Create the initial immutable dataset

Write a protocol JSON describing the **actual** acquisition conditions. Required fields are:

```json
{
  "protocol_id": "YOUR_VERIFIED_MEASUREMENT_PROTOCOL",
  "device_id": "YOUR_DEVICE_ID",
  "kernel_id": "YOUR_KERNEL_AND_BUILD_ID",
  "prompt_tokens": 49,
  "output_tokens": 32,
  "energy_target": "dynamic"
}
```

Replace the identity placeholders with verified identifiers. Do not reuse an identity across different kernel builds, devices or measurement methods. Add any other conditions that define comparability to the protocol object. Imports require an exact match of the whole object. A legacy CSV cannot prove its device/protocol identity; the operator must supply it correctly.

`energy_target` can be `dynamic` or `gross`. Both energy labels are prefill-inclusive per-output-token metrics. The current feature version deliberately supports only 49 actual prompt tokens and 32 outputs. Other workloads need a feature/protocol extension, not silent mixing.

To preserve the historical 1,564-architecture cohort and its 1,100/150/314 split:

```bash
python -m llmforge.hw.device.prediction dataset bootstrap \
  --previous runs/device/batch2_progress_632 \
  --protocol /path/to/protocol.json \
  --output runs/device/datasets/round_000.json
```

For another initial dataset, use the normalized schema documented below with explicit, architecture-disjoint split assignments.

### 2. Train and predict candidate architectures

```bash
python -m llmforge.hw.device.prediction train \
  --dataset runs/device/datasets/round_000.json \
  --family xgboost \
  --output runs/device/al_round_000

python -m llmforge.hw.device.prediction predict \
  --model runs/device/al_round_000/model.joblib \
  --configs /path/to/candidate_architectures.csv \
  --output runs/device/round_000_predictions.csv
```

Use `--family transformer` for the grouped Transformer. `--seed` selects one training seed; this command does not claim a multi-seed comparison. A run records its dataset snapshot/hash, protocol, model, split IDs, source hashes, library versions, held-out predictions and report.

Prediction column names follow the saved model's targets, including the distinction between gross and dynamic energy. Only trusted local joblib bundles should be loaded.

### 3. Measure selected candidates, then append a new round

Hardware measurements remain an explicit, separate step using the appropriate sweep protocol. Once a completed CSV batch has been reviewed:

```bash
python -m llmforge.hw.device.prediction dataset import-csv \
  --dataset runs/device/datasets/round_000.json \
  --configs /path/to/measured_architectures.csv \
  --measurements /path/to/completed_measurements.csv \
  --protocol /path/to/protocol.json \
  --round-id round_001 \
  --output runs/device/datasets/round_001.json

python -m llmforge.hw.device.prediction train \
  --dataset runs/device/datasets/round_001.json \
  --family xgboost \
  --output runs/device/al_round_001
```

The CSV adapter accepts the existing random-sweep field names and joins on `config_id`. It verifies the seven measured architecture dimensions against the configuration CSV. Gross energy is `total_energy_j * 1000 / output_tokens`; the stored dynamic-energy field is not recomputed. It refuses malformed or flagged rows rather than silently cleaning them. A different hardware output format needs an explicit adapter.

Measurement identity is `round_id` plus the supplied `measurement_id`/`run_id`, or otherwise `config_id:repeat`. Re-importing the same batch under the same round is rejected. Do not rename the round just to bypass duplicate detection. Repeated measurements require distinguishable repeat/run IDs. Source CSV hashes are retained.

### Split and provenance rules

- Dataset writes and run-directory creation are exclusive: existing versions are never overwritten.
- Appends preserve all earlier observations and record the parent dataset hash.
- New rows enter training only. An architecture already in validation/test cannot enter training, even under a new config ID.
- A config ID cannot refer to conflicting architectures. Duplicate measurement IDs are rejected.
- Nonpositive/nonfinite selected log targets and unsupported architecture shapes are rejected.
- All repeats of one architecture stay in one split. They are still separate, equally weighted observations; repeat aggregation and repeat-aware weighting are not implicitly chosen.
- Hardware calibration and neural scalers use training rows only. Validation controls early stopping. Measured power, temperatures, batch ID and acquisition order are never features.
- Repeatedly inspecting the same test across active-learning rounds remains exploratory. Keep a final prospective acquisition block untouched if you need an unbiased final assessment.

An accuracy-oriented active-learning controller now builds on this ingestion/retraining boundary. It selects architecture-only committee-disagreement, coverage and random-exploration batches, optionally measures them with the existing sweep protocol, validates measurements, and retrains XGBoost on immutable dataset versions. Reference architectures and their drift gate are disabled by default (`--anchors 0`); they remain opt-in for control experiments. Hardware execution is opt-in. See [the active-learning guide](active_learning/README.md) for preparation, launch, audited no-anchor migration, recovery, budgets and limitations. This is not Pareto/architecture optimization, calibrated uncertainty, automatic charging, or automatic production-model promotion.

## Python API and normalized batch schema

```python
from llmforge.hw.device.prediction.data.dataset import MeasurementDataset
from llmforge.hw.device.prediction.training.pipeline import fit_dataset
from llmforge.hw.device.prediction.inference.predictor import predict_bundle, bundle_targets
from llmforge.hw.device.prediction.models.serialization import load_bundle

dataset = MeasurementDataset.load("runs/device/datasets/round_000.json")
next_dataset = dataset.append_training(batch)
next_dataset.save("runs/device/datasets/round_001.json")
model = fit_dataset(next_dataset, family="xgboost", seed=42)
predictions = predict_bundle(model, candidate_configs)
targets = bundle_targets(model)
```

A normalized dataset has `schema_version: 1`, a `protocol` object, and an `observations` list. Each observation contains:

```text
measurement_id   unique measurement identity
config_id        architecture identity
round_id         acquisition round
split            train / validation / test
architecture     n_layer, d_model, n_h, n_kv, d_qk, d_v, d_mlp;
                 optional vocab_size (default 50257), q8_group_size (validated)
metrics          decode_tok_s, ttft_ms, and the selected energy metric:
                 dynamic_energy_per_token_mj or gross_energy_per_token_mj
```

A normalized append batch contains `protocol`, `observations`, and optional `source_hashes`. Its split may be omitted and is set to training. It can be imported with `dataset append --dataset ... --batch ... --output ...`.

## Historical artifacts and compatibility

Historical experiment directories are not part of this repository. The historical experiment commands expect them under `runs/device/`. Recorded historical hashes are unchanged.

Old reports may show former commands/source paths. Use the new package CLI rather than editing those historical records. Trusted older checkpoints load through `models.serialization.load_bundle`, which supplies their old flat Python module names. Newly saved bundles use the package module paths.

Shared measurement and configuration files live under `src/llmforge/hw/device/measurement/sweep/`. The inference C sources belong to the pinned device runtime, see `docs/hw_device.md`.

## Tests

```bash
python -m unittest discover -s src/llmforge/hw/device/prediction/tests -t src -v
```

Tests include historical checkpoint prediction equivalence, immutable rounds, protocol mismatches, duplicate IDs, architecture leakage, energy target labeling, CSV ingestion and a synthetic train/report run. The historical checkpoint tests run only when `LLMFORGE_DEVICE_HISTORY` points to the historical outputs. Tests never benchmark the watch or retrain production checkpoints.
