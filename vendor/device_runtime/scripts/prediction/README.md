# Hardware performance prediction

**Current delivery / latest reported result (2026-09-22):** [TPOT XGBoost refit](../../diliverable/layerwise_tpot_2000_20260922/README.md),
implemented in the isolated [TPOT training/reporting module](../prediction_research/layerwise_tpot/README.md).
Outputs are **TPOT (not throughput), TTFT, dynamic energy**, all lower-is-better.
Using 110 features and 1600/200/200 splits, three-seed mean test MAPE is
**15.50% / 16.52% / 13.97%**. The final checkpoint is validation-selected seed 2026.
Its individual test MAPE is 15.46% / 16.49% / 13.97%; do not confuse this with the
three-seed mean. See the [delivery index and file checklist](../../diliverable/README.md)
for the current model, labels, raw evidence, test predictions, and checksum manifests.
Old AL and throughput predictors below retain their original target schemas; they are not migrated or overwritten.

Latest data-only release: [2,000 layerwise points after the GS64 refresh](../../diliverable/layerwise_2000_gs64_refresh_20260922/README.md).
It replaces 664 complete measurement rows, retains 1,336 old rows and all original
splits, and records mixed kernel provenance. All 2,000 energy labels are now valid.
Use its `dataset_snapshot.json` explicitly with the TPOT runner's `--snapshot`;
default live-registry reads still use old measurements. Historical checkpoints and
the throughput accuracy results below have **not** been retrained on this refresh.

A reusable prediction package, separate from device measurement code in `scripts/sweep/`.

The isolated [compact-surrogate study](../prediction_research/compact_surrogates/README.md)
compares 1–2-layer 32/64-wide Transformers and two-hidden-layer MLPs on the frozen
1,564-architecture cohort (1,100/150/314 split). Its definitions, training and
reporting live outside this package so existing AL source contracts remain valid.
The local `outputs/compact_surrogates_1564/README.md` records validation-selected
three-seed results: small Transformer MAPE 22.23/18.83/26.96%, two-hidden-layer MLP
22.47/19.36/28.64% (throughput/TTFT/dynamic energy). Historical XGBoost remains
20.80/19.06/26.69%. This is exploratory test reuse, not a production-model promotion
or a pure capacity ablation (the neural stopping criterion also differs).

The new [layerwise 500-candidate sampling preparation](../sweep/layerwise/README.md)
is a separate hardware campaign with a confirmed fixed skeleton. Existing predictor checkpoints and this
package's homogeneous architecture schema do not accept its ordered layer lists;
do not ingest those candidates into the old AL workspace. Its explicit runner
collects new labels; no layerwise predictor is trained by that measurement tool.
An independent [layerwise XGBoost refit](../prediction_research/layerwise_refit/README.md)
now snapshots these registries, preserves their grouped holdouts, and saves
architecture-only checkpoints without changing live collection or AL state.
The [2,000-architecture frozen release](../../diliverable/layerwise_2000_20260921/README.md)
uses identical 110 features and 1600/200/200 grouped splits, with three seeds.
Mean test MAPE (throughput/TTFT/dynamic energy) is 26.06/21.95/11.57% for XGBoost,
26.46/22.63/12.13% for Transformer 1x32, and 26.52/22.73/11.78% for Transformer 2x64.
The final validation-selected XGBoost seed-2026 checkpoint has individual test MAPE
25.91/22.29/11.61%; it is saved with the dataset, SQLite backups, raw evidence and reports.
The [earlier 1,745-row comparison](../prediction_research/layerwise_compare/README.md)
is preserved as historical context; its 184-row test set differs from the current 200.

Active learning now defaults to baseline-subtracted **dynamic energy**. The [active-learning guide](active_learning/README.md) documents the audited `active switch-energy` transition, resumable `active run` command, and target-specific `active monitor` accuracy reports. Existing gross experiments are preserved rather than overwritten.

```text
scripts/prediction/
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
└── outputs/       Local models, snapshots, predictions and reports (Git-ignored)
```

Core modules do not import experiment scripts. Define a model in `models/`, fit it in `training/`, and expose its prediction in `inference/`. Shared architecture transformations belong in `features/`. Historical ablations retain their experiment-specific settings rather than silently changing old results.

## Curated dataset and model delivery

The repository-root [diliverable directory](../../diliverable/README.md) is the curated, non-ignored export; its spelling follows the requested folder name. The **2026-09-14 snapshot** contains:

| Artifact under `diliverable/` | Meaning |
|---|---|
| `data/dataset_latest.json` | Dynamic round 9: 1,552 training + 150 validation + 314 test = 2,016 valid observations |
| `models/predictor_best.joblib` | Dynamic round 1: lowest mean of the three fixed-validation MAPEs among the stage's initial and completed checkpoints |
| `data/dataset_best_model.json` | Exact dataset matching that best checkpoint: 1,472 training + the same 464 holdout observations |
| `models/predictor_latest.joblib` | Dynamic round 9 checkpoint, matching `dataset_latest.json` |

Best energy validation MAPE is **43.558%**; three-target mean MAPE is **29.322%**. This is the best within the current dynamic stage, not across incompatible historical experiments, and not an independent test-set claim. The best checkpoint was **not** trained on the latest 1,552-row training set. For checkpoint-based initialization, always use its matching dataset; for prediction, either checkpoint accepts architecture configurations alone.

The export includes selection results, source/protocol metadata, package versions, SHA-256 checks and the quarantined zero-energy record. See its README for inference examples and scope. Original raw traces and interrupted experiment state remain in the local workspace; this export is not a resumable AL workspace or a full raw-data backup.

To create a later snapshot without accessing hardware or retraining, run from the repository root and choose a new destination:

```bash
python -m scripts.package_prediction_deliverable --destination diliverable_next
```

The exporter verifies committed history, requires the source workspace lock, reevaluates fixed validation only, and refuses to overwrite an existing destination. It copies artifacts; it does not move or delete experiment data.

Both `/scripts/prediction/outputs/` and `/scripts/sweep/outputs/` are ignored by Git. Previously tracked outputs were removed from the index only, keeping all local files and existing Git history. A fresh checkout will therefore not supply those local historical artifacts or a resumable experiment workspace; use the curated delivery for prediction, and retain/back up the original workspace separately for resume or historical reproductions.

## Commands

Run from the repository root in the existing `nanollmforge` environment:

```bash
python -m scripts.prediction --help
python -m scripts.prediction train --help
python -m scripts.prediction predict --help
python -m scripts.prediction dataset --help
python -m scripts.prediction active --help
```

The former flat `python scripts/prediction/<name>.py` commands are replaced by `python -m scripts.prediction <name-with-hyphens>`. For example:

```bash
python -m scripts.prediction compare-surrogates \
  --output scripts/prediction/outputs/surrogate_comparison_rerun

python -m scripts.prediction train-proposed-physics-surrogate \
  --output scripts/prediction/outputs/proposed_physics_surrogate_rerun

python -m scripts.prediction evaluate-gross-energy \
  --output scripts/prediction/outputs/gross_energy_comparison_1564_rerun
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
python -m scripts.prediction dataset bootstrap \
  --previous scripts/prediction/outputs/batch2_progress_632 \
  --protocol /path/to/protocol.json \
  --output scripts/prediction/datasets/round_000.json
```

For another initial dataset, use the normalized schema documented below with explicit, architecture-disjoint split assignments.

### 2. Train and predict candidate architectures

```bash
python -m scripts.prediction train \
  --dataset scripts/prediction/datasets/round_000.json \
  --family xgboost \
  --output scripts/prediction/outputs/al_round_000

python -m scripts.prediction predict \
  --model scripts/prediction/outputs/al_round_000/model.joblib \
  --configs /path/to/candidate_architectures.csv \
  --output scripts/prediction/outputs/round_000_predictions.csv
```

Use `--family transformer` for the grouped Transformer. `--seed` selects one training seed; this command does not claim a multi-seed comparison. A run records its dataset snapshot/hash, protocol, model, split IDs, source hashes, library versions, held-out predictions and report.

Prediction column names follow the saved model's targets, including the distinction between gross and dynamic energy. Only trusted local joblib bundles should be loaded.

### 3. Measure selected candidates, then append a new round

Hardware measurements remain an explicit, separate step using the appropriate sweep protocol. Once a completed CSV batch has been reviewed:

```bash
python -m scripts.prediction dataset import-csv \
  --dataset scripts/prediction/datasets/round_000.json \
  --configs /path/to/measured_architectures.csv \
  --measurements /path/to/completed_measurements.csv \
  --protocol /path/to/protocol.json \
  --round-id round_001 \
  --output scripts/prediction/datasets/round_001.json

python -m scripts.prediction train \
  --dataset scripts/prediction/datasets/round_001.json \
  --family xgboost \
  --output scripts/prediction/outputs/al_round_001
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
from scripts.prediction.data.dataset import MeasurementDataset
from scripts.prediction.training.pipeline import fit_dataset
from scripts.prediction.inference.predictor import predict_bundle, bundle_targets
from scripts.prediction.models.serialization import load_bundle

dataset = MeasurementDataset.load("scripts/prediction/datasets/round_000.json")
next_dataset = dataset.append_training(batch)
next_dataset.save("scripts/prediction/datasets/round_001.json")
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

The existing historical experiment directories remain locally in the Git-ignored `outputs/`. Models, snapshots, measurements, reports and recorded historical hashes are unchanged. Their availability is not guaranteed in a fresh checkout. `relocation_manifest.json` describes the preceding move from `sweep/`; it is historical provenance, not a list of current module locations.

Old reports may show former commands/source paths. Use the new package CLI rather than editing those historical records. Trusted older checkpoints load through `models.serialization.load_bundle`, which supplies their old flat Python module names. Newly saved bundles use the package module paths.

Shared measurement/configuration files remain under `scripts/sweep/`. Stability-verification archives and inference C sources are untouched.

## Tests

Offline recorded-start-temperature ablations are isolated under
[`prediction_research/state_inputs`](../prediction_research/state_inputs/README.md).
They preserve production/AL source contracts and do not promote diagnostic models.

```bash
python -m unittest discover -s scripts/prediction/tests -t . -v
```

Tests include historical checkpoint prediction equivalence, immutable rounds, protocol mismatches, duplicate IDs, architecture leakage, energy target labeling, CSV ingestion and a synthetic train/report run. The historical checkpoint tests require the existing local `outputs/` artifacts. Tests never benchmark the watch or retrain production checkpoints.
