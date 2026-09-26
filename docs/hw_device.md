# On-device target: Pixel Watch 5

The on-device target predicts how fast and how energy-efficiently a uniform decoder-only language model runs on a Google Pixel Watch 5. A tree ensemble fitted to measurements taken on the watch maps an architecture to three metrics, and the search uses it as a hardware evaluator. The measurement harness and the active-learning loop that collected the labels ship with the predictor, so the fit can be reproduced or extended on a device.

Code lives in `src/llmforge/hw/device/`, the fitted model and its data in `assets/device/pixel_watch5/`, and tests in `tests/device/`.

## What is measured

Every label comes from one inference run of a randomly initialized INT8 model on the watch.

| Setting | Value |
|---|---|
| Prompt | 49 tokens from a nominal 48-token prompt |
| Output | 32 tokens, the first after prefill and then 31 decode steps |
| Weights | INT8 with group size 16, 32 or 64, the largest that divides every matrix width |
| Threads | 4 OpenMP threads |
| Power | battery current and voltage sampled at 10 Hz |
| Admission | CPU below the temperature ceiling, no cooling or frequency clamp, battery above 30 percent |

| Target | Unit | Definition |
|---|---|---|
| `decode_tok_s` | tokens per second | decode throughput reported by the runtime |
| `ttft_ms` | milliseconds | time to first token |
| `dynamic_energy_per_token_mj` | millijoules per output token | `max(active_power_w - baseline_power_w, 0) * duration_s * 1000 / 32` |

Baseline power is the median over the idle window before inference, after one second of settling. Active power is the mean over the inference window. The energy label includes prefill, so it is not decode-only energy. Measured power, temperature, battery state and acquisition order are never model inputs. The active-learning adapter also refuses to measure while a charger is connected.

The recorded protocol keeps the historical admission ceiling of 40 °C. Later rounds used a strict ceiling below 45 °C, and each of those observations records its thermal policy in its measurement context.

## Predictor

Inputs are the physics32 features, computed from the architecture alone in `src/llmforge/hw/device/prediction/features/physics.py`:

- the shape fields `n_layer`, `d_model`, `n_h`, `n_kv`, `d_qk`, `d_v`, `d_mlp`, the vocabulary size and the INT8 group size
- parameter count, model and per-layer weight bytes, and KV-cache bytes per token
- how much of one layer fits in a shared 512 KiB L2 cache, the spilled bytes and the cache pressure
- flags for widths aligned to 64
- decode and prefill multiply-accumulate counts, and roofline times from a calibrated hardware profile

The hardware profile fits four effective constants by non-negative least squares on training rows only. Memory bandwidth and a per-layer synchronization time describe decode. A compute rate and a per-layer dispatch time describe prefill.

Each target has its own XGBoost regressor on the logarithm of the target, with 1,500 trees, learning rate 0.03, depth 3, minimum child weight 3, L2 penalty 5, row subsampling 0.85 and column subsampling 0.9. Training stops early after 50 rounds without validation improvement.

## Shipped checkpoints

| Checkpoint | Round | Training rows | Matching dataset |
|---|---:|---:|---|
| `models/predictor_best.joblib` | 1 | 1,472 | `data/dataset_best_model.json` |
| `models/predictor_latest.joblib` | 9 | 1,552 | `data/dataset_latest.json` |

Both datasets share the same holdouts of 150 validation and 314 test observations, and no holdout architecture appears in training. The best checkpoint has the lowest mean validation MAPE over the three targets among the initial checkpoint and the nine completed rounds of the dynamic-energy stage. The test split played no part in the selection and was not evaluated.

Validation MAPE in percent, from `evaluation/model_selection.json`:

| Round | Throughput | TTFT | Energy | Mean |
|---:|---:|---:|---:|---:|
| 0 | 22.139 | 22.414 | 44.051 | 29.535 |
| 1 | 22.024 | 22.384 | 43.558 | 29.322 |
| 2 | 22.106 | 22.360 | 43.819 | 29.428 |
| 3 | 21.996 | 22.402 | 44.291 | 29.563 |
| 4 | 22.091 | 22.687 | 44.195 | 29.658 |
| 5 | 22.003 | 22.525 | 44.377 | 29.635 |
| 6 | 21.977 | 22.402 | 44.477 | 29.619 |
| 7 | 22.057 | 22.466 | 44.560 | 29.694 |
| 8 | 22.113 | 22.416 | 44.066 | 29.532 |
| 9 | 22.059 | 22.369 | 44.100 | 29.510 |

For the best checkpoint, validation MAE is 5.490 tokens per second for throughput with R² 0.586, 197.7 ms for TTFT with R² 0.734, and 5.940 mJ per token for energy with R² 0.842. All ten checkpoints lie within 0.4 points of mean MAPE.

## Training domain

Ranges over all 2,016 observations in `data/dataset_latest.json`. The training split covers the same ranges, except that its smallest `d_mlp` is 512.

| Field | Min | Max | Distinct values |
|---|---:|---:|---:|
| `n_layer` | 3 | 28 | 26 |
| `d_model` | 256 | 640 | 13 |
| `n_h` | 1 | 16 | 7 |
| `n_kv` | 1 | 16 | 8 |
| `d_qk` | 16 | 64 | 4 |
| `d_v` | 16 | 64 | 4 |
| `d_mlp` | 480 | 3520 | 95 |
| `vocab_size` | 50257 | 50257 | 1 |
| `q8_group_size` | 16 | 64 | 3 |
| parameters in millions | 50.0 | 149.8 | 1,963 |

Tree models do not extrapolate beyond the feature values seen in training. The full SmolLM2-135M shape illustrates the limit. Its 134.5M parameters lie inside the domain, but its 30 layers and its 49,152-token vocabulary lie outside. Predictions for such architectures are extrapolations and should be reported as such.

Ranges are not the whole domain. The sweep drew query head counts from 1, 2, 4, 6, 8, 12 and 16 only, so a head count of 3 or 9 lies inside the range but appears in no observation. The SmolLM2-135M search space uses 3, 6 and 9 heads. `device_on_support` marks an architecture that lies inside every range and uses only observed values of `n_h`, `n_kv`, `d_qk`, `d_v` and `d_model`.

The INT8 group size enters the features but follows from the architecture: 64 when `d_model`, `n_h * d_v` and `d_mlp` are all multiples of 64, and 32 or 16 otherwise. Its measured effect is large. Within each model width with enough observations, those with group size 64 decode at 0.35 to 0.41 times the throughput of those with group size 32 and use 2.3 to 3.2 times the dynamic energy per token, after a least-squares control for parameter count and depth on logarithms. Predictions therefore jump whenever a change of head count or value head dimension changes the group size, by up to 2.8x in either direction on the SmolLM2-135M uniform grid. No architecture was measured twice, so the data give no direct estimate of repeat noise.

## Uniform architectures only

The features describe one layer shape repeated `n_layer` times. The predictor has no input for differences between layers, so it applies only to architectures whose layers share `n_h`, `n_kv`, `d_qk`, `d_v` and `d_mlp`. Callers must reject a heterogeneous architecture rather than average it into a uniform one.

## Using the predictor

```python
from llmforge.hw.device.prediction.inference.predictor import bundle_targets, predict_bundle
from llmforge.hw.device.prediction.models.serialization import load_bundle

pack = load_bundle('assets/device/pixel_watch5/models/predictor_best.joblib')
config = dict(n_layer=24, d_model=512, n_h=8, n_kv=4, d_qk=64, d_v=64, d_mlp=1536)
decode_tok_s, ttft_ms, energy_mj = predict_bundle(pack, [config])[0]
print(bundle_targets(pack))
```

Load only trusted checkpoints, because unpickling a joblib file can execute code. Checkpoints written under the earlier module layout still load, since `load_bundle` maps their old module names for the duration of the load.

The command-line equivalent reads a CSV of architectures:

```bash
python -m llmforge.hw.device.prediction predict \
  --model assets/device/pixel_watch5/models/predictor_best.joblib \
  --configs ARCHITECTURES.csv --output runs/device/predictions.csv
```

## How the labels were collected

1. A random sweep measured 1,564 architectures between 50M and 150M parameters. They were split into 1,100 training, 150 validation and 314 test architectures.
2. An active-learning loop then added training architectures in rounds. Each round fits a five-member bootstrap committee of XGBoost models on training rows and draws a batch from a pool of 5,000 unmeasured candidates. A batch of ten takes six candidates by committee disagreement, three by distance from measured architectures and one at random, and the measurement order is shuffled. Holdout architectures never enter the pool.
3. After 36 rounds that targeted gross energy, the loop switched to baseline-subtracted dynamic energy. The switch recomputed labels from the recorded power fields and restarted the round count. One training row with zero dynamic energy is quarantined in `provenance/energy_transition.json`.
4. The dynamic stage began with 1,926 observations and completed nine rounds of ten candidates, which gives the 2,016 observations shipped here.

Early measurements used unseeded random weights. Active rounds seed the weights from a hash of the architecture. Each round freezes its predictions before measuring, validates every result against the architecture and the energy formula, and records dataset and checkpoint hashes. Operation is described in `src/llmforge/hw/device/prediction/active_learning/README.md`.

## Measurement harness and runtime

`llmforge.hw.device.measurement.sweep.run_sweep_configs` measures a CSV of configurations and writes one result row per configuration. For each configuration it

1. writes a random-weight checkpoint of the requested shape and exports it to the INT8 format of the runtime,
2. waits until the admission conditions hold,
3. runs inference on the watch while the power sampler records current and voltage, with four seconds of idle sampling before and after,
4. reads throughput and TTFT from the runtime log and computes the power and energy fields.

The runtime is a C inference engine derived from llama2.c, vendored at `vendor/device_runtime` and licensed MIT. The kernel identity in the dataset protocol is the SHA-256 of `src/runq_llmforge.c`, and `provenance/contract.json` records the 70 source hashes of the runtime tree the measurements ran on. Fifteen of those files were changed for anonymous review, in comments, docstrings, error strings, report text, environment names, lock-file names and file names only, with no change to any numeric path. The manifests carry the current digests so that verification passes on the released tree, and keep the measurement-time digests under `anonymization.sha256_before_anonymization`. `LLMFORGE_DEVICE_RUNTIME` overrides the runtime root. Building needs an Android NDK through `NDK` or `ANDROID_NDK_HOME`, and measuring needs `adb`.

The device is always named explicitly with `--serial` or `ANDROID_SERIAL`. Temporary build and trace files go to `scratch/device/sweep`, and results go to `runs/device/sweeps`. A live active-learning run pins the physical device identity inside its own workspace, which is not part of the shipped assets.

## Release copy of the assets

The shipped assets were rewritten once for release.

- Absolute locations on the acquisition machine were removed from provenance strings and keys. Paths are relative to the directory that held the acquisition workspaces.
- The device identity record, which holds the device serial and build fingerprint, is withheld.
- Dataset fingerprints were recomputed along the full chain of round datasets, which rebuilds exactly from `data/dataset_latest.json`. The parent pointers, the fingerprints in `evaluation/model_selection.json`, the state record, the manifest and the `dataset_sha256` field inside both checkpoints were updated to match, and file digests were recomputed.
- Predictions of both checkpoints are identical to the originals on all 2,016 shipped architectures.
- Digests of files that are not part of this package refer to the original files.

`tests/device/test_assets_integrity.py` checks these properties.

## Tests

```bash
PYTHONPATH=src python -m pytest tests/device -q
PYTHONPATH=src python -m unittest discover -s src/llmforge/hw/device/prediction/tests -t src -v
```

The historical checkpoint regression tests run only when `LLMFORGE_DEVICE_HISTORY` points to the historical prediction outputs, which are not part of this repository.
