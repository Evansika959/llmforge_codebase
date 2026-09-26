# Pixel Watch 5 per-layer predictor: TPOT, TTFT and dynamic energy

This is the Backend D predictor. `python -m llmforge.search.cosearch --hw device` loads
`models/predictor_final.joblib` by default and hands it each candidate with its layer shapes intact.

Dataset: 2,000 per-layer architectures of SmolLM2-135M and SmolLM2-360M measured on a Pixel Watch 5,
under a fixed grouped split of 1,600 for training, 200 for validation and early stopping and 200 for test.
The predictor reads 110 architecture-only features, with no temperature, battery, voltage or measured
timing among its inputs. Each target has its own XGBoost regressor fitted in log space, with depth 3, up
to 1,500 trees, learning rate 0.03, L2 penalty 5, row and column subsampling 0.85 and 0.9, and early
stopping after 50 rounds without improvement in validation log-RMSE. Seeds 42, 123 and 2026 were trained.
All three targets are lower-is-better.

## Test results, mean across three seeds

| Target | MAPE | Spearman | Kendall τ-b | Pairwise accuracy | Recall@32 |
|---|---:|---:|---:|---:|---:|
| tpot_ms | 15.50% | 0.8426 | 0.6666 | 83.33% | 97.92% |
| ttft_ms | 16.52% | 0.9225 | 0.7777 | 88.88% | 93.75% |
| dynamic_energy_per_token_mj | 13.97% | 0.9378 | 0.7919 | 89.59% | 62.50% |

Pairwise accuracy is over the 19,900 pairs of test architectures.

## Checkpoint used by the search

`models/predictor_final.joblib` is seed 2026, selected only by the mean validation MAPE across the three
targets before the test set was evaluated. It is one seed, not an ensemble and not refitted on
validation or test rows. The other seeds are kept under `models/`.

| Target | Test MAPE of the selected checkpoint |
|---|---:|
| tpot_ms | 15.46% |
| ttft_ms | 16.49% |
| dynamic_energy_per_token_mj | 13.97% |

## Definitions and limitations

- The workload is a 48-token prompt, 32 output tokens, a batch of one and four CPU threads, on the INT8
  runtime in `vendor/device_runtime`.
- TPOT is 1000 × (end − prefill end) / 31 in milliseconds per decode token.
- Dynamic energy covers prefill and decode, with the idle baseline subtracted, divided by the 32 output
  tokens. It is not decode-only energy.
- Group size 64 rows use the optimized kernel and group sizes 16 and 32 keep earlier measurements, so the
  dataset mixes kernels and measurement dates. `manifest.json` records the kernel digest of each group.
- 417 rows carry baseline-drift warnings, which are kept. All 2,000 labels are positive and finite.
- The test rows were fixed before this refit and reused from earlier evaluations, so the test result is
  exploratory rather than a fresh confirmatory holdout.

## Files

- `dataset_snapshot.json`: every architecture, label, split and measurement protocol record.
- `labels.json`: the three training targets and the preserved splits.
- `metrics.json`, `selection.json`: per-seed validation and test metrics, and the checkpoint selection.
- `test_predictions.npz`: the recorded test predictions of every seed.
- `source_snapshot/`: the exact feature and training code the bundle was fitted with. The inference path
  in `src/llmforge/hw/device/prediction/inference/layerwise.py` imports its feature code from here.
- `manifest.json`, `release_hashes.json`, `verification.json`: provenance, digests and delivery checks.
  `manifest.json` also records what was changed for anonymous review.

## Reproduce

The training code is vendored under `vendor/device_runtime`. From that directory:

```bash
python -m scripts.prediction_research.layerwise_tpot.run \
  --snapshot ../../assets/device/layerwise_tpot_2000_20260922/dataset_snapshot.json \
  --output scripts/prediction/outputs/layerwise_tpot_reproduction
```

The output directory must not exist yet.
