# Vendored third-party code

Two external projects are vendored here rather than fetched, so that the released tree builds and runs
without network access and so that every file a measurement depended on is present. Both are MIT licensed
and both keep their upstream `LICENSE` file. `THIRD_PARTY_NOTICES.md` at the repository root lists them
alongside the rest of the third-party material.

## `gpt_model`

A nanoGPT derivative. The ZEUS GPU target instantiates its model definitions to measure a candidate
architecture, so it is the model implementation behind every number the GPU backend reports.

Taken: `model.py`, `gpt_conf.py`, `shared_param_utils.py`, the `variations/`, `initializations/` and
`quantization/` packages, and `LICENSE`. That is 19 Python files. Nothing else from upstream is needed,
and the training and evaluation scripts, the datasets and the checkpoints were left behind.

Changed: nothing. Every file is byte identical to upstream.

## `device_runtime`

A llama2.c derivative, extended upstream of this project with an exporter and a heterogeneous kernel so
that a per-block elastic architecture can run in the C engine. It is the runtime of the on-device
measurement harness, which is what `assets/device/pixel_watch5` records.

Taken: the whole project, 216 files, including the C kernels under `src/`, the export and parity tooling
under `llmforge_bridge/`, the PyTorch reference under `pytorch/`, the measurement and prediction scripts
under `scripts/`, the documentation under `doc/`, and the example tokenizer under `models/`. Model weights
are not included, since they are regenerated from a supernet checkpoint by
`llmforge_bridge/export_llmforge_hetero.py`.

Changed: project identifiers were renamed for anonymous review. The bridge package moved from its former
name to `llmforge_bridge/`, seven files were renamed to carry `llmforge` instead of the former project
name, the corresponding make targets became `runllmforge`, `runqllmforge` and `runllmforgeomp`, and
comments, docstrings and error strings were rewritten to match. No numeric path was touched, so the
measurements in `assets/device/pixel_watch5` still describe these files. Nine of the renamed files are
recorded in the measurement manifests, whose digests were recomputed for the released tree with the
measurement-time digests preserved under `anonymization.sha256_before_anonymization`. See
`docs/hw_device.md` for the verification path.
