# Third-party notices

## Included in this repository

| component | location | origin | license |
|---|---|---|---|
| Timeloop architecture, mapper and problem specifications | `src/llmforge/hw/timeloop/specs` | adapted from the Timeloop project examples | MIT |
| Held-out evaluation documents | `assets/heldout` | FineWeb-Edu, sample-10BT | ODC-By 1.0 |
| A nanoGPT derivative | `vendor/gpt_model` | the model definitions the ZEUS GPU target instantiates | MIT |
| A llama2.c derivative | `vendor/device_runtime` | the C inference runtime of the device measurement harness | MIT |

Both vendored trees keep their upstream `LICENSE` file. `vendor/README.md` records what was taken and what was changed.

## Built by `scripts/setup/install_timeloop.sh`, not included

| component | purpose | license |
|---|---|---|
| Timeloop | mapper and model behind the simulator targets | BSD-3-Clause |
| timeloopfe | Python front end of Timeloop | MIT |
| Accelergy with its Library, CACTI, table-based, Aladdin, NeuroSim and ADC plug-ins | energy and area estimation | MIT |
| NeuroSim, DNN_NeuroSim V1.3, compiled by the NeuroSim plug-in | circuit estimates for adders, multipliers and registers | CC BY-NC 4.0 |
| CACTI | memory energy model called by the CACTI plug-in | BSD-3-Clause |
| barvinok | polyhedral counting library linked by Timeloop | GPL-2.0 |
| isl and PolyLib, bundled with barvinok | polyhedral libraries | MIT |
| NTL, from the system package manager | number theory library linked by barvinok | LGPL-2.1-or-later |

NeuroSim is licensed for non-commercial use only.

## Downloaded at run time

| component | purpose | license |
|---|---|---|
| SmolLM2-135M and SmolLM2-360M | supernet base checkpoints | Apache-2.0 |
| Qwen3-0.6B-Base, Qwen3-1.7B-Base and Qwen3-4B-Base | supernet base checkpoints | Apache-2.0 |
| FineWeb-Edu | supernet training data | ODC-By 1.0 |

## Python dependencies

| package | license |
|---|---|
| PyTorch | BSD-3-Clause |
| Transformers, Datasets, Accelerate, Hugging Face Hub, Safetensors | Apache-2.0 |
| ZEUS | Apache-2.0 |
| XGBoost | Apache-2.0 |
| scikit-learn, NumPy, pandas, joblib | BSD-3-Clause |
| PyArrow | Apache-2.0 |
| PyYAML | MIT |
