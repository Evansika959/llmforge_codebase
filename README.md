# LLMForge

LLMForge searches for language model architectures that are accurate and cheap on a given piece of
hardware. It explores the slices of a trained elastic supernet with NSGA-II. The supernet scores the
quality of every candidate without training it, and a hardware target in the loop measures or
predicts what the candidate costs to run.

## Repository layout

| path | content |
|---|---|
| `src/llmforge/search` | elastic search space, NSGA-II, and the co-search dispatcher |
| `src/llmforge/supernet` | elastic supernets over pretrained checkpoints: training, slicing, extraction |
| `src/llmforge/evaluators` | software and hardware evaluators behind one interface |
| `src/llmforge/hw` | hardware targets: `zeus` for GPUs, `timeloop` and `rdxe` simulators, `device` for the Pixel Watch 5 |
| `configs/search_spaces` | one YAML per supernet and partition, validated against the supernet at load |
| `experiments/hw_nas` | hardware-aware search experiments: design, queue, analysis |
| `experiments/supernet_fidelity` | experiments on how well slice scores stand in for training |
| `assets` | frozen held-out documents, dedicated-training reference losses, the device predictor |
| `docs` | method documentation |
| `scripts` | environment setup and supernet training launchers |
| `tests` | CPU tests, with GPU tests skipped when no GPU is present |

`runs/` holds outputs and evaluation caches and `scratch/` holds temporary files. Git ignores both.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[supernet,zeus,device,dev]"
scripts/setup/fetch_third_party.sh
```

The GPU target builds models from a pinned ReaLLM-Forge checkout. The simulator targets also need
Timeloop, see `docs/hw_simulators.md`.

## Search

```bash
python -m llmforge.search.cosearch \
    --space configs/search_spaces/smollm2-135m.yaml \
    --supernet runs/supernet/sl135_lr8e-3/step2500 \
    --hw zeus --zeus-batch 64 --prefill-len 512 --decode-len 128 \
    --objectives val_loss energy_per_token_uJ \
    --pop 32 --generations 25 --seed 0 \
    --out runs/search/example
```

| `--hw` | target | cost source |
|---|---|---|
| `analytic` | none | parameter, FLOP, KV-cache and MAC estimates |
| `zeus` | local NVIDIA GPU | latency and energy measured through NVML |
| `device` | Pixel Watch 5 | predictor fitted to on-device measurements, uniform architectures |
| `timeloop` | accelerator substrates | Timeloop mapping of a per-layer GEMM decomposition |
| `rdxe` | ring-configured decoder accelerator | simulator with an inner chip-configuration search |

A run writes its arguments, every evaluation, a per-generation hypervolume trace, population
checkpoints and the final front to `--out`. Evaluations are cached across runs, so an interrupted run
resumes by rerunning the same command.

## Documentation

- `docs/search.md` specifies the search space and the search method.
- `docs/supernet.md` covers supernet training, slicing and ground truth.
- `docs/hw_gpu.md` covers the GPU measurement method and its decode implementation.
- `docs/hw_device.md` covers the Pixel Watch 5 measurement method and predictor.
- `docs/hw_simulators.md` covers the Timeloop and rDXE targets.
- `experiments/hw_nas/README.md` covers the experiment design and the GPU protocol pilot.

## Tests

```bash
CUDA_VISIBLE_DEVICES="" pytest tests
```

## License

MIT, see `LICENSE`. Third-party components and their licenses are listed in
`THIRD_PARTY_NOTICES.md`.
