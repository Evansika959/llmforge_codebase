# rDXE ring simulator

A graph-level simulator for the DXE decoder accelerator and its scaled multi-chip ring extension. It models full LLM inference, prefill and decode, with a KV cache, fused VRC ops for Softmax and RMSNorm, GQA through VLINK, WMEM layer-switch reload, and inter-chip pipeline communication.

The simulator wraps Timeloop per-GEMM energy and latency results and adds post-hoc corrections for features Timeloop cannot model: KV cache routing, asymmetric iWuR access, GQA sharing, fused non-linear ops, and multi-chip pipelining.

Reference for the DXE design: withheld for anonymous review.

See `docs/hw_simulators.md` at the repository root for how the search uses this package.

## Architecture: 3 layers

```
+------------------------------------------------------------+
| Layer 3: RingSimulator                                     |
|   N chips in pipeline ring, token-level scheduling,        |
|   inter-chip comm, pipeline bubble analysis                |
+------------------------------------------------------------+
| Layer 2: ChipSimulator                                     |
|   Full prefill+decode loop on 1 chip, KV cache state,      |
|   layer graph composition, WMEM reload, fusion             |
+------------------------------------------------------------+
| Layer 1: OpModel                                           |
|   Wraps Timeloop GEMM results + post-hoc                   |
|   corrections (KV cache energy swap, VLINK, VRC)           |
+------------------------------------------------------------+
| Underlying: llmforge.hw.timeloop.gemm + cached mapper runs |
+------------------------------------------------------------+
```

## File layout

```
llmforge/hw/rdxe/
    __init__.py              re-exports OpModel, ChipSimulator, RingSimulator, DXEConfig, ...
    cli.py                   eval, ring, and compare commands
    workflow.py              profile, pack, and ring-simulate models: the three-stage workflow
    cosearch.py              inner chip co-search, called once per candidate architecture
    scaling.py               analytical edge-LLM scaling on chip-scaled rings

    core/
        constants.py         hardware parameters of the reference configuration
        config.py            DXEConfig dataclass and YAML loader
        op_model.py          Layer 1: OpModel, Timeloop wrapper plus corrections
        timeloop_evaluator.py  cached Timeloop GEMM oracle with fast and loose mapper passes
        analytical_model.py  tiling-aware analytical GEMM model, the fallback oracle
        kv_cache_model.py    KVCacheState: per-user occupancy tracker
        layer_graph.py       LayerGraph: 7-GEMM plus VRC op DAG per layer
        scaled_arch.py       ScaledChipSpec and the EDGE_MODELS catalog

    simulator/
        chip_sim.py          Layer 2: ChipSimulator, full inference on 1 chip
        ring_sim.py          Layer 3: RingSimulator, N-chip pipeline ring
        layer_eval_timeloop.py  Timeloop-backed per-layer decode and prefill costs

    reporting/
        output.py            CSV writers and matplotlib plots

    configs/
        mt5_small.yaml       example YAML config
```

Outputs land in `runs/rdxe/results` and `runs/rdxe/plots`. The Timeloop cache lives under `runs/cache/timeloop` unless `LLMFORGE_TIMELOOP_WORK` points elsewhere.

**Import paths**:
```python
from llmforge.hw.rdxe import OpModel, ChipSimulator, RingSimulator, DXEConfig
from llmforge.hw.rdxe.core import EDGE_MODELS, ScaledChipSpec
from llmforge.hw.rdxe.cosearch import run_rdxe_eval
```

## Quick start

### 1. Single-chip MT5-small inference

```bash
python -m llmforge.hw.rdxe.cli eval --model mt5_small --prefill-length 960 --decode-length 64
```

Reports per-token energy split into compute, KV cache, VRC, and WMEM reload, plus TTFT, TPOT, and throughput. Saves per-token traces and a 4-panel plot.

### 2. Multi-chip ring

```bash
python -m llmforge.hw.rdxe.cli ring --model mt5_small --n-chips 8 --decode-length 64
```

Reports per-chip energy, utilization, layer assignment, and steady-state throughput.

### 3. Timeloop-only versus corrected simulation

```bash
python -m llmforge.hw.rdxe.cli compare --model mt5_small --decode-length 64
```

Side-by-side comparison showing the impact of the KV cache swap, VLINK, and VRC corrections.

### 4. Profile, pack, and simulate a model zoo

```bash
python -m llmforge.hw.rdxe.workflow --ctx 2048 --prefill 512 --decode 256 --workers 8
```

### 5. Edge LLM scaling on chip-scaled rings

```bash
python -m llmforge.hw.rdxe.scaling --prefill 128 --decode 64 --context 256
python -m llmforge.hw.rdxe.scaling --models smollm_135m,qwen25_05b,llama32_1b
```

Each model's chip is scaled so its WMEM fits exactly one layer's weights, forming a ring with one chip per layer.

## Modeled features

| Feature | Implementation |
|---------|---------------|
| **KV cache energy** | Swap DRAM (64 pJ/byte) for KV-cache SRAM (0.24 pJ/byte) on QK_attn and PV_attn |
| **iWuR (asym. K/V)** | 1.2x energy penalty on column-wise V reads (multi-bank activation) |
| **VLINK GQA sharing** | When `n_heads > n_kv_groups`, divide KV reads by the sharing ratio |
| **VRC fused ops** | Analytical: 0.5 pJ/elem softmax, 0.3 pJ/elem RMSNorm |
| **WMEM layer reload** | Triggers when total weights exceed 3 MB: `weight_bytes x 64 pJ/byte`, latency = bytes / QSPI bandwidth |
| **Inter-chip comm** | `n_embd x 8 bits x 1 pJ/bit` per hop, 10 cycles + serialization |
| **Pipeline bubbles** | Prefill ramp = (n_chips - 1) stages of bubble |
| **Multi-tenant KV** | Up to 4 users x 128 entries/user per core |
| **Operator fusion** | `compute_fusion_savings()` from `llmforge.hw.timeloop.gemm` |

## Reference configuration

- **Compute**: 8 DXT x 16 VAC x 16 MAC = **2048 INT8 MACs** at 200 MHz (5 ns)
- **WMEM**: 24 KB per core (128 cores) = **3 MB total**, weight-stationary
- **KV Cache**: 8 KB per core = **1 MB total**, 16 internal banks, iWuR layout
- **Global SRAM**: 8 KB (relaxed) / 512 B (original)
- **Head SRAM**: 2 KB per DXT (relaxed) / 512 B (original)
- **Acc Buffer**: register file per core, 64 entries (relaxed) / 16 entries (original)
- **DRAM**: LPDDR4, 4 B/cycle (relaxed) / 2 B/cycle (original)

## Configuration

YAML config example (`configs/mt5_small.yaml`):

```yaml
arch: dxe_relaxed
features:
  kv_correction: true
  vlink_gqa: true
  vrc: true
  wmem_reload: true
  fusion: true
simulation:
  prefill_length: 960
  decode_length: 64
multi_chip:
  n_chips: 8
  layer_assignment: round_robin   # round_robin | balanced | single_layer
```

Programmatic use:

```python
from llmforge.hw.rdxe import DXEConfig, OpModel, RingSimulator, EDGE_MODELS

config = DXEConfig(prefill_length=128, decode_length=64)
op_model = OpModel(config)
ring = RingSimulator(n_chips=8, op_model=op_model, config=config)
result = ring.run_ring_inference(EDGE_MODELS['mt5_small'])

print(f"E/token: {result.energy_per_token_uJ:.3f} uJ")
print(f"TTFT:    {result.ttft_ms:.3f} ms")
print(f"TPOT:    {result.tpot_us:.2f} us")
```

## Dependencies

- Python 3.10+
- `numpy`, `matplotlib`, `pyyaml`
- Optional: `timeloopfe` and the `timeloop-mapper` binary for Timeloop-mapped GEMMs. Without them, GEMM shapes without a cached mapper result use the analytical model, and results count those ops in `ops_fallback`.

## Caveats

- **Energy is an approximation**: Timeloop energy tables for compute and SRAM, analytical models for VRC and inter-chip communication.
- **Dynamic energy only**: no leakage, clock tree, control logic, or I/O pad modeling.
- **Single-chip mode requires WMEM reload** for any model above 3 MB, which covers most edge LLMs. The ring topology with one layer per chip is the design's intended operating mode.
- **Context length must be Timeloop-mappable**: small contexts are rounded up to powers of 2 up to 128, larger ones to multiples of 128, for DXE's spatial constraint.
- **Co-search profile is homogeneous**: `cosearch.run_rdxe_eval` costs every active layer like the first active layer.
