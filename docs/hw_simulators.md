# Simulator hardware targets

LLMForge prices candidate architectures on two families of simulated accelerators. The Timeloop backend maps every matrix multiplication of a transformer layer onto a published or custom accelerator substrate. The rDXE backend goes one step further and co-searches the chip configuration of a multi-chip ring accelerator for every candidate architecture. Both plug into the search through the evaluator interface in `src/llmforge/evaluators/base.py` and report the standard keys `energy_per_token_uJ`, `ttft_ms`, `tpot_ms`, and `hw_feasible`.

| Backend | Evaluator | Hardware model | Clock |
|---|---|---|---|
| Timeloop | `HwTimeloop` in `evaluators/hw_timeloop.py` | one substrate from the registry below | 1 GHz reference, 1 ns per cycle |
| rDXE | `HwRdxeInner` in `evaluators/hw_rdxe.py` | DXE ring with the chip configuration searched per architecture | 200 MHz, 5 ns per cycle |

The evaluator arguments used by the experiments live in `configs/hw/timeloop.yaml` and `configs/hw/rdxe.yaml`.

## Substrates

`ARCH_CONFIGS` in `src/llmforge/hw/timeloop/gemm.py` registers every substrate. Their spec files live under `src/llmforge/hw/timeloop/specs/`.

| Substrate | Spec | Model | DRAM bandwidth in bytes per cycle |
|---|---|---|---|
| `gemmini` | `arch/system_gemmini.yaml` | legacy Gemmini configuration with its original technology setting | 4 |
| `gemmini_16nm` | `arch/gemmini/` | Gemmini systolic array | 4 |
| `eyeriss` | `arch/eyeriss/` | Eyeriss row-stationary dataflow with 168 PEs | 4 |
| `simba` | `arch/simba/` | one Simba output-stationary chiplet with 256 MACs | 4 |
| `simba_edge` | `arch/simba_edge/` | Simba chiplet with edge-sized buffers | 4 |
| `flat_edge` | `arch/flat_edge/` | FLAT fused-attention dataflow on a 1024-MAC edge template | 25 |
| `dxe` | `arch/DXE/` | rDXE decoder engine with its original strict spatial constraints | 2 |
| `dxe_relaxed` | `arch/dxe_relaxed/` | DXE with relaxed spatial constraints for general GEMM shapes, 2048 MACs | 4 |
| `dxe_relaxed_m32` | `arch/dxe_relaxed_m32/` | `dxe_relaxed` with 32 MACs per core lane, 4096 MACs | 4 |
| `dxe_relaxed_m64` | `arch/dxe_relaxed_m64/` | `dxe_relaxed` with 64 MACs per core lane, 8192 MACs | 4 |

Every substrate except the legacy `gemmini` entry uses the same technology setting, so energy differences come from architecture and dataflow alone. The four DXE variants fix their spatial mesh at 128 output lanes. When a GEMM's output dimension exceeds 128 and is not a multiple of 128, the backend rounds it up to the next multiple and records the original and padded sizes under `padded_ops`.

## Seven-GEMM layer model

`evaluate_layer` decomposes a layer with infinite-head attention into seven GEMMs. A GEMM maps K input channels to N output channels over L rows, and Timeloop maps each GEMM onto the substrate.

| # | Op | K to N | L in prefill | L in decode | Repeats |
|---|---|---|---|---|---|
| 0 | `QK_gen` | `n_embd` to `n_qk_head_dim * (n_head + n_kv_group)` | prompt length | 1 | 1 |
| 1 | `V_gen` | `n_embd` to `n_v_head_dim * n_kv_group` | prompt length | 1 | 1 |
| 2 | `QK_attn` | `n_qk_head_dim` to context | `n_head / n_kv_group` | `n_head / n_kv_group` | `n_kv_group` |
| 3 | `PV_attn` | context to `n_v_head_dim` | `n_head / n_kv_group` | `n_head / n_kv_group` | `n_kv_group` |
| 4 | `ATTN_proj` | `n_v_head_dim * n_head` to `n_embd` | prompt length | 1 | 1 |
| 5 | `MLP_FC1` | `n_embd` to `mlp_size` | prompt length | 1 | 1 |
| 6 | `MLP_FC2` | `mlp_size` to `n_embd` | prompt length | 1 | 1 |

The context of the attention GEMMs equals the `block_size` of the pass. Energy, cycles, op counts, and memory accesses of the two attention GEMMs are multiplied by `n_kv_group`. A layer whose `attention_variant` is not `infinite` contributes only the two MLP GEMMs. An individual costs the sum over the layers enabled in `layer_mask`.

**Operator fusion.** Six edges pass a tensor directly from one GEMM to the next: `QK_gen` to `QK_attn`, `V_gen` to `PV_attn`, `QK_attn` to `PV_attn`, `PV_attn` to `ATTN_proj`, `ATTN_proj` to `MLP_FC1`, and `MLP_FC1` to `MLP_FC2`. A fused edge keeps that tensor on chip. For every edge the backend removes the DRAM energy of the producer's Outputs dataspace and of the consumer's Inputs dataspace. It also removes the matching DRAM cycles, computed as output writes divided by the DRAM write bandwidth plus output and input reads divided by the DRAM read bandwidth. A dataspace is removed once even when two edges touch it, as both incoming edges of `PV_attn` do. The savings of the two attention GEMMs are multiplied by `n_kv_group`, because each runs once per KV group, while the savings of the other GEMMs count once. The total saving is capped at 90 percent of the unfused energy of the layer as a safety bound. Before 2026-09-15 the backend multiplied every edge that touched an attention GEMM by `n_kv_group`, counted the attention GEMMs twice inside the cap, and removed the Inputs of `PV_attn` twice, so fused energies came out too low. In one long-prompt case fusion removed 96 percent of the prefill energy. The removed amounts are reported as `fusion_saved_energy_uJ` and `fusion_saved_cycles`.

**DRAM statistics.** `parse_dram_dataspace_stats` in `src/llmforge/hw/timeloop/stats.py` reads the per-dataspace block of the DRAM level in `timeloop-mapper.stats.txt`. It returns one dict per dataspace with `energy_pJ`, `scalar_reads`, `scalar_fills`, `scalar_updates`, `energy_per_scalar_access_pJ`, the bandwidth fields, and totals over instances. Every substrate names its off-chip level DRAM, so one parser covers all of them. The general `parse_buffer_stats` returns the same statistics for every level together with each level's leakage and total energy.

The parser was checked against 874 cached mapper outputs. Per-access energy times accesses reproduces every DRAM dataspace energy exactly. The DRAM dataspace energies match the DRAM entry of the fJ-per-compute table that Timeloop prints to within 4e-5 relative error. Level energies including leakage sum to the summary energy of every file within its printed rounding.

## Two-pass prefill and decode protocol

`HwTimeloop` prices every individual in two passes.

1. The prefill pass sets `block_size` to `prefill_len` and maps every GEMM at the prompt length. Its cycles give the time to first token.
2. The decode pass sets `block_size` to `decode_len`. Projections run on one token and the attention GEMMs read a context of `decode_len` tokens, so the pass prices one generated token.

The passes combine as follows, with one cycle equal to one nanosecond.

| Key | Definition |
|---|---|
| `energy_uJ` | prefill energy + `decode_len` * decode energy |
| `cycles` | prefill cycles + `decode_len` * decode cycles |
| `energy_per_token_uJ` | decode energy of one generated token |
| `session_e_per_tok_uJ` | `energy_uJ` / total tokens, where total tokens = `prefill_len` + `decode_len` |
| `ttft_ms`, `ttft` | prefill cycles / 1e6 in milliseconds and / 1e9 in seconds |
| `tpot_ms`, `tpot` | decode cycles / 1e6 in milliseconds and / 1e9 in seconds |
| `cycles_per_token`, `token_delay` | `cycles` / total tokens, and that value / 1e9 |
| `prefill_energy_uJ`, `prefill_cycles`, `decode_energy_uJ`, `decode_cycles` | per-pass values |
| `fusion_saved_energy_uJ`, `fusion_saved_cycles` | prefill value + `decode_len` * decode value |
| `padded_ops`, `padded_op_count` | GEMMs padded on DXE substrates, listed per pass |
| `hw_feasible` | False when the mapper failed on any GEMM of the individual |

The decode context equals `decode_len` and does not include the prompt. A failed mapping marks only that individual infeasible with infinite metrics and a `timeloop_error` message, so the generation continues. Setting `prefill_len` or `decode_len` to 0 runs a single prefill pass at each individual's own `block_size`.

## rDXE ring co-search

`HwRdxeInner` calls `run_rdxe_eval` in `src/llmforge/hw/rdxe/cosearch.py`, which runs an inner hardware search for every architecture that the outer search proposes.

**Profile.** Every active layer keeps its own shape. `workflow.profile_layers` turns each layer into weight bytes, KV-cache bytes at the configured context and user count, and a MAC estimate.

**Chip grid.** The inner search sweeps 45 chip configurations.

| Knob | Values |
|---|---|
| MACs per VAC core | 16, 32, 64 |
| Maximum chips in the ring | 8, 16, 32, capped at the number of active layers |
| WMEM per core | 24, 48, 96, 192, 384 KB |

**Pack.** For each configuration `workflow.pack_balanced` fills chips with contiguous layers until the next layer's weights would exceed the chip's WMEM with a 15 percent overhead, then sizes KV$ per core to the largest group. It grows the core count in powers of two and keeps the packing with the smallest total estimated area. A configuration that cannot place every layer within the chip limit is dropped.

**Simulate.** `workflow.simulate_ring` costs every distinct layer shape once. A decode token passes every layer and every inter-chip hop, which gives time per output token. The prompt is token-level pipelined: its first token passes the whole ring, every later token follows one bottleneck stage behind, and the first output token leaves with the last prompt token, which gives time to first token. Prompt tokens are priced like decode tokens at half the prompt length of context. Energy per output token adds the hops and the leakage of every chip over the decode step. Decode power in watts is `per_tok_uJ * n_users / tpot_ms * 1e-3`.

**Select.** The Pareto front over `per_tok_uJ`, `tpot_ms`, and `ttft_ms` is computed on all packed configurations and returned as `chip_pareto`. With `envelope_filter` on, the selected chip is the configuration with the lowest `select_by` value among those whose estimated area and decode power fall inside the envelope. When none fits, the evaluator returns the configuration with the smallest summed relative overshoot of area and power and sets `envelope_feasible` to False. With `envelope_filter` off, selection takes the lowest `select_by` value over all configurations and `envelope_feasible` reports whether that choice fits. The metrics of the selected chip become the hardware metrics of the architecture.

`HwRdxeInner` defaults to a prefill of 128 tokens, 32 decode tokens, a context of 2048, one user, `select_by=per_tok_uJ`, `weight_memory=wmem`, an area envelope from 0 to 800 mm2, and a power envelope from 0 to 100 W. Direct calls to `run_rdxe_eval` default to a prefill of 512 tokens, 256 decode tokens, a context of 2048, an area envelope from 100 to 2500 mm2, and a power envelope from 0.005 to 2 W.

**Layer costs.** The dxe_relaxed specification routes weights through DRAM so that one reference chip can map layers larger than its WMEM. The packing sizes every ring chip so that its layers fit in WMEM, so with `weight_memory=wmem` a GEMM costs the dynamic energy of its on-chip levels and the cycles of its slowest on-chip level, and the DRAM level of the mapping counts in neither. `weight_memory=dram` keeps the mapped DRAM traffic in both. `simulator/layer_eval_timeloop.py` adds three DXE corrections to the mapping.

- In QK_attn and PV_attn the mapped weight matrix is the K or V cache. Timeloop reads it once per query row and VLINK multicast reads it once per KV group, so these reads are divided by `n_head / n_kv_group` and cost KV$ energy, with a 1.2x iWuR penalty on V.
- WMEM and KV$ access energy scale with memory depth as `depth^0.78`.
- MAC, accumulator and WMEM cycles follow the output-channel tiling of the scaled core count, `ceil(D / n_cores)` columns per core against `ceil(D / 128)` in the mapping. Head and global SRAM keep their mapped cycles.

VRC fuses softmax and RMSNorm into the GEMVs, so it adds energy but no cycles. KV writes cost KV$ energy.

**GEMM costs.** `TimeloopEvaluator` maps each padded decode shape once per MAC width with fast mapper settings. It retries with loose settings when the on-chip energy exceeds twice an analytical baseline, marks a loose mapping as final so that later calls read it from the cache, and marks a shape unmappable when both passes fail. Each MAC width keeps its own cache directory. A shape that Timeloop cannot price, or whose on-chip energy exceeds ten times the tiling-aware analytical model in `core/analytical_model.py` at the same padded shape, uses that analytical model. `ops_timeloop` and `ops_fallback` count the GEMMs of all active layers that each source priced for the selected chip. The mapping constraints fix a 16-wide reduction per core, so the mapped decode cycles do not change with the MAC width, and a wider MAC array adds area without lowering TPOT.

**Estimated area and leakage.** `ScaledChipSpec.estimated_area_mm2` and `ScaledChipSpec.leakage_pJ_per_cycle` add up the component estimates that Accelergy reports for the dxe_relaxed specification: MACs, accumulators, WMEM and KV$ per byte, head SRAM per DXT and global SRAM. `core/constants.py` holds the estimates, and `tests/hw_sim/test_rdxe_costs.py` re-derives them from a mapper statistics file. The specification has no pads, I/O, vector engine or bus, so the estimates cover compute and on-chip memory only.

**Before 2026-09-15.** The backend costed every layer like the first active layer, used the query and key head dimension in place of the value head dimension, combined on-chip energy with DRAM-bound cycles, scaled area with SRAM alone, and re-mapped cached shapes on every call. rDXE results from before that date are superseded.

| Key | Meaning |
|---|---|
| `energy_per_token_uJ` | decode energy of one generated token for one user, with hops and leakage |
| `session_e_per_tok_uJ` | decode energy plus prefill energy amortized over the generated tokens |
| `ttft_ms`, `tpot_ms`, `ttft`, `tpot` | latency at 5 ns per cycle, in milliseconds and seconds |
| `total_area_mm2`, `power_W`, `mac_util_pct` | estimated area of all chips, decode power, MAC utilization |
| `n_chips`, `chip_macs`, `pipeline_depth` | ring size and MACs per chip |
| `selected_mac_per_vac`, `selected_max_chips`, `selected_wmem_KB` | grid point of the selected chip |
| `envelope_feasible` | whether the selected chip fits the area and power envelope |
| `tokens_per_second`, `inter_chip_comm_energy_uJ` | saturated ring throughput and hop energy per token |
| `ops_timeloop`, `ops_fallback` | GEMM cost sources over all active layers of the selected chip |
| `chip_pareto` | Pareto front over the chip grid |
| `hw_feasible` | False only when no configuration packs the architecture |

## Installing Timeloop

The tests run without Timeloop. Real mappings need the `timeloop-mapper` binary on `PATH`, and the `timeloopfe` front end, Accelergy and its estimation plug-ins in the same Python environment as LLMForge. `scripts/setup/install_timeloop.sh` builds all of them from source at the versions pinned by the Accelergy-Timeloop infrastructure repository, commit `6e6186f`.

```bash
scripts/setup/install_timeloop.sh deps       # apt packages, needs sudo
scripts/setup/install_timeloop.sh barvinok   # barvinok 0.41.8 with its bundled isl, installed under PREFIX
scripts/setup/install_timeloop.sh timeloop   # timeloop-mapper, timeloop-model and timeloop-metrics
scripts/setup/install_timeloop.sh python     # Accelergy, every estimation plug-in and timeloopfe
scripts/setup/install_timeloop.sh check      # prints whether the backend can run
```

`PREFIX` defaults to `/usr/local`, `PYTHON` to `.venv/bin/python` and `JOBS` to 8. Sources build under the git-ignored `third_party/timeloop-build`. The script departs from the infrastructure Dockerfile in three places. barvinok comes from the SourceForge project mirror, because the URL in the Dockerfile now refuses downloads. Ubuntu 24.04 also needs `libboost-thread-dev`. The CACTI build flags are patched with `sed`, the same edit that `cacti.patch` makes in the infrastructure repository.

The script installs every plug-in of the infrastructure image, because the substrates rely on different estimators. The NeuroSim plug-in estimates the adders, multipliers and registers of the Eyeriss, FLAT and DXE specifications, CACTI estimates SRAM and DRAM, and the Library plug-in covers the remaining components. NeuroSim is licensed for non-commercial use only, as `THIRD_PARTY_NOTICES.md` records.

Timeloop finds Accelergy through `which accelergy`. The backend adds the directory of the running Python interpreter to `PATH` when Accelergy is installed there, so a job that runs `.venv/bin/python` without activating the environment still reaches it. A mapper run that leaves no energy reference table raises an error instead of reporting zero energy. Check an installation with

```bash
python -c "from llmforge.hw.timeloop.gemm import timeloop_available; print(timeloop_available())"
```

**Validation.** On 2026-09-14 the installation reproduced mapper outputs cached by the earlier LLMForge code exactly, with identical energy, cycles, GFLOPs, utilization and energy reference tables. The checks covered two prefill GEMMs and one decode GEMM on `gemmini`, one GEMM on `flat_edge`, and one prefill and one decode GEMM on `dxe_relaxed`. The DXE references predate the change described in `specs/arch/dxe_relaxed/constraints.yaml`, which moved weights from on-chip weight memory to DRAM, so they reproduce only with the earlier constraint and 4 mapper threads. `eyeriss` has no cached reference and was checked for a successful mapping only.

Without Timeloop, constructing `HwTimeloop` raises an error, so a search never starts with every architecture marked infeasible. The rDXE backend uses any cached mapper result and prices the remaining shapes with the analytical model, which `ops_fallback` reports.

## Caches and outputs

| Path | Content |
|---|---|
| `runs/cache/timeloop/<substrate>/gemm_*` | rDXE mapper cache, one directory per padded GEMM shape |
| `runs/cache/timeloop/<substrate>/prefill/gemm_*` and `.../decode/gemm_*` | `HwTimeloop` mapper cache |
| `runs/cache/timeloop/arch_variants/` | DXE spec clones for MAC widths without a shipped spec |
| `runs/rdxe/results` and `runs/rdxe/plots` | CSVs and plots of the rDXE command-line tools |

`LLMFORGE_TIMELOOP_WORK` moves the Timeloop cache and `LLMFORGE_RUNS` moves every run output.

## Tests

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src python -m pytest tests/hw_sim
```

The tests cover the stats parsers on two cached mapper outputs, the substrate registry and its spec files, the seven-GEMM decomposition and fusion arithmetic with a fake mapper, the two-pass aggregation of `HwTimeloop`, and an end-to-end rDXE co-search on the analytical fallback. A mapper smoke test runs only when Timeloop is installed.
