"""Per-layer rDXE decode cost from Timeloop mappings of the dxe_relaxed substrate.

The dxe_relaxed specification routes weights through DRAM so that one reference chip can map layers
larger than its WMEM, as its constraints.yaml explains. The ring co-search sizes every chip so that its
layers fit in WMEM, so the default `weight_memory="wmem"` drops the DRAM level from energy and time:

  1. Energy is the dynamic energy of the on-chip levels: MACs, accumulators, WMEM, head SRAM and global
     SRAM. WMEM reads scale with the per-core WMEM depth (ScaledChipSpec.wmem_energy_scale). Leakage is
     charged per chip by the ring simulation, not per GEMM.
  2. In QK_attn and PV_attn the Weights dataspace is the K or V cache. Timeloop reads it once per query
     row, while VLINK multicast reads it once per KV group, so the reads divide by n_head / n_kv_group
     and cost KV$ energy, with the iWuR penalty on V.
  3. Cycles are those of the slowest on-chip level. MACs, accumulators and WMEM are per core and follow
     the output-channel tiling of the scaled chip, ceil(D / n_cores) columns per core against
     ceil(D / 128) on the mapped chip. Head and global SRAM are shared and keep their mapped cycles.
     VRC is pipelined with the GEMV it fuses with, so it adds energy but no cycles.
  4. A shape that Timeloop cannot map, or whose on-chip energy exceeds SANITY_ENERGY_MULT times the
     analytical model of the same padded shape, uses that analytical model. Every op records its source.

`weight_memory="dram"` keeps the mapped DRAM traffic in both energy and cycles instead.
"""

import math
from typing import Dict

from ..core.analytical_model import analytical_gemm
from ..core.constants import (
    KV_CACHE_ENERGY_PER_ACCESS_PJ, VRC_SOFTMAX_PJ_PER_ELEMENT, VRC_RMSNORM_PJ_PER_ELEMENT,
)
from ..core.scaled_arch import ScaledChipSpec
from ..core.timeloop_evaluator import (
    CORES_REF, TimeloopEvaluator, enumerate_gemm_shapes_decode, pad_for_dxe,
)

IWUR_V_PENALTY = 1.2
KV_WRITE_PENALTY = 1.06
WEIGHT_MEMORIES = ("wmem", "dram")
ATTENTION_OPS = ("QK_attn", "PV_attn")
CORE_LEVELS = ("mac", "acc_buffer", "wmem")
SHARED_LEVELS = ("head_sram", "global_sram")

# A well-mapped GEMM cannot cost more than SANITY_ENERGY_MULT times the analytical model of the same
# padded shape. The mapper already retries schedules above twice its own baseline, so this bound only
# catches a broken mapping or a broken analytical model.
SANITY_ENERGY_MULT = 10.0


def _tiling_scale(D: int, chip: ScaledChipSpec) -> float:
    """Cycles of a per-core level on `chip` relative to the mapped 128-core chip."""
    return math.ceil(D / max(1, chip.n_cores)) / math.ceil(D / CORES_REF)


def _analytical_cost(K: int, N: int, M: int, chip: ScaledChipSpec) -> Dict[str, float]:
    ar = analytical_gemm(M=M, K=K, N=N, n_cores=chip.n_cores, pe_width=chip.n_mac_per_vac)
    return dict(gemm_pJ=ar.total_energy_pJ, kv_read_pJ=0.0, cycles=float(ar.total_cycles))


def _mapped_cost(res, chip: ScaledChipSpec, op: str, share: int,
                 weight_memory: str) -> Dict[str, float]:
    """Energy and cycles of one mapped GEMM on `chip`."""
    if weight_memory == "dram":
        return dict(gemm_pJ=res.energy_pJ, kv_read_pJ=0.0, cycles=res.cycles)
    levels = {name: lv for name, lv in res.levels.items() if name != "DRAM"}
    weights = levels.get("wmem", {}).get("dataspaces", {}).get("Weights", {})
    gemm_pJ = sum(lv["dynamic_pJ"] for lv in levels.values()) - weights.get("energy_pJ", 0.0)
    kv_read_pJ = 0.0
    if op in ATTENTION_OPS:
        per_read_pJ = KV_CACHE_ENERGY_PER_ACCESS_PJ * chip.kv_energy_scale
        if op == "PV_attn":
            per_read_pJ *= IWUR_V_PENALTY
        kv_read_pJ = weights.get("total_reads", 0.0) / max(1, share) * per_read_pJ
    else:
        gemm_pJ += weights.get("energy_pJ", 0.0) * chip.wmem_energy_scale
    core = max((levels[n]["cycles"] for n in CORE_LEVELS if n in levels), default=0.0)
    shared = max((levels[n]["cycles"] for n in SHARED_LEVELS if n in levels), default=0.0)
    return dict(gemm_pJ=gemm_pJ, kv_read_pJ=kv_read_pJ,
                cycles=max(core * _tiling_scale(res.N, chip), shared))


def _evaluate_shape(evaluator: TimeloopEvaluator, op: str, K: int, N: int, M: int,
                    chip: ScaledChipSpec, share: int, weight_memory: str) -> Dict[str, float]:
    """Mapped cost of one GEMM, or the analytical model of the same padded shape."""
    K_pad, N_pad, _ = pad_for_dxe(K, N, M)
    ana = _analytical_cost(K_pad, N_pad, M, chip)
    try:
        res = evaluator.evaluate(K, N, M)
    except Exception:
        return {**ana, "source": "analytical_fallback_exception"}
    if not (math.isfinite(res.energy_pJ) and res.levels):
        return {**ana, "source": "analytical_fallback_mapper_failed"}
    cost = _mapped_cost(res, chip, op, share, weight_memory)
    mapped_pJ = cost["gemm_pJ"] + cost["kv_read_pJ"]
    if weight_memory == "wmem" and mapped_pJ > SANITY_ENERGY_MULT * ana["gemm_pJ"]:
        return {**ana, "source": "analytical_fallback_sanity", "timeloop_pJ": mapped_pJ}
    return {**cost, "source": "timeloop"}


def timeloop_layer_decode(layer_spec: dict, n_embd: int, ctx: int, chip: ScaledChipSpec,
                          evaluator: TimeloopEvaluator, n_users: int = 1,
                          weight_memory: str = "wmem") -> Dict[str, float]:
    """Energy and cycles of one layer that generates one token per user at KV context `ctx`.

    `n_users` scales the row count M of every GEMM, so the users share one pass over the weights.
    Attention GEMMs are mapped per KV group and repeat n_kv_group times.
    """
    if weight_memory not in WEIGHT_MEMORIES:
        raise ValueError(f"weight_memory must be one of {WEIGHT_MEMORIES}, got {weight_memory!r}")
    nh = layer_spec['n_head']
    nkv = layer_spec['n_kv_group']
    qk = layer_spec['n_qk_head_dim']
    vd = layer_spec['n_v_head_dim']
    share = max(1, nh // max(1, nkv))

    e_gemm_pJ = e_kv_read_pJ = cycles = 0.0
    per_op_sources = {}
    for name, K, N, M in enumerate_gemm_shapes_decode(layer_spec, n_embd, ctx):
        info = _evaluate_shape(evaluator, name, K, N, max(1, M * n_users), chip, share, weight_memory)
        per_op_sources[name] = info['source']
        repeats = nkv if name in ATTENTION_OPS else 1
        e_gemm_pJ += info['gemm_pJ'] * repeats
        e_kv_read_pJ += info['kv_read_pJ'] * repeats
        cycles += info['cycles'] * repeats

    # The new token's K and V of every group, written at KV$ speed.
    kv_writes = nkv * (qk + vd) * n_users
    e_kv_write_pJ = kv_writes * KV_CACHE_ENERGY_PER_ACCESS_PJ * chip.kv_energy_scale * KV_WRITE_PENALTY
    # Softmax over the context and the two RMSNorms, fused into the GEMVs by VRC.
    e_vrc_pJ = (ctx * nh * n_users * VRC_SOFTMAX_PJ_PER_ELEMENT
                + 2 * n_embd * n_users * VRC_RMSNORM_PJ_PER_ELEMENT)

    return dict(
        total_energy_pJ=e_gemm_pJ + e_kv_read_pJ + e_kv_write_pJ + e_vrc_pJ,
        cycles=cycles,
        gemm_energy_pJ=e_gemm_pJ,
        kv_read_energy_pJ=e_kv_read_pJ,
        kv_write_energy_pJ=e_kv_write_pJ,
        vrc_energy_pJ=e_vrc_pJ,
        per_op_sources=per_op_sources,
    )
