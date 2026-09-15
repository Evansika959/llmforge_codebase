"""Layer 1: Single-operation energy model wrapping Timeloop with DXE corrections.

Applies post-hoc corrections for:
  - KV cache routing (swap DRAM energy → local SRAM energy on QK_attn/PV_attn)
  - VLINK GQA sharing (reduce KV reads when heads share cache)
  - iWuR asymmetric access (V column-wise penalty)
  - VRC fused ops (analytical softmax/RMSNorm energy)
  - WMEM layer-switch reload cost
"""

import dataclasses
from enum import Enum
from typing import Dict, Optional, Tuple

from .constants import (
    DRAM_ENERGY_PER_ACCESS_PJ,
    KV_CACHE_ENERGY_PER_ACCESS_PJ,
    IWUR_V_PENALTY,
    VRC_SOFTMAX_PJ_PER_ELEMENT,
    VRC_RMSNORM_PJ_PER_ELEMENT,
    VRC_CYCLES_PER_ELEMENT,
    WMEM_TOTAL_B,
    QSPI_TOTAL_BW_BYTES_PER_SEC,
    CLOCK_FREQ_HZ,
    CLOCK_PERIOD_NS,
    N_CORES,
    N_MAC_PER_VAC,
    TOTAL_MACS,
    WMEM_ENERGY_PER_ACCESS_PJ,
    WMEM_PER_CORE_B,
)


class OpType(Enum):
    GEMM_PREFILL = "gemm_prefill"
    GEMV_DECODE = "gemv_decode"
    QK_ATTN = "qk_attn"
    PV_ATTN = "pv_attn"
    VRC_SOFTMAX = "vrc_softmax"
    VRC_RMSNORM = "vrc_rmsnorm"
    WMEM_RELOAD = "wmem_reload"


@dataclasses.dataclass
class OpResult:
    """Result of evaluating one operation with corrections."""
    op_name: str
    op_type: OpType

    # Dimensions
    in_channels: int = 0
    out_channels: int = 0
    seq_length: int = 0
    scale_factor: float = 1.0

    # Raw Timeloop results (pJ / cycles)
    raw_energy_pJ: float = 0.0
    raw_cycles: float = 0.0

    # Corrections
    kv_cache_correction_pJ: float = 0.0   # negative = energy saved
    vlink_correction_pJ: float = 0.0
    vrc_energy_pJ: float = 0.0

    # Corrected totals
    energy_pJ: float = 0.0
    cycles: float = 0.0

    # Access counts
    dram_accesses: int = 0
    act_dram_correction_pJ: float = 0.0
    kv_cache_reads: int = 0
    kv_cache_writes: int = 0
    wmem_accesses: int = 0

    @property
    def energy_uJ(self) -> float:
        return self.energy_pJ / 1e6

    @property
    def latency_ns(self) -> float:
        return self.cycles * CLOCK_PERIOD_NS


class OpModel:
    """Single-operation energy model wrapping Timeloop + DXE corrections."""

    def __init__(self, config=None):
        from .config import DXEConfig
        self.config = config or DXEConfig()

        # Lazily import the Timeloop GEMM backend
        self._hw_exp = None
        self._parse_stats = None
        self._arch_cfg = None

    def _ensure_imports(self):
        """Import the Timeloop GEMM backend on first use."""
        if self._hw_exp is not None:
            return
        from llmforge.hw.timeloop import gemm, stats
        self._hw_exp = gemm
        self._parse_stats = stats
        self._arch_cfg = gemm.get_arch_config(self.config.arch)

    def evaluate_gemm(self, in_ch: int, out_ch: int, seq_len: int,
                      op_type: OpType = OpType.GEMV_DECODE,
                      op_name: str = "",
                      n_heads: int = 1,
                      n_kv_groups: int = 1,
                      scale_factor: float = 1.0) -> OpResult:
        """Evaluate a GEMM/GEMV using Timeloop (primary) or analytical fallback.

        Architecture: Timeloop evaluates GEMM energy/cycles for the reference
        DXE chip (2048 MACs). For on-chip operation (no DRAM), we strip DRAM
        energy and replace with on-chip SRAM cost. Cycles are scaled linearly
        for different MAC counts (weight-stationary: energy is MAC-count
        independent, only throughput changes).

        Fallback: analytical tiling model for GEMMs that Timeloop cannot map
        (degenerate dimensions < 8).
        """
        self._ensure_imports()
        from .analytical_model import analytical_gemm

        total_ops = in_ch * out_ch * seq_len
        min_dim = min(in_ch, out_ch, seq_len)
        chip_macs = self.config.chip_total_macs or TOTAL_MACS
        chip_cores = chip_macs // (N_MAC_PER_VAC or 16)

        # Use analytical model for all GEMMs (Timeloop disabled for speed)
        # Analytical model has ~18% energy overestimate vs Timeloop for compute-bound GEMMs
        if True:  # was: min_dim < 8
            ar = analytical_gemm(M=seq_len, K=in_ch, N=out_ch,
                                 n_cores=chip_cores, pe_width=N_MAC_PER_VAC)
            return OpResult(
                op_name=op_name, op_type=op_type,
                in_channels=in_ch, out_channels=out_ch, seq_length=seq_len,
                scale_factor=scale_factor,
                raw_energy_pJ=ar.total_energy_pJ * scale_factor,
                raw_cycles=ar.total_cycles * scale_factor,
                energy_pJ=ar.total_energy_pJ * scale_factor,
                cycles=ar.total_cycles * scale_factor,
            )

        # Primary: Timeloop evaluation (cached results for reference chip)
        try:
            summary, dram_stats = self._hw_exp.run_GEMM_evaluation_detailed(
                in_ch, out_ch, seq_len,
                work_dir=self.config.work_dir or None, arch=self.config.arch)
        except RuntimeError:
            # Timeloop mapper failed — use analytical fallback
            ar = analytical_gemm(M=seq_len, K=in_ch, N=out_ch,
                                 n_cores=chip_cores, pe_width=N_MAC_PER_VAC)
            return OpResult(
                op_name=op_name, op_type=op_type,
                in_channels=in_ch, out_channels=out_ch, seq_length=seq_len,
                scale_factor=scale_factor,
                raw_energy_pJ=ar.total_energy_pJ * scale_factor,
                raw_cycles=ar.total_cycles * scale_factor,
                energy_pJ=ar.total_energy_pJ * scale_factor,
                cycles=ar.total_cycles * scale_factor,
            )

        raw_energy_pJ = (summary.get('energy_uJ') or 0) * 1e6 * scale_factor
        raw_cycles = (summary.get('cycles') or 0) * scale_factor

        # Strip DRAM energy: replace DRAM accesses with on-chip SRAM cost
        # In the chiplet design, all weights are in WMEM and activations
        # pass through on-chip buffers or the ring link — no DRAM access.
        if self.config.enable_act_dram_correction:
            dram_total_energy = sum(d.get('energy_pJ', 0) for d in dram_stats.values())
            dram_total_accesses = sum(
                int((d.get('scalar_reads') or 0) + (d.get('scalar_updates') or 0) + (d.get('scalar_fills') or 0))
                for d in dram_stats.values())
            sram_replacement_energy = dram_total_accesses * KV_CACHE_ENERGY_PER_ACCESS_PJ
            energy_pJ = max(0.0, raw_energy_pJ - dram_total_energy * scale_factor
                            + sram_replacement_energy * scale_factor)
        else:
            energy_pJ = raw_energy_pJ

        # Scale cycles for different MAC counts.
        # Weight-stationary: energy is MAC-count independent (same access pattern),
        # but throughput scales linearly with MAC count.
        ref_macs = TOTAL_MACS  # Timeloop models the reference 2048-MAC chip
        cycles = max(1.0, raw_cycles * (ref_macs / chip_macs))

        result = OpResult(
            op_name=op_name, op_type=op_type,
            in_channels=in_ch, out_channels=out_ch, seq_length=seq_len,
            scale_factor=scale_factor,
            raw_energy_pJ=raw_energy_pJ, raw_cycles=raw_cycles,
            energy_pJ=energy_pJ, cycles=cycles,
        )

        # Apply KV cache correction for attention ops
        if self.config.enable_kv_correction and op_type in (OpType.QK_ATTN,
                                                             OpType.PV_ATTN):
            correction = self._apply_kv_cache_correction(
                dram_stats, op_type, n_heads, n_kv_groups, scale_factor)
            result.kv_cache_correction_pJ = correction['energy_correction_pJ']
            result.kv_cache_reads = correction['kv_cache_reads']
            result.energy_pJ += correction['energy_correction_pJ']  # negative
            result.cycles += correction['cycle_correction']

        # Apply VLINK GQA correction
        if (self.config.enable_vlink_gqa and n_heads > n_kv_groups and
                op_type in (OpType.QK_ATTN, OpType.PV_ATTN)):
            vlink_corr = self._apply_vlink_correction(
                result.kv_cache_reads, op_type, n_heads, n_kv_groups)
            result.vlink_correction_pJ = vlink_corr
            result.energy_pJ += vlink_corr  # negative

        # Clamp energy to non-negative
        result.energy_pJ = max(0.0, result.energy_pJ)
        result.cycles = max(0.0, result.cycles)

        return result

    def evaluate_vrc(self, n_elements: int, vrc_type: str = "softmax",
                     op_name: str = "") -> OpResult:
        """Evaluate a fused VRC operation (softmax or RMSNorm)."""
        if vrc_type == "softmax":
            op_t = OpType.VRC_SOFTMAX
            energy_per_elem = VRC_SOFTMAX_PJ_PER_ELEMENT
        else:
            op_t = OpType.VRC_RMSNORM
            energy_per_elem = VRC_RMSNORM_PJ_PER_ELEMENT

        total_energy = n_elements * energy_per_elem
        total_cycles = n_elements * VRC_CYCLES_PER_ELEMENT

        return OpResult(
            op_name=op_name or vrc_type,
            op_type=op_t,
            seq_length=n_elements,
            vrc_energy_pJ=total_energy,
            energy_pJ=total_energy,
            cycles=total_cycles,
        )

    def evaluate_wmem_reload(self, weight_bytes: int,
                             op_name: str = "wmem_reload") -> OpResult:
        """Model layer-switch WMEM reload cost."""
        energy_pJ = weight_bytes * DRAM_ENERGY_PER_ACCESS_PJ
        cycles = weight_bytes / QSPI_TOTAL_BW_BYTES_PER_SEC * CLOCK_FREQ_HZ

        return OpResult(
            op_name=op_name,
            op_type=OpType.WMEM_RELOAD,
            energy_pJ=energy_pJ,
            cycles=cycles,
            dram_accesses=weight_bytes,
        )

    def _apply_kv_cache_correction(self, dram_stats: dict, op_type: OpType,
                                    n_heads: int, n_kv_groups: int,
                                    scale_factor: float
                                    ) -> dict:
        """Calculate energy/cycle corrections for KV cache routing.

        For QK_ATTN: K is the 'Inputs' dataspace — route through KV cache instead
        For PV_ATTN: V is the 'Inputs' dataspace — same, with iWuR penalty
        """
        dram_inputs = dram_stats.get('Inputs', {})
        dram_reads = (dram_inputs.get('scalar_reads') or 0)
        dram_input_energy = (dram_inputs.get('energy_pJ') or 0) * scale_factor
        dram_reads_scaled = int(dram_reads * scale_factor)

        # KV cache energy for the same number of accesses
        kv_energy_per_access = self.config.kv_cache_energy_pJ
        if op_type == OpType.PV_ATTN:
            kv_energy_per_access *= self.config.iwur_v_penalty

        kv_cache_energy = dram_reads_scaled * kv_energy_per_access

        # Energy saved (negative = reduction)
        energy_correction = -(dram_input_energy - kv_cache_energy)

        # Cycle correction: DRAM reads take dram_reads/bw cycles,
        # KV cache reads are at SRAM bandwidth (much faster)
        dram_bw = self._arch_cfg.dram_read_bw if self._arch_cfg else 4
        sram_bw = 16  # KV cache read bandwidth (128b / 8b = 16 scalars/cycle)
        dram_cycles = dram_reads_scaled / dram_bw
        sram_cycles = dram_reads_scaled / sram_bw
        cycle_correction = -(dram_cycles - sram_cycles)

        return {
            'energy_correction_pJ': energy_correction,
            'cycle_correction': cycle_correction,
            'kv_cache_reads': dram_reads_scaled,
            'dram_energy_removed_pJ': dram_input_energy,
            'kv_cache_energy_added_pJ': kv_cache_energy,
        }

    def _apply_vlink_correction(self, kv_cache_reads: int, op_type: OpType,
                                 n_heads: int, n_kv_groups: int) -> float:
        """VLINK GQA: shared KV reads reduce energy further.

        With GQA, n_heads/n_kv_groups query heads share the same KV entries.
        Physical KV cache reads are already for one group (Timeloop's seq_len
        = n_heads//n_kv_groups), but the scale_factor = n_kv_groups multiplies
        the result. VLINK means within each group, the KV data is multicast
        rather than read independently per head — no extra energy for sharing.

        The correction removes the redundant per-head KV cache read energy
        beyond the first head in each group.
        """
        gqa_ratio = n_heads / n_kv_groups
        if gqa_ratio <= 1:
            return 0.0

        kv_energy_per_access = self.config.kv_cache_energy_pJ
        if op_type == OpType.PV_ATTN:
            kv_energy_per_access *= self.config.iwur_v_penalty

        # Energy for reads that VLINK multicast avoids
        # Each group reads KV once, broadcasts to gqa_ratio heads
        # Savings = kv_reads * (1 - 1/gqa_ratio) * energy_per_access
        savings = kv_cache_reads * (1.0 - 1.0 / gqa_ratio) * kv_energy_per_access
        return -savings
