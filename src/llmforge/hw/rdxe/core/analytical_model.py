"""Tiling-aware analytical energy and latency model for rDXE GEMM/GEMV operations.

Models the DXE weight-stationary systolic array with explicit tiling,
SRAM access counting, and utilization penalties.

DXE Architecture:
  - PE array: n_dxt × n_vac × n_mac_per_vac (default 8×16×16 = 2048 MACs)
  - Weight-stationary: weights pinned in WMEM, streamed once per tile
  - WMEM: per-core SRAM, 128-bit (16 byte) read port, row-major
  - Accumulator: per-core register file for partial sums
  - Activation buffer: shared head SRAM for input/output vectors

GEMM: C[M,N] = A[M,K] × B[K,N]
  - Weight matrix B[K,N] stored in WMEM (weight-stationary)
  - Input activation A[M,K] streamed from head SRAM
  - Output activation C[M,N] accumulated in accumulator, written to head SRAM

Tiling for the PE array (n_mac_per_vac = PE_WIDTH):
  - K dimension: tiled by PE_WIDTH (weight vector width)
  - N dimension: tiled across n_cores (each core holds a column slice of B)
  - M dimension: streamed one row at a time (batch/sequence dimension)

Energy sources:
  1. MAC compute: padded_ops × energy_per_mac (includes wasted MACs on padding)
  2. WMEM reads: weight tile loaded once per K-tile, reused across all M rows
  3. Accumulator writes/reads: partial sum accumulation across K tiles
  4. Head SRAM (activation I/O): input read once total, output write once
  5. Utilization penalty: reflected in padded_ops and accumulator access counts

Cycle model:
  - K-tiles outer, M-rows inner, N-cols innermost (weight-stationary optimal)
  - Double-buffering: next tile's weight load overlaps with current tile's compute
  - First tile load is always exposed; subsequent stalls only if memory-bound
"""

import math
from dataclasses import dataclass

from .constants import (
    N_MAC_PER_VAC,
    N_CORES,
    CLOCK_PERIOD_NS,
    WMEM_ENERGY_PER_ACCESS_PJ,
    HEAD_SRAM_ENERGY_PER_ACCESS_PJ,
)

# Accumulator (register file) energy — cheaper than SRAM
ACC_ENERGY_PER_ACCESS_PJ = 0.05  # register-file access, from the Timeloop energy tables
MAC_ENERGY_PJ = 0.2  # INT8 MAC, from the Timeloop energy reference table

# WMEM read granularity: 128-bit = 16 bytes per read
WMEM_READ_WIDTH_BYTES = 16


@dataclass
class GEMMAnalyticalResult:
    """Analytical model output for one GEMM/GEMV."""
    # Dimensions
    M: int  # batch/sequence (rows of A and C)
    K: int  # inner dimension (cols of A, rows of B)
    N: int  # output dimension (cols of B and C)

    # Compute
    total_ops: int
    useful_ops: int  # after utilization penalty
    utilization: float  # 0-1

    # Cycles
    compute_cycles: float
    memory_stall_cycles: float
    total_cycles: float

    # Energy (pJ)
    mac_energy_pJ: float
    wmem_energy_pJ: float
    acc_energy_pJ: float
    act_input_energy_pJ: float
    act_output_energy_pJ: float
    total_energy_pJ: float

    # Latency
    latency_ns: float

    # Access counts
    wmem_reads: int
    acc_accesses: int
    act_input_reads: int
    act_output_writes: int

    # Regime
    is_compute_bound: bool


def analytical_gemm(M: int, K: int, N: int,
                    n_cores: int = N_CORES,
                    pe_width: int = N_MAC_PER_VAC) -> GEMMAnalyticalResult:
    """Tiling-aware analytical model for GEMM C[M,N] = A[M,K] × B[K,N].

    Weight-stationary dataflow:
      B[K,N] is distributed across cores. Each core holds a column slice.
      A[M,K] is broadcast/streamed to all cores.
      C[M,N] is accumulated locally, then gathered.

    Tiling:
      - N tiled across n_cores: each core handles ceil(N/n_cores) columns
      - K tiled by pe_width: each MAC lane processes pe_width elements
      - M streamed: one row at a time (or small batch)
    """
    if M == 0 or K == 0 or N == 0:
        return GEMMAnalyticalResult(
            M=M, K=K, N=N, total_ops=0, useful_ops=0, utilization=0,
            compute_cycles=0, memory_stall_cycles=0, total_cycles=0,
            mac_energy_pJ=0, wmem_energy_pJ=0, acc_energy_pJ=0,
            act_input_energy_pJ=0, act_output_energy_pJ=0, total_energy_pJ=0,
            latency_ns=0, wmem_reads=0, acc_accesses=0,
            act_input_reads=0, act_output_writes=0, is_compute_bound=True,
        )

    total_ops = M * K * N

    # ── Tiling ──
    # N dimension across cores
    n_cols_per_core = math.ceil(N / n_cores)
    n_active_cores = math.ceil(N / n_cols_per_core) if n_cols_per_core > 0 else 0
    # K dimension across PE width (MAC lanes)
    k_tiles = math.ceil(K / pe_width)
    k_util = K / (k_tiles * pe_width) if k_tiles > 0 else 1.0

    # Core utilization: fraction of cores actually doing useful work
    core_util = n_active_cores / n_cores if n_cores > 0 else 1.0

    # Per-core column utilization
    col_util = N / (n_active_cores * n_cols_per_core) if n_active_cores * n_cols_per_core > 0 else 1.0

    # Overall utilization
    utilization = k_util * core_util * col_util

    # Useful ops (accounting for wasted compute on padding)
    padded_ops = M * (k_tiles * pe_width) * (n_active_cores * n_cols_per_core)
    useful_ops = total_ops

    # ── Cycles ──
    # Weight-stationary tiling order: K-tiles outer, M-rows inner, N-cols innermost.
    # Per K-tile:
    #   1. Load weights from WMEM: n_cols_per_core × pe_width bytes per core
    #   2. Stream M rows of activations, computing pe_width MACs per cycle per core
    #      Each core processes n_cols_per_core columns sequentially: M × n_cols_per_core cycles
    # Total compute per K-tile = M × n_cols_per_core cycles (all active cores in parallel).
    # Total compute = k_tiles × M × n_cols_per_core.

    compute_per_tile = M * n_cols_per_core  # cycles of useful compute per K-tile
    compute_cycles = k_tiles * compute_per_tile

    # Weight loading latency per K-tile:
    # Per core: n_cols_per_core × pe_width bytes, read at WMEM_READ_WIDTH_BYTES per cycle.
    wmem_reads_per_tile = math.ceil(n_cols_per_core * pe_width / WMEM_READ_WIDTH_BYTES)

    # First K-tile: weight load cannot overlap with compute (nothing to compute yet).
    # Subsequent K-tiles: weight load can overlap with previous tile's compute
    #   if double-buffering is available. Model both regimes:
    #   - If compute_per_tile >= wmem_reads_per_tile: memory-bound stalls are hidden.
    #   - Otherwise: stalls = (wmem_reads_per_tile - compute_per_tile) per transition.
    # First tile load is always exposed.
    first_tile_load = wmem_reads_per_tile
    if k_tiles > 1:
        # Per subsequent tile: max(compute, load) determines throughput.
        # Exposed stall = max(0, load_time - compute_time) per tile transition.
        stall_per_transition = max(0, wmem_reads_per_tile - compute_per_tile)
        memory_stall_cycles = first_tile_load + (k_tiles - 1) * stall_per_transition
    else:
        memory_stall_cycles = first_tile_load

    total_cycles = compute_cycles + memory_stall_cycles

    # ── SRAM Access Counts ──

    # WMEM reads: weight matrix B[K,N] read once (weight-stationary, pinned)
    # Weight-stationary: weights loaded once into PE registers, reused across all M rows.
    # Per core: n_cols_per_core columns × K rows of INT8 weights.
    # Read granularity: WMEM_READ_WIDTH_BYTES (16 bytes) per read.
    # Weights are read per-column in K-chunks of pe_width. Each chunk is ceil(pe_width/16) reads.
    # Total reads per core = n_cols_per_core × k_tiles × ceil(pe_width / WMEM_READ_WIDTH_BYTES)
    wmem_reads_per_col_per_ktile = math.ceil(pe_width / WMEM_READ_WIDTH_BYTES)
    wmem_reads_per_core = n_cols_per_core * k_tiles * wmem_reads_per_col_per_ktile
    wmem_reads_total = wmem_reads_per_core * n_active_cores

    # Accumulator accesses: partial sum read-modify-write per K-tile (except first).
    # The hardware accumulates for every column assigned to a core (including padded ones).
    # Per output position (padded): (k_tiles - 1) reads + k_tiles writes.
    n_output_positions_padded = M * n_active_cores * n_cols_per_core
    acc_reads = n_output_positions_padded * max(0, k_tiles - 1)
    acc_writes = n_output_positions_padded * k_tiles
    acc_accesses = acc_reads + acc_writes

    # Activation input reads: A[M,K] streamed from head SRAM.
    # With K-outer tiling: for each K-tile, read M × pe_width bytes (broadcast to all cores).
    # Total bytes from head SRAM = M × K (each activation byte read exactly once).
    act_input_reads = M * K

    # Activation output writes: C[M,N] written to head SRAM
    # Each output element written once after all K-tiles accumulated
    act_output_writes = M * N

    # ── Energy ──
    # MAC energy: hardware executes padded ops (including wasted MACs on padding).
    # The PE array fires every cycle regardless of whether the output is useful.
    mac_energy = padded_ops * MAC_ENERGY_PJ
    wmem_energy = wmem_reads_total * WMEM_READ_WIDTH_BYTES * WMEM_ENERGY_PER_ACCESS_PJ
    acc_energy = acc_accesses * ACC_ENERGY_PER_ACCESS_PJ
    act_input_energy = act_input_reads * HEAD_SRAM_ENERGY_PER_ACCESS_PJ
    act_output_energy = act_output_writes * HEAD_SRAM_ENERGY_PER_ACCESS_PJ
    total_energy = mac_energy + wmem_energy + acc_energy + act_input_energy + act_output_energy
    latency_ns = total_cycles * CLOCK_PERIOD_NS

    # Determine compute-bound vs memory-bound regime
    is_compute_bound = compute_per_tile >= wmem_reads_per_tile

    return GEMMAnalyticalResult(
        M=M, K=K, N=N,
        total_ops=total_ops, useful_ops=useful_ops, utilization=utilization,
        compute_cycles=compute_cycles, memory_stall_cycles=memory_stall_cycles,
        total_cycles=total_cycles,
        mac_energy_pJ=mac_energy,
        wmem_energy_pJ=wmem_energy,
        acc_energy_pJ=acc_energy,
        act_input_energy_pJ=act_input_energy,
        act_output_energy_pJ=act_output_energy,
        total_energy_pJ=total_energy,
        latency_ns=latency_ns,
        wmem_reads=wmem_reads_total,
        acc_accesses=acc_accesses,
        act_input_reads=act_input_reads,
        act_output_writes=act_output_writes,
        is_compute_bound=is_compute_bound,
    )
