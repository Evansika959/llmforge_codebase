"""DXE hardware constants of the reference configuration.

Values follow the reference DXE design and its Timeloop specs in llmforge/hw/timeloop/specs/arch/DXE.
"""

# ---------------------------------------------------------------------------
# Energy per scalar (1 byte) access, from the Timeloop energy reference tables
# ---------------------------------------------------------------------------
DRAM_ENERGY_PER_ACCESS_PJ = 64.0       # 512 pJ / 8-wide block
KV_CACHE_ENERGY_PER_ACCESS_PJ = 0.24   # same SRAM class as WMEM (smartbuffer_SRAM)
WMEM_ENERGY_PER_ACCESS_PJ = 0.243      # 3.89 pJ / 16 scalars per 128b vector
HEAD_SRAM_ENERGY_PER_ACCESS_PJ = 0.14  # from Timeloop stats
GLOBAL_SRAM_ENERGY_PER_ACCESS_PJ = 0.17

# iWuR: V stored column-by-column requires multi-bank reads (16 banks)
# Slight penalty vs row-major K access
IWUR_V_PENALTY = 1.2  # 20% extra energy for column-wise V reads

# ---------------------------------------------------------------------------
# Compute topology
# ---------------------------------------------------------------------------
N_DXT = 8                           # DXT tiles
N_VAC_PER_DXT = 16                  # VAC cores per DXT
N_MAC_PER_VAC = 16                  # MACs per VAC core
N_CORES = N_DXT * N_VAC_PER_DXT    # 128
TOTAL_MACS = N_CORES * N_MAC_PER_VAC  # 2048

# ---------------------------------------------------------------------------
# Clock
# ---------------------------------------------------------------------------
CLOCK_PERIOD_NS = 5.0    # 200 MHz nominal
CLOCK_FREQ_HZ = 200e6

# ---------------------------------------------------------------------------
# Per-core storage (bytes)
# ---------------------------------------------------------------------------
WMEM_PER_CORE_B = 24 * 1024        # 24 KB
KV_CACHE_PER_CORE_B = 8 * 1024     # 8 KB
WMEM_TOTAL_B = WMEM_PER_CORE_B * N_CORES     # 3,145,728 = 3 MB
KV_CACHE_TOTAL_B = KV_CACHE_PER_CORE_B * N_CORES  # 1,048,576 = 1 MB

# ---------------------------------------------------------------------------
# KV cache geometry
# ---------------------------------------------------------------------------
KV_CACHE_DEPTH = 512            # entries per core
KV_CACHE_WIDTH_BITS = 128       # bits per entry (2 x 64b macros)
KV_CACHE_BANKS = 16             # internal banks, one per MAC
MAX_CONTEXT_LENGTH = 256        # per user
MAX_CONTEXT_LENGTH_GQA = 512    # with GQA
MAX_NUM_USERS = 4

# ---------------------------------------------------------------------------
# VRC (fused Softmax / RMSNorm), analytical estimates
# ---------------------------------------------------------------------------
VRC_SOFTMAX_PJ_PER_ELEMENT = 0.5   # from design power analysis
VRC_RMSNORM_PJ_PER_ELEMENT = 0.3
VRC_CYCLES_PER_ELEMENT = 1         # pipelined with GEMV output

# ---------------------------------------------------------------------------
# Weight loading interface (layer switch)
# ---------------------------------------------------------------------------
QSPI_CHANNELS = 16
QSPI_RATE_MBPS = 800  # per channel
QSPI_TOTAL_BW_BYTES_PER_SEC = QSPI_CHANNELS * QSPI_RATE_MBPS * 1e6 / 8

# ---------------------------------------------------------------------------
# Inter-chip communication (ring topology)
# ---------------------------------------------------------------------------
INTER_CHIP_ENERGY_PJ_PER_BIT = 1.0   # on-package wire estimate
INTER_CHIP_BW_BYTES_PER_CYCLE = 16   # 128b link @ 1 cycle
INTER_CHIP_LATENCY_CYCLES = 10       # link setup + serialization

# ---------------------------------------------------------------------------
# Area and leakage of a scaled chip
# ---------------------------------------------------------------------------
# Component estimates that Accelergy reports for the dxe_relaxed Timeloop specification, read from the
# SPECS blocks of its mapper statistics. Leakage is per clock cycle, the level's leakage energy divided
# by the mapped cycles and the level's instances. SRAM scales per byte from the 24 KB WMEM macro, and KV$
# uses the same SRAM class. The specification has no pads, I/O, vector engine or bus, so an estimated
# area covers the MAC arrays, accumulators and on-chip memories only.
AREA_MAC_UM2 = 135.685             # one INT8 MAC
AREA_ACC_UM2 = 5474.79             # accumulator of one VAC core
AREA_SRAM_UM2_PER_B = 0.711316     # WMEM or KV$, per byte
AREA_HEAD_SRAM_UM2 = 1453.80       # head SRAM of one DXT
AREA_GLOBAL_SRAM_UM2 = 4824.14     # global SRAM of one chip
LEAK_SRAM_PJ_PER_B_CYCLE = 5.9070e-7
LEAK_ACC_PJ_PER_CYCLE = 1.7709e-3        # per VAC core
LEAK_HEAD_SRAM_PJ_PER_CYCLE = 1.7442e-3  # per DXT
LEAK_GLOBAL_SRAM_PJ_PER_CYCLE = 5.3914e-3
