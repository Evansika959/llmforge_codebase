"""Scaled DXE architecture generator.

Scales the DXE chip dimensions (MAC array, WMEM, KV cache, buffers)
to fit a given model layer's weights entirely on-chip. Used for
ring-topology analysis where each chip holds exactly one layer.

Scaling philosophy:
  - The DXE is a weight-stationary design: WMEM must hold ALL weights
    for the layer(s) assigned to that chip.
  - MAC array scales to maintain compute throughput proportional to
    the weight capacity (more weights = more compute needed per token).
  - KV cache scales with context length and head dimensions.
  - Spatial dimensions (DXT, VAC, MAC lane) scale in powers of 2.
  - Energy scales with CACTI models: SRAM energy ~ depth^0.78 * width.

Reference DXE configuration:
  8 DXT × 16 VAC × 16 MAC = 2048 MACs
  WMEM: 24KB/core (3MB total), KV$: 8KB/core (1MB total)
  Target model: d_model=512, 1 layer ≈ 1.8MB weights
"""

import dataclasses
import math
from typing import Dict, List, Optional

from .constants import (
    N_DXT, N_VAC_PER_DXT, N_MAC_PER_VAC, N_CORES, TOTAL_MACS,
    WMEM_PER_CORE_B, KV_CACHE_PER_CORE_B, WMEM_TOTAL_B, KV_CACHE_TOTAL_B,
    CLOCK_PERIOD_NS, CLOCK_FREQ_HZ,
    AREA_MAC_UM2, AREA_ACC_UM2, AREA_SRAM_UM2_PER_B, AREA_HEAD_SRAM_UM2, AREA_GLOBAL_SRAM_UM2,
    LEAK_SRAM_PJ_PER_B_CYCLE, LEAK_ACC_PJ_PER_CYCLE, LEAK_HEAD_SRAM_PJ_PER_CYCLE,
    LEAK_GLOBAL_SRAM_PJ_PER_CYCLE,
)


@dataclasses.dataclass
class ScaledChipSpec:
    """Specification for a scaled DXE chip."""
    name: str

    # Spatial dimensions
    n_dxt: int = N_DXT
    n_vac_per_dxt: int = N_VAC_PER_DXT
    n_mac_per_vac: int = N_MAC_PER_VAC

    # Derived
    @property
    def n_cores(self) -> int:
        return self.n_dxt * self.n_vac_per_dxt

    @property
    def total_macs(self) -> int:
        return self.n_cores * self.n_mac_per_vac

    # Per-core storage (bytes)
    wmem_per_core_B: int = WMEM_PER_CORE_B
    kv_cache_per_core_B: int = KV_CACHE_PER_CORE_B

    @property
    def wmem_total_B(self) -> int:
        return self.wmem_per_core_B * self.n_cores

    @property
    def kv_cache_total_B(self) -> int:
        return self.kv_cache_per_core_B * self.n_cores

    @property
    def total_sram_B(self) -> int:
        return self.wmem_total_B + self.kv_cache_total_B

    # Clock (same technology, same frequency)
    clock_ns: float = CLOCK_PERIOD_NS

    @property
    def peak_gops(self) -> float:
        return self.total_macs * (1e9 / self.clock_ns) / 1e9

    @property
    def estimated_area_mm2(self) -> float:
        """Area from the Accelergy component estimates in constants.py.

        MACs and accumulators scale with the core count and the MACs per core, WMEM and KV$ with
        their capacity, and head SRAM with the DXT count.
        """
        per_core_um2 = (self.n_mac_per_vac * AREA_MAC_UM2 + AREA_ACC_UM2
                        + (self.wmem_per_core_B + self.kv_cache_per_core_B) * AREA_SRAM_UM2_PER_B)
        um2 = self.n_cores * per_core_um2 + self.n_dxt * AREA_HEAD_SRAM_UM2 + AREA_GLOBAL_SRAM_UM2
        return um2 / 1e6

    @property
    def leakage_pJ_per_cycle(self) -> float:
        """Leakage of the whole chip in one clock cycle, from the same component estimates."""
        per_core = (LEAK_ACC_PJ_PER_CYCLE
                    + (self.wmem_per_core_B + self.kv_cache_per_core_B) * LEAK_SRAM_PJ_PER_B_CYCLE)
        return (self.n_cores * per_core + self.n_dxt * LEAK_HEAD_SRAM_PJ_PER_CYCLE
                + LEAK_GLOBAL_SRAM_PJ_PER_CYCLE)

    # Energy scaling factor for SRAM accesses
    # CACTI: energy ~ depth^0.78 for same width
    @property
    def wmem_energy_scale(self) -> float:
        ref_depth = 1536  # reference WMEM depth
        new_depth = self.wmem_per_core_B * 8 // 128  # depth = bytes * 8 / width_bits
        return (new_depth / ref_depth) ** 0.78

    @property
    def kv_energy_scale(self) -> float:
        ref_depth = 512  # reference KV cache depth
        new_depth = self.kv_cache_per_core_B * 8 // 128
        return (new_depth / ref_depth) ** 0.78

    def summary(self) -> str:
        lines = [
            f"=== {self.name} ===",
            f"  MACs: {self.n_dxt} DXT × {self.n_vac_per_dxt} VAC × "
            f"{self.n_mac_per_vac} MAC = {self.total_macs}",
            f"  Cores: {self.n_cores}",
            f"  WMEM: {self.wmem_per_core_B//1024}KB/core = "
            f"{self.wmem_total_B/1024/1024:.2f}MB total",
            f"  KV$:  {self.kv_cache_per_core_B//1024}KB/core = "
            f"{self.kv_cache_total_B/1024/1024:.2f}MB total",
            f"  Total SRAM: {self.total_sram_B/1024/1024:.2f}MB",
            f"  Peak: {self.peak_gops:.1f} GOPS @ {1e3/self.clock_ns:.0f}MHz",
            f"  Area: ~{self.estimated_area_mm2:.1f} mm²",
            f"  WMEM energy scale: {self.wmem_energy_scale:.2f}x",
        ]
        return "\n".join(lines)


# Reference DXE configuration
DXE_REFERENCE = ScaledChipSpec(name="DXE-ref")


def layer_weight_bytes(layer_spec: dict, n_embd: int) -> int:
    """Compute weight bytes for one layer (INT8)."""
    nh = layer_spec['n_head']
    nkv = layer_spec['n_kv_group']
    qk = layer_spec['n_qk_head_dim']
    vd = layer_spec['n_v_head_dim']
    mlp = layer_spec['mlp_size']
    attn = layer_spec.get('attention_variant', 'infinite')

    if attn == 'infinite':
        return (n_embd * qk * (nh + nkv) +   # QK_gen
                n_embd * vd * nkv +            # V_gen
                vd * nh * n_embd +              # ATTN_proj
                n_embd * mlp +                  # MLP_FC1
                mlp * n_embd)                   # MLP_FC2
    else:
        return n_embd * mlp + mlp * n_embd


def kv_bytes_per_token_per_layer(layer_spec: dict) -> int:
    """KV cache bytes per token for one layer."""
    nkv = layer_spec['n_kv_group']
    qk = layer_spec['n_qk_head_dim']
    vd = layer_spec['n_v_head_dim']
    attn = layer_spec.get('attention_variant', 'infinite')
    if attn != 'infinite':
        return 0
    return nkv * (qk + vd)


def scale_chip_for_layer(layer_spec: dict, n_embd: int,
                         max_context: int = 256,
                         n_users: int = 1,
                         overhead_factor: float = 1.15,
                         name: str = "") -> ScaledChipSpec:
    """Design a scaled DXE chip that fits one layer's weights on-chip.

    Scaling rules:
      1. WMEM total must hold all layer weights (with 15% overhead for alignment)
      2. KV cache must hold max_context * n_users tokens
      3. MAC count scales proportionally to sqrt(weight_count) for balanced compute
      4. Spatial dims (DXT, VAC) scale in powers of 2
      5. MAC lane stays at 16 (reduction dimension, tied to datapath width)

    Args:
        layer_spec: Layer architecture specification
        n_embd: Model embedding dimension
        max_context: Maximum context length per user
        n_users: Number of concurrent users
        overhead_factor: SRAM overhead for alignment/metadata (default 15%)
        name: Chip name label
    """
    wb = layer_weight_bytes(layer_spec, n_embd)
    kv_per_tok = kv_bytes_per_token_per_layer(layer_spec)

    # Required WMEM: layer weights + overhead
    required_wmem = int(wb * overhead_factor)

    # Required KV cache: context * users * kv_bytes_per_token
    required_kv = kv_per_tok * max_context * n_users if kv_per_tok > 0 else 0

    # Start from reference and scale up
    # MAC lane = 16 (fixed, tied to 128-bit datapath)
    n_mac = N_MAC_PER_VAC  # 16, always

    # Scale cores (n_dxt * n_vac) to accommodate WMEM
    # Reference: 128 cores × 24KB = 3MB
    # Distribute weights evenly across cores
    # Minimum WMEM per core: 24KB (reference), scale up if needed

    # Strategy: keep per-core WMEM manageable (24-96KB), scale core count
    # Per-core WMEM sizes to consider: 24KB, 48KB, 96KB, 192KB
    wmem_options = [24*1024, 48*1024, 96*1024, 192*1024]

    best = None
    for wmem_per_core in wmem_options:
        # How many cores needed?
        n_cores_wmem = max(1, math.ceil(required_wmem / wmem_per_core))

        # KV cache per core
        if required_kv > 0:
            kv_per_core = max(8*1024, math.ceil(required_kv / n_cores_wmem))
            # Round up to power of 2 KB
            kv_kb = max(8, 1 << math.ceil(math.log2(kv_per_core / 1024)))
            kv_per_core = kv_kb * 1024
        else:
            kv_per_core = 8 * 1024

        # Factor n_cores into DXT × VAC (both powers of 2)
        # Prefer more VACs than DXTs (VAC is the inner loop)
        n_dxt, n_vac = _factor_cores(n_cores_wmem)
        n_cores = n_dxt * n_vac

        spec = ScaledChipSpec(
            name=name or f"DXE-scaled-{n_dxt}x{n_vac}",
            n_dxt=n_dxt,
            n_vac_per_dxt=n_vac,
            n_mac_per_vac=n_mac,
            wmem_per_core_B=wmem_per_core,
            kv_cache_per_core_B=kv_per_core,
        )

        # Check: does it fit?
        if spec.wmem_total_B >= required_wmem:
            if best is None or spec.total_sram_B < best.total_sram_B:
                best = spec

    if best is None:
        # Fallback: largest option
        n_cores_needed = math.ceil(required_wmem / wmem_options[-1])
        n_dxt, n_vac = _factor_cores(n_cores_needed)
        best = ScaledChipSpec(
            name=name or f"DXE-scaled-{n_dxt}x{n_vac}",
            n_dxt=n_dxt,
            n_vac_per_dxt=n_vac,
            n_mac_per_vac=n_mac,
            wmem_per_core_B=wmem_options[-1],
            kv_cache_per_core_B=max(8*1024, math.ceil(required_kv / (n_dxt * n_vac))),
        )

    return best


def _factor_cores(n_cores: int) -> tuple:
    """Factor n_cores into (n_dxt, n_vac) where both are powers of 2.

    Prefer n_vac >= n_dxt (inner parallelism).
    Minimum: n_dxt=1, n_vac=1.
    """
    # Round up to next power of 2
    n = max(1, 1 << math.ceil(math.log2(max(n_cores, 1))))

    # Factor: try to keep n_vac >= n_dxt
    # n = n_dxt * n_vac
    best_dxt, best_vac = 1, n
    for dxt_exp in range(int(math.log2(n)) + 1):
        dxt = 1 << dxt_exp
        if n % dxt == 0:
            vac = n // dxt
            # VAC per DXT should be power of 2
            if vac & (vac - 1) == 0:
                if vac >= dxt:  # prefer more VACs
                    best_dxt, best_vac = dxt, vac

    return best_dxt, best_vac


def scale_ring_for_model(model_spec: dict,
                         max_context: int = 256,
                         n_users: int = 1,
                         ) -> List[ScaledChipSpec]:
    """Design a ring of scaled DXE chips for a full model.

    One chip per layer. Each chip is independently scaled to fit
    its layer's weights.

    Returns list of ScaledChipSpec, one per layer.
    """
    n_embd = model_spec['n_embd']
    layers = model_spec['layers']

    chips = []
    for li, layer in enumerate(layers):
        spec = scale_chip_for_layer(
            layer, n_embd,
            max_context=max_context,
            n_users=n_users,
            name=f"chip_{li}",
        )
        chips.append(spec)

    return chips


# -----------------------------------------------------------------------
# Edge LLM model definitions
# -----------------------------------------------------------------------

EDGE_MODELS = {
    'mt5_small': {
        'name': 'MT5-small',
        'n_embd': 512,
        'layers': [
            {'n_head': 6, 'n_kv_group': 6, 'n_qk_head_dim': 64,
             'n_v_head_dim': 64, 'mlp_size': 1024, 'n_cproj': 1,
             'attention_variant': 'infinite'}
        ] * 8,
    },
    'smollm_135m': {
        'name': 'SmolLM-135M',
        'n_embd': 576,
        'layers': [
            {'n_head': 9, 'n_kv_group': 3, 'n_qk_head_dim': 64,
             'n_v_head_dim': 64, 'mlp_size': 1536, 'n_cproj': 1,
             'attention_variant': 'infinite'}
        ] * 30,
    },
    'smollm_360m': {
        'name': 'SmolLM-360M',
        'n_embd': 960,
        'layers': [
            {'n_head': 15, 'n_kv_group': 5, 'n_qk_head_dim': 64,
             'n_v_head_dim': 64, 'mlp_size': 2560, 'n_cproj': 1,
             'attention_variant': 'infinite'}
        ] * 32,
    },
    'mobilellm_125m': {
        'name': 'MobileLLM-125M',
        'n_embd': 576,
        'layers': [
            {'n_head': 9, 'n_kv_group': 9, 'n_qk_head_dim': 64,
             'n_v_head_dim': 64, 'mlp_size': 1536, 'n_cproj': 1,
             'attention_variant': 'infinite'}
        ] * 24,
    },
    'mobilellm_350m': {
        'name': 'MobileLLM-350M',
        'n_embd': 960,
        'layers': [
            {'n_head': 15, 'n_kv_group': 15, 'n_qk_head_dim': 64,
             'n_v_head_dim': 64, 'mlp_size': 2560, 'n_cproj': 1,
             'attention_variant': 'infinite'}
        ] * 32,
    },
    'qwen25_05b': {
        'name': 'Qwen2.5-0.5B',
        'n_embd': 896,
        'layers': [
            {'n_head': 14, 'n_kv_group': 2, 'n_qk_head_dim': 64,
             'n_v_head_dim': 64, 'mlp_size': 4864, 'n_cproj': 1,
             'attention_variant': 'infinite'}
        ] * 24,
    },
    'tinyllama_1b': {
        'name': 'TinyLlama-1.1B',
        'n_embd': 2048,
        'layers': [
            {'n_head': 32, 'n_kv_group': 4, 'n_qk_head_dim': 64,
             'n_v_head_dim': 64, 'mlp_size': 5632, 'n_cproj': 1,
             'attention_variant': 'infinite'}
        ] * 22,
    },
    'llama32_1b': {
        'name': 'Llama-3.2-1B',
        'n_embd': 2048,
        'layers': [
            {'n_head': 32, 'n_kv_group': 8, 'n_qk_head_dim': 64,
             'n_v_head_dim': 64, 'mlp_size': 8192, 'n_cproj': 1,
             'attention_variant': 'infinite'}
        ] * 16,
    },
}
