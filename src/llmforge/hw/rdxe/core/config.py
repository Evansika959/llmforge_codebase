"""DXE simulator configuration."""

import dataclasses

import yaml

from llmforge import paths

from .constants import (
    KV_CACHE_ENERGY_PER_ACCESS_PJ,
    DRAM_ENERGY_PER_ACCESS_PJ,
    IWUR_V_PENALTY,
)


@dataclasses.dataclass
class DXEConfig:
    """Configuration for DXE simulator."""

    # Architecture variant (Timeloop backend)
    arch: str = "dxe_relaxed"
    # Timeloop mapper cache. Empty uses the substrate cache under llmforge.paths.TIMELOOP_WORK.
    work_dir: str = ""

    # Feature toggles
    enable_kv_correction: bool = True
    enable_vlink_gqa: bool = True
    enable_vrc: bool = True
    enable_wmem_reload: bool = True
    enable_fusion: bool = True
    enable_act_dram_correction: bool = True  # Remove DRAM activation energy (on-chip activations)
    chip_wmem_bytes: int = 0  # Per-chip WMEM capacity (0 = use reference 3MB)
    chip_total_macs: int = 0  # Per-chip MAC count (0 = use reference 2048)

    # Energy overrides (pJ per scalar access)
    kv_cache_energy_pJ: float = KV_CACHE_ENERGY_PER_ACCESS_PJ
    dram_energy_pJ: float = DRAM_ENERGY_PER_ACCESS_PJ
    iwur_v_penalty: float = IWUR_V_PENALTY

    # Simulation parameters
    prefill_length: int = 0
    decode_length: int = 128
    user_idx: int = 0

    # Multi-chip
    n_chips: int = 1
    layer_assignment: str = "round_robin"  # round_robin | balanced | single_layer

    # Output
    output_dir: str = ""   # CSVs land here, default <runs>/rdxe/results
    plot_dir: str = ""     # PNGs land here, default <runs>/rdxe/plots
    save_traces: bool = True
    plot: bool = True

    def __post_init__(self):
        if not self.output_dir:
            self.output_dir = str(paths.RUNS / 'rdxe' / 'results')
        if not self.plot_dir:
            self.plot_dir = str(paths.RUNS / 'rdxe' / 'plots')

    @classmethod
    def from_yaml(cls, path: str) -> 'DXEConfig':
        with open(path) as f:
            d = yaml.safe_load(f)

        features = d.get('features', {})
        energy = d.get('energy_overrides', {})
        sim = d.get('simulation', {})
        multi = d.get('multi_chip', {})
        output = d.get('output', {})

        return cls(
            arch=d.get('arch', 'dxe_relaxed'),
            work_dir=d.get('work_dir', ''),
            enable_kv_correction=features.get('kv_correction', True),
            enable_vlink_gqa=features.get('vlink_gqa', True),
            enable_vrc=features.get('vrc', True),
            enable_wmem_reload=features.get('wmem_reload', True),
            enable_fusion=features.get('fusion', True),
            kv_cache_energy_pJ=energy.get('kv_cache_pJ',
                                          KV_CACHE_ENERGY_PER_ACCESS_PJ),
            dram_energy_pJ=energy.get('dram_pJ', DRAM_ENERGY_PER_ACCESS_PJ),
            iwur_v_penalty=energy.get('iwur_v_penalty', IWUR_V_PENALTY),
            prefill_length=sim.get('prefill_length', 0),
            decode_length=sim.get('decode_length', 128),
            user_idx=sim.get('user_idx', 0),
            n_chips=multi.get('n_chips', 1),
            layer_assignment=multi.get('layer_assignment', 'round_robin'),
            output_dir=output.get('dir', ''),
            plot_dir=output.get('plot_dir', ''),
            save_traces=output.get('save_traces', True),
            plot=output.get('plot', True),
        )
