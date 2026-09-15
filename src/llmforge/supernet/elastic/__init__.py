"""Elastic slicing of a Qwen3- or Llama-family model (see families.py).

    attention.py  per-layer qk/v dims and query-head selection
    mlp.py        nested MLP width
    patch.py      install the forwards, and select one sub-network
    permute.py    reorder the units a prefix slice selects (exact full-width symmetry)
    sampler.py    sandwich-rule configuration sampling

Import the public API from this package rather than a submodule, so the split stays an
implementation detail.
"""
from .attention import (N_GRID, grid_index, add_elastic_temperature, build_pair_order, pair_order_for, head_index,
                        qk_gather_index, set_elastic_backend)
from .mlp import elastic_mlp_forward
from .patch import add_kv_alignment, disable_elastic, enable_elastic, set_elastic_config
from .permute import (KNOBS, apply_permutation, identity_perms, permuted,
                      random_perms, units)

__all__ = ["build_pair_order", "pair_order_for", "qk_gather_index", "head_index", "set_elastic_backend",
           "add_elastic_temperature", "N_GRID", "grid_index", "elastic_mlp_forward",
           "enable_elastic", "disable_elastic", "set_elastic_config",
           "KNOBS", "apply_permutation", "permuted", "random_perms", "identity_perms",
           "units"]
