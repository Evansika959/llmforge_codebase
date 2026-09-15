"""KV cache state tracker for autoregressive decode simulation."""

import dataclasses
from typing import Dict

from .constants import (
    N_CORES,
    KV_CACHE_DEPTH,
    KV_CACHE_ENERGY_PER_ACCESS_PJ,
    MAX_CONTEXT_LENGTH,
    MAX_NUM_USERS,
    IWUR_V_PENALTY,
)


@dataclasses.dataclass
class KVCacheState:
    """Tracks KV cache occupancy across decode steps.

    Models the per-core KV cache (8KB, 512 depth) across all layers and users.
    Each token adds one K row + one V column per layer per KV group.
    """

    n_cores: int = N_CORES
    depth_per_core: int = KV_CACHE_DEPTH
    max_context: int = MAX_CONTEXT_LENGTH
    max_users: int = MAX_NUM_USERS

    # occupancy[layer_idx][user_idx] = number of tokens stored
    _occupancy: Dict[int, Dict[int, int]] = dataclasses.field(
        default_factory=dict)

    def add_token(self, layer_idx: int, user_idx: int = 0) -> int:
        """Record one new K,V pair written to cache. Returns new occupancy."""
        if layer_idx not in self._occupancy:
            self._occupancy[layer_idx] = {}
        occ = self._occupancy[layer_idx].get(user_idx, 0) + 1
        self._occupancy[layer_idx][user_idx] = occ
        return occ

    def add_tokens(self, layer_idx: int, n_tokens: int,
                   user_idx: int = 0) -> int:
        """Add multiple tokens at once (prefill). Returns new occupancy."""
        if layer_idx not in self._occupancy:
            self._occupancy[layer_idx] = {}
        occ = self._occupancy[layer_idx].get(user_idx, 0) + n_tokens
        self._occupancy[layer_idx][user_idx] = occ
        return occ

    def get_context(self, layer_idx: int, user_idx: int = 0) -> int:
        """Current number of cached tokens = context length for attention."""
        return self._occupancy.get(layer_idx, {}).get(user_idx, 0)

    def check_capacity(self, layer_idx: int, user_idx: int = 0) -> bool:
        """Check if KV cache has room for another token."""
        return self.get_context(layer_idx, user_idx) < self.max_context

    @staticmethod
    def kv_bytes_per_token(n_kv_groups: int, qk_dim: int, v_dim: int) -> int:
        """Bytes of K + V stored per token per layer."""
        return n_kv_groups * (qk_dim + v_dim)

    @staticmethod
    def kv_write_energy_pJ(n_kv_groups: int, qk_dim: int, v_dim: int,
                           kv_energy_pJ: float = KV_CACHE_ENERGY_PER_ACCESS_PJ
                           ) -> float:
        """Energy (pJ) for writing one token's K,V into cache.

        K write: n_kv_groups * qk_dim scalars (row-major, single bank per write)
        V write: n_kv_groups * v_dim scalars (column-major, iWuR penalty)
        """
        k_write_energy = n_kv_groups * qk_dim * kv_energy_pJ
        v_write_energy = n_kv_groups * v_dim * kv_energy_pJ * IWUR_V_PENALTY
        return k_write_energy + v_write_energy

    def total_bytes_used(self, n_kv_groups: int, qk_dim: int,
                         v_dim: int) -> int:
        """Total KV cache bytes across all layers and users."""
        bpt = self.kv_bytes_per_token(n_kv_groups, qk_dim, v_dim)
        total = 0
        for layer_occ in self._occupancy.values():
            for occ in layer_occ.values():
                total += occ * bpt
        return total

    def reset(self):
        """Clear all cached tokens (new sequence)."""
        self._occupancy.clear()
