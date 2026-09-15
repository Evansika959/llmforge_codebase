"""Analytic metrics, merged into every record whatever the hardware target.

    params_M           weights including embeddings, counted once when tied, millions
    params             alias of params_M kept for older constraint configs
    nonembed_params_M  weights excluding embeddings, millions
    flops_per_token    matmul FLOPs per token over seq_len tokens
    kv_cache_MB        KV cache for seq_len tokens at 2 bytes per scalar, megabytes
    decode_macs_M      multiply-accumulates per generated token at context seq_len, LM head
                       included, millions
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..search.individual import Individual


class HwAnalytic:
    def __init__(self, seq_len: int = 1024):
        self.seq_len = int(seq_len)
        self.settings = {"backend": "analytic", "seq_len": self.seq_len}

    def evaluate(self, inds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for d in inds:
            ind = d if isinstance(d, Individual) else Individual.from_dict(d)
            params = ind.estimate_params()
            kv = ind.estimate_kv_cache_size(seq_len=self.seq_len)
            out.append({
                "params_M": params / 1e6,
                "params": params / 1e6,
                "nonembed_params_M": (params - ind.embedding_params()) / 1e6,
                "flops_per_token": ind.estimate_flops(seq_len=self.seq_len) / self.seq_len,
                "kv_cache_MB": kv * 2 / 1e6,
                "decode_macs_M": ind.decode_macs(self.seq_len) / 1e6,
            })
        return out
