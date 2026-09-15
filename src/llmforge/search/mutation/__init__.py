"""Surrogate-informed mutation operators for NSGA cosearch.

Strict graph-distance neighborhood: identity-attention and disabled-layer
states are reachable only through small steps, not random flips.
"""
from .neighbors import enumerate_neighbors, GeneRef, IDENTITY_DEAD_GENES
from .sensitivity import (
    compute_sensitivity_scores,
    mutate_sensitivity,
)

__all__ = [
    "enumerate_neighbors",
    "GeneRef",
    "IDENTITY_DEAD_GENES",
    "compute_sensitivity_scores",
    "mutate_sensitivity",
]
