"""Evaluator interfaces shared by the search dispatcher and every backend.

A software evaluator scores model quality. A hardware evaluator measures or predicts the cost of
running an architecture on one target. The dispatcher always merges the chosen hardware backend
with the analytic backend, so parameter counts and KV-cache sizes are present in every record.

Individuals are plain dicts with two keys, `globals` and `layers` (see llmforge.search.individual).

Standard hardware keys. A backend fills the keys it can produce and omits the rest. Objectives
and constraints refer to metrics by these names.

    energy_per_token_uJ   energy per generated token, microjoules
    ttft_ms               time to first token, milliseconds
    tpot_ms               time per output token after the first, milliseconds
    hw_feasible           False when the backend cannot evaluate the architecture, for example a
                          non-uniform architecture on the device predictor

Backend-specific keys keep their own names and state their units in the backend docstring.
"""
from __future__ import annotations

from typing import Any, Dict, List, Protocol, Tuple

STANDARD_HW_KEYS = ("energy_per_token_uJ", "ttft_ms", "tpot_ms", "hw_feasible")


class SwEvaluator(Protocol):
    def evaluate(self, inds: List[Dict[str, Any]]) -> Tuple[List[float], List[float]]:
        """Return (mu, sigma) loss predictions aligned to `inds`.

        Deterministic backends return sigma = [0.0] * len(inds).
        """
        ...


class HwEvaluator(Protocol):
    def evaluate(self, inds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return one metrics dict per individual, aligned to `inds`."""
        ...


def merge_hw_dicts(primary: List[Dict[str, Any]],
                   secondary: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Combine two aligned metric lists. Keys in `primary` win on collision."""
    if len(primary) != len(secondary):
        raise ValueError(f"merge_hw_dicts: length mismatch ({len(primary)} vs {len(secondary)})")
    return [{**sec, **pri} for pri, sec in zip(primary, secondary)]
