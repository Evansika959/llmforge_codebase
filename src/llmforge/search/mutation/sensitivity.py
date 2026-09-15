"""Sensitivity scoring + softmax-sampled mutation.

For a parent arch x:
  1. Enumerate strict 1-step neighborhood (see neighbors.py).
  2. Surrogate-score the batch → predicted Δval_loss per candidate.
  3. Score = −Δval_loss (positive = improvement). No σ confidence weight,
     no direction bias — the score is the raw predicted improvement.
  4. Convert scores into a sampling weight via softmax(score / T).
  5. Sample 1-3 neighbors and apply them sequentially as the offspring.
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from llmforge.search.hetero_space import HeteroSearchSpace, Individual
from .neighbors import GeneRef, enumerate_neighbors


def compute_sensitivity_scores(
    parent: Individual,
    sw_eval,
    space: HeteroSearchSpace,
    *,
    return_candidates: bool = False,
) -> Tuple[List[float], List[GeneRef], Optional[List[Individual]]]:
    """Score every 1-step neighbor of `parent` by predicted Δval_loss.

    score = −Δval_loss, so "higher = better improvement".
    Raw signal only — no σ weighting, no direction bias.

    Returns (scores, refs, candidates_or_None).
    """
    candidates_refs = enumerate_neighbors(parent, space)
    if not candidates_refs:
        return [], [], ([] if return_candidates else None)

    candidates = [c for c, _ in candidates_refs]
    refs = [r for _, r in candidates_refs]

    # One batched surrogate call for parent + all candidates.
    inds_to_score = [parent] + candidates
    mus, _ = sw_eval.evaluate(inds_to_score)
    mu_parent = float(mus[0])
    mu_cands = np.asarray(mus[1:], dtype=np.float64)

    delta = mu_cands - mu_parent          # negative = improvement
    scores_arr = -delta                    # positive = improvement

    scores_list = [float(s) for s in scores_arr]
    if return_candidates:
        return scores_list, refs, candidates
    return scores_list, refs, None


def _softmax_sample(scores: List[float], temperature: float,
                    n: int, rng: random.Random) -> List[int]:
    """Sample n distinct indices, weighted by exp(score / T)."""
    if not scores:
        return []
    arr = np.asarray(scores, dtype=np.float64)
    if temperature <= 0.0:
        # greedy: top-n by score
        return list(np.argsort(-arr)[:n])
    # Numerically stable softmax
    z = arr / max(temperature, 1e-8)
    z -= z.max()
    p = np.exp(z)
    p_sum = p.sum()
    if not np.isfinite(p_sum) or p_sum <= 0:
        return rng.sample(range(len(arr)), min(n, len(arr)))
    p /= p_sum
    n = min(n, len(arr))
    # Without replacement via numpy
    return list(np.random.choice(len(arr), size=n, replace=False, p=p))


def mutate_sensitivity(
    parent: Individual,
    sw_eval,
    space: HeteroSearchSpace,
    *,
    n_mutations: int = 1,
    temperature: float = 0.5,
    random_fraction: float = 0.10,
    legacy_mutation_rate: float = 0.3,
    rng: Optional[random.Random] = None,
) -> Tuple[Individual, List[GeneRef]]:
    """Produce a mutated offspring from `parent` using surrogate guidance.

    With probability `random_fraction`, falls back to the legacy random
    `space.mutate` so the search keeps an unfiltered exploration arm.

    `n_mutations` neighbors are sampled (softmax over sensitivity scores)
    and applied sequentially. Sequential application means later moves
    see the partial-mutation state; this is conservative because the
    sensitivity was computed at the original parent.
    """
    rng = rng or random.Random()

    if rng.random() < random_fraction:
        # Unfiltered exploration arm — preserves the old behaviour
        return space.mutate(parent, legacy_mutation_rate), []

    scores, refs, candidates = compute_sensitivity_scores(
        parent, sw_eval, space, return_candidates=True)

    if not scores or not candidates:
        # Defensive: nothing to mutate (e.g. all genes at boundary).
        return space.mutate(parent, legacy_mutation_rate), []

    picks = _softmax_sample(scores, temperature, n_mutations, rng)
    applied_refs: List[GeneRef] = []

    # Compose the picked edges onto an evolving child. Each pick is applied
    # GENE-PRECISELY: we copy only the field(s)/bit the edge actually
    # changes, never the whole layer or whole mask. Copying whole
    # layers/masks (the prior implementation) silently reverted earlier
    # picks that touched the same layer or other mask bits, so "n_mutations
    # edits" sometimes collapsed to fewer effective edits.
    #
    # Remaining approximation (intentional, documented): the sensitivity
    # scores were computed once at the original parent, so picks 2..n are
    # judged against the parent's geometry, not the partially-mutated child.
    # That is staleness, not clobbering — picks no longer cancel each other.
    current = candidates[picks[0]]
    applied_refs.append(refs[picks[0]])

    # Variant-flip field sets, by direction. to_thin_attn resets the dims;
    # to_identity only flips the categorical field.
    _VARIANT_FIELDS = {
        "to_thin_attn": ("attention_variant", "n_head",
                         "n_qk_head_dim", "n_v_head_dim"),
        "to_identity":  ("attention_variant",),
    }

    for idx in picks[1:]:
        ref = refs[idx]
        cand = candidates[idx]
        # Deep-copy globals incl. the mask list so per-bit edits below do
        # not alias `current`'s structures.
        merged = {
            "globals": {k: (list(v) if isinstance(v, list) else v)
                        for k, v in current["globals"].items()},
            "layers": [dict(li) for li in current["layers"]],
        }
        if ref.scope == "global":
            merged["globals"][ref.gene] = cand["globals"][ref.gene]
        elif ref.scope == "layer":
            merged["layers"][ref.index][ref.gene] = \
                cand["layers"][ref.index][ref.gene]
        elif ref.scope == "layer_variant":
            src = cand["layers"][ref.index]
            for k in _VARIANT_FIELDS.get(ref.direction, ("attention_variant",)):
                if k in src:
                    merged["layers"][ref.index][k] = src[k]
        elif ref.scope == "layer_mask":
            # Flip only this single mask bit (do NOT overwrite the whole
            # mask — that would revert other picks' enable/disable flips).
            merged["globals"]["layer_mask"][ref.index] = \
                cand["globals"]["layer_mask"][ref.index]
            if ref.direction == "enable":
                # `enable` lands the woken layer in the floor config; the
                # slot was inactive in the parent so no earlier pick edited
                # it, making a whole-layer copy safe here.
                merged["layers"][ref.index] = dict(cand["layers"][ref.index])
        current = space.repair(merged)
        applied_refs.append(ref)

    return current, applied_refs
