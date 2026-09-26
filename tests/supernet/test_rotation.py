"""Regression tests for the KV-head rotation alignment, CPU-only.

These exist because an invariance check was written twice in a form that measured nothing:
it computed rotations with `groups_for(n_kv, n_kv)`, which makes every group a singleton, so
every rotation was the identity and the test reported a perfect score. A valid invariance
check must compute rotations on a POOLED grouping and apply them WITHOUT merging.
"""
import numpy as np
import pytest
import torch

from llmforge.supernet.config import SPECS
from llmforge.supernet.eval.pilot_rotation_merge import apply_pair_rotation, groups_for, pair_rotations


def test_singleton_grouping_yields_no_rotation():
    """The exact trap: groups_for(n, n) is all singletons, so any 'invariance' measured
    against it is vacuous. Pinned so nobody rebuilds the same broken check."""
    spec = SPECS["qwen3-4b"]
    grp = groups_for(spec.n_kv, spec.n_kv)
    assert all(len(g) == 1 for g in grp)
    K = torch.randn(spec.n_kv, 64, spec.head_dim, dtype=torch.float64)
    ang = pair_rotations(K, grp, spec.head_dim // 2)
    assert torch.allclose(ang, torch.zeros_like(ang)), "singleton groups must give identity"


def test_pooled_grouping_yields_real_rotation():
    """A genuinely pooled grouping must produce non-trivial angles -- otherwise the
    alignment is silently doing nothing."""
    spec = SPECS["qwen3-4b"]
    torch.manual_seed(0)
    K = torch.randn(spec.n_kv, 64, spec.head_dim, dtype=torch.float64)
    ang = pair_rotations(K, groups_for(spec.n_kv, 2), spec.head_dim // 2)
    assert ang.abs().max() > 1e-3, "pooled groups must produce non-identity rotations"


def test_rotation_is_orthogonal_preserves_inner_products():
    """The whole method rests on this: rotating a key and its queries by the same angle
    leaves q.k unchanged. If it ever stops holding, alignment is not free."""
    torch.manual_seed(0)
    hd = 128
    half = hd // 2
    q = torch.randn(hd, 512, dtype=torch.float64)
    k = torch.randn(hd, 512, dtype=torch.float64)
    ang = torch.randn(half, dtype=torch.float64)
    before = (q * k).sum(0)
    after = (apply_pair_rotation(q, ang, half) * apply_pair_rotation(k, ang, half)).sum(0)
    assert torch.allclose(before, after, atol=1e-10)


def test_rotation_within_group_aligns_heads():
    """Aligned heads must agree more than unaligned ones -- the premise of merging."""
    spec = SPECS["qwen3-4b"]
    torch.manual_seed(0)
    hd, half = spec.head_dim, spec.head_dim // 2
    # two heads that are rotations of one another: alignment should recover that
    shared = torch.randn(64, hd, dtype=torch.float64)
    phi = torch.rand(half, dtype=torch.float64) * 2 - 1
    rotated = apply_pair_rotation(shared.T, phi, half).T
    K = torch.zeros(spec.n_kv, 64, hd, dtype=torch.float64)
    K[0], K[1] = shared, rotated
    ang = pair_rotations(K, [[0, 1]] + [[g] for g in range(2, spec.n_kv)], half)
    a0 = apply_pair_rotation(K[0].T, ang[0], half).T
    a1 = apply_pair_rotation(K[1].T, ang[1], half).T
    disagree_before = (K[0] - K[1]).norm()
    disagree_after = (a0 - a1).norm()
    assert disagree_after < disagree_before, "alignment must reduce within-group disagreement"
