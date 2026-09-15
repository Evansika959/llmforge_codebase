"""Pareto utilities for minimization: non-dominated filtering and exact hypervolume."""
from __future__ import annotations

from typing import List, Sequence

import numpy as np


def dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def non_dominated(points: Sequence[Sequence[float]]) -> List[int]:
    """Indices of points that no other point dominates. Duplicates are all kept."""
    P = [list(map(float, p)) for p in points]
    return [i for i, p in enumerate(P) if not any(dominates(q, p) for j, q in enumerate(P) if j != i)]


def hypervolume(points: Sequence[Sequence[float]], ref: Sequence[float]) -> float:
    """Dominated hypervolume with respect to `ref`. Exact for two or three objectives."""
    ref = np.asarray(ref, float)
    P = np.array([p for p in np.asarray(points, float).reshape(-1, len(ref)) if np.all(p < ref)])
    if len(P) == 0:
        return 0.0
    if P.shape[1] == 1:
        return float(ref[0] - P[:, 0].min())
    P = P[np.lexsort(tuple(P[:, i] for i in range(P.shape[1] - 1, -1, -1)))]
    if P.shape[1] == 2:
        hv, prev = 0.0, ref[1]
        for x, y in P:
            if y < prev:
                hv += (ref[0] - x) * (prev - y)
                prev = y
        return float(hv)
    if P.shape[1] == 3:
        hv, zs = 0.0, sorted({p[2] for p in P})
        for i, z in enumerate(zs):
            nxt = zs[i + 1] if i + 1 < len(zs) else ref[2]
            sl = np.array([p[:2] for p in P if p[2] <= z])
            hv += hypervolume(sl, ref[:2]) * max(0.0, nxt - z)
        return float(hv)
    raise ValueError("hypervolume supports up to three objectives")


def normalized_hypervolume(points: Sequence[Sequence[float]], lo: Sequence[float],
                           hi: Sequence[float], margin: float = 0.1) -> float:
    """Hypervolume after scaling each objective so `lo` maps to 0 and `hi` to 1.

    The reference point sits at 1 + margin on every axis, and the result is divided by the volume
    of that box, so a value of 1 would mean a point at the ideal corner.
    """
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    span = np.where(hi > lo, hi - lo, 1.0)
    P = (np.asarray(points, float).reshape(-1, len(lo)) - lo) / span
    ref = np.full(len(lo), 1.0 + margin)
    return hypervolume(P, ref) / float(np.prod(ref))
