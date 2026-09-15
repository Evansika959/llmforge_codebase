"""Supernet data and run roots, derived from llmforge.paths.

    ROOT    repository root
    PACKED  packed token buckets for supernet training   ($LLMFORGE_DATA/packed)
    RAW     raw jsonl or parquet training sources          ($LLMFORGE_DATA/raw)
    RUNS    supernet checkpoints and supernet-side results ($LLMFORGE_RUNS/supernet)
"""
from ..paths import DATA, ROOT
from ..paths import RUNS as _RUNS

PACKED = DATA / "packed"
RAW = DATA / "raw"
RUNS = _RUNS / "supernet"


def describe() -> str:
    return "\n".join(
        f"  {n:8} {p}{'' if p.exists() else '   [missing]'}"
        for n, p in (("ROOT", ROOT), ("PACKED", PACKED), ("RAW", RAW), ("RUNS", RUNS)))
