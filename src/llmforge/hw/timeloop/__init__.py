"""Timeloop simulator targets.

    gemm.py   per-layer seven-GEMM evaluation with operator fusion, on published and custom substrates
    stats.py  parsers for timeloop-mapper.stats.txt
    specs/    architecture, constraint, mapper, and problem specifications for every substrate

Nothing here imports timeloopfe at module import time. Only running the mapper needs it.
"""
