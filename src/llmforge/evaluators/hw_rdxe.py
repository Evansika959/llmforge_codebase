"""rDXE inner chip co-search hardware evaluator.

For each candidate architecture, llmforge.hw.rdxe.cosearch.run_rdxe_eval sweeps a grid of chip
configurations (MACs per VAC core, maximum ring size, WMEM per core), packs and simulates the ring
for each, and keeps the chip Pareto front. The front is preserved per individual under
`chip_pareto`, and the metrics of the chip selected by `select_by` are promoted to top-level keys
for the outer search.

Units. The ring simulator counts cycles at the DXE clock of 200 MHz, so one cycle is 5 ns.
Latencies are reported in milliseconds (ttft_ms, tpot_ms) and seconds (ttft, tpot).
energy_per_token_uJ is the decode energy of one generated token for one user, in microjoules.
session_e_per_tok_uJ adds the closed-form prefill energy, amortized over the generated tokens.

Standard keys (see llmforge.evaluators.base): energy_per_token_uJ, ttft_ms, tpot_ms, hw_feasible.
hw_feasible is False only when no chip configuration can pack the architecture. An architecture
whose best chip violates the area or power envelope stays hw_feasible and carries
envelope_feasible=False, so a search can constrain on envelope_feasible explicitly.
ops_timeloop and ops_fallback count the GEMMs of the selected ring that Timeloop mapped and
the GEMMs that used the analytical fallback.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

_INF = float("inf")


class HwRdxeInner:
    """rDXE inner chip co-search.

    Args:
        prefill_len, decode_len, n_users, ctx: workload knobs. ctx=None uses the co-search
            default context of 2048 tokens.
        select_by: which Pareto metric the outer NSGA inherits per arch
            (per_tok_uJ, tpot_ms, or ttft_ms).
        envelope_filter: if True, restricts selection to chips that fit the
            (area, power) edge envelope.
        area_max_mm2 / area_min_mm2 / power_max_W / power_min_W: envelope.
        verbose: per-shape and per-individual progress.
        n_workers: parallel Timeloop mapper processes during prefetch.
        weight_memory: "wmem" prices weights in on-chip WMEM for both energy and time, as the ring
            packing assumes. "dram" keeps the DRAM traffic of the Timeloop mappings instead.
    """

    def __init__(self, *, prefill_len: int = 128, decode_len: int = 32,
                 n_users: int = 1, ctx: Optional[int] = None,
                 select_by: str = "per_tok_uJ",
                 envelope_filter: bool = True,
                 area_max_mm2: float = 800.0, area_min_mm2: float = 0.0,
                 power_max_W: float = 100.0, power_min_W: float = 0.0,
                 verbose: bool = False, n_workers: int = 8, weight_memory: str = "wmem"):
        self.prefill_len = int(prefill_len)
        self.decode_len = int(decode_len)
        self.n_users = int(n_users)
        self.ctx = ctx
        self.select_by = select_by
        self.envelope_filter = bool(envelope_filter)
        self.area_max_mm2 = float(area_max_mm2)
        self.area_min_mm2 = float(area_min_mm2)
        self.power_max_W = float(power_max_W)
        self.power_min_W = float(power_min_W)
        self.verbose = bool(verbose)
        self.n_workers = int(n_workers)
        if weight_memory not in ("wmem", "dram"):
            raise ValueError(f"weight_memory must be 'wmem' or 'dram', got {weight_memory!r}")
        self.weight_memory = weight_memory

    def evaluate(self, ind_dicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        from llmforge.hw.rdxe.cosearch import run_rdxe_eval
        raw = run_rdxe_eval(
            list(ind_dicts),
            prefill_len=self.prefill_len,
            decode_len=self.decode_len,
            n_users=self.n_users,
            ctx=self.ctx,
            select_by=self.select_by,
            envelope_filter=self.envelope_filter,
            area_max_mm2=self.area_max_mm2,
            area_min_mm2=self.area_min_mm2,
            power_max_W=self.power_max_W,
            power_min_W=self.power_min_W,
            verbose=self.verbose,
            n_workers=self.n_workers,
            weight_memory=self.weight_memory,
        )
        # run_rdxe_eval already promotes the selected chip's metrics to
        # top-level. Rename `pareto_points` -> `chip_pareto` and add the
        # standard keys; keep everything else as-is.
        out = []
        for r in raw:
            r2 = dict(r)
            if "pareto_points" in r2:
                r2["chip_pareto"] = r2.pop("pareto_points")
            if "ttft_ms" not in r2:
                r2["ttft_ms"] = r2.get("ttft", _INF) * 1e3
            if "tpot_ms" not in r2:
                r2["tpot_ms"] = r2.get("tpot", _INF) * 1e3
            values = (r2.get("energy_per_token_uJ"), r2["ttft_ms"], r2["tpot_ms"])
            r2["hw_feasible"] = all(v is not None and math.isfinite(v) for v in values)
            out.append(r2)
        return out
