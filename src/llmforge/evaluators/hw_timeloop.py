"""Timeloop hardware evaluator: costs every individual on one published or custom substrate.

Each individual goes through llmforge.hw.timeloop.gemm in up to two passes:

    prefill  every GEMM at the prompt length, with block_size = prefill_len
    decode   projections at one token and attention against a context of decode_len tokens,
             so the pass costs one generated token

Units. Timeloop counts cycles at a 1 GHz reference clock, so one cycle is one nanosecond.
ttft = prefill cycles / 1e9 seconds and tpot = decode cycles of one token / 1e9 seconds.
ttft_ms and tpot_ms state the same in milliseconds. Energies are microjoules.

Standard keys (see llmforge.evaluators.base):
    energy_per_token_uJ   decode energy of one generated token
    ttft_ms, tpot_ms      prefill latency and per-token decode latency
    hw_feasible           False when the mapper failed on a GEMM of the individual

session_e_per_tok_uJ is (prefill energy + decode_len x decode energy per token) divided by
prefill_len + decode_len, the session metric the ZEUS backend also reports. Earlier versions of
this evaluator reported that session value under energy_per_token_uJ.

With prefill_len or decode_len set to 0 the evaluator runs one prefill pass at each individual's
own block_size, and energy_per_token_uJ is then the energy per prompt token.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, List, Optional

from llmforge import paths

# Substrates exposed at the CLI -> ARCH_CONFIGS key in llmforge.hw.timeloop.gemm
SUBSTRATE_MAP = {
    "eyeriss": "eyeriss",
    "simba": "simba",
    "simba_edge": "simba_edge",
    "gemmini": "gemmini",
    "gemmini_16nm": "gemmini_16nm",
    "flat_edge": "flat_edge",
    # dxe (strict) has rigid mapper constraints; many search-space GEMM
    # shapes (e.g. n_head x n_qk_head_dim values that don't tile into the
    # PE array) can't be mapped -> mapper exhausts candidates -> exception.
    # dxe_relaxed has the same chip but loosened mapper constraints; this
    # is what the rDXE inner search uses (llmforge.hw.rdxe.cosearch).
    "dxe": "dxe",
    "dxe_relaxed": "dxe_relaxed",
    "dxe_relaxed_m32": "dxe_relaxed_m32",
    "dxe_relaxed_m64": "dxe_relaxed_m64",
}

_INF = float("inf")
_SUMMED_KEYS = ("total_ops", "total_memory_accesses",
                "fusion_saved_energy_uJ", "fusion_saved_cycles")


def _failed(reason: str) -> Dict[str, Any]:
    return {"energy_uJ": _INF, "cycles": _INF,
            "energy_per_token_uJ": _INF, "session_e_per_tok_uJ": _INF,
            "ttft": _INF, "tpot": _INF, "ttft_ms": _INF, "tpot_ms": _INF,
            "hw_feasible": False, "timeloop_error": reason[:500]}


def _finite(*values) -> bool:
    return all(v is not None and math.isfinite(v) for v in values)


class HwTimeloop:
    """Run Timeloop on a fixed substrate. Two-pass (prefill+decode) when
    both lengths are positive, single-pass otherwise.

    Args:
        substrate: key of SUBSTRATE_MAP.
        prefill_len, decode_len: workload of the two passes.
        fused: subtract DRAM traffic that operator fusion keeps on chip.
        work_dir: root of the mapper cache, default llmforge.paths.TIMELOOP_WORK. Results land in
            <work_dir>/<substrate>/prefill and <work_dir>/<substrate>/decode.
        require_timeloop: raise at construction when timeloopfe or timeloop-mapper is missing,
            instead of marking every individual infeasible during the search.
    """

    def __init__(self, substrate: str, prefill_len: int = 128, decode_len: int = 32,
                 fused: bool = True, work_dir: Optional[str] = None,
                 require_timeloop: bool = True):
        if substrate not in SUBSTRATE_MAP:
            raise ValueError(
                f"Unknown timeloop substrate '{substrate}'. "
                f"Available: {sorted(SUBSTRATE_MAP)}")
        self.substrate = substrate
        self.arch = SUBSTRATE_MAP[substrate]
        self.prefill_len = int(prefill_len)
        self.decode_len = int(decode_len)
        self.fused = bool(fused)
        self.work_dir = None if work_dir is None else str(work_dir)
        if require_timeloop:
            from llmforge.hw.timeloop.gemm import timeloop_available
            if not timeloop_available():
                raise RuntimeError(
                    "The Timeloop backend needs timeloopfe and the timeloop-mapper binary "
                    "on PATH. See docs/hw_simulators.md.")

    def _pass_dir(self, mode: str) -> str:
        root = self.work_dir or str(paths.TIMELOOP_WORK)
        return os.path.join(root, self.arch, mode)

    def _run(self, ind: Dict[str, Any], block_size: int, mode: str) -> Dict[str, Any]:
        from llmforge.hw.timeloop import gemm
        # A shallow view with its own globals, so the caller's individual is never mutated.
        view = {"globals": {**ind["globals"], "block_size": int(block_size)},
                "layers": ind["layers"]}
        return gemm.eval_individual(view, work_dir=self._pass_dir(mode), fused=self.fused,
                                    arch=self.arch, mode=mode)

    def evaluate(self, ind_dicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        start = time.time()
        out = [self._evaluate_one(d) for d in ind_dicts]
        n_failed = sum(1 for r in out if not r.get("hw_feasible", False))
        print(f"  [timeloop:{self.arch}] {len(ind_dicts)} inds ({n_failed} failed) "
              f"in {time.time()-start:.1f}s")
        return out

    def _evaluate_one(self, ind: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if self.prefill_len > 0 and self.decode_len > 0:
                pf = self._run(ind, self.prefill_len, "prefill")
                dc = self._run(ind, self.decode_len, "decode")
                return self._combine(pf, dc)
            rec = dict(self._run(ind, ind["globals"].get("block_size", 0), "prefill"))
            cycles = rec.get("cycles")
            if cycles is not None:
                rec["ttft"] = cycles / 1e9
                rec["ttft_ms"] = cycles / 1e6
            rec["hw_feasible"] = _finite(rec.get("energy_uJ"), cycles)
            return rec
        except Exception as e:   # one unmappable architecture must not stop a generation
            return _failed(f"{type(e).__name__}: {e}")

    def _combine(self, p: Dict[str, Any], d: Dict[str, Any]) -> Dict[str, Any]:
        total_tokens = self.prefill_len + self.decode_len
        pf_e = (p.get("energy_uJ", 0) if p else 0)
        pf_c = (p.get("cycles", 0) if p else 0)
        dc_e = (d.get("energy_uJ", 0) if d else 0)
        dc_c = (d.get("cycles", 0) if d else 0)
        rec: Dict[str, Any] = {
            "energy_uJ": pf_e + dc_e * self.decode_len,
            "cycles": pf_c + dc_c * self.decode_len,
        }
        for k in _SUMMED_KEYS:
            rec[k] = (p.get(k, 0) if p else 0) + (d.get(k, 0) if d else 0) * self.decode_len
        if total_tokens > 0:
            rec["session_e_per_tok_uJ"] = rec["energy_uJ"] / total_tokens
            rec["cycles_per_token"] = rec["cycles"] / total_tokens
            rec["token_delay"] = rec["cycles_per_token"] / 1e9
        rec["edp"] = rec["energy_uJ"] * rec["cycles"] / 10e6
        if p:
            rec["prefill_energy_uJ"] = pf_e
            rec["prefill_cycles"] = pf_c
            rec["ttft"] = pf_c / 1e9
            rec["ttft_ms"] = pf_c / 1e6
        if d:
            rec["decode_energy_uJ"] = dc_e
            rec["decode_cycles"] = dc_c
            rec["tpot"] = dc_c / 1e9
            rec["tpot_ms"] = dc_c / 1e6
            rec["energy_per_token_uJ"] = dc_e
        # Surface D-axis padding annotations from gemm's `_pad_D_for_arch`
        # (only set on substrates with a strict spatial mesh, currently the
        # four DXE variants), so downstream consumers can tell which ops needed
        # padding and by how much. Prefill and decode both contribute.
        pf_pads = (p.get("padded_ops") or []) if p else []
        dc_pads = (d.get("padded_ops") or []) if d else []
        if pf_pads or dc_pads:
            rec["padded_ops"] = {"prefill": pf_pads, "decode": dc_pads}
            rec["padded_op_count"] = len(pf_pads) + len(dc_pads)
        rec["hw_feasible"] = _finite(rec.get("energy_per_token_uJ"),
                                     rec.get("ttft_ms"), rec.get("tpot_ms"))
        return rec
