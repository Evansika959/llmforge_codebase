"""rDXE architecture x chip co-search: the inner hardware loop run for every candidate architecture.

For each architecture proposed by the outer search, run_rdxe_eval

  1. profiles the active layers into per-layer WMEM, KV-cache, and MAC demand,
  2. sweeps a grid of chip configurations: MACs per VAC core, maximum ring size, and WMEM per core,
  3. packs the layers onto each feasible ring and simulates it with Timeloop-backed GEMM costs
     (workflow.profile_model, workflow.pack_balanced, workflow.simulate_ring),
  4. keeps the Pareto front over (per_tok_uJ, tpot_ms, ttft_ms) across the whole grid, and
  5. returns the candidate with the lowest `select_by` value as the architecture's hardware
     metrics, restricted to the area and power envelope when envelope_filter is on.

Decode GEMMs are mapped with Timeloop through core/timeloop_evaluator.py, with an analytical
fallback per shape, and simulator/layer_eval_timeloop.py turns the mappings into per-layer costs.
Every active layer is profiled and costed with its own shape, and prompt tokens are priced like
decode tokens. The full chip Pareto front is attached to each result under `pareto_points`.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import workflow as wf
from .core.timeloop_evaluator import TimeloopEvaluator, enumerate_gemm_shapes_decode

# ---- HW design-space grid for per-individual Pareto sweep ----
# Each (mac_per_vac, max_chips, wmem_per_core_KB) triple is one HW candidate, and the grid size
# multiplies the per-individual eval cost. Deep or heterogeneous models can fail to pack when
# max_chips forces more than 2 layers per chip onto a small-WMEM config, so the grid spans
# 3 x 3 x 5 = 45 configs. Prefetch cost amortizes over the whole population.
RDXE_SWEEP_MAC_PER_VAC = (16, 32, 64)
RDXE_SWEEP_MAX_CHIPS   = (8, 16, 32)
RDXE_SWEEP_WMEM_KB     = (24, 48, 96, 192, 384)

# ---- Canonical workload defaults ----
# Pinned to match the edge-LLM chat workload profile:
#   ctx=2048, prefill=512, decode=256, users=1.
RDXE_DEFAULT_CTX      = 2048
RDXE_DEFAULT_PREFILL  = 512
RDXE_DEFAULT_DECODE   = 256
RDXE_DEFAULT_N_USERS  = 1

# Edge HW envelope, which filters the Pareto candidate set BEFORE selection.
# If an individual has no candidate that fits the envelope, the least-violating candidate is
# returned with envelope_feasible=False. These are soft defaults.
RDXE_DEFAULT_AREA_MAX_MM2 = 2500.0
RDXE_DEFAULT_AREA_MIN_MM2 = 100.0
RDXE_DEFAULT_POWER_MAX_W  = 2.0
RDXE_DEFAULT_POWER_MIN_W  = 0.005

_INF = float("inf")


def _layer_dict_to_spec(layer: dict, default_n_embd: int) -> dict:
    """Convert an Individual layer dict into the rDXE layer-spec format."""
    d  = default_n_embd
    nh = int(layer.get("n_head", 8))
    return {
        "n_head":            nh,
        "n_kv_group":        int(layer.get("n_kv_group", nh)),
        "n_qk_head_dim":     int(layer.get("n_qk_head_dim", d // max(1, nh))),
        "n_v_head_dim":      int(layer.get("n_v_head_dim",
                                           layer.get("n_qk_head_dim", d // max(1, nh)))),
        "mlp_size":          int(layer.get("mlp_size", 4 * d)),
        "n_cproj":           1,
        "attention_variant": layer.get("attention_variant", "infinite"),
    }


def _pareto_front(points: List[dict], keys: List[str]) -> List[dict]:
    """Return the non-dominated subset (minimize every key)."""
    front = []
    for p in points:
        dominated = False
        for q in points:
            if q is p: continue
            if all(q[k] <= p[k] for k in keys) and any(q[k] < p[k] for k in keys):
                dominated = True
                break
        if not dominated:
            front.append(p)
    return front


def _infeasible() -> Dict[str, Any]:
    return {"ttft": _INF, "tpot": _INF, "energy_per_token_uJ": _INF, "pareto_points": []}


def run_rdxe_eval(individuals: Sequence[Dict[str, Any]],
                  n_chips: int = 8,                # retained for CLI compat
                  layer_assignment: str = "round_robin",  # (legacy, unused)
                  prefill_len: int = RDXE_DEFAULT_PREFILL,
                  decode_len: int = RDXE_DEFAULT_DECODE,
                  n_users: int = RDXE_DEFAULT_N_USERS,
                  ctx: Optional[int] = RDXE_DEFAULT_CTX,
                  pareto_objs: Tuple[str, ...] = ("per_tok_uJ",
                                                   "tpot_ms", "ttft_ms"),
                  select_by: str = "per_tok_uJ",
                  area_max_mm2: float = RDXE_DEFAULT_AREA_MAX_MM2,
                  area_min_mm2: float = RDXE_DEFAULT_AREA_MIN_MM2,
                  power_max_W:  float = RDXE_DEFAULT_POWER_MAX_W,
                  power_min_W:  float = RDXE_DEFAULT_POWER_MIN_W,
                  envelope_filter: bool = True,
                  verbose: bool = False,
                  n_workers: int = 8,
                  weight_memory: str = "wmem") -> List[dict]:
    """Timeloop-backed rDXE ring evaluator for a list of Individual dicts.

    For each individual we:
      1. Profile the active layers -> per-layer WMEM / KV$ / MAC demand.
      2. Sweep the HW grid (n_mac_per_vac, max_chips, wmem_per_core).
      3. For each feasible config, pack layers onto chips and run the
         Timeloop-backed ring simulation (DXE KV/VLINK/iWuR corrections,
         per-layer decode costs, token-level pipelined prefill).
      4. Compute the Pareto front over (per_tok_uJ, tpot_ms, ttft_ms).
      5. Return the `select_by`-argmin point as the primary hw_data dict.
         The full Pareto front is attached under key `pareto_points` for
         logging / post-hoc analysis.

    `weight_memory="wmem"` prices weights in the on-chip WMEM that the packing sizes, for both energy
    and time. "dram" keeps the DRAM traffic of the Timeloop mappings instead.

    Returned scalars keep the names the search consumes (ttft, tpot,
    energy_per_token_uJ, ...). Individuals with no active layer, or with no chip
    configuration that packs them, get infinite metrics.

    `individuals` may also be a population object with `.gen`, `.individuals`, and
    `.offspring`, as in the original driver: generation 0 evaluates the individuals and
    later generations evaluate the offspring.
    """
    if ctx is None:
        # Fall back to canonical default if caller did not supply one.
        ctx = RDXE_DEFAULT_CTX

    # ---- Pick individuals that need evaluation ----
    if hasattr(individuals, "individuals"):
        pop = individuals
        individuals = pop.individuals if getattr(pop, "gen", 0) == 0 else pop.offspring
    individuals = list(individuals)

    # ---- Pre-build the HW sweep list ----
    sweep = [(mpv, mc, wkb)
             for mpv in RDXE_SWEEP_MAC_PER_VAC
             for mc in RDXE_SWEEP_MAX_CHIPS
             for wkb in RDXE_SWEEP_WMEM_KB]

    # ---- One evaluator per n_mac_per_vac variant (Timeloop re-maps each) ----
    evaluators = {
        mpv: TimeloopEvaluator(arch='dxe_relaxed', verbose=False,
                               n_mac_per_vac=mpv)
        for mpv in RDXE_SWEEP_MAC_PER_VAC
    }

    # ---- Gather the decode shapes of every distinct layer across the population ----
    # Prompt tokens are priced like decode tokens at half the prompt length of context, so both
    # contexts are prefetched.
    contexts = (ctx, max(1, prefill_len // 2)) if prefill_len > 0 else (ctx,)
    decode_shapes_all = set()
    active_layers = []
    for ind in individuals:
        g = ind["globals"]
        layers_raw = ind["layers"]
        mask = g.get("layer_mask", [True] * len(layers_raw))
        active = [_layer_dict_to_spec(l, g["n_embd"]) for l, m in zip(layers_raw, mask) if m]
        if not active:
            active_layers.append(None)
            continue
        active_layers.append((active, g["n_embd"]))
        for spec in {wf.layer_key(s): s for s in active}.values():
            for c in contexts:
                for (_, ic, oc, sl) in enumerate_gemm_shapes_decode(spec, g["n_embd"], c):
                    decode_shapes_all.add((ic, oc, max(1, sl * n_users)))

    all_shapes = decode_shapes_all
    if verbose:
        print(f"  [rDXE] unique decode shapes to map: {len(all_shapes)}")
    for mpv, ev in evaluators.items():
        if verbose:
            print(f"  [rDXE] prefetch variant mac_per_vac={mpv} ...")
        ev.prefetch(list(all_shapes), n_workers=n_workers)

    # ---- Per-individual Pareto sweep ----
    start = time.time()
    hw_data = []
    n = len(individuals)
    for i, (ind, layer_info) in enumerate(zip(individuals, active_layers)):
        if layer_info is None:
            # Empty / all-masked individual -> inf metrics keep the search honest
            hw_data.append(_infeasible())
            continue
        active, n_embd = layer_info
        n_active = len(active)
        profile, info = wf.profile_layers(active, n_embd, ctx=ctx, n_users=n_users)

        candidates = []
        for (mpv, max_chips, wmem_kb) in sweep:
            saved = wf.WMEM_PER_CORE_OPTIONS
            wf.WMEM_PER_CORE_OPTIONS = [wmem_kb * 1024]
            try:
                packing = wf.pack_balanced(
                    profile,
                    max_chips=min(max_chips, n_active),
                    max_wmem_total_B=None,
                    n_mac_per_vac=mpv,
                )
            finally:
                wf.WMEM_PER_CORE_OPTIONS = saved
            if packing is None:
                continue
            r = wf.simulate_ring(info, packing, ctx, evaluators[mpv],
                                 prefill_length=prefill_len,
                                 decode_length=decode_len,
                                 n_users=n_users,
                                 weight_memory=weight_memory)
            r["sweep_mac_per_vac"]    = mpv
            r["sweep_max_chips"]      = max_chips
            r["sweep_wmem_per_core_KB"] = wmem_kb
            # Power (W) during decode = per-token energy x steady-state rate.
            #   per_tok_uJ (uJ) x n_users / tpot_ms (ms) -> mW -> /1000 -> W
            r["power_W"] = (r["per_tok_uJ"] * n_users
                            / max(1e-9, r["tpot_ms"])) * 1e-3
            candidates.append(r)

        if not candidates:
            hw_data.append(_infeasible())
            continue

        # ---- Optional envelope filter: restricts selection to HW configs ----
        # that fit the edge budget (area + power). When disabled, the selector
        # sees the full Pareto candidate set (raw argmin on `select_by`).
        #
        # Rationale:
        #   ON  : production runs with a known HW budget. The Pareto-set to scalar
        #         mapping picks the best operating point that fits, and falls back to
        #         the least-violating point if nothing fits (a search constraint on
        #         envelope_feasible can then mark the individual infeasible).
        #   OFF : HW-agnostic exploration and sensitivity studies. The search ranks
        #         models by their globally-best HW config, and envelope_feasible still
        #         reports whether the selected point fits.
        if envelope_filter:
            feasible = [r for r in candidates
                        if area_min_mm2 <= r["total_area_mm2"] <= area_max_mm2
                        and power_min_W  <= r["power_W"]        <= power_max_W]

            if feasible:
                candidates_for_selection = feasible
                envelope_feasible = True
            else:
                # Least-violating: min overshoot on area + power (relative)
                def _overshoot(r):
                    a_over = max(0, r["total_area_mm2"] - area_max_mm2) / area_max_mm2
                    a_under = max(0, area_min_mm2 - r["total_area_mm2"]) / max(1, area_min_mm2)
                    p_over = max(0, r["power_W"] - power_max_W) / power_max_W
                    p_under = max(0, power_min_W - r["power_W"]) / max(1e-9, power_min_W)
                    return a_over + a_under + p_over + p_under
                candidates_for_selection = [min(candidates, key=_overshoot)]
                envelope_feasible = False
        else:
            # No envelope filter: raw argmin on the full candidate set.
            candidates_for_selection = candidates
            # Still report whether the final selection fits the envelope
            # (computed after the argmin below).
            envelope_feasible = None

        # Pareto front is still computed on the FULL candidate set for
        # downstream analysis (`pareto_points`), not just the feasible
        # subset. This preserves visibility into area/power tradeoffs.
        front = _pareto_front(candidates, list(pareto_objs))
        sel   = min(candidates_for_selection, key=lambda r: r[select_by])

        # When envelope_filter was off, retrospectively flag whether the
        # selected point actually fits the envelope (useful for reporting).
        if envelope_feasible is None:
            envelope_feasible = (area_min_mm2 <= sel["total_area_mm2"] <= area_max_mm2
                                 and power_min_W <= sel["power_W"] <= power_max_W)

        hw_data.append({
            # Scalar fields consumed by search objectives and constraints.
            "ttft":                  sel["ttft_ms"] / 1e3,     # seconds
            "tpot":                  sel["tpot_ms"] / 1e3,     # seconds
            "ttft_ms":               sel["ttft_ms"],
            "tpot_ms":               sel["tpot_ms"],
            "energy_per_token_uJ":   sel["per_tok_uJ"],
            # prefill+decode session-amortized E/tok kept in the aux record
            # but NOT used as a primary objective (prefill is analytical).
            "session_e_per_tok_uJ":  sel.get("session_e_per_tok_uJ",
                                              sel["per_tok_uJ"]),
            "total_area_mm2":        sel["total_area_mm2"],
            "power_W":               sel["power_W"],
            "mac_util_pct":          sel["mac_util_pct"],
            "n_chips":               sel["n_chips"],
            "chip_macs":             sel["chip_macs"],
            "selected_mac_per_vac":  sel["sweep_mac_per_vac"],
            "selected_max_chips":    sel["sweep_max_chips"],
            "selected_wmem_KB":      sel["sweep_wmem_per_core_KB"],
            # Envelope-feasibility flag: True if the selected point fits
            # the edge HW budget, False means we returned the least-violating point.
            "envelope_feasible":     envelope_feasible,
            # GEMM cost provenance of the selected point: ops mapped by Timeloop versus
            # ops that used the analytical fallback, per simulated layer.
            "ops_timeloop":          sel.get("ops_timeloop"),
            "ops_fallback":          sel.get("ops_fallback"),
            # Legacy compatibility (aliases)
            "tokens_per_second":     sel["throughput_tps"],
            "inter_chip_comm_energy_uJ": sel["e_hop_uJ"],
            "pipeline_depth":        sel["n_chips"],
            # Pareto front, keyed by (TTFT, TPOT, E/tok) for this arch.
            # Also carries area + power for post-hoc HW-budget filtering.
            "pareto_points": [
                {k: p[k] for k in ("sweep_mac_per_vac", "sweep_max_chips",
                                    "sweep_wmem_per_core_KB",
                                    "ttft_ms", "tpot_ms",
                                    "per_tok_uJ", "session_e_per_tok_uJ",
                                    "total_area_mm2", "power_W",
                                    "mac_util_pct",
                                    "n_chips", "chip_macs")}
                for p in front
            ],
        })
        if verbose:
            print(f"\r  rDXE eval [{i+1}/{n}]  "
                  f"Pareto={len(front):>2d}/{len(candidates)} configs  ",
                  end="", flush=True)
    if verbose:
        print()

    elapsed = time.time() - start
    # Terse single-line summary even when quiet: how many individuals
    # evaluated, how many sit inside the envelope, and wall time.
    n_feas = sum(1 for h in hw_data
                 if h.get("envelope_feasible") is True)
    print(f"[rDXE] eval: {len(hw_data)} ind | {n_feas} envelope-feasible | "
          f"{len(sweep)} HW configs/ind | {elapsed:.1f}s")
    return hw_data
