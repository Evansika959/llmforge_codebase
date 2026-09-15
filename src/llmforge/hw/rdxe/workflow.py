"""Timeloop-backed rDXE workflow: profile → pack → ring-sim → Pareto.

Mirrors the three-stage figure:
  (1) Model Profiling  — per-layer WMEM / KV$ / MAC demand
  (2) Multi-Resource Balanced Packing — greedy-contiguous layer groups bound
       by a shared DXE config (WMEM_per_core, KV_per_core, MAC count),
       selected via binary search over per-core WMEM until the worst chip
       fits.
  (3) Ring Simulation — steady-state TTFT/TPOT/E-per-token using the fast
       Timeloop evaluator as the single-DXE energy/latency oracle.

Run:
    python -m llmforge.hw.rdxe.workflow
    python -m llmforge.hw.rdxe.workflow --workers 8
"""

import argparse
import csv
import math
import os
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Tuple

from llmforge import paths

from .core.constants import (
    CLOCK_PERIOD_NS, N_MAC_PER_VAC, TOTAL_MACS,
    INTER_CHIP_ENERGY_PJ_PER_BIT, INTER_CHIP_BW_BYTES_PER_CYCLE,
    INTER_CHIP_LATENCY_CYCLES,
)
from .core.scaled_arch import (
    ScaledChipSpec, layer_weight_bytes, kv_bytes_per_token_per_layer,
    _factor_cores,
)
from .core.timeloop_evaluator import (
    TimeloopEvaluator, enumerate_gemm_shapes_decode,
)
from .simulator.layer_eval_timeloop import timeloop_layer_decode


# ---------------------------------------------------------------------------
# Model zoo (~200M each)
# ---------------------------------------------------------------------------
MODELS = {
    "A_ultra_thin":   dict(n_layer=48, n_embd=512,  n_head=8,  n_kv_group=2, hd=64,  mlp=3328),
    "B_thin_deep":    dict(n_layer=32, n_embd=768,  n_head=12, n_kv_group=4, hd=64,  mlp=3072),
    "C_balanced":     dict(n_layer=18, n_embd=1024, n_head=16, n_kv_group=4, hd=64,  mlp=4096),
    "D_wide_shallow": dict(n_layer=12, n_embd=1280, n_head=20, n_kv_group=5, hd=64,  mlp=5120),
    "E_ultra_wide":   dict(n_layer=8,  n_embd=1536, n_head=24, n_kv_group=6, hd=64,  mlp=6144),
}


def layer_of(model_cfg: dict) -> dict:
    return {
        'n_head':            model_cfg['n_head'],
        'n_kv_group':        model_cfg['n_kv_group'],
        'n_qk_head_dim':     model_cfg['hd'],
        'n_v_head_dim':      model_cfg['hd'],
        'mlp_size':          model_cfg['mlp'],
        'n_cproj':           1,
        'attention_variant': 'infinite',
    }


# ---------------------------------------------------------------------------
# Stage 1 — Model Profiling
# ---------------------------------------------------------------------------
@dataclass
class LayerResource:
    layer_idx:     int
    weight_B:      int      # WMEM demand (INT8 bytes)
    kv_bytes_ctx:  int      # KV$ demand = kv_per_tok × ctx × n_users
    macs:          int      # rough compute for ranking
    layer_spec:    dict     # for Timeloop shape enumeration


def layer_key(layer: dict) -> tuple:
    """Hashable shape of one layer spec, so each distinct shape is costed once."""
    return (layer['n_head'], layer['n_kv_group'], layer['n_qk_head_dim'], layer['n_v_head_dim'],
            layer['mlp_size'], layer.get('attention_variant', 'infinite'))


def profile_layers(layers: List[dict], n_embd: int, ctx: int, n_users: int = 1
                   ) -> Tuple[List[LayerResource], dict]:
    """Per-layer resource profile of the active layers, which may all differ."""
    profile = []
    for i, layer in enumerate(layers):
        mac_est = (n_embd * layer['n_qk_head_dim'] * (layer['n_head'] + layer['n_kv_group'])
                   + n_embd * layer['n_v_head_dim'] * (layer['n_kv_group'] + layer['n_head'])
                   + 2 * n_embd * layer['mlp_size'])
        profile.append(LayerResource(
            layer_idx=i, weight_B=layer_weight_bytes(layer, n_embd),
            kv_bytes_ctx=kv_bytes_per_token_per_layer(layer) * ctx * n_users,
            macs=mac_est, layer_spec=layer))
    return profile, dict(n_layer=len(layers), n_embd=n_embd, layers=list(layers))


def profile_model(model_cfg: dict, ctx: int, n_users: int = 1
                  ) -> Tuple[List[LayerResource], dict]:
    """Per-layer resource profile of a MODELS entry, whose layers share one shape."""
    return profile_layers([layer_of(model_cfg)] * model_cfg['n_layer'], model_cfg['n_embd'],
                          ctx, n_users)


# ---------------------------------------------------------------------------
# Stage 2 — Multi-Resource Balanced Packing
# ---------------------------------------------------------------------------
WMEM_PER_CORE_OPTIONS = [24*1024, 48*1024, 96*1024, 192*1024, 384*1024]


def _build_chip(n_cores: int, wmem_per_core_B: int,
                kv_per_core_B: int,
                n_mac_per_vac: int = N_MAC_PER_VAC) -> ScaledChipSpec:
    n_dxt, n_vac = _factor_cores(n_cores)
    return ScaledChipSpec(
        name=f"DXE-{n_dxt}x{n_vac}x{n_mac_per_vac}-w{wmem_per_core_B//1024}",
        n_dxt=n_dxt, n_vac_per_dxt=n_vac, n_mac_per_vac=n_mac_per_vac,
        wmem_per_core_B=wmem_per_core_B,
        kv_cache_per_core_B=kv_per_core_B,
    )


@dataclass
class Packing:
    chip:          ScaledChipSpec
    groups:        List[List[int]]         # layer indices per chip group
    group_wmem_B:  List[int]               # WMEM used per group
    group_kv_B:    List[int]               # KV$ used per group
    util_pct:      List[float]             # WMEM utilization % per group


def pack_balanced(profile: List[LayerResource],
                  max_chips: int = None,
                  max_wmem_total_B: int = None,
                  n_mac_per_vac: int = N_MAC_PER_VAC,
                  overhead: float = 1.15) -> Packing:
    """Greedy-contiguous layer-to-chip packing + binary search over shared
    DXE config so every group's weights fit in WMEM with 15% overhead. KV$ per
    core is then sized to the largest group.

    Strategy:
      1. Pick a candidate shared chip (wmem_per_core, n_cores).
      2. Greedily pack contiguous layers into a group until the next layer
         would exceed the chip's WMEM; start a new group.
      3. Accept the smallest config that packs all layers in ≤ max_chips
         AND wmem_total ≤ max_wmem_total_B (if set).
    """
    n_layers = len(profile)
    if max_chips is None:
        max_chips = n_layers

    # Budget per group: WMEM_total(1-wmem_kv_split) ≥ group_weights*overhead.
    # Enumerate (wmem_per_core, n_cores) from smallest upward.
    best: Packing = None

    for wmem_pc in WMEM_PER_CORE_OPTIONS:
        # ceil(largest_layer / wmem_pc) gives core-count floor per group
        largest_layer = max(p.weight_B for p in profile)
        cores_min = max(1, math.ceil(largest_layer * overhead / wmem_pc))

        # Grow core-count in powers of 2 until a valid packing exists
        n_cores = 1
        while n_cores < cores_min:
            n_cores *= 2

        found = False
        for _ in range(6):  # allow up to 6 doublings
            wmem_total = n_cores * wmem_pc
            usable     = int(wmem_total / overhead)

            # Enforce per-chip SRAM cap
            if max_wmem_total_B is not None and wmem_total > max_wmem_total_B:
                break

            groups, g_w, g_kv, g_util = [], [], [], []
            cur_group, cur_w, cur_kv = [], 0, 0
            for p in profile:
                w_next = cur_w + p.weight_B
                # Close the current group when the next layer's weights do not fit. KV$ per core is
                # sized to the packed layers afterwards, so it does not bound a group.
                if cur_group and w_next > usable:
                    groups.append(cur_group)
                    g_w.append(cur_w); g_kv.append(cur_kv)
                    g_util.append(100.0 * cur_w / wmem_total)
                    cur_group, cur_w, cur_kv = [], 0, 0
                cur_group.append(p.layer_idx)
                cur_w += p.weight_B
                cur_kv += p.kv_bytes_ctx
            if cur_group:
                groups.append(cur_group)
                g_w.append(cur_w); g_kv.append(cur_kv)
                g_util.append(100.0 * cur_w / wmem_total)

            if len(groups) <= max_chips and all(w <= usable for w in g_w):
                kv_per_core = max(8*1024, math.ceil(max(g_kv) * overhead / n_cores))
                # round up to power-of-2 KB
                kv_kb = max(8, 1 << math.ceil(math.log2(kv_per_core / 1024)))
                kv_per_core = kv_kb * 1024
                chip = _build_chip(n_cores, wmem_pc, kv_per_core,
                                   n_mac_per_vac=n_mac_per_vac)
                pk = Packing(chip=chip, groups=groups, group_wmem_B=g_w,
                             group_kv_B=g_kv, util_pct=g_util)
                if best is None or (len(groups) * chip.estimated_area_mm2 <
                                    len(best.groups) * best.chip.estimated_area_mm2):
                    best = pk
                found = True
                break
            n_cores *= 2

        if found and best and len(best.groups) == 1:
            break  # can't do better than one chip

    return best


# ---------------------------------------------------------------------------
# Stage 3 — Ring Simulation
# ---------------------------------------------------------------------------
def simulate_ring(model_info: dict, packing: Packing, ctx: int,
                  evaluator: TimeloopEvaluator,
                  prefill_length: int = 0,
                  decode_length: int = 512,
                  n_users: int = 1,
                  weight_memory: str = "wmem") -> dict:
    """TPOT, TTFT and energy per token on the packed ring.

    Every active layer is costed with its own shape, each distinct shape once. A chip runs its layers
    in sequence and hands the hidden state to the next chip, so a decode token passes every layer and
    every hop. The prompt is token-level pipelined: its first token passes the whole ring, every later
    token follows one bottleneck stage behind, and the first output token leaves with the last prompt
    token. Prompt tokens are priced like decode tokens at half the prompt length of context. Every chip
    leaks for the whole decode step and for the whole prompt.

    Args:
        ctx:            context window used for decode KV reads.
        prefill_length: prompt tokens; drives TTFT.
        decode_length:  number of output tokens (for session amortization).
        n_users:        concurrent users in one forward pass (decode batch).
        weight_memory:  "wmem" or "dram", see simulator/layer_eval_timeloop.py.
    """
    layers  = model_info['layers']
    n_embd  = model_info['n_embd']
    chip    = packing.chip
    n_layer = len(layers)
    n_chips = len(packing.groups)

    def layer_costs(context: int) -> List[dict]:
        memo = {}
        for layer in layers:
            key = layer_key(layer)
            if key not in memo:
                memo[key] = timeloop_layer_decode(layer, n_embd, context, chip, evaluator,
                                                  n_users=n_users, weight_memory=weight_memory)
        return [memo[layer_key(layer)] for layer in layers]

    dec = layer_costs(ctx)

    # Inter-chip hop of one hidden state per user
    hop_e_pJ = n_embd * 8 * INTER_CHIP_ENERGY_PJ_PER_BIT * n_users
    hop_cy = INTER_CHIP_LATENCY_CYCLES + n_embd * n_users / INTER_CHIP_BW_BYTES_PER_CYCLE
    ring_leak_pJ_per_cy = chip.leakage_pJ_per_cycle * n_chips

    # Decode: one step through every layer and every hop
    stage_dec_cy = [sum(dec[i]['cycles'] for i in group) for group in packing.groups]
    step_cy = sum(stage_dec_cy) + hop_cy * (n_chips - 1)
    leak_e_pJ = ring_leak_pJ_per_cy * step_cy
    per_tok_pJ = (sum(d['total_energy_pJ'] for d in dec) + hop_e_pJ * (n_chips - 1)
                  + leak_e_pJ) / n_users
    tpot_us = step_cy * CLOCK_PERIOD_NS / 1e3

    # Prefill: token-level pipeline over the chips
    if prefill_length > 0:
        pre = layer_costs(max(1, prefill_length // 2))
        stage_pre_cy = [sum(pre[i]['cycles'] for i in group) for group in packing.groups]
        spacing_cy = max(stage_pre_cy) + (hop_cy if n_chips > 1 else 0.0)
        ttft_cy = sum(stage_pre_cy) + hop_cy * (n_chips - 1) + (prefill_length - 1) * spacing_cy
        prefill_total_e_pJ = (prefill_length * (sum(p['total_energy_pJ'] for p in pre)
                                                + hop_e_pJ * (n_chips - 1))
                              + ring_leak_pJ_per_cy * ttft_cy)
    else:
        ttft_cy = step_cy
        prefill_total_e_pJ = 0.0
    ttft_us = ttft_cy * CLOCK_PERIOD_NS / 1e3
    prefill_per_user_pJ = prefill_total_e_pJ / n_users

    # Session-amortized E/tok: (prefill_per_user + decode_length x per_tok) / decode_length
    if decode_length > 0:
        session_e_per_tok_pJ = (prefill_per_user_pJ + per_tok_pJ * decode_length) / decode_length
    else:
        session_e_per_tok_pJ = per_tok_pJ

    # Saturated token-level pipeline: one token per bottleneck stage per user
    bottleneck_cy = max(stage_dec_cy) + (hop_cy if n_chips > 1 else 0.0)
    throughput_tps = n_users / (bottleneck_cy * CLOCK_PERIOD_NS * 1e-9)

    # MAC utilization over the cycles in which the MAC array computes
    useful_macs = 0
    for layer in layers:
        nh, nkv = layer['n_head'], layer['n_kv_group']
        qk, vd, mlp = layer['n_qk_head_dim'], layer['n_v_head_dim'], layer['mlp_size']
        useful_macs += (n_embd * qk * (nh + nkv) + n_embd * vd * nkv + vd * nh * n_embd
                        + 2 * n_embd * mlp + nh * (qk + vd) * ctx) * n_users
    mac_util_pct = 100.0 * useful_macs / (chip.total_macs * max(1.0, sum(d['cycles'] for d in dec)))

    sources = [s for d in dec for s in d['per_op_sources'].values()]

    return dict(
        n_chips           = n_chips,
        n_layer           = n_layer,
        layers_per_chip   = max(len(g) for g in packing.groups),
        chip_macs         = chip.total_macs,
        chip_n_dxt        = chip.n_dxt,
        chip_n_vac        = chip.n_vac_per_dxt,
        chip_n_mac_per_vac= chip.n_mac_per_vac,
        chip_wmem_MB      = chip.wmem_total_B / 1e6,
        chip_kv_MB        = chip.kv_cache_total_B / 1e6,
        prefill_length    = prefill_length,
        decode_length     = decode_length,
        n_users           = n_users,
        tpot_ms           = tpot_us / 1e3,
        ttft_ms           = ttft_us / 1e3,
        throughput_tps    = throughput_tps,
        per_tok_uJ        = per_tok_pJ / 1e6,      # per output token
        session_e_per_tok_uJ = session_e_per_tok_pJ / 1e6,
        prefill_e_per_user_uJ = prefill_per_user_pJ / 1e6,
        total_area_mm2    = chip.estimated_area_mm2 * n_chips,
        e_gemm_uJ         = sum(d['gemm_energy_pJ'] for d in dec) / (n_users * 1e6),
        e_kv_read_uJ      = sum(d['kv_read_energy_pJ'] for d in dec) / (n_users * 1e6),
        e_kv_write_uJ     = sum(d['kv_write_energy_pJ'] for d in dec) / (n_users * 1e6),
        e_vrc_uJ          = sum(d['vrc_energy_pJ'] for d in dec) / (n_users * 1e6),
        e_hop_uJ          = hop_e_pJ * (n_chips - 1) / (n_users * 1e6),
        e_leak_uJ         = leak_e_pJ / (n_users * 1e6),
        packing_util_pct  = sum(packing.util_pct)/len(packing.util_pct),
        mac_util_pct      = mac_util_pct,
        ops_timeloop      = sum(1 for s in sources if s == 'timeloop'),
        ops_fallback      = sum(1 for s in sources if 'analytical' in s),
        per_op_sources    = {s: sources.count(s) for s in sorted(set(sources))},
    )


# ---------------------------------------------------------------------------
# Orchestration + reporting
# ---------------------------------------------------------------------------
def run_all(args):
    evaluator = TimeloopEvaluator(arch=args.arch, verbose=True,
                                  n_mac_per_vac=args.mac_per_vac)

    # ---- Phase A: collect all shapes across every (model, chip) pair ----
    all_shapes = set()
    packings   = {}
    profiles   = {}
    max_wmem_B = int(args.max_wmem_MB * 1024 * 1024)
    for name, cfg in MODELS.items():
        profile, info = profile_model(cfg, ctx=args.ctx)
        # Platform caps: at most args.max_chips chips, no chip above max_wmem_B
        chip_cap = min(args.max_chips, cfg['n_layer'])
        packing = pack_balanced(profile, max_chips=chip_cap,
                                max_wmem_total_B=max_wmem_B,
                                n_mac_per_vac=args.mac_per_vac)
        if packing is None:
            raise SystemExit(
                f"Infeasible: {name} cannot be packed into ≤{chip_cap} chips "
                f"with ≤{args.max_wmem_MB:.0f} MB WMEM/chip. "
                f"Relax --max-chips or --max-wmem-MB.")
        profiles[name] = (profile, info)
        packings[name] = packing
        # Decode shapes of every distinct layer, at the decode context and at the prompt context
        contexts = (args.ctx, max(1, args.prefill // 2)) if args.prefill > 0 else (args.ctx,)
        for spec in {layer_key(l): l for l in info['layers']}.values():
            for c in contexts:
                for (_, ic, oc, sl) in enumerate_gemm_shapes_decode(spec, info['n_embd'], c):
                    all_shapes.add((ic, oc, sl))

        print(f"  [profile] {name}: {cfg['n_layer']} layers × {cfg['n_embd']}-dim "
              f"→ {len(packing.groups)} chips, "
              f"{packing.chip.total_macs} MACs, "
              f"{packing.chip.wmem_total_B/1e6:.1f} MB WMEM  "
              f"(util {sum(packing.util_pct)/len(packing.util_pct):.0f}%)")

    # Decode shapes scale with n_users in M (seq_len).  Also add shapes
    # for n_users > 1 so prefetch covers them.
    if args.users > 1:
        extra = set()
        for (ic, oc, sl) in all_shapes:
            extra.add((ic, oc, max(1, sl * args.users)))
        all_shapes = all_shapes | extra

    print(f"\n  [Timeloop] unique GEMM shapes across all models: {len(all_shapes)}")
    t0 = time.time()
    evaluator.prefetch(list(all_shapes), n_workers=args.workers)
    print(f"  [Timeloop] prefetch total time: {time.time()-t0:.1f}s")
    print(f"  {evaluator.summary()}")

    # ---- Phase B: ring simulation using cached Timeloop results ----
    print(f"\n{'='*120}")
    print(f"  Stage 3 — Ring Simulation (Timeloop backend)")
    print(f"    ctx={args.ctx}   prefill={args.prefill}   "
          f"decode={args.decode}   users={args.users}")
    print(f"{'='*120}")
    print(f"{'model':<18s}|{'L×d':>10s}|{'chips':>6s}|{'L/ch':>5s}|"
          f"{'MACs':>6s}|{'WMEM':>8s}|"
          f"{'E/tok':>8s}|"
          f"{'TPOT':>8s}|{'TTFT':>9s}|{'Tput':>9s}|{'Area':>7s}|"
          f"{'MACu':>6s}|{'src':>6s}")
    print('-'*120)

    results = {}
    for name, cfg in MODELS.items():
        _, info   = profiles[name]
        packing   = packings[name]
        r = simulate_ring(info, packing, args.ctx, evaluator,
                          prefill_length=args.prefill,
                          decode_length=args.decode,
                          n_users=args.users)
        if 'error' in r:
            print(f"  {name}: {r['error']}")
            continue
        r['name'] = name
        r['n_embd'] = cfg['n_embd']
        results[name] = r
        print(f"{name:<18s}|{cfg['n_layer']:>3d}×{cfg['n_embd']:<6d}|"
              f"{r['n_chips']:>5d} |{r['layers_per_chip']:>4d} |"
              f"{r['chip_macs']:>5d} |"
              f"{r['chip_wmem_MB']:>6.1f}MB|"
              f"{r['per_tok_uJ']:>6.2f}uJ|"
              f"{r['tpot_ms']:>6.2f}ms|"
              f"{r['ttft_ms']:>6.1f}ms|"
              f"{r['throughput_tps']:>7.0f}/s|"
              f"{r['total_area_mm2']:>5.0f}mm²|"
              f"{r['mac_util_pct']:>5.1f}%|"
              f"{r['ops_timeloop']:>2d}T+{r['ops_fallback']:<2d}f")

    print("\n  Energy breakdown (µJ / decode token, full ring):")
    cols = ['e_gemm_uJ','e_kv_read_uJ','e_kv_write_uJ','e_vrc_uJ','e_hop_uJ','e_leak_uJ']
    labels = ['GEMM','KV-read','KV-write','VRC','hop','leakage']
    print(f"  {'model':<18s}  " + "  ".join(f"{l:>15s}" for l in labels) + "    Total")
    for name, r in results.items():
        vals = "  ".join(f"{r[c]:>15.3f}" for c in cols)
        print(f"  {name:<18s}  {vals}    {r['per_tok_uJ']:>6.2f}")

    # ---- Outputs ----
    results_dir = str(paths.RUNS / 'rdxe' / 'results')
    plot_dir    = str(paths.RUNS / 'rdxe' / 'plots')
    os.makedirs(results_dir, exist_ok=True)

    tag = args.tag or f"pf{args.prefill}_dec{args.decode}_u{args.users}_ctx{args.ctx}"
    csv_path = os.path.join(results_dir, f'timeloop_workflow_{tag}.csv')
    if results:
        fields = list(next(iter(results.values())).keys())
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in results.values():
                w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                            for k, v in r.items()})
        print(f"\nCSV: {csv_path}")

    # ---- Pareto plot ----
    if results:
        plot_pareto(results, profiles, packings, args.ctx, plot_dir, tag=tag)

    if args.skip_iha:
        return

    # ---- Phase C: IHA variant isolation at baseline C ----
    print(f"\n{'='*110}")
    print(f"  IHA Variant Isolation (Timeloop backend, fixed chip)")
    print(f"{'='*110}")
    backbone = MODELS['C_balanced']
    iha_variants = {
        'baseline   (nkv=4)': dict(),
        'MQA (nkv=1)':        dict(n_kv_group=1),
        'GQA8 (nkv=2)':       dict(n_kv_group=2),
        'MHA (nkv=16)':       dict(n_kv_group=16),
        'qk=32':              dict(hd_qk=32),
        'qk=128':             dict(hd_qk=128),
        'v=32':               dict(hd_v=32),
        'v=128':              dict(hd_v=128),
        'asymm (qk=64,v=128)':dict(hd_v=128),
    }
    # Chip sized for worst-case variant
    worst = layer_of({**backbone, 'hd': 128})
    worst['n_kv_group'] = 16
    from .core.scaled_arch import scale_chip_for_layer
    iha_chip = scale_chip_for_layer(worst, backbone['n_embd'],
                                    max_context=args.ctx, n_users=1,
                                    name='iha_fixed_chip')

    # Enumerate + prefetch all variant shapes
    iha_shapes = set()
    variant_specs = {}
    for name, ov in iha_variants.items():
        spec = layer_of(backbone)
        if 'n_kv_group' in ov: spec['n_kv_group'] = ov['n_kv_group']
        if 'hd_qk'     in ov: spec['n_qk_head_dim'] = ov['hd_qk']
        if 'hd_v'      in ov: spec['n_v_head_dim']  = ov['hd_v']
        variant_specs[name] = spec
        for (_, ic, oc, sl) in enumerate_gemm_shapes_decode(spec, backbone['n_embd'], args.ctx):
            iha_shapes.add((ic, oc, sl))
    evaluator.prefetch(list(iha_shapes), n_workers=args.workers)

    print(f"\n{'variant':<22s}|{'qk':>3s}|{'v':>3s}|{'nkv':>4s}|"
          f"{'E/tok':>8s}|{'ΔE%':>6s}|{'E_kv_rd':>8s}|"
          f"{'src':>6s}")
    print('-'*80)
    base_e = None
    iha_results = {}
    for name, spec in variant_specs.items():
        dec = timeloop_layer_decode(spec, backbone['n_embd'], args.ctx,
                                    iha_chip, evaluator, n_users=args.users)
        nl = backbone['n_layer']
        tot_uJ = dec['total_energy_pJ'] * nl / 1e6
        kv_uJ  = dec['kv_read_energy_pJ']  * nl / 1e6
        if base_e is None and 'baseline' in name:
            base_e = tot_uJ
        deltapct = 100 * (tot_uJ - base_e) / base_e if base_e else 0
        srcs = dec.get('per_op_sources', {})
        n_tl = sum(1 for s in srcs.values() if s == 'timeloop')
        n_fb = sum(1 for s in srcs.values() if 'analytical' in s)
        iha_results[name] = dict(
            qk=spec['n_qk_head_dim'], v=spec['n_v_head_dim'],
            nkv=spec['n_kv_group'], per_tok_uJ=tot_uJ, kv_read_uJ=kv_uJ,
            delta_pct=deltapct, ops_tl=n_tl, ops_fb=n_fb,
        )
        print(f"{name:<22s}|{spec['n_qk_head_dim']:>3d}|{spec['n_v_head_dim']:>3d}|"
              f"{spec['n_kv_group']:>4d}|"
              f"{tot_uJ:>6.2f}uJ|{deltapct:>+5.1f}%|"
              f"{kv_uJ:>7.2f}|"
              f"{n_tl:>2d}T+{n_fb:<2d}f")

    print(f"\n  {evaluator.summary()}")

    if iha_results:
        iha_csv = os.path.join(results_dir, f'timeloop_iha_variants_{tag}.csv')
        fields = list(next(iter(iha_results.values())).keys()) + ['name']
        with open(iha_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for nm, r in iha_results.items():
                w.writerow({**{k: (f"{v:.6f}" if isinstance(v, float) else v)
                               for k, v in r.items()}, 'name': nm})
        print(f"CSV: {iha_csv}")


def plot_pareto(results: dict, profiles: dict, packings: dict,
                ctx: int, out_dir: str, tag: str = ''):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    mpl.rcParams.update({
        'font.family':'serif','font.size':10,'axes.labelsize':11,
        'legend.fontsize':9,'legend.frameon':False,
        'axes.grid':True,'grid.alpha':0.3,'grid.linestyle':'--',
        'figure.dpi':150,'savefig.dpi':300,'savefig.bbox':'tight',
    })
    os.makedirs(out_dir, exist_ok=True)

    names  = list(results.keys())
    colors = plt.cm.viridis(np.linspace(0.12, 0.85, len(names)))

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.5))

    # Panel 1: per-layer resource profile (stacked bars, one subplot)
    ax = axes[0]
    for (name, c) in zip(names, colors):
        prof, info = profiles[name]
        xs = np.arange(len(prof))
        wmem_MB = [p.weight_B/1e6 for p in prof]
        # Single bar per model at x=n_layer marker
        ax.plot(xs, wmem_MB, '-', color=c, alpha=0.8,
                label=f"{name} ({info['n_embd']}-dim)")
    ax.set_xlabel('Layer index')
    ax.set_ylabel('Per-layer WMEM demand (MB)')
    ax.set_title('Stage 1 — Per-Layer Resource Profile')
    ax.legend(fontsize=7.5, loc='upper right')

    # Panel 2: TPOT vs E/tok (Pareto)
    ax = axes[1]
    for name, c in zip(names, colors):
        r = results[name]
        ax.scatter(r['tpot_ms'], r['per_tok_uJ'], s=180, c=[c],
                   edgecolor='k', linewidth=0.7, zorder=3)
        ax.annotate(f"{name}\n{r['n_chips']}ch × {r['layers_per_chip']}L",
                    (r['tpot_ms'], r['per_tok_uJ']),
                    fontsize=8, xytext=(7, -4), textcoords='offset points')
    ax.set_xlabel('Per-user TPOT (ms)')
    ax.set_ylabel('Energy / decode token (µJ)')
    ax.set_title('Stage 3 — Ring PPA Pareto (Timeloop-backed)')

    # Panel 3: stacked energy breakdown
    ax = axes[2]
    comps = ['e_gemm_uJ','e_kv_read_uJ','e_kv_write_uJ','e_vrc_uJ','e_hop_uJ','e_leak_uJ']
    lbls  = ['GEMM','KV-rd','KV-wr','VRC','hop','leak']
    pal   = ['#4e79a7','#f28e2b','#e15759','#59a14f','#9467bd','#bab0ac']
    xs = np.arange(len(names))
    bot = np.zeros(len(names))
    for c, lab, col in zip(comps, lbls, pal):
        vals = np.array([results[n][c] for n in names])
        ax.bar(xs, vals, 0.72, bottom=bot, label=lab,
               color=col, edgecolor='k', linewidth=0.4)
        bot += vals
    ax.set_xticks(xs)
    ax.set_xticklabels([n.split('_',1)[-1] for n in names],
                       rotation=15, ha='right')
    ax.set_ylabel('Energy / decode token (µJ)')
    ax.set_title('Energy Breakdown (per-GEMM from Timeloop)')
    ax.legend(loc='upper right', fontsize=8)

    plt.suptitle(f'rDXE Workflow — Timeloop GEMM oracle + balanced packing '
                 f'(~200M, ctx={ctx})',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    fname = f'timeloop_workflow_pareto_{tag}.png' if tag else 'timeloop_workflow_pareto.png'
    p = os.path.join(out_dir, fname)
    plt.savefig(p); plt.close()
    print(f"Plot: {p}")


def _print_config_banner(args):
    print(f"\n{'#'*96}")
    print(f"# rDXE Timeloop workflow — configuration")
    print(f"#   ctx            = {args.ctx}")
    print(f"#   prefill_length = {args.prefill}")
    print(f"#   decode_length  = {args.decode}")
    print(f"#   n_users        = {args.users}")
    print(f"#   max_chips      = {args.max_chips}  (platform cap)")
    print(f"#   max_wmem_MB    = {args.max_wmem_MB}  (per-chip SRAM cap)")
    print(f"#   mac_per_vac    = {args.mac_per_vac}  (E-dim reduction width)")
    print(f"#   models         = {args.models or 'ALL'}")
    print(f"#   arch           = {args.arch}")
    print(f"#   workers        = {args.workers}")
    print(f"#   tag            = {args.tag or '(auto)'}")
    print(f"# Reproduce:")
    repro = (f"#   python -m llmforge.hw.rdxe.workflow "
             f"--ctx {args.ctx} --prefill {args.prefill} --decode {args.decode} "
             f"--users {args.users} --workers {args.workers} --arch {args.arch} "
             f"--max-chips {args.max_chips} --max-wmem-MB {args.max_wmem_MB} "
             f"--mac-per-vac {args.mac_per_vac}")
    if args.models:
        repro += f" --models {','.join(args.models)}"
    if args.skip_iha: repro += " --skip-iha"
    if args.tag:      repro += f" --tag {args.tag}"
    print(repro)
    print(f"{'#'*96}\n")


def _parse_sweep(spec):
    """'prefill=512,1024,2048' -> ('prefill', [512,1024,2048])."""
    k, v = spec.split('=', 1)
    return k.strip(), [int(x) for x in v.split(',')]


def main():
    ap = argparse.ArgumentParser(description=(
        "rDXE ring workflow with Timeloop-backed GEMMs. "
        "Tune prefill / decode / users / ctx to explore TTFT/TPOT/E-per-token "
        "under different serving regimes. Every run writes a CSV + PNG "
        "tagged by config so results are reproducible and diffable."))
    # --- serving-scenario knobs -------------------------------------------
    ap.add_argument('--ctx', type=int, default=2048,
                    help='Context window (KV length during decode)')
    ap.add_argument('--prefill', type=int, default=512,
                    help='Prompt (prefill) tokens; drives TTFT')
    ap.add_argument('--decode', type=int, default=256,
                    help='Decode horizon (tokens). Default 256 matches '
                         'typical chat-response length. Decode-only E/tok '
                         'reported here is invariant to this value.')
    ap.add_argument('--users', type=int, default=1,
                    help='Concurrent users in one forward pass (decode batch)')
    # --- platform feasibility constraints ---------------------------------
    ap.add_argument('--max-chips', type=int, default=16,
                    help='Upper bound on ring size (e.g. 16 = edge device). '
                         'The packer must fit every model into ≤ this many chips.')
    ap.add_argument('--max-wmem-MB', type=float, default=64.0,
                    help='Upper bound on per-chip WMEM (SRAM) in MB. '
                         '64 MB ≈ high-end mobile / Jetson Orin class.')
    ap.add_argument('--mac-per-vac', type=int, default=N_MAC_PER_VAC,
                    help=f'MACs per VAC core (E-dim reduction width). '
                         f'Default {N_MAC_PER_VAC} matches the reference design. '
                         f'Common sweep values: 16, 32, 64.')
    # --- model selection --------------------------------------------------
    ap.add_argument('--models', type=str, default=None,
                    help='Comma-separated subset of MODELS to run '
                         '(default: all 5). Example: '
                         'A_ultra_thin,C_balanced,E_ultra_wide')
    ap.add_argument('--skip-iha', action='store_true',
                    help='Skip Phase C (IHA variant isolation)')
    # --- Timeloop / infra -------------------------------------------------
    ap.add_argument('--workers', type=int, default=8,
                    help='Parallel Timeloop mappers during prefetch')
    ap.add_argument('--arch', type=str, default='dxe_relaxed')
    ap.add_argument('--tag', type=str, default='',
                    help='Suffix for output files (default: auto from args)')
    # --- batch sweep ------------------------------------------------------
    ap.add_argument('--sweep', type=str, default=None,
                    help=('Run multiple configurations sequentially by varying '
                          'one knob, e.g. --sweep prefill=128,512,2048   '
                          'or --sweep users=1,2,4,8'))
    args = ap.parse_args()

    # Restrict MODELS by --models if requested
    if args.models:
        wanted = [m.strip() for m in args.models.split(',')]
        for m in wanted:
            if m not in MODELS:
                raise SystemExit(f"Unknown model: {m}. "
                                 f"Known: {list(MODELS.keys())}")
        for k in list(MODELS.keys()):
            if k not in wanted:
                del MODELS[k]
        args.models = wanted

    if args.sweep:
        key, vals = _parse_sweep(args.sweep)
        if not hasattr(args, key):
            raise SystemExit(f"Cannot sweep unknown knob: {key}")
        base_tag = args.tag
        for v in vals:
            setattr(args, key, v)
            args.tag = f"{base_tag}_{key}{v}" if base_tag else f"{key}{v}"
            _print_config_banner(args)
            run_all(args)
    else:
        _print_config_banner(args)
        run_all(args)


if __name__ == '__main__':
    main()
