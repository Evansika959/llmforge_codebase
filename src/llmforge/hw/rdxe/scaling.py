"""Edge LLM scaling analysis on scaled DXE ring architecture.

For each model, scales the DXE chip to fit one layer per chip,
then simulates a ring of N chips (N = n_layers). Reports
energy, TTFT, TPOT, memory stats, and chip area.

Usage:
    python -m llmforge.hw.rdxe.scaling
    python -m llmforge.hw.rdxe.scaling --models smollm_135m,qwen25_05b,llama32_1b
    python -m llmforge.hw.rdxe.scaling --context 256,512,1024
"""

import argparse
import csv
import math
import os
import time

from llmforge import paths

from .core.constants import (
    CLOCK_PERIOD_NS, CLOCK_FREQ_HZ,
    WMEM_ENERGY_PER_ACCESS_PJ, KV_CACHE_ENERGY_PER_ACCESS_PJ,
    DRAM_ENERGY_PER_ACCESS_PJ, INTER_CHIP_ENERGY_PJ_PER_BIT,
    INTER_CHIP_BW_BYTES_PER_CYCLE, INTER_CHIP_LATENCY_CYCLES,
    VRC_SOFTMAX_PJ_PER_ELEMENT, VRC_RMSNORM_PJ_PER_ELEMENT,
)
from .core.scaled_arch import (
    ScaledChipSpec, DXE_REFERENCE,
    EDGE_MODELS, layer_weight_bytes, kv_bytes_per_token_per_layer,
    scale_chip_for_layer, scale_ring_for_model,
)


def compute_layer_decode_energy(layer_spec: dict, n_embd: int,
                                context: int, chip: ScaledChipSpec
                                ) -> dict:
    """Analytically compute one layer's decode energy and cycles.

    All weights on-chip (WMEM), KV cache on-chip.
    No DRAM access during decode. No Timeloop — pure analytical model
    based on MAC count, SRAM access patterns, and chip dimensions.

    For decode (L=1):
      Each GEMV reads ALL weights once from WMEM and produces outputs.
      Total weight reads = layer_weight_bytes
      Total MACs = layer_weight_bytes (1 MAC per weight for GEMV)
      Cycles = total_MACs / chip.total_macs (fully utilized)
    """
    nh = layer_spec['n_head']
    nkv = layer_spec['n_kv_group']
    qk = layer_spec['n_qk_head_dim']
    vd = layer_spec['n_v_head_dim']
    mlp = layer_spec['mlp_size']
    attn = layer_spec.get('attention_variant', 'infinite')

    # Projection GEMVs: weight reads = weight bytes, MACs = weight bytes
    if attn == 'infinite':
        # Projection weights
        proj_weights = (n_embd * qk * (nh + nkv) +  # QK_gen
                        n_embd * vd * nkv +           # V_gen
                        vd * nh * n_embd +             # ATTN_proj
                        n_embd * mlp +                 # MLP_FC1
                        mlp * n_embd)                  # MLP_FC2

        # Attention GEMVs (not weight-bound, KV-cache-bound)
        # QK_attn: q[qk] × K_cache[qk, ctx] → scores[ctx]
        # MACs = qk * ctx * n_head
        qk_attn_macs = qk * context * nh
        # PV_attn: scores[ctx] × V_cache[ctx, vd] → attended[vd]
        pv_attn_macs = context * vd * nh

        # KV cache reads
        # K: qk * context * n_kv_groups (shared via VLINK for GQA)
        kv_k_reads = qk * context * nkv
        # V: context * vd * n_kv_groups (column-wise, iWuR)
        kv_v_reads = context * vd * nkv
        # KV writes (new token): K + V for all kv groups
        kv_writes = nkv * (qk + vd)

        # VRC
        softmax_elements = context * nh
        rmsnorm_elements = n_embd * 2  # post-attn + post-mlp
    else:
        proj_weights = n_embd * mlp + mlp * n_embd
        qk_attn_macs = 0
        pv_attn_macs = 0
        kv_k_reads = 0
        kv_v_reads = 0
        kv_writes = 0
        softmax_elements = 0
        rmsnorm_elements = n_embd

    total_macs = proj_weights + qk_attn_macs + pv_attn_macs

    # Cycles: limited by compute or memory bandwidth, whichever is slower
    # Compute cycles = total_macs / chip.total_macs
    compute_cycles = math.ceil(total_macs / chip.total_macs)

    # WMEM bandwidth: each core reads at 16 B/cycle (128b width / 8b datawidth)
    # Total WMEM read bandwidth = n_cores * 16 B/cycle
    wmem_bw = chip.n_cores * 16  # bytes/cycle
    wmem_read_cycles = math.ceil(proj_weights / wmem_bw)

    # KV cache bandwidth: each core reads at 16 B/cycle
    kv_read_bytes = kv_k_reads + kv_v_reads
    kv_bw = chip.n_cores * 16
    kv_read_cycles = math.ceil(kv_read_bytes / kv_bw) if kv_read_bytes > 0 else 0

    # Actual cycles = max(compute, memory) — compute and memory overlap
    # For projection GEMVs: compute-bound (weight streaming matches MAC rate)
    # For attention: KV-cache-bandwidth-bound for large contexts
    layer_cycles = max(compute_cycles, wmem_read_cycles + kv_read_cycles)

    # Energy
    # WMEM read energy: scale by chip's WMEM size
    wmem_energy_pJ = proj_weights * WMEM_ENERGY_PER_ACCESS_PJ * chip.wmem_energy_scale

    # KV cache read energy
    kv_read_energy_pJ = (kv_k_reads * KV_CACHE_ENERGY_PER_ACCESS_PJ *
                         chip.kv_energy_scale)
    kv_read_energy_pJ += (kv_v_reads * KV_CACHE_ENERGY_PER_ACCESS_PJ *
                          chip.kv_energy_scale * 1.2)  # iWuR V penalty

    # KV cache write energy
    kv_write_energy_pJ = (kv_writes * KV_CACHE_ENERGY_PER_ACCESS_PJ *
                          chip.kv_energy_scale * 1.06)  # write slightly costlier

    # MAC compute energy: ~0.07 pJ/MAC (from ERT)
    mac_energy_pJ = total_macs * 0.07

    # VRC energy
    vrc_energy_pJ = (softmax_elements * VRC_SOFTMAX_PJ_PER_ELEMENT +
                     rmsnorm_elements * VRC_RMSNORM_PJ_PER_ELEMENT)

    total_energy_pJ = (wmem_energy_pJ + kv_read_energy_pJ +
                       kv_write_energy_pJ + mac_energy_pJ + vrc_energy_pJ)

    return {
        'total_macs': total_macs,
        'proj_weight_bytes': proj_weights,
        'kv_k_reads': kv_k_reads,
        'kv_v_reads': kv_v_reads,
        'kv_writes': kv_writes,
        'cycles': layer_cycles,
        'compute_cycles': compute_cycles,
        'wmem_read_cycles': wmem_read_cycles,
        'kv_read_cycles': kv_read_cycles,
        'total_energy_pJ': total_energy_pJ,
        'wmem_energy_pJ': wmem_energy_pJ,
        'kv_read_energy_pJ': kv_read_energy_pJ,
        'kv_write_energy_pJ': kv_write_energy_pJ,
        'mac_energy_pJ': mac_energy_pJ,
        'vrc_energy_pJ': vrc_energy_pJ,
    }


def compute_prefill_layer_energy(layer_spec: dict, n_embd: int,
                                  prefill_len: int, chip: ScaledChipSpec
                                  ) -> dict:
    """Analytically compute one layer's prefill energy (batch GEMM, L tokens)."""
    nh = layer_spec['n_head']
    nkv = layer_spec['n_kv_group']
    qk = layer_spec['n_qk_head_dim']
    vd = layer_spec['n_v_head_dim']
    mlp = layer_spec['mlp_size']
    attn = layer_spec.get('attention_variant', 'infinite')

    L = prefill_len

    if attn == 'infinite':
        # Projection GEMMs: weight bytes reused L times
        proj_weights = (n_embd * qk * (nh + nkv) +
                        n_embd * vd * nkv +
                        vd * nh * n_embd +
                        n_embd * mlp +
                        mlp * n_embd)
        proj_macs = proj_weights * L

        # Attention: Q[L, qk] × K[qk, L] → [L, L] for each head group
        qk_attn_macs = qk * L * L * nh
        pv_attn_macs = L * vd * L * nh

        # KV cache: populated during prefill, no reads from cache
        # (Q, K, V computed fresh in prefill)
        kv_writes = nkv * (qk + vd) * L

        softmax_elements = L * L * nh
        rmsnorm_elements = n_embd * L * 2
    else:
        proj_weights = n_embd * mlp + mlp * n_embd
        proj_macs = proj_weights * L
        qk_attn_macs = 0
        pv_attn_macs = 0
        kv_writes = 0
        softmax_elements = 0
        rmsnorm_elements = n_embd * L

    total_macs = proj_macs + qk_attn_macs + pv_attn_macs
    compute_cycles = math.ceil(total_macs / chip.total_macs)

    # WMEM reads: each weight read L times (temporal reuse over sequence)
    wmem_reads = proj_weights * L
    wmem_energy_pJ = wmem_reads * WMEM_ENERGY_PER_ACCESS_PJ * chip.wmem_energy_scale
    mac_energy_pJ = total_macs * 0.07
    kv_write_energy_pJ = (kv_writes * KV_CACHE_ENERGY_PER_ACCESS_PJ *
                          chip.kv_energy_scale * 1.06)
    vrc_energy_pJ = (softmax_elements * VRC_SOFTMAX_PJ_PER_ELEMENT +
                     rmsnorm_elements * VRC_RMSNORM_PJ_PER_ELEMENT)

    total_energy_pJ = wmem_energy_pJ + mac_energy_pJ + kv_write_energy_pJ + vrc_energy_pJ

    return {
        'total_macs': total_macs,
        'cycles': compute_cycles,
        'total_energy_pJ': total_energy_pJ,
        'wmem_energy_pJ': wmem_energy_pJ,
        'mac_energy_pJ': mac_energy_pJ,
        'kv_write_energy_pJ': kv_write_energy_pJ,
        'vrc_energy_pJ': vrc_energy_pJ,
    }


def simulate_ring(model_spec: dict, chips: list,
                  prefill_len: int, decode_len: int,
                  max_context: int) -> dict:
    """Simulate full inference on a scaled ring.

    Each chip holds one layer. Token-level pipeline.
    All weights and KV cache on-chip. Zero DRAM during inference.
    """
    n_embd = model_spec['n_embd']
    layers = model_spec['layers']
    n_layers = len(layers)
    total_tokens = prefill_len + decode_len

    # Per-layer decode cost at average context
    avg_ctx = prefill_len + decode_len // 2
    avg_ctx = min(avg_ctx, max_context)
    avg_ctx = max(avg_ctx, 1)

    layer_decode_stats = []
    for li, layer in enumerate(layers):
        s = compute_layer_decode_energy(layer, n_embd, avg_ctx, chips[li])
        layer_decode_stats.append(s)

    # Per-layer prefill cost
    layer_prefill_stats = []
    if prefill_len > 0:
        for li, layer in enumerate(layers):
            s = compute_prefill_layer_energy(layer, n_embd, prefill_len, chips[li])
            layer_prefill_stats.append(s)

    # Pipeline: stage time = max layer cycle time + inter-chip transfer
    transfer_bits = n_embd * 8
    transfer_energy_pJ = transfer_bits * INTER_CHIP_ENERGY_PJ_PER_BIT
    transfer_cycles = INTER_CHIP_LATENCY_CYCLES + n_embd / INTER_CHIP_BW_BYTES_PER_CYCLE

    # Decode stage: bottleneck chip
    decode_layer_cycles = [s['cycles'] for s in layer_decode_stats]
    max_decode_cycles = max(decode_layer_cycles)
    decode_stage_cycles = max_decode_cycles + transfer_cycles

    # Prefill stage: bottleneck chip
    if prefill_len > 0:
        prefill_layer_cycles = [s['cycles'] for s in layer_prefill_stats]
        max_prefill_cycles = max(prefill_layer_cycles)
        prefill_stage_cycles = max_prefill_cycles + transfer_cycles
    else:
        prefill_stage_cycles = 0

    # TTFT = prefill pipeline drain + first decode stage
    # Prefill: each token must traverse all N chips
    # Pipeline: first token takes N stages, subsequent tokens overlap
    if prefill_len > 0:
        # All prefill tokens through pipeline: (prefill_len + N - 1) stages
        prefill_total_cycles = (prefill_len + n_layers - 1) * prefill_stage_cycles
        ttft_cycles = prefill_total_cycles + decode_stage_cycles * n_layers
    else:
        prefill_total_cycles = 0
        ttft_cycles = decode_stage_cycles * n_layers

    ttft_ms = ttft_cycles * CLOCK_PERIOD_NS / 1e6

    # TPOT = one decode stage (steady state)
    tpot_cycles = decode_stage_cycles
    tpot_us = tpot_cycles * CLOCK_PERIOD_NS / 1e3

    # Decode total
    decode_total_cycles = decode_len * decode_stage_cycles

    # Total latency
    total_cycles = prefill_total_cycles + decode_total_cycles
    total_latency_ms = total_cycles * CLOCK_PERIOD_NS / 1e6

    # Energy per decode token (sum across all layers + inter-chip comm)
    decode_energy_per_token = sum(s['total_energy_pJ'] for s in layer_decode_stats)
    comm_energy_per_token = transfer_energy_pJ * (n_layers - 1)  # N-1 hops
    total_decode_e_per_token = decode_energy_per_token + comm_energy_per_token

    # Prefill energy
    if prefill_len > 0:
        prefill_total_energy = sum(s['total_energy_pJ'] for s in layer_prefill_stats)
        prefill_comm_energy = transfer_energy_pJ * (n_layers - 1) * prefill_len
        prefill_total_energy += prefill_comm_energy
    else:
        prefill_total_energy = 0

    # Amortized energy per token
    total_energy_pJ = prefill_total_energy + total_decode_e_per_token * decode_len
    amortized_e_per_token = total_energy_pJ / total_tokens if total_tokens > 0 else 0

    # Throughput
    throughput = total_tokens / (total_latency_ms / 1e3) if total_latency_ms > 0 else 0
    steady_state_tps = 1.0 / (tpot_cycles * CLOCK_PERIOD_NS * 1e-9) if tpot_cycles > 0 else 0

    # Memory access totals (per decode token)
    total_wmem_reads = sum(s['proj_weight_bytes'] for s in layer_decode_stats)
    total_kv_reads = sum(s['kv_k_reads'] + s['kv_v_reads'] for s in layer_decode_stats)
    total_kv_writes = sum(s['kv_writes'] for s in layer_decode_stats)
    total_macs_decode = sum(s['total_macs'] for s in layer_decode_stats)

    # Arithmetic intensity
    total_bytes = total_wmem_reads + total_kv_reads + total_kv_writes
    ai = total_macs_decode * 2 / total_bytes if total_bytes > 0 else 0  # ops = 2 * MACs

    # Per-chip utilization
    per_chip_util = [s['cycles'] / max_decode_cycles if max_decode_cycles > 0 else 0
                     for s in layer_decode_stats]

    # Energy breakdown (per decode token)
    wmem_e = sum(s['wmem_energy_pJ'] for s in layer_decode_stats)
    kv_read_e = sum(s['kv_read_energy_pJ'] for s in layer_decode_stats)
    kv_write_e = sum(s['kv_write_energy_pJ'] for s in layer_decode_stats)
    mac_e = sum(s['mac_energy_pJ'] for s in layer_decode_stats)
    vrc_e = sum(s['vrc_energy_pJ'] for s in layer_decode_stats)

    # Chip area
    total_area = sum(c.estimated_area_mm2 for c in chips)
    total_sram = sum(c.total_sram_B for c in chips)
    total_macs_hw = sum(c.total_macs for c in chips)

    return {
        'model': model_spec.get('name', ''),
        'n_layers': n_layers,
        'n_embd': n_embd,
        'n_chips': n_layers,
        'prefill': prefill_len,
        'decode': decode_len,
        'max_context': max_context,
        'total_tokens': total_tokens,
        # Chip specs (use bottleneck chip as representative)
        'chip_macs': chips[0].total_macs,
        'chip_cores': chips[0].n_cores,
        'chip_wmem_KB': chips[0].wmem_per_core_B // 1024,
        'chip_kv_KB': chips[0].kv_cache_per_core_B // 1024,
        'chip_sram_MB': chips[0].total_sram_B / 1e6,
        'total_sram_MB': total_sram / 1e6,
        'total_macs_hw': total_macs_hw,
        'total_area_mm2': total_area,
        # Serving metrics
        'energy_per_token_uJ': amortized_e_per_token / 1e6,
        'decode_e_per_token_uJ': total_decode_e_per_token / 1e6,
        'ttft_ms': ttft_ms,
        'tpot_us': tpot_us,
        'total_latency_ms': total_latency_ms,
        'throughput_tps': throughput,
        'ss_throughput_tps': steady_state_tps,
        # Energy breakdown (per decode token, uJ)
        'wmem_energy_uJ': wmem_e / 1e6,
        'kv_read_energy_uJ': kv_read_e / 1e6,
        'kv_write_energy_uJ': kv_write_e / 1e6,
        'mac_energy_uJ': mac_e / 1e6,
        'vrc_energy_uJ': vrc_e / 1e6,
        'comm_energy_uJ': comm_energy_per_token / 1e6,
        # Memory access (per decode token)
        'wmem_reads': total_wmem_reads,
        'kv_reads': total_kv_reads,
        'kv_writes': total_kv_writes,
        'total_macs_per_tok': total_macs_decode,
        'arithmetic_intensity': ai,
        'dram_accesses': 0,  # all on-chip
        # Utilization
        'avg_util': sum(per_chip_util) / len(per_chip_util) if per_chip_util else 0,
        'bottleneck_chip': decode_layer_cycles.index(max_decode_cycles),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Edge LLM Scaling on DXE Ring Architecture")
    parser.add_argument("--models", type=str,
                        default=",".join(EDGE_MODELS.keys()),
                        help="Comma-separated model names")
    parser.add_argument("--context", type=str, default="256",
                        help="Comma-separated max context lengths")
    parser.add_argument("--prefill", type=int, default=128,
                        help="Prefill token count")
    parser.add_argument("--decode", type=int, default=64,
                        help="Decode token count")
    parser.add_argument("--n-users", type=int, default=1,
                        help="Concurrent users")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--no-plot", action="store_true")

    args = parser.parse_args()

    model_names = [m.strip() for m in args.models.split(',')]
    ctx_list = [int(c) for c in args.context.split(',')]
    output_dir = args.output_dir or str(paths.RUNS / 'rdxe' / 'results')
    plot_dir = str(paths.RUNS / 'rdxe' / 'plots')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    print("=" * 100)
    print("Edge LLM Scaling Analysis — DXE Ring Architecture")
    print("=" * 100)
    print(f"Models: {model_names}")
    print(f"Context: {ctx_list}, Prefill: {args.prefill}, Decode: {args.decode}")
    print(f"Users: {args.n_users}")
    print()

    all_results = []

    for model_name in model_names:
        if model_name not in EDGE_MODELS:
            print(f"  Unknown model: {model_name}, skipping")
            continue

        model = EDGE_MODELS[model_name]
        n_embd = model['n_embd']
        layers = model['layers']
        n_layers = len(layers)

        # Per-layer weight size
        layer_wb = layer_weight_bytes(layers[0], n_embd)
        total_wb = layer_wb * n_layers
        kv_per_tok = kv_bytes_per_token_per_layer(layers[0])

        print(f"{'='*80}")
        print(f"  {model['name']}: {n_layers}L, d={n_embd}, "
              f"{total_wb/1e6:.1f}MB weights ({layer_wb/1024:.1f}KB/layer)")
        print(f"  KV/token/layer: {kv_per_tok} bytes")

        for max_ctx in ctx_list:
            # Scale chips for this model
            chips = scale_ring_for_model(model, max_context=max_ctx,
                                         n_users=args.n_users)
            chip0 = chips[0]

            print(f"\n  Context={max_ctx}:")
            print(f"    Chip: {chip0.n_dxt}×{chip0.n_vac_per_dxt}×{chip0.n_mac_per_vac} = "
                  f"{chip0.total_macs} MACs, "
                  f"WMEM={chip0.wmem_per_core_B//1024}KB/core "
                  f"({chip0.wmem_total_B/1e6:.2f}MB), "
                  f"KV$={chip0.kv_cache_per_core_B//1024}KB/core, "
                  f"area~{chip0.estimated_area_mm2:.1f}mm²")

            r = simulate_ring(model, chips, args.prefill, args.decode, max_ctx)
            all_results.append(r)

            print(f"    E/tok: {r['energy_per_token_uJ']:.3f} uJ | "
                  f"TTFT: {r['ttft_ms']:.3f} ms | "
                  f"TPOT: {r['tpot_us']:.2f} us | "
                  f"Tput: {r['ss_throughput_tps']:.0f} tok/s")

    print()

    # Summary table
    print("=" * 140)
    print(f"{'Model':>18s} | {'Layers':>6s} | {'d':>5s} | "
          f"{'Wt(MB)':>7s} | {'Chips':>5s} | {'MACs':>6s} | "
          f"{'SRAM/chip':>9s} | {'Area(mm²)':>9s} | "
          f"{'E/tok(uJ)':>10s} | {'TTFT(ms)':>9s} | {'TPOT(us)':>9s} | "
          f"{'Tput(t/s)':>9s} | {'AI':>5s} | {'DRAM':>4s}")
    print("-" * 140)

    for r in all_results:
        total_wb = r['wmem_reads']  # = layer weights * n_layers (per token)
        per_layer_wb = total_wb / r['n_layers'] if r['n_layers'] > 0 else 0

        print(f"{r['model']:>18s} | {r['n_layers']:>6d} | {r['n_embd']:>5d} | "
              f"{per_layer_wb*r['n_layers']/1e6:>7.1f} | {r['n_chips']:>5d} | "
              f"{r['chip_macs']:>6d} | "
              f"{r['chip_sram_MB']:>7.2f}MB | {r['total_area_mm2']:>9.1f} | "
              f"{r['energy_per_token_uJ']:>10.3f} | {r['ttft_ms']:>9.3f} | "
              f"{r['tpot_us']:>9.2f} | {r['ss_throughput_tps']:>9.0f} | "
              f"{r['arithmetic_intensity']:>5.1f} | {'None':>4s}")

    print()

    # Energy breakdown table
    print("Energy Breakdown per Decode Token (uJ):")
    print(f"{'Model':>18s} | {'WMEM':>8s} | {'KV-Read':>8s} | {'KV-Write':>8s} | "
          f"{'MAC':>8s} | {'VRC':>8s} | {'Comm':>8s} | {'Total':>8s}")
    print("-" * 90)
    for r in all_results:
        print(f"{r['model']:>18s} | {r['wmem_energy_uJ']:>8.4f} | "
              f"{r['kv_read_energy_uJ']:>8.4f} | {r['kv_write_energy_uJ']:>8.4f} | "
              f"{r['mac_energy_uJ']:>8.4f} | {r['vrc_energy_uJ']:>8.4f} | "
              f"{r['comm_energy_uJ']:>8.4f} | {r['decode_e_per_token_uJ']:>8.4f}")

    print()

    # Memory access table
    print("Memory Access per Decode Token:")
    print(f"{'Model':>18s} | {'WMEM Reads':>12s} | {'KV Reads':>12s} | "
          f"{'KV Writes':>10s} | {'Total MACs':>12s} | {'AI(Op/B)':>8s}")
    print("-" * 85)
    for r in all_results:
        def fmt(v):
            if v >= 1e6: return f"{v/1e6:.2f}M"
            if v >= 1e3: return f"{v/1e3:.1f}K"
            return f"{v:.0f}"
        print(f"{r['model']:>18s} | {fmt(r['wmem_reads']):>12s} | "
              f"{fmt(r['kv_reads']):>12s} | {fmt(r['kv_writes']):>10s} | "
              f"{fmt(r['total_macs_per_tok']):>12s} | {r['arithmetic_intensity']:>8.2f}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'edge_llm_scaling.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        w.writeheader()
        for r in all_results:
            w.writerow({k: f"{v:.6f}" if isinstance(v, float) else v
                        for k, v in r.items()})
    print(f"\nCSV: {csv_path}")

    # Plot
    if not args.no_plot and len(all_results) > 1:
        plot_scaling(all_results, plot_dir)


def plot_scaling(results, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    models = [r['model'] for r in results]
    n = len(models)
    x = np.arange(n)

    fig, axes = plt.subplots(2, 3, figsize=(22, 12), dpi=150)
    fig.suptitle("Edge LLM Scaling on DXE Ring Architecture\n"
                 "1 layer/chip, all weights + KV on SRAM, zero DRAM during inference",
                 fontsize=13, fontweight='bold')

    bar_w = 0.6
    label_rot = 30

    # Energy per token
    ax = axes[0, 0]
    vals = [r['energy_per_token_uJ'] for r in results]
    bars = ax.bar(x, vals, bar_w, color='#4e79a7', edgecolor='k', linewidth=0.5)
    for i, v in enumerate(vals):
        ax.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=label_rot, ha='right', fontsize=7)
    ax.set_ylabel('Energy/Token (uJ)'); ax.set_title('Energy per Token')

    # TTFT
    ax = axes[0, 1]
    vals = [r['ttft_ms'] for r in results]
    ax.bar(x, vals, bar_w, color='#f28e2b', edgecolor='k', linewidth=0.5)
    for i, v in enumerate(vals):
        ax.text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=label_rot, ha='right', fontsize=7)
    ax.set_ylabel('TTFT (ms)'); ax.set_title('Time to First Token')

    # TPOT
    ax = axes[0, 2]
    vals = [r['tpot_us'] for r in results]
    ax.bar(x, vals, bar_w, color='#e15759', edgecolor='k', linewidth=0.5)
    for i, v in enumerate(vals):
        ax.text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=label_rot, ha='right', fontsize=7)
    ax.set_ylabel('TPOT (us)'); ax.set_title('Time per Output Token')

    # Chip area
    ax = axes[1, 0]
    vals = [r['total_area_mm2'] for r in results]
    ax.bar(x, vals, bar_w, color='#76b7b2', edgecolor='k', linewidth=0.5)
    for i, v in enumerate(vals):
        ax.text(i, v, f'{v:.0f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=label_rot, ha='right', fontsize=7)
    ax.set_ylabel('Total Ring Area (mm²)'); ax.set_title('Estimated Area (all chips)')

    # Energy breakdown stacked
    ax = axes[1, 1]
    wmem = [r['wmem_energy_uJ'] for r in results]
    kv = [r['kv_read_energy_uJ'] + r['kv_write_energy_uJ'] for r in results]
    mac = [r['mac_energy_uJ'] for r in results]
    vrc = [r['vrc_energy_uJ'] for r in results]
    comm = [r['comm_energy_uJ'] for r in results]

    bottoms = np.zeros(n)
    for vals, label, color in [
        (mac, 'MAC Compute', '#4e79a7'),
        (wmem, 'WMEM Read', '#f28e2b'),
        (kv, 'KV Cache R+W', '#e15759'),
        (vrc, 'VRC', '#76b7b2'),
        (comm, 'Inter-chip', '#59a14f'),
    ]:
        ax.bar(x, vals, bar_w, bottom=bottoms, color=color, edgecolor='white',
               linewidth=0.3, label=label)
        bottoms += np.array(vals)
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=label_rot, ha='right', fontsize=7)
    ax.set_ylabel('Energy (uJ/decode tok)'); ax.set_title('Decode Energy Breakdown')
    ax.legend(fontsize=7)

    # Arithmetic intensity
    ax = axes[1, 2]
    vals = [r['arithmetic_intensity'] for r in results]
    ax.bar(x, vals, bar_w, color='#59a14f', edgecolor='k', linewidth=0.5)
    for i, v in enumerate(vals):
        ax.text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=label_rot, ha='right', fontsize=7)
    ax.set_ylabel('Op/Byte'); ax.set_title('Arithmetic Intensity (decode)')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(output_dir, 'edge_llm_scaling.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"Plot: {path}")


if __name__ == "__main__":
    main()
