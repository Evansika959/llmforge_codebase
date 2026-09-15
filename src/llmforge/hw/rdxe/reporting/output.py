"""Output: CSV export and matplotlib plots for simulation results."""

import csv
import os
from typing import List, Optional

from ..simulator.chip_sim import InferenceResult, TokenTrace
from ..simulator.ring_sim import RingResult


def save_inference_csv(result: InferenceResult, path: str):
    """Save per-token trace to CSV."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    fields = ['token_idx', 'phase', 'context_length',
              'energy_pJ', 'cycles', 'latency_us',
              'compute_pJ', 'kv_cache_pJ', 'kv_write_pJ',
              'dram_pJ', 'vrc_pJ', 'wmem_reload_pJ',
              'kv_correction_pJ', 'vlink_correction_pJ']

    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for t in result.token_traces:
            w.writerow({
                'token_idx': t.token_idx,
                'phase': t.phase,
                'context_length': t.context_length,
                'energy_pJ': f'{t.total_energy_pJ:.2f}',
                'cycles': f'{t.total_cycles:.0f}',
                'latency_us': f'{t.latency_us:.3f}',
                'compute_pJ': f'{t.compute_energy_pJ:.2f}',
                'kv_cache_pJ': f'{t.kv_cache_energy_pJ:.2f}',
                'kv_write_pJ': f'{t.kv_cache_write_energy_pJ:.2f}',
                'dram_pJ': f'{t.dram_energy_pJ:.2f}',
                'vrc_pJ': f'{t.vrc_energy_pJ:.2f}',
                'wmem_reload_pJ': f'{t.wmem_reload_energy_pJ:.2f}',
                'kv_correction_pJ': f'{t.kv_correction_pJ:.2f}',
                'vlink_correction_pJ': f'{t.vlink_correction_pJ:.2f}',
            })


def save_summary_csv(result: InferenceResult, path: str):
    """Save aggregated summary."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'value', 'unit'])
        w.writerow(['model', result.model_name, ''])
        w.writerow(['n_layers', result.n_layers, ''])
        w.writerow(['n_embd', result.n_embd, ''])
        w.writerow(['prefill_length', result.prefill_length, 'tokens'])
        w.writerow(['decode_length', result.decode_length, 'tokens'])
        w.writerow(['total_tokens', result.total_tokens, 'tokens'])
        w.writerow(['total_energy', f'{result.total_energy_uJ:.4f}', 'uJ'])
        w.writerow(['energy_per_token', f'{result.energy_per_token_uJ:.4f}', 'uJ/tok'])
        w.writerow(['total_latency', f'{result.total_latency_ms:.4f}', 'ms'])
        w.writerow(['latency_per_token', f'{result.latency_per_token_us:.3f}', 'us/tok'])
        w.writerow(['tokens_per_second', f'{result.tokens_per_second:.0f}', 'tok/s'])
        w.writerow(['compute_energy', f'{result.compute_energy_uJ:.4f}', 'uJ'])
        w.writerow(['kv_cache_energy', f'{result.kv_cache_energy_uJ:.4f}', 'uJ'])
        w.writerow(['dram_energy', f'{result.dram_energy_uJ:.4f}', 'uJ'])
        w.writerow(['vrc_energy', f'{result.vrc_energy_uJ:.4f}', 'uJ'])
        w.writerow(['wmem_reload_energy', f'{result.wmem_reload_energy_uJ:.4f}', 'uJ'])
        w.writerow(['kv_correction', f'{result.total_kv_correction_uJ:.4f}', 'uJ'])
        w.writerow(['vlink_correction', f'{result.total_vlink_correction_uJ:.4f}', 'uJ'])


def plot_inference(result: InferenceResult, output_dir: str):
    """Plot per-token energy trace and breakdown."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)

    decode_traces = [t for t in result.token_traces if t.phase == "decode"]
    if not decode_traces:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
    fig.suptitle(
        f"DXE-Sim: {result.model_name} ({result.n_layers}L, d={result.n_embd})\n"
        f"Prefill={result.prefill_length}, Decode={result.decode_length} | "
        f"E/tok={result.energy_per_token_uJ:.3f} uJ",
        fontsize=12, fontweight='bold')

    # Plot 1: Energy per token vs token index
    ax = axes[0, 0]
    tok_idx = [t.token_idx for t in decode_traces]
    energies = [t.total_energy_pJ / 1e6 for t in decode_traces]
    ax.plot(tok_idx, energies, 'o-', markersize=3, linewidth=1.5, color='#1f77b4')
    ax.set_xlabel('Token Index')
    ax.set_ylabel('Energy per Token (uJ)')
    ax.set_title('Energy vs Token Position (Decode)')
    ax.grid(alpha=0.3)

    # Plot 2: Stacked energy breakdown per token
    ax = axes[0, 1]
    compute = np.array([t.compute_energy_pJ / 1e6 for t in decode_traces])
    kv_read = np.array([t.kv_cache_energy_pJ / 1e6 for t in decode_traces])
    kv_write = np.array([t.kv_cache_write_energy_pJ / 1e6 for t in decode_traces])
    vrc = np.array([t.vrc_energy_pJ / 1e6 for t in decode_traces])
    wmem = np.array([t.wmem_reload_energy_pJ / 1e6 for t in decode_traces])
    x = np.arange(len(decode_traces))

    ax.fill_between(x, 0, compute, alpha=0.7, label='Compute+SRAM', color='#4e79a7')
    ax.fill_between(x, compute, compute + kv_read, alpha=0.7,
                    label='KV Cache Read', color='#f28e2b')
    ax.fill_between(x, compute + kv_read, compute + kv_read + kv_write, alpha=0.7,
                    label='KV Cache Write', color='#e15759')
    ax.fill_between(x, compute + kv_read + kv_write,
                    compute + kv_read + kv_write + vrc, alpha=0.7,
                    label='VRC (Softmax/RMSNorm)', color='#76b7b2')
    if wmem.any():
        ax.fill_between(x, compute + kv_read + kv_write + vrc,
                        compute + kv_read + kv_write + vrc + wmem, alpha=0.7,
                        label='WMEM Reload', color='#59a14f')
    ax.set_xlabel('Decode Step')
    ax.set_ylabel('Energy (uJ)')
    ax.set_title('Energy Breakdown per Token')
    ax.legend(fontsize=7, loc='upper left')

    # Plot 3: Pie chart — average decode token breakdown
    ax = axes[1, 0]
    avg_compute = float(np.mean(compute))
    avg_kv = float(np.mean(kv_read + kv_write))
    avg_vrc = float(np.mean(vrc))
    avg_wmem = float(np.mean(wmem))

    sizes = [avg_compute, avg_kv, avg_vrc]
    labels = [f'Compute+SRAM\n{avg_compute:.4f} uJ',
              f'KV Cache\n{avg_kv:.4f} uJ',
              f'VRC\n{avg_vrc:.4f} uJ']
    colors = ['#4e79a7', '#f28e2b', '#76b7b2']
    if avg_wmem > 0:
        sizes.append(avg_wmem)
        labels.append(f'WMEM Reload\n{avg_wmem:.4f} uJ')
        colors.append('#59a14f')

    total = sum(sizes)
    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, colors=colors,
        autopct=lambda p: f'{p:.1f}%' if p > 2 else '',
        startangle=90, pctdistance=0.75,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    ax.legend(wedges, labels, fontsize=7, loc='center left',
              bbox_to_anchor=(-0.2, 0.5))
    ax.set_title(f'Average Decode Token Energy\nTotal: {total:.4f} uJ')

    # Plot 4: Latency per token vs context
    ax = axes[1, 1]
    contexts = [t.context_length for t in decode_traces]
    latencies = [t.latency_us for t in decode_traces]
    ax.plot(contexts, latencies, 'o-', markersize=3, linewidth=1.5, color='#d62728')
    ax.set_xlabel('Context Length (tokens in KV cache)')
    ax.set_ylabel('Latency per Token (us)')
    ax.set_title('Decode Latency vs Context Length')
    ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plot_path = os.path.join(output_dir, f'rdxe_{result.model_name}.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Plot: {plot_path}")


def plot_ring(result: RingResult, output_dir: str):
    """Plot ring simulation results."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)
    fig.suptitle(
        f"DXE Ring Pipeline — {result.n_chips} Chips\n"
        f"Throughput: {result.steady_state_throughput_tps:.0f} tok/s | "
        f"E/tok: {result.energy_per_token_uJ:.3f} uJ | "
        f"Latency: {result.total_latency_ms:.2f} ms",
        fontsize=12, fontweight='bold')

    # Per-chip energy
    ax = axes[0]
    x = np.arange(result.n_chips)
    ax.bar(x, result.per_chip_energy_uJ, color='#4e79a7', edgecolor='k',
           linewidth=0.5)
    for i, e in enumerate(result.per_chip_energy_uJ):
        ax.text(i, e, f'{e:.3f}', ha='center', va='bottom', fontsize=8)
    ax.set_xlabel('Chip ID')
    ax.set_ylabel('Energy per Token (uJ)')
    ax.set_title('Per-Chip Energy')
    ax.set_xticks(x)

    # Per-chip utilization
    ax = axes[1]
    ax.bar(x, [u * 100 for u in result.per_chip_utilization],
           color='#59a14f', edgecolor='k', linewidth=0.5)
    for i, u in enumerate(result.per_chip_utilization):
        ax.text(i, u * 100, f'{u*100:.1f}%', ha='center', va='bottom', fontsize=8)
    ax.set_xlabel('Chip ID')
    ax.set_ylabel('Utilization (%)')
    ax.set_title('Pipeline Utilization')
    ax.set_xticks(x)
    ax.set_ylim(0, 110)

    # Layer assignment
    ax = axes[2]
    for ci, layers in enumerate(result.layers_per_chip):
        for li in layers:
            ax.barh(ci, 1, left=li, height=0.6, color='#f28e2b',
                    edgecolor='k', linewidth=0.3)
            ax.text(li + 0.5, ci, str(li), ha='center', va='center', fontsize=7)
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Chip ID')
    ax.set_title('Layer Assignment')
    ax.set_yticks(range(result.n_chips))

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    path = os.path.join(output_dir, f'dxe_ring_{result.n_chips}chips.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"Ring plot: {path}")
