"""Descriptive energy/latency correlations on the frozen historical JSON cohort."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.prediction.data.legacy import load_cohort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1] / 'prediction/outputs/batch2_progress_632'
    snapshot, _, rows, _, ix, y, gross, audit = load_cohort(source)
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    values = {k: np.array([float(r[k]) for r in rows]) for k in
              ['ttft_ms', 'tpot_ms', 'duration_s', 'temp_cpu_start_c']}
    values.update(dynamic_energy=y[:, 2].astype(float), gross_energy=gross.astype(float))
    assert all(np.isfinite(v).all() and (v > 0).all() for v in values.values())
    old_ids = {r['config_id'] for r in snapshot['records']['batch1']}
    old = np.array([r['config_id'] in old_ids for r in rows])
    subsets = {'all': np.arange(len(rows)), 'batch1': np.flatnonzero(old),
               'batch2': np.flatnonzero(~old), 'fixed_test': ix['pooled_test']}
    metrics = []
    for subset, ids in subsets.items():
        for energy in ['dynamic_energy', 'gross_energy']:
            for latency in ['ttft_ms', 'tpot_ms', 'duration_s']:
                a, b = values[energy][ids], values[latency][ids]
                metrics.append(dict(subset=subset, n=len(ids), energy=energy, latency=latency,
                                    pearson=float(pearsonr(a, b).statistic),
                                    spearman=float(spearmanr(a, b).statistic),
                                    log_pearson=float(pearsonr(np.log(a), np.log(b)).statistic)))
    # A linear diagnostic, not causal adjustment or a general nonlinear control.
    controls = np.column_stack([np.ones(len(rows)), values['temp_cpu_start_c'], old])
    residuals = {k: v - controls @ np.linalg.lstsq(controls, v, rcond=None)[0]
                 for k, v in values.items()}
    partial = {e: {l: float(pearsonr(residuals[e], residuals[l]).statistic)
                   for l in ['ttft_ms', 'tpot_ms']} for e in ['dynamic_energy', 'gross_energy']}
    summary = dict(audit=audit, metrics=metrics,
                   partial_pearson_controlling_linear_temperature_and_batch=partial,
                   ttft_tpot_pearson=float(pearsonr(values['ttft_ms'], values['tpot_ms']).statistic),
                   source_sha256={f: hashlib.sha256((source/f).read_bytes()).hexdigest()
                                  for f in ['input_snapshot.json', 'metadata.json']},
                   limits=['Descriptive analysis, no fitting or model selection on test labels.',
                           'Pooled architecture observations, not paired repeat measurements.',
                           'Energy labels share measured duration algebraically; association is not causation.',
                           'Historical energy is total measured inference-window energy divided by 32, not decode-only energy.'])
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    lines = ['# Energy–latency correlations: 1,564 historical architectures', '',
             'No extra filtering; 32 generated tokens. Gross = total_energy_j × 1000 / 32.', '',
             '| Subset | N | Energy | Latency | Pearson r | Spearman rho | Log Pearson r |',
             '|---|---:|---|---|---:|---:|---:|']
    for r in metrics:
        lines.append(f'| {r["subset"]} | {r["n"]} | {r["energy"]} | {r["latency"]} | '
                     f'{r["pearson"]:.4f} | {r["spearman"]:.4f} | {r["log_pearson"]:.4f} |')
    lines += ['', 'Partial Pearson after linear adjustment for starting temperature and batch:',
              '', '```json', json.dumps(partial, indent=2), '```', '',
              f'TTFT–TPOT Pearson r: {summary["ttft_tpot_pearson"]:.4f}.', '',
              'Energy = mean power × measured inference-window duration / 32. Dynamic power subtracts baseline. '
              'This creates shared duration dependence; it does not prove latency alone predicts energy accurately. '
              'These labels are not decode-only energy. Temperature adjustment is linear and non-causal. '
              'Results include training, validation and test observations; fixed-test statistics are separately shown.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for i, energy in enumerate(['dynamic_energy', 'gross_energy']):
        for j, latency in enumerate(['ttft_ms', 'tpot_ms']):
            ax = axes[i, j]
            for mask, label, color in [(old, 'Batch 1', '#247BA0'), (~old, 'Batch 2', '#F18F3B')]:
                ax.scatter(values[latency][mask], values[energy][mask], s=12,
                           alpha=.4, color=color, label=label, linewidths=0)
            r = next(r for r in metrics if r['subset']=='all' and r['energy']==energy and r['latency']==latency)
            ax.set_title(f'Pearson r = {r["pearson"]:.3f}  |  Spearman rho = {r["spearman"]:.3f}')
            ax.set_xlabel('TTFT (ms)' if latency=='ttft_ms' else 'TPOT (ms/token)')
            ax.set_ylabel(('Dynamic' if i==0 else 'Gross')+' energy (mJ/generated token)')
            ax.grid(alpha=.15)
            ax.spines[['top', 'right']].set_visible(False)
    axes[0, 0].legend(frameon=False)
    fig.suptitle('Energy vs. latency | 1,564 historical architectures\n32 generated tokens; linear axes; no additional filtering', fontsize=14)
    fig.savefig(out/'energy_latency.png', dpi=170)
    plt.close(fig)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
