"""Descriptive before/after audit from an immutable plotted_records.json snapshot."""
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plot_measurements import style


def main():
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--boundary', type=int, default=98)
    args = parser.parse_args()
    destination = args.report / 'session_comparison.png'
    if destination.exists() or destination.with_suffix('.json').exists():
        raise ValueError('Refusing to overwrite a previous comparison')
    d = pd.read_json(args.report / 'plotted_records.json')
    d['period'] = np.where(d.ordinal < args.boundary, 'earlier', 'latest')
    columns = ['params_m', 'decode_tok_s', 'ttft_s', 'dynamic_energy_per_token_mj', 'baseline_power_w']
    grouped = d.groupby(['family', 'group', 'period'])[columns].median()
    comparison = []
    for (family, group), part in d.groupby(['family', 'group']):
        a, b = (grouped.loc[(family, group, p)] for p in ['earlier', 'latest'])
        comparison.append(dict(family=family, group=int(group),
            n_earlier=int((part.period == 'earlier').sum()),
            n_latest=int((part.period == 'latest').sum()),
            earlier=a.to_dict(), latest=b.to_dict(),
            change_pct={k:100*(float(b[k])/float(a[k])-1) for k in columns}))
    summary = dict(boundary=args.boundary, n=len(d), comparison=comparison,
                   baseline_median=d.groupby('period').baseline_power_w.median().to_dict(),
                   limitation='Different architectures, not paired repeats or a causal session effect estimate.')
    destination.with_suffix('.json').write_text(json.dumps(summary, indent=2)+'\n')
    style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8))
    fig.subplots_adjust(left=.065, right=.985, top=.77, bottom=.24, wspace=.3)
    fig.suptitle('A collection-session shift needs investigation', x=.065, y=.98,
                 ha='left', fontsize=21, fontweight='bold')
    fig.text(.065, .90, f'{len(d)} measured architectures | Split at record #{args.boundary} | Group medians, not paired repeats',
             color='#697586', fontsize=11)
    colors = {'earlier':'#75859A', 'latest':'#168A85'}
    ax = axes[0]
    for period, part in d.groupby('period'):
        ax.scatter(part.ordinal, part.baseline_power_w, color=colors[period], s=19, alpha=.8,
                   label=f'Records {int(part.ordinal.min())}–{int(part.ordinal.max())}')
        ax.hlines(part.baseline_power_w.median(), part.ordinal.min(), part.ordinal.max(),
                  color=colors[period], linewidth=2)
    ax.axvline(args.boundary-.5, color='#C83550', linestyle='--', linewidth=1)
    ax.set(title='A  Pre-inference idle power', xlabel='Measurement order', ylabel='Watts', ylim=(0,.56))
    ax.legend(frameon=False, fontsize=9)
    labels = [r['family'].split('-')[-1].upper()+'\nGS'+str(r['group']) for r in comparison]
    x = np.arange(len(comparison))
    for ax, key, title, ylabel in [(axes[1], 'decode_tok_s', 'B  Decode throughput', 'Tokens / second'),
                                   (axes[2], 'ttft_s', 'C  Time to first token', 'Seconds')]:
        for delta, period in [(-.19,'earlier'),(.19,'latest')]:
            ax.bar(x+delta, [r[period][key] for r in comparison], width=.36, color=colors[period])
        ax.set_xticks(x, labels, fontsize=8)
        ax.set(title=title, ylabel=ylabel)
    for ax in axes:
        ax.grid(axis='y', alpha=.65)
        ax.set_axisbelow(True)
    fig.text(.065,.075,f'Comparison across record #{args.boundary}, within family × GS strata.\n'
             'Architecture distributions differ: changes do not establish a causal device/session effect.',
             fontsize=10, color='#697586', linespacing=1.6)
    fig.savefig(destination, dpi=180)
    fig.savefig(destination.with_suffix('.pdf'))
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
