"""Plot the committed dynamic-stage validation history without training or hardware."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.prediction.active_learning.engine import verify_workspace
from scripts.prediction.data.dataset import MeasurementDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve(strict=True)
    _, state, dataset = verify_workspace(workspace)
    source = workspace/'monitoring/accuracy_history.json'
    document = json.loads(source.read_text())
    history = document['history']
    rounds = np.array([r['round'] for r in history])
    assert rounds.tolist() == list(range(state['completed_rounds']+1))
    assert document['energy_target'] == 'dynamic'
    initial = MeasurementDataset.load(workspace/'dataset_000.json')
    n0 = sum(r['split']=='train' for r in initial.observations)
    counts = [n0]
    for row in history[1:]:
        folder = workspace/f'rounds/{row["round"]:04d}'
        completion = json.loads((folder/'completion.json').read_text())
        assert completion['model_sha256'] == row['model_sha256']
        for a, b in zip(completion['validation_after'], row['validation']):
            assert a['target']==b['target'] and a['mape']==b['mape'] and a['n']==b['n']==150
        d = MeasurementDataset.load(folder/'dataset_after.json')
        counts.append(sum(r['split']=='train' for r in d.observations))
    assert counts[-1] == sum(r['split']=='train' for r in dataset.observations)
    added = np.array(counts)-n0
    targets = document['targets']
    values = np.array([[next(m['mape'] for m in h['validation'] if m['target']==t)
                        for t in targets] for h in history])
    assert np.isfinite(values).all()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'history_snapshot.json').write_text(json.dumps(document, indent=2)+'\n')
    summary = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   dataset_sha256=dataset.fingerprint, initial_training=n0, final_training=counts[-1],
                   rounds=rounds.tolist(), training_counts=counts, validation_n=150,
                   metrics=[dict(target=t, initial_mape=float(values[0,j]), final_mape=float(values[-1,j]),
                                 reduction_pp=float(values[0,j]-values[-1,j])) for j,t in enumerate(targets)],
                   interpretation='Current physics32 XGBoost dynamic stage only; stage baseline already includes previous collected data. Not the offline three-feature ablation or historical 314-test comparison. Fixed validation reused for early stopping; no prospective/generalization claim.')
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 13, 'axes.labelsize': 11,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(14, 6.4))
    colors = ['#247BA0', '#168A75', '#A363C7']
    titles = ['Decode throughput', 'Time to first token (TTFT)', 'Dynamic energy / token']
    for j, ax in enumerate(axes):
        v = values[:,j]; color=colors[j]
        ax.axhline(v[0], color='#525C66', linestyle='--', linewidth=1.4, label='Initial checkpoint')
        ax.plot(rounds, v, color=color, linewidth=1.9, marker='o', markersize=3, label='After each AL round')
        ax.scatter([0,rounds[-1]], [v[0],v[-1]], s=[50,65], c=['#525C66',color], zorder=4)
        ax.set_title(titles[j], pad=49)
        ax.set_xlabel('Completed active-learning rounds')
        ax.set_ylabel('Validation MAPE (%) — lower is better')
        ax.set_xlim(-1, rounds[-1]+1)
        span=max(float(np.ptp(v)), .5)
        ax.set_ylim(v.min()-.25*span, v.max()+.25*span)
        ax.set_xticks(np.linspace(0,rounds[-1],6))
        ax.grid(axis='y', alpha=.18)
        ax.spines[['top','right']].set_visible(False)
        top=ax.secondary_xaxis('top')
        top.set_xticks(rounds[::10], [str(n) for n in added[::10]])
        top.set_xlabel('Added training architectures', fontsize=9)
        change=v[0]-v[-1]
        text=f'{v[0]:.2f}% → {v[-1]:.2f}%\n'+(f'Improved {change:.2f} pp' if change>=0 else f'Worsened {-change:.2f} pp')
        ax.text(.04,-.32,text,transform=ax.transAxes,va='top',fontsize=10,
                bbox=dict(facecolor='white',edgecolor='#E1E5EA',boxstyle='round,pad=.45',alpha=.95))
    handles, labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.10),ncol=2,frameon=False)
    fig.suptitle('Does active learning improve the predictor?', fontsize=18, y=.985)
    fig.text(.5,.925,f'Physics32 XGBoost · {n0:,} → {counts[-1]:,} training architectures · same 150 validation architectures',ha='center',fontsize=11)
    fig.text(.5,.058,'Round 0 = initial checkpoint of the baseline-subtracted energy stage. No test-set evaluation.',ha='center',fontsize=9,color='#505A64')
    fig.text(.5,.023,'Unsmoothed single-run history; independently zoomed y-axes. MAPE is prediction error, not classification accuracy.',ha='center',fontsize=9,color='#505A64')
    fig.subplots_adjust(left=.065,right=.985,bottom=.33,top=.73,wspace=.30)
    fig.savefig(out/'active_learning_accuracy.png',dpi=180)
    fig.savefig(out/'active_learning_accuracy.pdf')
    plt.close(fig)
    lines=['# Baseline and active-learning validation history','',
           f'Current dynamic stage: {n0} → {counts[-1]} training rows; {len(rounds)-1} completed rounds. Same 150 validation rows.','',
           '| Metric | Initial MAPE | Final MAPE | Reduction (pp) |','|---|---:|---:|---:|']
    for m in summary['metrics']:
        lines.append(f'| {m["target"]} | {m["initial_mape"]:.3f}% | {m["final_mape"]:.3f}% | {m["reduction_pp"]:+.3f} |')
    lines+=['',summary['interpretation'],'',
            'The plot shows actual unsmoothed recorded checkpoints, not seed means or confidence intervals. '
            'Axes are independently zoomed to reveal small fluctuations. No 100−MAPE accuracy conversion. '
            'Pending partial measurements are excluded until committed. Raw sources and model files are unchanged.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary['metrics'],indent=2))
    print(out/'active_learning_accuracy.png')


if __name__ == '__main__':
    main()
