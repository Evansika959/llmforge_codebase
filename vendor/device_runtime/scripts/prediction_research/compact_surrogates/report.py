"""Read-only evaluation of frozen selected predictions; never fits models."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TARGETS = ['decode_tok_s', 'ttft_ms', 'dynamic_energy_per_token_mj']


def report(out):
    out=Path(out)
    data=np.load(out/'test_predictions.npz',allow_pickle=False)
    y=data['actual']
    models=[k for k in data.files if k not in ['actual','config_ids','old_batch']]
    metrics=[]
    for name in models:
        for split,mask in [('pooled',np.ones(len(y),dtype=bool)),('old_batch',data['old_batch']),('new_batch',~data['old_batch'])]:
            for j,target in enumerate(TARGETS):
                a=y[mask,j]
                p=data[name][:,mask,j]
                values=np.mean(abs(p-a)/a,axis=1)*100
                metrics.append(dict(model=name,split=split,target=target,n=int(mask.sum()),
                    mape=float(values.mean()),seed_sd=float(values.std(ddof=1)),
                    mae=float(np.mean(abs(p-a))),r2=float(np.mean([r2_score(a,v) for v in p])),
                    spearman=float(np.mean([spearmanr(a,v).statistic for v in p]))))
    rng=np.random.default_rng(20260915)
    draws=rng.integers(0,len(y),size=(5000,len(y)))
    comparisons=[]
    for name in ['mlp','transformer']:
        for reference in ['xgboost14','transformer128x4']:
            for j,target in enumerate(TARGETS):
                own=np.mean(abs(data[name][:,:,j]-y[:,j])/y[:,j],axis=0)*100
                base=np.mean(abs(data[reference][:,:,j]-y[:,j])/y[:,j],axis=0)*100
                delta=base-own
                lo,hi=np.quantile(delta[draws].mean(axis=1),[.025,.975])
                comparisons.append(dict(model=name,reference=reference,target=target,
                    improvement_pp=float(delta.mean()),ci95=[float(lo),float(hi)]))
    summary=dict(metrics=metrics,paired_bootstrap=comparisons,
                 interpretation='Positive improvement favors compact model; intervals condition on one reused test split, average seed-specific errors, and are not multiple-comparison adjusted. Temporal correlation is not modeled.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    tuning=json.loads((out/'tuning.json').read_text())
    selected=json.loads((out/'selection.json').read_text())
    lines=['# Compact surrogate comparison: frozen 1564 architectures','',
           '1100 train / 150 validation / 314 test. Architecture-only physics32; log-space Smooth L1. No hardware accessed.','',
           'Each neural family has four capacity candidates, all evaluated with seeds 42, 123, 2026. Configuration selected by mean validation MAPE before test predictions. MLP has two hidden layers plus an output layer.','',
           '## Selected models','', '| Family | Configuration | Parameters | Validation MAPE |', '|---|---|---:|---:|']
    for family,name in selected.items():
        entries=[r for r in tuning if r['name']==name]
        lines.append(f'| {family} | {name} | {entries[0]["parameter_count"]:,} | {np.mean([r["validation_mape"] for r in entries]):.3f}% |')
    lines+=['','## Fixed pooled test','', '| Model | Throughput MAPE | TTFT MAPE | Dynamic energy MAPE |','|---|---:|---:|---:|']
    for name in models:
        rows=[next(r for r in metrics if r['model']==name and r['split']=='pooled' and r['target']==t) for t in TARGETS]
        lines.append('| '+name+' | '+' | '.join(f'{r["mape"]:.2f} ± {r["seed_sd"]:.2f}%' for r in rows)+' |')
    lines+=['','The ± values are training-seed SD, not confidence intervals over dataset splits. Historical baseline prediction reproduction was asserted within 0.005 percentage points per target/seed.','',
            'Smaller models use validation MAPE for stopping; the historical large Transformer used validation Smooth L1. This compares practical predictor recipes, not a pure capacity-only ablation.','',
            'No universal superiority or active-learning sample-efficiency claim follows from these results. The test was already inspected in earlier experiments. Separate old/new-batch metrics and unadjusted paired-bootstrap intervals are in summary.json. No checkpoint is automatically promoted into active learning.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(13,4.6))
    colors=['#168A85','#4776D0','#D67B24','#8795A5']
    names=['2-hidden-layer MLP','Small Transformer','XGBoost14','Large Transformer']
    for j,ax in enumerate(axes):
        rows=[next(r for r in metrics if r['model']==name and r['split']=='pooled' and r['target']==TARGETS[j]) for name in models]
        ax.bar(range(4),[r['mape'] for r in rows],yerr=[r['seed_sd'] for r in rows],color=colors,capsize=4)
        ax.set_xticks(range(4),['MLP','Small TF','XGB','Large TF'])
        ax.set_title(['Decode throughput','Time to first token','Dynamic energy'][j])
        ax.set_ylabel('Test MAPE (%) — lower is better')
        for i,r in enumerate(rows):
            ax.text(i,r['mape']+r['seed_sd']+.4,f'{r["mape"]:.2f}',ha='center',fontsize=9)
        ax.set_ylim(0,max(r['mape']+r['seed_sd'] for r in rows)*1.2)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Compact neural predictors | Same 314 test architectures',fontsize=15)
    fig.text(.5,.015,'3 training seeds; error bars = seed SD. Reused test set: exploratory, not prospective validation.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.92))
    fig.savefig(out/'comparison.png',dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    import sys
    report(sys.argv[1])
