"""Render diagnostics from the frozen results; never reread the live sweep."""
import csv
import json
import os
import tempfile
import argparse
from pathlib import Path
import numpy as np
from ..config import TARGETS
from ..data.io import write_csv


def main():
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);args=ap.parse_args();out=args.output
    with (out/'predictions.csv').open() as f:rows=list(csv.DictReader(f))
    meta=json.loads((out/'metadata.json').read_text());ids=meta['batch2_ids']
    rng=np.random.default_rng(42);boot=rng.integers(0,len(ids),size=(5000,len(ids)))
    changes=[]
    for target in TARGETS:
        errors={}
        for variant in ['architecture','architecture_temp']:
            records=[r for r in rows if r['protocol']=='new_batch_saved_models' and r['variant']==variant and r['target']==target]
            errors[variant]=np.array([np.mean([abs(float(r['prediction'])-float(r['actual']))/float(r['actual'])*100 for r in records if r['config_id']==i]) for i in ids])
        delta=errors['architecture']-errors['architecture_temp'];lo,hi=np.quantile(delta[boot].mean(1),[.025,.975])
        changes.append(dict(target=target,mape_improvement_pp=float(delta.mean()),ci95_lower=float(lo),ci95_upper=float(hi)))
    write_csv(out/'batch2_paired_bootstrap.csv',changes)
    records=[r for r in rows if r['protocol']=='held_out_order_block' and r['variant']=='architecture' and r['target']=='tpot_ms']
    assert len(records)==936 and len({r['config_id'] for r in records})==936
    order=np.array([int(r['config_id'].split('_')[1]) for r in records])
    temp=np.array([float(r['start_temp']) for r in records]);ratio=np.array([float(r['actual'])/float(r['prediction']) for r in records])
    cache=tempfile.mkdtemp(prefix='state-signal-mpl-');os.environ.setdefault('MPLCONFIGDIR',cache);os.environ.setdefault('XDG_CACHE_HOME',cache)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    im=axes[0].scatter(order,ratio,c=temp,cmap='coolwarm',s=12,alpha=.75)
    fig.colorbar(im,ax=axes[0],label='Recorded starting temperature (°C)')
    axes[0].set(xlabel='Acquisition/configuration index',ylabel='Actual TPOT / architecture-only prediction',title='Unexplained variation follows acquisition state')
    axes[1].scatter(temp,ratio,s=12,c='#64748b',alpha=.2)
    centers=[];medians=[]
    for lo,hi in [(29,35),(35,37),(37,39),(39,40.01)]:
        mask=(temp>=lo)&(temp<hi);centers.append(np.median(temp[mask]));medians.append(np.median(ratio[mask]))
    axes[1].plot(centers,medians,'o-',c='#2563eb',label='Temperature-bin median')
    axes[1].set(xlabel='Recorded starting temperature (°C)',ylabel='Actual TPOT / architecture-only prediction',title='Association; not a causal temperature estimate')
    axes[1].legend()
    for ax in axes:ax.axhline(1,c='black',ls='--',lw=1);ax.grid(alpha=.15)
    fig.savefig(out/'state_residuals.png',dpi=160)
    print(json.dumps(changes,indent=2));print('Batch-2 observed state ranges:',meta['batch2_ranges'])


if __name__=='__main__':main()
