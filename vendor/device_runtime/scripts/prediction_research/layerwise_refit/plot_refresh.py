"""Plot the published full dataset and paired GS64 replacements."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from scripts.sweep.layerwise.candidates import architecture_summary
from scripts.sweep.layerwise.reporting.plot_measurements import style, COLORS, MARKERS
from .data import TARGETS
from .refresh_gs64 import write


LABELS = ['Decode throughput', 'Time to first token', 'Dynamic energy per token']
UNITS = ['Tokens / second', 'Seconds', 'mJ / output token']
SCALE = [1, .001, 1]


def figures(folder):
    folder = Path(folder).resolve()
    data = json.loads((folder/'dataset_snapshot.json').read_text())
    old = json.loads((folder/'provenance/baseline_dataset_snapshot.json').read_text())
    rows = data['rows']; previous = {r['candidate_id']:r for r in old['rows']}
    records = [dict(candidate_id=r['candidate_id'],family=r['architecture']['family'],
                    group=r['architecture']['q8_group_size'],
                    params_m=architecture_summary(r['architecture'])['total_params']/1e6,
                    metrics=r['metrics'],old_metrics=previous[r['candidate_id']]['metrics'],
                    energy_warning=r['energy_warning']) for r in rows]
    out = folder/'plots'; out.mkdir(exist_ok=True)
    write(out/'plot_records.json',records)
    style()
    plt.rcParams.update({'font.size':10,'axes.titlesize':12,'axes.labelsize':10})
    fig, axes = plt.subplots(1,3,figsize=(16,6.3))
    fig.subplots_adjust(left=.065,right=.98,bottom=.22,top=.72,wspace=.28)
    fig.suptitle('Updated hardware dataset | 2,000 architectures',x=.065,y=.97,ha='left',fontsize=21,fontweight='bold')
    fig.text(.065,.90,'664 GS64 measurements replaced; 1,336 GS16/32 measurements retained',fontsize=11,color='#596778')
    handles=[Line2D([],[],marker='o',linestyle='',color=c,label=f'GS {g}',markersize=7) for g,c in COLORS.items()]
    handles += [Line2D([],[],marker=m,linestyle='',color='#6A7380',label=f'{f.split("-")[-1]} family',markersize=7) for f,m in MARKERS.items()]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.06,.845),ncol=5,frameon=False,columnspacing=2.4)
    for j,(ax,target) in enumerate(zip(axes,TARGETS)):
        for family,marker in MARKERS.items():
            for group,color in COLORS.items():
                take=[r for r in records if r['family']==family and r['group']==group]
                ax.scatter([r['params_m'] for r in take],[r['metrics'][target]*SCALE[j] for r in take],
                           s=19,color=color,marker=marker,alpha=.48,linewidths=0,rasterized=True)
        ax.set(title=LABELS[j]+'\n2,000 valid labels',xlabel='Actual parameters (millions)',ylabel=UNITS[j],ylim=(0,None))
        ax.grid(alpha=.55);ax.set_axisbelow(True)
    warned=sum(bool(r['energy_warning']) for r in rows)
    fig.text(.065,.115,f'49 prompt tokens, 32 output tokens. All {warned} energy-warning records are retained. No imputed labels.',fontsize=10,color='#596778')
    fig.text(.065,.055,'Two kernel revisions: baseline GS16/32 + optimized GS64. Group-size strata contain different architectures, not matched GS variants.',fontsize=9,color='#596778')
    for suffix in ('png','pdf'):fig.savefig(out/f'updated_2000_scatter.{suffix}',dpi=180)
    plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(16,6.3))
    fig.subplots_adjust(left=.065,right=.98,bottom=.23,top=.72,wspace=.29)
    fig.suptitle('GS64 refresh | paired measurements',x=.065,y=.97,ha='left',fontsize=21,fontweight='bold')
    fig.text(.065,.90,'Same architecture IDs and synthetic weights; each marker is one old / new pair',fontsize=11,color='#596778')
    palette={'smollm2-135m':'#168A85','smollm2-360m':'#7755A8'}
    handles=[Line2D([],[],marker=MARKERS[f],linestyle='',color=c,label=f'{f.split("-")[-1]} family',markersize=7) for f,c in palette.items()]
    handles += [Line2D([],[],color='#8C96A3',ls='--',label='Unchanged (y = x)')]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.06,.845),ncol=3,frameon=False,columnspacing=2.4)
    pair_summary=[]
    for j,(ax,target) in enumerate(zip(axes,TARGETS)):
        take=[r for r in records if r['group']==64 and r['old_metrics'][target] is not None and r['old_metrics'][target]>0]
        before=np.array([r['old_metrics'][target]*SCALE[j] for r in take])
        after=np.array([r['metrics'][target]*SCALE[j] for r in take])
        for family,color in palette.items():
            mask=np.array([r['family']==family for r in take])
            ax.scatter(before[mask],after[mask],s=20,color=color,marker=MARKERS[family],alpha=.5,linewidths=0,rasterized=True)
        limits=[min(before.min(),after.min())*.82,max(before.max(),after.max())*1.18]
        ax.plot(limits,limits,ls='--',lw=1,color='#8C96A3',zorder=0)
        ax.set(xscale='log',yscale='log',xlim=limits,ylim=limits,
               xlabel='Baseline ('+UNITS[j].lower()+')',ylabel='Updated ('+UNITS[j].lower()+')',
               title=LABELS[j]+f'\n{len(take)} matched pairs')
        ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.5)
        ratio=float(np.median(after/before));reduction=float(np.median(1-after/before)*100)
        note=f'Median speedup: {ratio:.2f}×' if j==0 else f'Median reduction: {reduction:.1f}%'
        ax.text(.96,.045,note,transform=ax.transAxes,va='bottom',ha='right',fontsize=10,
                bbox=dict(facecolor='white',alpha=.88,edgecolor='none',pad=3))
        pair_summary.append(dict(target=target,n=len(take),median_new_over_old=ratio,median_reduction_pct=reduction))
    fig.text(.065,.115,'Logarithmic axes. Energy uses 663 valid old/new pairs; the previously missing energy label is now measured.',fontsize=10,color='#596778')
    fig.text(.065,.055,'Historical and new measurements were collected at different times. Paired changes include environmental variation; warnings are not filtered.',fontsize=9,color='#596778')
    for suffix in ('png','pdf'):fig.savefig(out/f'gs64_paired_comparison.{suffix}',dpi=180)
    plt.close(fig)
    write(out/'paired_summary.json',pair_summary)
    print('Saved scatter and paired comparison:',out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset-dir',type=Path,required=True)
    figures(p.parse_args().dataset_dir)
