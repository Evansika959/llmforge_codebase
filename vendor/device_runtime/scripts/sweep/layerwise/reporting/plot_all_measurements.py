"""Scatter all accepted architectures across the four layerwise registries."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from .plot_measurements import load_measurements, style, COLORS, MARKERS
from ..database import ROOT, digest

RETEST='LW_47d4353f9819966c231b'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    base=ROOT/'scripts/sweep/outputs'
    paths=['watch5_layerwise_500_v1','watch5_layerwise_500_variable_kv_v2',
           'watch5_layerwise_1000_variable_kv_v3/part1','watch5_layerwise_1000_variable_kv_v3/part2']
    tables=[]; sources={}
    for name in paths:
        d,validation,hashes=load_measurements(base/name)
        d['campaign']=name;tables.append(d)
        sources[name]=dict(validation=validation,artifact_hashes=hashes,database_sha256=digest(base/name/'candidates.sqlite'))
    df=pd.concat(tables,ignore_index=True)
    if df.candidate_id.duplicated().any(): raise ValueError('Duplicate accepted architecture')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    style()
    fig,axes=plt.subplots(1,3,figsize=(16,6.5))
    fig.subplots_adjust(left=.06,right=.985,bottom=.22,top=.73,wspace=.27)
    fig.suptitle(f'Pixel Watch 5 | {len(df):,} measured architectures',x=.06,y=.97,ha='left',fontsize=22,fontweight='bold')
    fig.text(.06,.9,'49 prompt tokens  ·  32 generated tokens  ·  Baseline-subtracted energy',fontsize=11,color='#596778')
    handles=[Line2D([],[],marker='o',linestyle='',color=c,label=f'GS {g}',markersize=7) for g,c in COLORS.items()]
    handles += [Line2D([],[],marker=m,linestyle='',color='#6A7380',label=f'{f.split("-")[-1]} family',markersize=7) for f,m in MARKERS.items()]
    handles += [Line2D([],[],marker='*',linestyle='',markerfacecolor='#C32676',markeredgecolor='#202938',label='Retested point',markersize=13)]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.055,.85),ncol=6,frameon=False,columnspacing=2)
    counts={}
    metrics=[('decode_tok_s','Decode throughput','Tokens / second'),('ttft_s','Time to first token','Seconds'),
             ('dynamic_energy_per_token_mj','Dynamic energy per token','mJ / output token')]
    for ax,(metric,title,unit) in zip(axes,metrics):
        n=int(df[metric].notna().sum());counts[metric]=n
        for family,marker in MARKERS.items():
            for group,color in COLORS.items():
                d=df[(df.family==family)&(df.group==group)].dropna(subset=[metric])
                ax.scatter(d.params_m,d[metric],s=20,c=color,marker=marker,alpha=.48,linewidths=0,rasterized=True)
        target=df[df.candidate_id==RETEST].dropna(subset=[metric])
        ax.scatter(target.params_m,target[metric],s=220,c='#C32676',marker='*',edgecolors='#202938',linewidths=.8,zorder=10)
        ax.set(title=f'{title}\n{n:,} labels',xlabel='Actual parameters (millions)',ylabel=unit,ylim=(0,None))
        ax.grid(alpha=.65);ax.set_axisbelow(True)
    missing=int(df.dynamic_energy_per_token_mj.isna().sum())
    warned=int(df.energy_warning.notna().sum())
    fig.text(.06,.105,f'One accepted measurement per architecture; the retest replaces the original attempt for this point.\n'
             f'{missing} missing energy label(s) omitted only from the energy panel; {warned} energy-warning records retained.',
             fontsize=10,color='#596778',linespacing=1.5)
    fig.text(.06,.035,'GS groups contain different architectures. Dynamic energy covers prefill + decode and is divided by 32 output tokens.',
             fontsize=9,color='#596778')
    fig.savefig(out/'all_architectures_scatter.png',dpi=180)
    fig.savefig(out/'all_architectures_scatter.pdf')
    plt.close(fig)
    (out/'plotted_records.json').write_text(df.to_json(orient='records',indent=2)+'\n')
    summary=dict(architectures=len(df),label_counts=counts,energy_warnings=warned,
                 retested=df[df.candidate_id==RETEST].to_dict(orient='records'),
                 missing_energy=df[df.dynamic_energy_per_token_mj.isna()].candidate_id.tolist(),sources=sources)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='sources'},indent=2),flush=True)


if __name__=='__main__':main()
