"""Plot committed measurements only; never access ADB or mutate the source DB.

Usage: python -m scripts.sweep.layerwise.reporting.plot_measurements --output NEW_REPORT_DIR
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sqlite3

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from ..candidates import fingerprint
from ..database import ROOT, digest, validate_database

COLORS={16:'#168A85',32:'#4776D0',64:'#D67B24'}
MARKERS={'smollm2-135m':'o','smollm2-360m':'^'}
RED='#C83550'
METRICS=[('decode_tok_s','Decode throughput','Tokens / second'),
         ('ttft_s','Time to first token','Seconds'),
         ('dynamic_energy_per_token_mj','Dynamic energy / token','mJ / generated token')]


def load_measurements(folder):
    folder=Path(folder).resolve()
    validation=validate_database(folder/'candidates.sqlite')
    with sqlite3.connect((folder/'candidates.sqlite').as_uri()+'?mode=ro',uri=True) as con:
        con.row_factory=sqlite3.Row
        records=con.execute('''SELECT m.*,c.family,c.q8_group_size,c.pattern,c.architecture_json,
            c.summary_json,j.ordinal FROM measurements m JOIN candidates c USING(candidate_id)
            JOIN jobs j USING(candidate_id) WHERE m.accepted=1 AND j.status='complete'
            ORDER BY j.ordinal''').fetchall()
    if len({r['candidate_id'] for r in records})!=len(records):
        raise ValueError('Multiple accepted attempts for one architecture require explicit repeat handling')
    rows=[]; provenance_hashes={}
    for row in records:
        arch=json.loads(row['architecture_json']); stats=json.loads(row['summary_json'])
        artifact=(folder/row['artifact_path']).resolve()
        if not artifact.is_relative_to(folder): raise ValueError('Artifact outside campaign')
        result=json.loads((artifact/'result.json').read_text())
        provenance=json.loads((artifact/'provenance.json').read_text())
        if provenance['architecture_sha256']!=fingerprint(arch) or provenance['protocol_sha256']!=row['protocol_sha256']:
            raise ValueError('Attempt identity/protocol mismatch')
        for key in ['decode_tok_s','ttft_ms','dynamic_energy_per_token_mj','baseline_power_w','active_power_w','duration_s']:
            x,y=row[key],result[key]
            if x is None or y is None:
                if x!=y: raise ValueError('Database/result null mismatch')
            elif not math.isclose(x,y,rel_tol=1e-10,abs_tol=1e-9):
                raise ValueError('Database/result metric mismatch: '+key)
        t=result['timing']
        if [t[k] for k in ['prefill_tokens','output_tokens','decode_tokens']]!=[49,32,31]:
            raise ValueError('Mixed workloads')
        rows.append(dict(ordinal=row['ordinal'],candidate_id=row['candidate_id'],family=row['family'],
            group=row['q8_group_size'],pattern=row['pattern'],params_m=stats['total_params']/1e6,
            unique_layers=stats['unique_layer_shapes'],q8_mib=stats['q8_file_bytes']/1024**2,
            decode_tok_s=row['decode_tok_s'],ttft_s=row['ttft_ms']/1000,
            dynamic_energy_per_token_mj=row['dynamic_energy_per_token_mj'],
            gross_energy_per_token_mj=result.get('gross_energy_per_token_mj'),
            baseline_power_w=result.get('baseline_power_w'),post_baseline_power_w=result.get('post_baseline_power_w'),
            baseline_relative_drift=result.get('baseline_relative_drift'),energy_valid=result['energy_valid'],
            energy_warning=result.get('energy_warning'),temperature_c=t['admission_temperature_c'],
            battery_percent=provenance['admission']['battery_percent'],start=t['start'],end=t['end']))
        for name in ['result.json','provenance.json','timing.json']:
            path=artifact/name; provenance_hashes[str(path.relative_to(folder))]=digest(path)
    if not rows: raise ValueError('No committed measurements to plot')
    return pd.DataFrame(rows),validation,provenance_hashes


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
        'axes.titleweight':'bold','axes.labelcolor':'#354052','text.color':'#223047',
        'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#BEC7D2',
        'xtick.color':'#586476','ytick.color':'#586476','grid.color':'#E6EAF0',
        'grid.linewidth':.7,'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})


def scatter(ax,df,x,y,flag=False):
    for family,marker in MARKERS.items():
        for group,color in COLORS.items():
            d=df[(df.family==family)&(df.group==group)].dropna(subset=[x,y])
            ax.scatter(d[x],d[y],c=color,marker=marker,s=64,alpha=.87,edgecolors='white',linewidths=.7,zorder=3)
    if flag:
        d=df[df.energy_warning.notna()].dropna(subset=[x,y])
        ax.scatter(d[x],d[y],facecolors='none',edgecolors=RED,s=145,linewidths=1.7,zorder=5)
    ax.grid(alpha=.8,zorder=0)


def legend(fig,quality=False):
    handles=[Line2D([],[],marker='o',linestyle='',color=c,label=f'GS={g}',markersize=7) for g,c in COLORS.items()]
    handles += [Line2D([],[],marker=m,linestyle='',color='#707988',label=f'{f.split("-")[-1].upper()} family',markersize=7) for f,m in MARKERS.items()]
    handles += [Line2D([],[],marker='o',linestyle='',markerfacecolor='none',markeredgecolor=RED,label='Energy warning',markersize=9)]
    fig.legend(handles=handles,ncol=6,loc='upper center',bbox_to_anchor=(.5,.862 if quality else .895),frameon=False,columnspacing=1.5)


def save(fig,output,name):
    fig.savefig(output/(name+'.png'),dpi=180)
    fig.savefig(output/(name+'.pdf'))
    plt.close(fig)


def overview(df,output):
    fig,axes=plt.subplots(2,3,figsize=(15,9))
    fig.subplots_adjust(left=.068,right=.98,bottom=.12,top=.80,hspace=.43,wspace=.26)
    fig.suptitle(f'Pixel Watch 5 | First {len(df)} measured architectures',x=.068,y=.977,ha='left',fontsize=21,fontweight='bold')
    fig.text(.068,.921,'49 prompt tokens  /  32 generated tokens  /  One measurement per architecture',fontsize=11,color='#697586')
    legend(fig)
    rng=np.random.default_rng(20260914)
    for col,(key,title,unit) in enumerate(METRICS):
        ax=axes[0,col]; scatter(ax,df,'params_m',key,flag=col==2)
        ax.set(title=f'{chr(65+col)}  {title} vs model size',xlabel='Actual parameters (millions)',ylabel=unit)
        ax.set_ylim(bottom=0)
        ax=axes[1,col]
        arrays=[df.loc[df.group==g,key].dropna().to_numpy() for g in COLORS]
        bp=ax.boxplot(arrays,positions=[0,1,2],widths=.48,showfliers=False,patch_artist=True,
            medianprops=dict(color='#29344A',linewidth=1.8),whiskerprops=dict(color='#A0AABA'),capprops=dict(color='#A0AABA'))
        for box,color in zip(bp['boxes'],COLORS.values()): box.set(facecolor=color,alpha=.16,edgecolor=color)
        for i,(g,color) in enumerate(COLORS.items()):
            d=df[df.group==g].dropna(subset=[key]).copy(); d['jitter']=i+rng.uniform(-.16,.16,len(d))
            for family,marker in MARKERS.items():
                s=d[d.family==family]
                ax.scatter(s.jitter,s[key],c=color,s=42,marker=marker,edgecolors='white',linewidths=.4,alpha=.9,zorder=3)
            if col==2:
                s=d[d.energy_warning.notna()]
                ax.scatter(s.jitter,s[key],facecolors='none',edgecolors=RED,s=130,linewidths=1.6,zorder=4)
        ax.set_xticks([0,1,2],[f'GS={g}\nn={len(values)}' for g,values in zip(COLORS,arrays)])
        ax.set(title=f'{chr(68+col)}  Observed distribution by GS',ylabel=unit,ylim=axes[0,col].get_ylim())
        ax.grid(axis='y'); ax.set_axisbelow(True)
    fig.text(.068,.035,'Different architectures in each GS group: descriptive distributions, not a controlled GS comparison.\nDynamic energy includes prefill + decode, subtracts pre-idle baseline, and is divided by 32 output tokens.',fontsize=10,color='#697586',linespacing=1.6)
    save(fig,output,'performance_overview')


def quality(df,output):
    fig,axes=plt.subplots(1,3,figsize=(15,6.6))
    fig.subplots_adjust(left=.065,right=.945,bottom=.2,top=.73,wspace=.38)
    fig.suptitle('Energy accounting and measurement conditions',x=.065,y=.98,ha='left',fontsize=20,fontweight='bold')
    fig.text(.065,.906,f'{len(df)} committed records  /  {int(df.energy_warning.notna().sum())} energy-quality warning(s)  /  No warning points removed',fontsize=11,color='#697586')
    legend(fig,quality=True)
    ax=axes[0]; scatter(ax,df,'gross_energy_per_token_mj','dynamic_energy_per_token_mj',True)
    limit=1.08*max(df.gross_energy_per_token_mj.max(),df.dynamic_energy_per_token_mj.max())
    ax.plot([0,limit],[0,limit],'--',c='#8993A1',lw=1,label='No baseline subtraction')
    ax.set(xlim=(0,limit),ylim=(0,limit),xlabel='Gross energy (mJ / output token)',ylabel='Dynamic energy (mJ / output token)',title='A  Before / after subtraction')
    ax.text(.05,.9,'Dashed line: equal energy',transform=ax.transAxes,fontsize=9,color='#697586')
    ax=axes[1]; scatter(ax,df,'baseline_power_w','post_baseline_power_w',True)
    limit=1.15*max(df.baseline_power_w.max(),df.post_baseline_power_w.max())
    x=np.linspace(0,limit,100)
    ax.plot(x,x,c='#8993A1',lw=1); ax.plot(x,x*.75,'--',c='#A7AFBB',lw=1); ax.plot(x,x*1.25,'--',c='#A7AFBB',lw=1)
    ax.set(xlim=(0,limit),ylim=(0,limit),xlabel='Pre-inference idle power (W)',ylabel='Post-inference idle power (W)',title='B  Baseline drift audit')
    ax.text(.04,.92,'Dashed lines: +/-25%',transform=ax.transAxes,fontsize=9,color='#697586')
    warned=df[df.energy_warning.notna()].nlargest(3,'baseline_relative_drift')
    for i,(_,r) in enumerate(warned.iterrows()):
        ax.annotate(f'#{int(r.ordinal)}: {r.baseline_relative_drift:.0%} drift',(r.baseline_power_w,r.post_baseline_power_w),
                    xytext=(.04,.82-.10*i),textcoords='axes fraction',color=RED,fontsize=9,
                    arrowprops=dict(arrowstyle='-',color=RED,lw=.9))
    # Label an observed minimum, without inventing a new exclusion rule.
    low=df.loc[df.baseline_power_w.idxmin()]
    ax.annotate(f'#{int(low.ordinal)}: lowest idle power',
        (low.baseline_power_w,low.post_baseline_power_w),
        xytext=(low.baseline_power_w+.03,low.post_baseline_power_w+.085),fontsize=8,color='#697586',
        arrowprops=dict(arrowstyle='-',color='#8993A1',lw=.8))
    ax=axes[2]; right=ax.twinx(); right.spines['right'].set_visible(True)
    gaps=np.flatnonzero((df.start.to_numpy()[1:]-df.end.to_numpy()[:-1])>600)+1
    for segment in np.split(np.arange(len(df)),gaps):
        d=df.iloc[segment]
        ax.plot(d.ordinal,d.temperature_c,'o-',c=COLORS[16],lw=1.4,ms=3)
        right.plot(d.ordinal,d.battery_percent,'s-',c=COLORS[32],lw=1.3,ms=3,alpha=.75)
    for i in gaps:
        at=(df.iloc[i-1].ordinal+df.iloc[i].ordinal)/2
        ax.axvline(at,color='#B3BBC7',linestyle=':',lw=1)
        ax.text(at+.35,26,'>10 min gap',rotation=90,fontsize=8,color='#7B8593')
    ax.axhline(45,c=COLORS[16],ls='--',lw=1,alpha=.7)
    right.axhline(30,c=COLORS[32],ls='--',lw=1,alpha=.7)
    ax.set(title='C  Admission conditions',xlabel='Measurement order (not elapsed time)',ylabel='Pre-inference temperature (C)',ylim=(24,47))
    right.set(ylabel='Battery at host admission (%)',ylim=(20,105))
    ax.yaxis.label.set_color(COLORS[16]); right.yaxis.label.set_color(COLORS[32]); ax.grid(alpha=.6)
    ax.legend(handles=[Line2D([],[],c=COLORS[16],marker='o',ms=4,label='Temperature'),Line2D([],[],c=COLORS[32],marker='s',ms=4,label='Battery')],loc='lower left',fontsize=8,frameon=False)
    fig.text(.065,.064,f'{len(df)} completed records are not repeated measurements; this plot cannot estimate repeatability or predictor accuracy.\nTemperature is sampled before inference only. Uncompleted candidates are excluded.',fontsize=10,color='#697586',linespacing=1.6)
    save(fig,output,'energy_quality')


def summarize(df,validation,hashes,folder):
    summary=dict(campaign=str(folder),n=len(df),jobs=validation['jobs_by_status'],
        families=dict(Counter(df.family)),groups={str(k):v for k,v in Counter(df.group).items()},
        patterns=dict(Counter(df.pattern)),heterogeneous=int((df.unique_layers>1).sum()),
        warnings=dict(Counter(df.energy_warning.fillna('none'))),metrics={},groups_summary=[],
        observed_parameter_range_m=[df.params_m.min(),df.params_m.max()],
        temperature_range_c=[df.temperature_c.min(),df.temperature_c.max()],
        median_baseline_relative_drift=float(df.baseline_relative_drift.median()),
        median_subtracted_energy_fraction=float((1-df.dynamic_energy_per_token_mj/df.gross_energy_per_token_mj).median()),
        parameter_spearman={key:float(df.params_m.corr(df[key],method='spearman')) for key,_,_ in METRICS},
        source_files_sha256=hashes,database_sha256=digest(folder/'candidates.sqlite'),
        limitations=['Descriptive only; no repeated-measurement noise or predictor-accuracy estimate.',
            'GS strata contain different architectures and parameter distributions; no causal GS claim.',
            'Two protocol hashes differ by audited host transfer timeout only; kernel and energy accounting unchanged.',
            'Null energy labels are excluded only from energy panels; warned points remain visible.'])
    for key,_,_ in METRICS:
        s=df[key].dropna(); summary['metrics'][key]=dict(n=len(s),min=float(s.min()),median=float(s.median()),max=float(s.max()),q25=float(s.quantile(.25)),q75=float(s.quantile(.75)))
    for (family,g),d in df.groupby(['family','group']):
        summary['groups_summary'].append(dict(family=family,group=int(g),n=len(d),params_m_median=float(d.params_m.median()),
            **{key:float(d[key].median()) for key,_,_ in METRICS}))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,default=ROOT/'scripts/sweep/outputs/watch5_layerwise_500_v1')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); folder=args.campaign.resolve()
    df,validation,hashes=load_measurements(folder)
    args.output.mkdir(parents=True,exist_ok=False)
    summary=summarize(df,validation,hashes,folder)
    style(); overview(df,args.output); quality(df,args.output)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    (args.output/'plotted_records.json').write_text(df.to_json(orient='records',indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='source_files_sha256'},indent=2))


if __name__=='__main__': main()
