"""Diagnostic only: test state signal across partitions and a new-batch snapshot."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split,GroupKFold
from ..config import ROOT
from ..config import FEATURES
from ..config import TARGETS
from ..data.io import write_csv
from ..features.analytic import physical_features


def load(path):
    raw=path.read_bytes()
    return list(csv.DictReader(io.StringIO(raw.decode()))),hashlib.sha256(raw).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='scripts/prediction/outputs/state_signal_investigation_936')
    args=ap.parse_args();out=ROOT/args.output;out.mkdir(parents=True,exist_ok=False)
    rows,sourcehash=load(ROOT/'scripts/sweep/outputs/watch5_random_50M_150M_40C_decode32_results_clean.csv')
    cfg1,_=load(ROOT/'scripts/sweep/configs/watch5_random_1000_50M_150M_sweep.csv')
    cfg2,_=load(ROOT/'scripts/sweep/configs/watch5_random_1000_50M_150M_sweep_batch2.csv')
    cfg={r['config_id']:r for r in cfg1+cfg2}
    old=ROOT/'scripts/prediction/outputs/surrogate_comparison_936'
    original=json.loads((old/'metadata.json').read_text());assert sourcehash==original['source_sha256']
    b2raw,b2hash=load(ROOT/'scripts/sweep/outputs/watch5_random_50M_150M_40C_decode32_batch2_results.csv')
    # Freeze the new-batch snapshot before any models are evaluated on it.
    b2=[]; rejected=[]
    for r in b2raw:
        try:
            values=[float(r[k]) for k in TARGETS+['temp_cpu_start_c','voltage_v']]
            assert all(np.isfinite(v) and v>0 for v in values)
            b2.append(r)
        except (ValueError,TypeError,KeyError,AssertionError):rejected.append(r.get('config_id'))
    assert b2 and len(set(r['config_id'] for r in b2))==len(b2)
    assert not {r['config_id'] for r in rows}&{r['config_id'] for r in b2}
    def arrays(rs):
        x=np.array([[physical_features(cfg[r['config_id']])[0][f] for f in FEATURES] for r in rs],dtype=np.float32)
        y=np.array([[float(r[t]) for t in TARGETS] for r in rs],dtype=np.float32)
        temp=np.array([[float(r['temp_cpu_start_c'])] for r in rs],dtype=np.float32)
        voltage=np.array([[float(r['voltage_v'])] for r in rs],dtype=np.float32)
        order=np.array([[int(r['config_id'].split('_')[1])] for r in rs],dtype=np.float32)
        return dict(architecture=x,architecture_temp=np.column_stack([x,temp]),
            architecture_voltage_diagnostic=np.column_stack([x,voltage]),
            architecture_order_diagnostic=np.column_stack([x,order]),
            architecture_temp_voltage_diagnostic=np.column_stack([x,temp,voltage])),y
    XX,Y=arrays(rows); X2,Y2=arrays(b2);strata=XX['architecture'][:,8].astype(int)
    Z=np.log(Y);results=[];predictions=[];splits=[]
    def score(protocol,split,variant,target,actual,pred,rs):
        results.append(dict(protocol=protocol,split=split,variant=variant,target=target,n=len(actual),
            mape=float(np.mean(abs(pred-actual)/actual)*100),mae=float(np.mean(abs(pred-actual))),
            r2=float(r2_score(actual,pred)),spearman=float(spearmanr(actual,pred).statistic)))
        for r,a,p in zip(rs,actual,pred):
            predictions.append(dict(protocol=protocol,split=split,variant=variant,target=target,
                config_id=r['config_id'],actual=float(a),prediction=float(p),
                start_temp=float(r['temp_cpu_start_c']),ending_voltage=float(r['voltage_v'])))
    # New data: evaluate only old saved checkpoints, no refit or model selection.
    conditioned=ROOT/'scripts/prediction/outputs/xgboost_error_diagnosis_936'
    for seed in [42,123,2026]:
        for j,t in enumerate(TARGETS):
            for v in ['architecture','architecture_temp']:
                path=old/f'xgboost_seed{seed}_{t}.json' if v=='architecture' else conditioned/f'temperature_conditioned_seed{seed}_{t}.json'
                m=xgb.XGBRegressor();m.load_model(path)
                assert m.n_features_in_==X2[v].shape[1]
                p=np.exp(m.predict(X2[v]));score('new_batch_saved_models',seed,v,t,Y2[:,j],p,b2)
    write_csv(out/'metrics.csv',results)
    print('NEW BATCH EVALUATED',len(b2),'rejected',rejected,flush=True)
    ids=np.arange(len(rows))
    for seed in range(20260909,20260914):
        pool,test=train_test_split(ids,test_size=.2,random_state=seed,stratify=strata)
        tr,stop=train_test_split(pool,test_size=.2,random_state=seed,stratify=strata[pool])
        splits.append(('random',seed,tr,stop,test))
    # Contiguous ID blocks test transfer across acquisition order, NOT verified sessions.
    groups=np.array([(int(r['config_id'].split('_')[1])-1)//200 for r in rows])
    for fi,(pool,test) in enumerate(GroupKFold(5).split(ids,groups=groups)):
        tr,stop=train_test_split(pool,test_size=.2,random_state=42,stratify=strata[pool])
        splits.append(('held_out_order_block',fi,tr,stop,test))
    splits.append(('forward_order',0,ids[:598],ids[598:748],ids[748:]))
    manifest=[]
    for protocol,split,tr,stop,test in splits:
        assert not set(tr)&set(test) and not set(stop)&set(test)
        manifest.append(dict(protocol=protocol,split=int(split),train=[rows[i]['config_id'] for i in tr],
            early_stopping=[rows[i]['config_id'] for i in stop],test=[rows[i]['config_id'] for i in test]))
        for v,x in XX.items():
            for j,t in enumerate(TARGETS):
                m=xgb.XGBRegressor(n_estimators=1500,learning_rate=.03,tree_method='hist',n_jobs=4,
                    max_depth=2,min_child_weight=3,reg_lambda=5,subsample=.85,colsample_bytree=.9,
                    objective='reg:squarederror',early_stopping_rounds=50,random_state=42)
                m.fit(x[tr],Z[tr,j],eval_set=[(x[stop],Z[stop,j])],verbose=False)
                score(protocol,split,v,t,Y[test,j],np.exp(m.predict(x[test])),[rows[i] for i in test])
        write_csv(out/'metrics.csv',results)
        print('EVALUATED',protocol,split,flush=True)
    write_csv(out/'predictions.csv',predictions)
    summary=[]
    for protocol in ['random','held_out_order_block','forward_order','new_batch_saved_models']:
        for v in XX:
            for t in TARGETS:
                rs=[r for r in results if r['protocol']==protocol and r['variant']==v and r['target']==t]
                if not rs:continue
                summary.append(dict(protocol=protocol,variant=v,target=t,n_evaluations=len(rs),
                    mape_mean=float(np.mean([r['mape'] for r in rs])),mape_sd=float(np.std([r['mape'] for r in rs],ddof=1)) if len(rs)>1 else 0.,
                    mae_mean=float(np.mean([r['mae'] for r in rs])),r2_mean=float(np.mean([r['r2'] for r in rs]))))
    write_csv(out/'summary.csv',summary)
    # Architecture-only residual bins use each first-batch row exactly once in
    # blocked out-of-fold predictions; do not compare raw latency of different sizes.
    bins=[]
    for t in TARGETS:
        for low,high in [(29,35),(35,37),(37,39),(39,40.01)]:
            rs=[r for r in predictions if r['protocol']=='held_out_order_block' and r['variant']=='architecture' and r['target']==t and low<=r['start_temp']<high]
            if rs:bins.append(dict(target=t,temp_min=low,temp_max_exclusive=high,n=len(rs),
                median_actual_over_arch_prediction=float(np.median([r['actual']/r['prediction'] for r in rs]))))
    write_csv(out/'temperature_residual_bins.csv',bins)
    b2range={k:[min(float(r[k]) for r in b2),max(float(r[k]) for r in b2)] for k in ['temp_cpu_start_c','voltage_v']}
    (out/'metadata.json').write_text(json.dumps(dict(source_sha256=sourcehash,batch2_snapshot_sha256=b2hash,
        batch2_ids=[r['config_id'] for r in b2],batch2_excluded=rejected,batch2_ranges=b2range,
        split_manifests=manifest,architecture_features=FEATURES,
        note='Ending voltage/order are diagnostic only; blocked groups are inferred from ID ranges, not verified charging sessions. Existing checkpoints frozen for batch2 evaluation. No measurement scripts or search defaults changed.'),indent=2))
    lines=['# Investigating device-state signal','',
        f'New-batch snapshot: {len(b2)} valid rows, {len(rejected)} excluded for nonpositive/missing/nonfinite targets/state. Snapshot SHA256 and exact IDs are recorded. No batch-2 measurements are used to fit, early-stop or select models.',
        '', '| Protocol | Features | TPOT MAPE | TTFT MAPE | Energy MAPE |','|---|---|---:|---:|---:|']
    for protocol in ['random','held_out_order_block','forward_order','new_batch_saved_models']:
        for v in XX:
            rs=[next((r for r in summary if r['protocol']==protocol and r['variant']==v and r['target']==t),None) for t in TARGETS]
            if any(r is None for r in rs):continue
            lines.append(f'| {protocol} | {v} | '+' | '.join(f"{r['mape_mean']:.2f}%" for r in rs)+' |')
    lines+=['','## What this can and cannot establish','',
        '- Random tests use five splits with distinct stopping sets. Block tests hold out five contiguous acquisition-ID ranges; forward test trains on the earliest 598, stops on the next 150, and evaluates the final 188. These are not verified session boundaries.',
        '- All new fits use fixed original XGBoost settings and seed 42. New-batch evaluation uses original saved models for three seeds and no training on new-batch labels.',
        '- Starting temperature is recorded before model transfer, shell setup and pre-idle, not at the exact inference start. thermal_zone17 is hard-coded; its sensor type is not validated in the script. Ending voltage/frequency are not available to an architecture searcher.',
        '- Order and ending-voltage inputs are diagnostic only. They must not be mistaken for deployable architecture-only predictors.',
        '- Better predictions conditional on temperature show association, not a causal thermal law. Temperature may capture session, placement, background-load or frequency behavior.',
        '- New-batch snapshot is small and may occupy a narrow state range. Continued use for model selection would turn it into a validation set.',
        '- No active model or watch scripts are changed. Models with temperature are not replacements for the architecture-only search baseline.',
        '', '## Reproduce','', '```bash','conda activate nanollmforge',
        'python -m scripts.prediction investigate-state-signal --output scripts/prediction/outputs/state_signal_investigation_repeat','```']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print('COMPLETE',out,flush=True)


if __name__=='__main__':main()
