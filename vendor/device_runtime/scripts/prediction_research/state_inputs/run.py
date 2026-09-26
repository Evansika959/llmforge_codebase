"""Frozen-cohort XGBoost start-temperature ablation; no post-run state inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import r2_score
from scripts.prediction.config import ROOT, FEATURES, PHYSICS_TARGETS
from scripts.prediction.data.legacy import load_cohort
from scripts.prediction.features.analytic import physical_features
from scripts.prediction.features.physics import HardwareProfile, FEATURES32
from scripts.prediction.training.trees import fit_xgb
from scripts.prediction.inference.trees import tree_predict


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    previous=ROOT/'scripts/prediction/outputs/batch2_progress_632'
    _,meta,rows,configs,ix,y,_,audit=load_cohort(previous)
    tr=np.concatenate([ix['old_train'],ix['batch2_train']]); va=ix['fixed_validation']; te=ix['pooled_test']
    assert [len(tr),len(va),len(te)]==[1100,150,314]
    assert not set(tr)&set(te) and not set(va)&set(te) and not set(tr)&set(va)
    temperature=np.array([[float(r['temp_cpu_start_c'])] for r in rows],dtype=np.float32)
    assert np.isfinite(temperature).all() and (temperature>0).all()
    x14=np.array([[physical_features(c)[0][f] for f in FEATURES] for c in configs],dtype=np.float32)
    profile=HardwareProfile().fit([configs[i] for i in tr],y[tr]); x32=profile.transform(configs)
    legacy=np.log(np.array([[float(r[t]) for t in ['tpot_ms','ttft_ms','dynamic_energy_per_token_mj']] for r in rows],dtype=np.float32))
    variants={'architecture14':(x14,legacy,2,True,FEATURES),
              'architecture14_temperature':(np.column_stack([x14,temperature]),legacy,2,True,FEATURES+['temp_cpu_start_c']),
              'physics32':(x32,np.log(y),3,False,FEATURES32),
              'physics32_temperature':(np.column_stack([x32,temperature]),np.log(y),3,False,FEATURES32+['temp_cpu_start_c'])}
    seeds=[42,123,2026]
    manifest=dict(seeds=seeds,targets=PHYSICS_TARGETS,audit=audit,
        splits={k:[rows[i]['config_id'] for i in ids] for k,ids in [('train',tr),('validation',va),('test',te)]},
        variants={name:dict(features=value[4],max_depth=value[2],inverse_tpot=value[3]) for name,value in variants.items()},
        source_sha256={n:hashlib.sha256((previous/n).read_bytes()).hexdigest() for n in ['input_snapshot.json','metadata.json']},
        experiment_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        temperature_range_c=[float(temperature.min()),float(temperature.max())],
        missing_inputs=['pre-inference battery percentage','pre-inference voltage'],
        excluded_inputs=['voltage_v (post-run)','ending temperature','measured power','acquisition order'],
        limits=['Temperature is a recorded pre-run sample, not necessarily immediately before inference.',
                'One reused test split, three training seeds; exploratory association, not a causal thermal law.',
                'Not architecture-only; requires a specified device state. No production/AL promotion.'])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    predictions={}; validation={}
    for name,(x,z,depth,inverse,features) in variants.items():
        models=[]
        for seed in seeds:
            estimator=fit_xgb(x,z,tr,va,seed,depth)
            joblib.dump(dict(models=estimator,features=features,profile=profile if name.startswith('physics') else None,
                            inverse_tpot=inverse,targets=PHYSICS_TARGETS,requires_recorded_start_temperature=name.endswith('temperature')),
                        out/f'{name}_seed{seed}.joblib')
            models.append(estimator)
            v=tree_predict(estimator,x[va],inverse)
            print(name,seed,'validation MAPE',np.mean(abs(v-y[va])/y[va],axis=0)*100,flush=True)
        predictions[name]=np.stack([tree_predict(m,x[te],inverse) for m in models])
    metrics=[]
    for name,p in predictions.items():
        for label,mask in [('pooled',np.ones(len(te),dtype=bool)),('old_batch',np.isin(te,ix['old_batch_test'])),('new_batch',np.isin(te,ix['new_batch_test']))]:
            for j,target in enumerate(PHYSICS_TARGETS):
                a=y[te[mask],j]; q=p[:,mask,j]; error=np.mean(abs(q-a)/a,axis=1)*100
                metrics.append(dict(model=name,split=label,target=target,n=len(a),mape=float(error.mean()),
                    seed_sd=float(error.std(ddof=1)),mae=float(np.mean(abs(q-a))),r2=float(np.mean([r2_score(a,v) for v in q]))))
    historical=json.loads((previous/'metrics.json').read_text())
    for k,seed in enumerate(seeds):
        for j,t in enumerate(PHYSICS_TARGETS):
            expected=next(r['mape'] for r in historical if r['model']=='xgboost14' and r['stage']=='plus_all' and r['test']=='pooled_test' and r['seed']==seed and r['target']==t)
            actual=np.mean(abs(predictions['architecture14'][k,:,j]-y[te,j])/y[te,j])*100
            assert abs(actual-expected)<.005,(actual,expected)
    draws=np.random.default_rng(20260915).integers(0,len(te),size=(5000,len(te)))
    comparisons=[]
    for name in ['architecture14','physics32']:
        base=np.mean(abs(predictions[name]-y[te])/y[te],axis=0)*100
        temp=np.mean(abs(predictions[name+'_temperature']-y[te])/y[te],axis=0)*100
        for j,t in enumerate(PHYSICS_TARGETS):
            delta=base[:,j]-temp[:,j]
            comparisons.append(dict(model=name,target=t,improvement_pp=float(delta.mean()),
                                    ci95=np.quantile(delta[draws].mean(1),[.025,.975]).tolist()))
    np.savez(out/'test_predictions.npz',actual=y[te],config_ids=np.array(manifest['splits']['test']),**predictions)
    (out/'summary.json').write_text(json.dumps(dict(metrics=metrics,paired_bootstrap=comparisons),indent=2)+'\n')
    lines=['# Recorded starting-temperature ablation','',
        'Frozen 1,564 architectures: 1,100 train / 150 validation / 314 test; seeds 42, 123, 2026. No test-based tuning. Original XGBoost settings retained within each feature family. No hardware access or model deployment.','',
        '| Features | Throughput MAPE | TTFT MAPE | Dynamic energy MAPE |','|---|---:|---:|---:|']
    for name in variants:
        r=[next(v for v in metrics if v['model']==name and v['split']=='pooled' and v['target']==t) for t in PHYSICS_TARGETS]
        lines.append('| '+name+' | '+' | '.join(f'{v["mape"]:.2f} ± {v["seed_sd"]:.2f}%' for v in r)+' |')
    lines+=['','± is training-seed SD, not a split confidence interval. Bootstrap averages paired architecture errors over seeds; intervals are conditional on one reused test split and not multiple-comparison adjusted.',
        '', 'Battery percentage and genuinely initial voltage are absent from this frozen cohort. Stored voltage_v is post-run and was excluded. No battery values were inferred from voltage. Starting temperature is not verified as the exact inference-start temperature.',
        '', 'Temperature-conditioned performance is a different task from architecture-only search prediction. Temperature can proxy session/background/DVFS state; improvement does not establish causality. Existing blocked/forward diagnostics on 936 rows are in outputs/state_signal_investigation_936/README.md.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    main()
