"""XGBoost feature ablation using analytic workload proxies, never measured inputs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score
from ..config import ROOT
from ..config import FEATURES
from ..config import TARGETS
from ..data.io import write_csv


from ..features.analytic import physical_features


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='scripts/prediction/outputs/xgboost_physics_priors_936')
    args=parser.parse_args()
    out=ROOT/args.output
    out.mkdir(parents=True,exist_ok=False)
    previous=ROOT/'scripts/prediction/outputs/surrogate_comparison_936'
    meta=json.loads((previous/'metadata.json').read_text())
    source=Path(meta['source'])
    assert hashlib.sha256(source.read_bytes()).hexdigest()==meta['source_sha256']
    with source.open() as f: rows=list(csv.DictReader(f))
    cfg_path=ROOT/'scripts/sweep/configs/watch5_random_1000_50M_150M_sweep.csv'
    with cfg_path.open() as f: cfg={r['config_id']:r for r in csv.DictReader(f)}
    mapping={r['config_id']:i for i,r in enumerate(rows)}
    train,val,test=([mapping[i] for i in meta['splits'][k]] for k in ['train','validation','test'])
    assert not set(train)&set(test) and not set(val)&set(test)
    Y=np.array([[float(r[t]) for t in TARGETS] for r in rows],dtype=np.float32)
    logy=np.log(Y); ys=logy[train].std(0).clip(1e-6)
    derived=[physical_features(cfg[r['config_id']]) for r in rows]
    variants={'baseline':[], 'compute':[1], 'memory':[2], 'kernel':[3], 'all_physics':[1,2,3]}
    grids=[dict(max_depth=d,min_child_weight=w,reg_lambda=l) for d,w,l in [(2,3,5),(3,3,5),(4,5,10),(6,5,10)]]
    matrices={}; feature_names={}
    for name,groups in variants.items():
        names=FEATURES+sum([list(derived[0][g]) for g in groups],[])
        records=[]
        for parts in derived:
            d={**parts[0]}
            for g in groups: d.update(parts[g])
            records.append([d[k] for k in names])
        matrices[name]=np.asarray(records,dtype=np.float32)
        feature_names[name]=names
        assert np.isfinite(matrices[name]).all()

    def fit(X,hp,seed):
        models=[]
        for j,t in enumerate(TARGETS):
            model=xgb.XGBRegressor(n_estimators=1500,learning_rate=.03,tree_method='hist',n_jobs=4,
                objective='reg:squarederror',early_stopping_rounds=50,subsample=.85,colsample_bytree=.9,
                random_state=seed,**hp)
            model.fit(X[train],logy[train,j],eval_set=[(X[val],logy[val,j])],verbose=False)
            models.append(model)
        vp=np.column_stack([m.predict(X[val]) for m in models])
        return models,float(np.mean(((vp-logy[val])/ys)**2))

    chosen={}; tuning=[]
    for name,X in matrices.items():
        for hp in grids:
            _,score=fit(X,hp,42)
            tuning.append(dict(variant=name,params=json.dumps(hp),validation_standardized_log_mse=score))
            if name not in chosen or score<chosen[name]['score']:
                chosen[name]=dict(score=score,params=hp)
        print('VALIDATION',name,len(feature_names[name]),chosen[name],flush=True)
    selected=min(chosen,key=lambda name:chosen[name]['score'])
    manifest=dict(source_sha256=meta['source_sha256'],config_sha256=hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
        splits=meta['splits'],features=feature_names,chosen=chosen,validation_selected_variant=selected,
        workload=dict(prompt_tokens=49,decode_forward_calls=31,output_tokens=32,vocab=50257,context_capacity=256),
        xgboost_version=xgb.__version__,notes='Analytic MAC/logical-byte proxies, not measured FLOPs or DRAM traffic; same baseline split/grid/seeds. Previously inspected test set is reused for exploratory comparison, not fresh confirmatory testing.')
    (out/'metadata.json').write_text(json.dumps(manifest,indent=2))
    write_csv(out/'tuning.csv',tuning)
    metrics=[]; predictions=[]; importance=[]; test_preds={}
    for name,X in matrices.items():
        test_preds[name]=[]
        for seed in [42,123,2026]:
            models,_=fit(X,chosen[name]['params'],seed)
            pred=np.exp(np.column_stack([m.predict(X[test]) for m in models]))
            test_preds[name].append(pred)
            for j,(t,model) in enumerate(zip(TARGETS,models)):
                model.save_model(out/f'{name}_seed{seed}_{t}.json')
                reloaded=xgb.XGBRegressor(); reloaded.load_model(out/f'{name}_seed{seed}_{t}.json')
                np.testing.assert_allclose(model.predict(X[test]),reloaded.predict(X[test]),rtol=1e-6)
                actual=Y[test,j]; estimated=pred[:,j]
                metrics.append(dict(variant=name,seed=seed,target=t,mae=float(mean_absolute_error(actual,estimated)),
                    mape_percent=float(np.mean(abs(estimated-actual)/actual)*100),r2=float(r2_score(actual,estimated)),
                    spearman=float(spearmanr(actual,estimated).statistic)))
                for i,a,p in zip(test,actual,estimated):
                    predictions.append(dict(variant=name,seed=seed,target=t,config_id=rows[i]['config_id'],actual=float(a),prediction=float(p)))
                for f,gain in zip(feature_names[name],model.feature_importances_):
                    importance.append(dict(variant=name,seed=seed,target=t,feature=f,normalized_gain=float(gain)))
        print('TESTED',name,flush=True)
    # Confirm that differences cannot be attributed to an accidentally changed baseline.
    with (previous/'metrics.csv').open() as f: old=[r for r in csv.DictReader(f) if r['model']=='xgboost']
    for r in old:
        now=next(m for m in metrics if m['variant']=='baseline' and m['seed']==int(r['seed']) and m['target']==r['target'])
        np.testing.assert_allclose(now['mae'],float(r['mae']),rtol=1e-6)
    summary=[]
    for name in variants:
        for t in TARGETS:
            group=[m for m in metrics if m['variant']==name and m['target']==t]
            record=dict(variant=name,target=t,n_features=len(feature_names[name]))
            for k in ['mae','mape_percent','r2','spearman']:
                record[k+'_mean']=float(np.mean([m[k] for m in group]))
                record[k+'_std']=float(np.std([m[k] for m in group],ddof=1))
            summary.append(record)
    rng=np.random.default_rng(20260909); boot=rng.integers(0,len(test),size=(5000,len(test)))
    intervals=[]
    for name in variants:
        if name=='baseline':continue
        for j,t in enumerate(TARGETS):
            base=np.mean(np.abs(np.asarray(test_preds['baseline'])[:,:,j]-Y[test,j])/Y[test,j]*100,axis=0)
            new=np.mean(np.abs(np.asarray(test_preds[name])[:,:,j]-Y[test,j])/Y[test,j]*100,axis=0)
            delta=base-new; low,high=np.quantile(delta[boot].mean(1),[.025,.975])
            intervals.append(dict(variant=name,target=t,mape_improvement_pp=float(delta.mean()),ci95_lower=float(low),ci95_upper=float(high)))
    for filename,records in [('metrics',metrics),('summary',summary),('test_predictions',predictions),('feature_importance',importance),('paired_bootstrap',intervals)]:
        write_csv(out/f'{filename}.csv',records)
    lines=['# XGBoost physical-prior feature ablation','',
        f'Validation-selected variant: **{selected}**. Fixed 598/150/188 train/validation/test split, three training seeds, four hyperparameter candidates per variant. Baseline metrics reproduce the preceding experiment.',
        '', '| Variant | Target | Features | MAPE (%) | MAE | R² |','|---|---|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['variant']} | {r['target']} | {r['n_features']} | {r['mape_percent_mean']:.2f} ± {r['mape_percent_std']:.2f} | {r['mae_mean']:.3f} | {r['r2_mean']:.3f} |")
    lines+=['','## Feature definitions','',
        '- Baseline: the original 14 architecture/derived features.',
        '- Compute: analytical MAC counts for projections, MLP, classifier and causal attention; prefill/decode separation, output-element count, and compute fractions.',
        '- Memory: INT8 weights plus FP32 group scales, dequantized embedding storage, KV logical reads/writes/allocation, layer footprints and MAC/byte ratios.',
        '- Kernel: MACs interacting with GS=16/32 specialized paths versus GS=64 general loop, number of quantization groups, approximate OpenMP region count and GQA reuse.',
        '- All physics: union of the preceding features. This is feature engineering, not a physics-constrained loss or calibrated roofline time model.',
        '', '## Assumptions and limitations','',
        'Fixed vocabulary 50,257, prompt 49, 31 decode forward calls (normally 32 output tokens), context capacity 256. MAC counts exclude normalization, nonlinear activation and sampling. Bytes describe logical footprints/accesses, not measured memory traffic. Prefill weight reuse and grouped KV cache reuse depend on the implementation and hardware. Kernel path features do not assert that compiler auto-vectorization is absent.',
        'Only architecture/workload inputs are used. No measured performance, temperature, frequency, voltage, config ID or run order is a feature. Correlated feature gain is descriptive, not causal importance.',
        'The existing test set has already been inspected. This is exploratory ablation, not an untouched confirmatory experiment. Bootstrap intervals condition on this split, are not corrected for multiple comparisons, and do not establish cross-session generalization. Seed SD is not uncertainty calibration.',
        'Checkpoints are trained on 598 rows with validation early stopping; all 936 rows participate in the experiment but are not all training rows.',
        '', '## Reproduce','', '```bash', 'conda activate nanollmforge',
        'python -m scripts.prediction compare-physics-priors --output scripts/prediction/outputs/xgboost_physics_priors_repeat','```']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print('SELECTED',selected,flush=True)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
