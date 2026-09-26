"""Learning curves, state diagnostics, and train-only CV tuning of XGBoost."""
import argparse
import csv
import hashlib
import json
import itertools
from pathlib import Path
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from ..config import ROOT
from ..config import FEATURES
from ..config import TARGETS
from ..data.io import write_csv
from ..features.analytic import physical_features


from ..evaluation.metrics import metrics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='scripts/prediction/outputs/xgboost_error_diagnosis_936')
    args=parser.parse_args()
    out=ROOT/args.output; out.mkdir(parents=True,exist_ok=False)
    prior=ROOT/'scripts/prediction/outputs/surrogate_comparison_936'
    meta=json.loads((prior/'metadata.json').read_text())
    src=Path(meta['source'])
    assert hashlib.sha256(src.read_bytes()).hexdigest()==meta['source_sha256']
    with src.open() as f: rows=list(csv.DictReader(f))
    with (ROOT/'scripts/sweep/configs/watch5_random_1000_50M_150M_sweep.csv').open() as f: configs={r['config_id']:r for r in csv.DictReader(f)}
    index={r['config_id']:i for i,r in enumerate(rows)}
    tr,va,te=(np.array([index[i] for i in meta['splits'][k]]) for k in ['train','validation','test'])
    parts=[physical_features(configs[r['config_id']]) for r in rows]
    names={'baseline':FEATURES,'physics':FEATURES+sum([list(parts[0][g]) for g in [1,2,3]],[])}
    matrices={}
    for name,fields in names.items():
        matrices[name]=np.asarray([[{**p[0],**p[1],**p[2],**p[3]}[f] for f in fields] for p in parts],dtype=np.float32)
    X=matrices['baseline']; Y=np.asarray([[float(r[t]) for t in TARGETS] for r in rows],dtype=np.float32); Z=np.log(Y)
    strata=X[:,8].astype(int)
    common=dict(n_estimators=1500,learning_rate=.03,tree_method='hist',n_jobs=4,early_stopping_rounds=50)
    base=dict(max_depth=2,min_child_weight=3,reg_lambda=5,subsample=.85,colsample_bytree=.9,objective='reg:squarederror')

    def fit(x,j,train,stop,hp,seed=42,eval_metric=None):
        m=xgb.XGBRegressor(**common,**hp,random_state=seed,eval_metric=eval_metric)
        m.fit(x[train],Z[train,j],eval_set=[(x[stop],Z[stop,j])],verbose=False)
        return m

    # The validation set stays fixed; different seeds vary stratified training subsets.
    curves=[]
    for n in [120,240,360,480,len(tr)]:
        for seed in [42,123,2026]:
            subset=tr if n==len(tr) else train_test_split(tr,train_size=n,random_state=seed,stratify=strata[tr])[0]
            for j,t in enumerate(TARGETS):
                m=fit(X,j,subset,va,base,seed)
                for split,ids in [('train',subset),('validation',va)]:
                    curves.append(dict(n_train=n,seed=seed,target=t,split=split,**metrics(Y[ids,j],np.exp(m.predict(X[ids])))))
        print('LEARNING CURVE',n,flush=True)
        write_csv(out/'learning_curve.csv',curves)

    # Diagnostics only: state features are deliberately excluded from deployable models.
    state=np.array([[float(r['temp_cpu_start_c']),float(r['voltage_v'])] for r in rows],dtype=np.float32)
    stateX=np.column_stack([X,state]); contextual=[]; residuals=[]; baseline_preds={}
    for seed in [42,123,2026]:
        for j,t in enumerate(TARGETS):
            for name,x in [('baseline',X),('state_diagnostic_only',stateX)]:
                m=fit(x,j,tr,va,base,seed)
                pred=np.exp(m.predict(x[te]))
                contextual.append(dict(variant=name,seed=seed,target=t,**metrics(Y[te,j],pred)))
                if name=='baseline':
                    baseline_preds[seed,j]=pred
                    for k,field in enumerate(['temp_cpu_start_c','voltage_v']):
                        residuals.append(dict(seed=seed,target=t,state=field,
                            signed_log_residual_spearman=float(spearmanr(np.log(pred)-Z[te,j],state[te,k]).statistic)))
    write_csv(out/'state_diagnostic.csv',contextual); write_csv(out/'residual_correlations.csv',residuals)
    print('STATE DIAGNOSTICS COMPLETE',flush=True)

    # Candidate selection uses only the 598 training rows. A separate 15% early-stop
    # subset inside each training fold leaves its scoring fold untouched.
    folds=[]
    for a,b in StratifiedKFold(3,shuffle=True,random_state=20260910).split(tr,strata[tr]):
        fit_ids,stop_ids=train_test_split(tr[a],test_size=.15,random_state=42,stratify=strata[tr[a]])
        folds.append((fit_ids,stop_ids,tr[b]))
    candidates=[]
    for name in matrices:
        for depth,child,objective in itertools.product([2,4,6],[1,5],['reg:squarederror','reg:absoluteerror']):
            candidates.append(dict(features=name,hp=dict(max_depth=depth,min_child_weight=child,reg_lambda=5,
                subsample=.9,colsample_bytree=1.,objective=objective)))
    candidates.append(dict(features='baseline',hp=base))
    scores=[]; selected={}
    for j,t in enumerate(TARGETS):
        for ci,candidate in enumerate(candidates):
            values=[]
            for fi,(a,b,c) in enumerate(folds):
                m=fit(matrices[candidate['features']],j,a,b,candidate['hp'],eval_metric='mae')
                values.append(metrics(Y[c,j],np.exp(m.predict(matrices[candidate['features']][c])))['mape'])
            score=float(np.mean(values))
            scores.append(dict(target=t,candidate=ci,features=candidate['features'],params=json.dumps(candidate['hp']),cv_mape=score))
            if t not in selected or score<selected[t]['cv_mape']:
                selected[t]=dict(**candidate,cv_mape=score)
        print('CV SELECTED',t,selected[t],flush=True)
        write_csv(out/'cv_tuning.csv',scores)
    # Freeze selected settings before looking at their test predictions.
    manifest=dict(source_sha256=meta['source_sha256'],splits=meta['splits'],features=names,selected=selected,
                  cv_seed=20260910,cv_folds=3,candidates_per_target=len(candidates),
                  notes='Target-specific train-only CV minimizes MAPE; early stopping on log MAE. Test set reused from preceding experiments; exploratory results.')
    (out/'metadata.json').write_text(json.dumps(manifest,indent=2))
    final=[]; predictions=[]; bootrows=[]
    rng=np.random.default_rng(42); boot=rng.integers(0,len(te),size=(5000,len(te)))
    for j,t in enumerate(TARGETS):
        candidate=selected[t]; x=matrices[candidate['features']]; deltas=[]
        for seed in [42,123,2026]:
            m=fit(x,j,tr,va,candidate['hp'],seed,eval_metric='mae')
            pred=np.exp(m.predict(x[te])); m.save_model(out/f'optimized_seed{seed}_{t}.json')
            reload=xgb.XGBRegressor(); reload.load_model(out/f'optimized_seed{seed}_{t}.json')
            np.testing.assert_allclose(pred,np.exp(reload.predict(x[te])),rtol=1e-6)
            for family,p in [('baseline',baseline_preds[seed,j]),('optimized',pred)]:
                final.append(dict(variant=family,seed=seed,target=t,**metrics(Y[te,j],p)))
                for i,a,estimate in zip(te,Y[te,j],p):
                    predictions.append(dict(variant=family,seed=seed,target=t,config_id=rows[i]['config_id'],actual=float(a),prediction=float(estimate)))
            deltas.append((abs(baseline_preds[seed,j]-Y[te,j])-abs(pred-Y[te,j]))/Y[te,j]*100)
        delta=np.mean(deltas,axis=0); lo,hi=np.quantile(delta[boot].mean(1),[.025,.975])
        bootrows.append(dict(target=t,mape_improvement_pp=float(delta.mean()),ci95_lower=float(lo),ci95_upper=float(hi)))
    write_csv(out/'metrics.csv',final); write_csv(out/'test_predictions.csv',predictions); write_csv(out/'paired_bootstrap.csv',bootrows)
    print('FINAL METRICS',json.dumps(final),flush=True)
    print('BOOTSTRAP',json.dumps(bootrows),flush=True)


if __name__=='__main__':
    main()
