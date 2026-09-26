"""Architecture-only regression and ranking experiments for candidate selection."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import joblib
import numpy as np
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import r2_score
from ..config import ROOT
from ..config import FEATURES
from ..config import TARGETS
from ..data.io import write_csv
from ..features.analytic import physical_features


def search_metrics(actual,score):
    """Both actual and score are lower-is-better. Stable tie ordering."""
    k=math.ceil(len(actual)*.1)
    order=np.argsort(score,kind='stable'); truth=np.argsort(actual,kind='stable')
    best=float(np.min(actual))
    return dict(spearman=float(spearmanr(actual,score).statistic),
        top10pct_recall=float(len(set(order[:k])&set(truth[:k]))/k),
        best_of_5_regret_percent=float((min(actual[order[:5]])/best-1)*100),
        best_of_10_regret_percent=float((min(actual[order[:10]])/best-1)*100),
        selected_top10pct_mean=float(np.mean(actual[order[:k]])),
        oracle_top10pct_mean=float(np.mean(actual[truth[:k]])))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',default='scripts/prediction/outputs/architecture_search_models_936')
    args=ap.parse_args(); out=ROOT/args.output; out.mkdir(parents=True,exist_ok=False)
    old=ROOT/'scripts/prediction/outputs/surrogate_comparison_936'
    metadata=json.loads((old/'metadata.json').read_text()); source=Path(metadata['source'])
    assert hashlib.sha256(source.read_bytes()).hexdigest()==metadata['source_sha256']
    with source.open() as f: rows=list(csv.DictReader(f))
    with (ROOT/'scripts/sweep/configs/watch5_random_1000_50M_150M_sweep.csv').open() as f: cfg={r['config_id']:r for r in csv.DictReader(f)}
    pos={r['config_id']:i for i,r in enumerate(rows)}
    tr,va,te=(np.array([pos[i] for i in metadata['splits'][s]]) for s in ['train','validation','test'])
    parts=[physical_features(cfg[r['config_id']]) for r in rows]
    X=np.array([[p[0][k] for k in FEATURES] for p in parts],dtype=np.float32)
    Y=np.array([[float(r[t]) for t in TARGETS] for r in rows],dtype=np.float32)
    Z=np.log(Y); gs=X[:,FEATURES.index('q8_group_size')]
    # Ridge log-load basis: all terms are architecture/workload-derived. GS one-hot
    # and GS*log(work) express different cost scaling by implementation path.
    bases=[]
    for j in range(3):
        basis=[]
        for p in parts:
            b,c,m,k=p
            compute=[c['decode_total_mac'],c['prefill_total_mac'],c['sequence_mac']][j]
            memory=[m['decode_logical_bytes'],m['q8_weight_bytes']+49*m['kv_write_bytes'],32*m['q8_weight_bytes']+31*m['kv_read_per_head_bytes']][j]
            values=np.log([compute,memory,k['omp_region_count'],b['d_model'],b['d_mlp'],b['n_layer']])
            g=b['q8_group_size']
            basis.append(list(values)+[float(g==16),float(g==64),float(g==16)*values[0],float(g==64)*values[0]])
        bases.append(np.asarray(basis))
    common=dict(n_estimators=1500,learning_rate=.03,tree_method='hist',n_jobs=4,
                early_stopping_rounds=50,subsample=.85,colsample_bytree=.9,reg_lambda=5)

    def fit_predict(family,hp,j,a,b,c,seed):
        """a=train, b=early stopping, c=prediction only. Returns lower-is-better."""
        pack=dict(family=family,params=hp,target=TARGETS[j],features=FEATURES)
        if family=='ranker':
            cut=np.quantile(Z[a,j],np.linspace(0,1,11)[1:-1])
            relevance=lambda z:9-np.searchsorted(cut,z,side='right')
            m=xgb.XGBRanker(**common,objective='rank:pairwise',max_depth=hp['depth'],min_child_weight=3,
                random_state=seed,eval_metric='ndcg@10')
            m.fit(X[a],relevance(Z[a,j]),group=[len(a)],eval_set=[(X[b],relevance(Z[b,j]))],eval_group=[[len(b)]],verbose=False)
            pack['model']=m; pack['relevance_cutpoints']=cut
            return -m.predict(X[c]),pack
        if family=='group_experts':
            result=np.zeros(len(c)); models={}
            for g in [16,32,64]:
                aa=a[gs[a]==g]; bb=b[gs[b]==g]; mask=gs[c]==g
                assert len(aa)>0 and len(bb)>0
                m=xgb.XGBRegressor(**common,objective='reg:squarederror',max_depth=hp['depth'],
                    min_child_weight=hp['child'],random_state=seed)
                m.fit(X[aa],Z[aa,j],eval_set=[(X[bb],Z[bb,j])],verbose=False)
                if mask.any():result[mask]=np.exp(m.predict(X[c[mask]]))
                models[g]=m
            pack['models']=models
            return result,pack
        if family in ['physics_ridge','physics_residual']:
            ridge=make_pipeline(StandardScaler(),Ridge(alpha=hp['alpha']))
            ridge.fit(bases[j][a],Z[a,j]); offsets=ridge.predict(bases[j]); pack['ridge']=ridge
            if family=='physics_ridge':return np.exp(offsets[c]),pack
            residual=Z[:,j]-offsets
            m=xgb.XGBRegressor(**common,objective='reg:squarederror',max_depth=hp['depth'],min_child_weight=3,random_state=seed)
            m.fit(X[a],residual[a],eval_set=[(X[b],residual[b])],verbose=False)
            pack['model']=m
            return np.exp(offsets[c]+m.predict(X[c])),pack
        m=xgb.XGBRegressor(**common,objective='reg:squarederror',max_depth=2,min_child_weight=3,random_state=seed)
        m.fit(X[a],Z[a,j],eval_set=[(X[b],Z[b,j])],verbose=False)
        pack['model']=m
        return np.exp(m.predict(X[c])),pack

    grids=dict(baseline=[{}],group_experts=[dict(depth=d,child=w) for d,w in [(1,3),(2,3),(3,5)]],
        physics_ridge=[dict(alpha=a) for a in [1.,10.,100.]],
        physics_residual=[dict(alpha=a,depth=d) for a,d in [(1.,1),(10.,1),(10.,2),(100.,2)]],
        ranker=[dict(depth=d) for d in [1,2,4]])
    folds=[]
    for a,c in StratifiedKFold(3,shuffle=True,random_state=20260911).split(tr,gs[tr]):
        aa,bb=train_test_split(tr[a],test_size=.15,random_state=42,stratify=gs[tr[a]])
        folds.append((aa,bb,tr[c]))
    choices={}; cv=[]
    for family,grid in grids.items():
        choices[family]={}
        for j,t in enumerate(TARGETS):
            for hp in grid:
                losses=[]
                for a,b,c in folds:
                    pred,_=fit_predict(family,hp,j,a,b,c,42)
                    loss=1-search_metrics(Y[c,j],pred)['top10pct_recall'] if family=='ranker' else float(np.mean(abs(pred-Y[c,j])/Y[c,j]))
                    losses.append(loss)
                value=float(np.mean(losses)); cv.append(dict(family=family,target=t,params=json.dumps(hp),cv_loss=value,
                    objective='1-top10pct_recall' if family=='ranker' else 'MAPE_fraction'))
                if t not in choices[family] or value<choices[family][t]['cv_loss']:
                    choices[family][t]=dict(params=hp,cv_loss=value)
        print('SELECTED',family,choices[family],flush=True)
        write_csv(out/'cv_tuning.csv',cv)
    manifest=dict(source_sha256=metadata['source_sha256'],splits=metadata['splits'],features=FEATURES,choices=choices,
        cv_seed=20260911,training_seeds=[42,123,2026],physics_basis='See compare_search_surrogates.py: log MAC, logical bytes, OpenMP regions, dimensions/layers and Q8-group interactions',
        notes='Architecture/workload-only inputs. Test reused for exploratory comparison. Ranking scores are not latency or energy predictions. Ridge coefficients are learned only from training labels.')
    (out/'metadata.json').write_text(json.dumps(manifest,indent=2))
    scores=[]; records=[]; predictions={}
    for family in grids:
        for j,t in enumerate(TARGETS):
            predictions[family,j]=[]
            for seed in [42,123,2026]:
                p,pack=fit_predict(family,choices[family][t]['params'],j,tr,va,te,seed)
                assert np.isfinite(p).all()
                joblib.dump(pack,out/f'{family}_seed{seed}_{t}.joblib')
                sm=search_metrics(Y[te,j],p)
                sm.update(mae=None if family=='ranker' else float(np.mean(abs(p-Y[te,j]))),
                    mape=None if family=='ranker' else float(np.mean(abs(p-Y[te,j])/Y[te,j])*100),
                    r2=None if family=='ranker' else float(r2_score(Y[te,j],p)))
                scores.append(dict(family=family,seed=seed,target=t,**sm))
                predictions[family,j].append(p)
                for i,actual,estimate in zip(te,Y[te,j],p):
                    records.append(dict(family=family,seed=seed,target=t,config_id=rows[i]['config_id'],actual=float(actual),prediction=float(estimate)))
        print('EVALUATED',family,flush=True)
    write_csv(out/'metrics.csv',scores);write_csv(out/'test_predictions.csv',records)
    summary=[]
    for family in grids:
        for t in TARGETS:
            rs=[r for r in scores if r['family']==family and r['target']==t]
            summary.append(dict(family=family,target=t,**{k:None if rs[0][k] is None else float(np.mean([r[k] for r in rs])) for k in rs[0] if k not in ['family','target','seed']}))
    write_csv(out/'summary.csv',summary)
    # Conditional paired bootstrap for regression error. Does not correct repeated
    # experimentation on this test set or multiple comparisons.
    rng=np.random.default_rng(42); boot=rng.integers(0,len(te),size=(5000,len(te))); cis=[]
    for family in grids:
        if family in ['baseline','ranker']:continue
        for j,t in enumerate(TARGETS):
            baseline=np.mean(abs(np.array(predictions['baseline',j])-Y[te,j])/Y[te,j]*100,axis=0)
            new=np.mean(abs(np.array(predictions[family,j])-Y[te,j])/Y[te,j]*100,axis=0)
            delta=baseline-new;lo,hi=np.quantile(delta[boot].mean(1),[.025,.975])
            cis.append(dict(family=family,target=t,mape_improvement_pp=float(delta.mean()),ci95_lower=float(lo),ci95_upper=float(hi)))
    write_csv(out/'paired_bootstrap.csv',cis)
    lines=['# Architecture-only search models','',
        'All models use the original 598 training / 150 validation / 188 test architectures. Hyperparameters are chosen by three-fold CV inside training, with a separate early-stopping subset per fold. Final fitting uses the original validation rows only for early stopping. Test results are exploratory because this set was inspected in earlier experiments.',
        '', '| Model | Target | MAPE % | Spearman | Recall of best 10% | Best-of-10 regret % |', '|---|---|---:|---:|---:|---:|']
    for r in summary:
        mape='n/a' if r['mape'] is None else f"{r['mape']:.2f}"
        lines.append(f"| {r['family']} | {r['target']} | {mape} | {r['spearman']:.3f} | {100*r['top10pct_recall']:.1f}% | {r['best_of_10_regret_percent']:.2f} |")
    lines+=['','## Definitions and restrictions','',
        '- Baseline: original XGBoost log-target regressors. Group experts fit separate models for GS=16, 32, 64.',
        '- Physics ridge: regularized linear model of log physical workload; physics residual adds an XGBoost residual model. No measured bandwidth or cache capacity is assumed.',
        '- Ranker: XGBoost pairwise ranking with training-defined relevance deciles. Lower reported scores are better. Scores have no physical units; MAE/MAPE/R² are intentionally unavailable.',
        '- Best 10% means 19 of 188 held-out candidates. Recall is overlap with the 19 truly best measured candidates. Best-of-10 regret compares the best actual result among 10 recommended candidates with the measured optimum in this test set.',
        '- Means over three training seeds. Evaluation uses noisy single-run labels; identifying the recorded optimum is not proof of the true repeatable optimum.',
        '- Inputs exclude temperatures, voltages, power measurements, performance measurements, run order and config ID. Labels enter only training or evaluation.',
        '- Joblib checkpoints contain trusted local Python objects. They remain trained on the training partition, not all 936 examples.',
        '', '## Reproduce','', '```bash','conda activate nanollmforge',
        'python -m scripts.prediction compare-search-surrogates --output scripts/prediction/outputs/architecture_search_models_repeat','```']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
