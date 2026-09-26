"""Paired architecture-only learning curves with separate early-stop/test rows."""
import argparse
import csv
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from ..config import ROOT
from ..config import FEATURES
from ..config import TARGETS
from ..models.neural import TinyTransformer
from ..data.io import write_csv
from ..features.analytic import physical_features


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',default='scripts/prediction/outputs/transformer_learning_curve_936')
    args=ap.parse_args(); out=ROOT/args.output;out.mkdir(parents=True,exist_ok=False)
    src=ROOT/'scripts/sweep/outputs/watch5_random_50M_150M_40C_decode32_results_clean.csv'
    old=ROOT/'scripts/prediction/outputs/surrogate_comparison_936'
    previous=json.loads((old/'metadata.json').read_text())
    assert hashlib.sha256(src.read_bytes()).hexdigest()==previous['source_sha256']
    hp=json.loads((old/'selected_hyperparameters.json').read_text())
    with src.open() as f:rows=list(csv.DictReader(f))
    with (ROOT/'scripts/sweep/configs/watch5_random_1000_50M_150M_sweep.csv').open() as f:cfg={r['config_id']:r for r in csv.DictReader(f)}
    X=np.array([[physical_features(cfg[r['config_id']])[0][k] for k in FEATURES] for r in rows],dtype=np.float32)
    Y=np.array([[float(r[t]) for t in TARGETS] for r in rows],dtype=np.float32);Z=np.log(Y)
    assert len(rows)==936 and len(np.unique(X[:,:7],axis=0))==936
    assert np.isfinite(X).all() and np.isfinite(Y).all() and (Y>0).all()
    strata=X[:,8].astype(int);torch.set_num_threads(4)
    sizes=[120,240,360,480,598]; seeds=[42,123]; split_seeds=[20260909,20260910,20260911,20260912,20260913]
    manifests=[]; metrics=[]; details=[]; training=[]
    def evaluate(family,split_seed,seed,n,partition,ids,pred):
        assert np.isfinite(pred).all() and (pred>0).all()
        for j,t in enumerate(TARGETS):
            y=Y[ids,j];p=pred[:,j]
            metrics.append(dict(model=family,split_seed=split_seed,training_seed=seed,n_train=n,partition=partition,target=t,
                mape=float(np.mean(abs(p-y)/y)*100),mae=float(np.mean(abs(p-y))),
                log_rmse=float(np.sqrt(np.mean((np.log(p)-np.log(y))**2))),r2=float(r2_score(y,p)),
                spearman=float(spearmanr(y,p).statistic)))
            if partition=='test':
                for i,a,b in zip(ids,y,p):details.append(dict(model=family,split_seed=split_seed,training_seed=seed,n_train=n,
                    config_id=rows[i]['config_id'],target=t,actual=float(a),prediction=float(b)))
    for split_seed in split_seeds:
        pool,test=train_test_split(np.arange(len(rows)),test_size=.2,random_state=split_seed,stratify=strata)
        pool,stop=train_test_split(pool,test_size=.2,random_state=split_seed,stratify=strata[pool])
        assert len(pool)==598 and len(stop)==150 and len(test)==188
        assert not(set(pool)&set(test)) and not(set(stop)&set(test))
        for seed in seeds:
            rng=np.random.default_rng(seed)
            # Interleave independently shuffled group lists by normalized rank.
            # Prefixes are nested and approximately stratified at every size.
            schedule=[]
            for g in np.unique(strata):
                group=rng.permutation(pool[strata[pool]==g])
                schedule.extend(((k+.5)/len(group),int(i)) for k,i in enumerate(group))
            ordering=np.array([i for _,i in sorted(schedule)])
            manifests.append(dict(split_seed=split_seed,training_seed=seed,
                train_order=[rows[i]['config_id'] for i in ordering],
                early_stopping=[rows[i]['config_id'] for i in stop],test=[rows[i]['config_id'] for i in test]))
            for n in sizes:
                start=time.monotonic();train=ordering[:n]
                xm,xs=X[train].mean(0),X[train].std(0).clip(1e-6)
                ym,ys=Z[train].mean(0),Z[train].std(0).clip(1e-6)
                xn=torch.tensor((X-xm)/xs);yn=torch.tensor((Z-ym)/ys)
                torch.manual_seed(seed);h=hp['transformer']
                model=TinyTransformer(len(FEATURES),h['width'],h['dropout'])
                optimizer=torch.optim.AdamW(model.parameters(),lr=h['lr'],weight_decay=.01)
                best=float('inf');state=None;best_epoch=0
                for epoch in range(1,501):
                    model.train();order=torch.tensor(train)[torch.randperm(len(train))]
                    for batch in order.split(128):
                        optimizer.zero_grad();loss=((model(xn[batch])-yn[batch])**2).mean()
                        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                    model.eval()
                    with torch.no_grad():score=float(((model(xn[stop])-yn[stop])**2).mean())
                    if score<best-1e-5:
                        best,best_epoch=score,epoch;state={k:v.detach().clone() for k,v in model.state_dict().items()}
                    if epoch-best_epoch>=40:break
                model.load_state_dict(state);model.eval()
                for partition,ids in [('train',train),('test',test)]:
                    with torch.no_grad():pred=np.exp(model(xn[ids]).numpy()*ys+ym)
                    evaluate('transformer',split_seed,seed,n,partition,ids,pred)
                training.append(dict(model='transformer',split_seed=split_seed,training_seed=seed,n_train=n,
                    target='joint',best_iteration=best_epoch,total_iterations=epoch,hit_cap=epoch==500))
                # Same subsets, stop rows and test rows for the XGBoost control.
                models=[]
                for j,t in enumerate(TARGETS):
                    m=xgb.XGBRegressor(n_estimators=1500,learning_rate=.03,tree_method='hist',n_jobs=4,
                        objective='reg:squarederror',early_stopping_rounds=50,subsample=.85,
                        colsample_bytree=.9,random_state=seed,**hp['xgboost'])
                    m.fit(X[train],Z[train,j],eval_set=[(X[stop],Z[stop,j])],verbose=False);models.append(m)
                    training.append(dict(model='xgboost',split_seed=split_seed,training_seed=seed,n_train=n,target=t,
                        best_iteration=m.best_iteration+1,total_iterations=m.get_booster().num_boosted_rounds(),
                        hit_cap=m.get_booster().num_boosted_rounds()==1500))
                for partition,ids in [('train',train),('test',test)]:
                    pred=np.exp(np.column_stack([m.predict(X[ids]) for m in models]))
                    evaluate('xgboost',split_seed,seed,n,partition,ids,pred)
                write_csv(out/'metrics.csv',metrics);write_csv(out/'training.csv',training)
                print(f'Completed split={split_seed} seed={seed} n={n}, TF best/total epochs={best_epoch}/{epoch}, {time.monotonic()-start:.1f}s',flush=True)
    write_csv(out/'test_predictions.csv',details)
    (out/'metadata.json').write_text(json.dumps(dict(source_sha256=previous['source_sha256'],features=FEATURES,targets=TARGETS,
        sizes=sizes,hyperparameters=hp,split_manifests=manifests,
        notes='Frozen previously chosen hyperparameters; 5 overlapping random splits x 2 subset/training seeds. Test never used for early stopping. Existing dataset already explored, not prospective confirmation. Early-stop set fixed at 150 for every size.'),indent=2))
    summary=[]
    for family in ['transformer','xgboost']:
        for t in TARGETS:
            for n in sizes:
                for partition in ['train','test']:
                    rs=[r for r in metrics if r['model']==family and r['target']==t and r['n_train']==n and r['partition']==partition]
                    row=dict(model=family,target=t,n_train=n,partition=partition)
                    for key in ['mape','mae','log_rmse','r2','spearman']:
                        means=[np.mean([r[key] for r in rs if r['split_seed']==s]) for s in split_seeds]
                        row[key+'_mean']=float(np.mean(means));row[key+'_split_sd']=float(np.std(means,ddof=1))
                    summary.append(row)
    write_csv(out/'summary.csv',summary)
    # Report paired changes, not significance from correlated overlapping splits.
    improvements=[]
    for family in ['transformer','xgboost']:
        for t in TARGETS:
            for begin in [120,240,480]:
                changes=[]
                for s in split_seeds:
                    def avg(n):return np.mean([r['mape'] for r in metrics if r['model']==family and r['target']==t and r['n_train']==n and r['partition']=='test' and r['split_seed']==s])
                    changes.append(avg(begin)-avg(598))
                improvements.append(dict(model=family,target=t,from_n=begin,to_n=598,
                    mean_mape_improvement_pp=float(np.mean(changes)),min_split_improvement_pp=float(min(changes)),
                    max_split_improvement_pp=float(max(changes)),splits_improved=int(sum(d>0 for d in changes))))
    write_csv(out/'paired_changes.csv',improvements)
    cache=tempfile.mkdtemp(prefix='learning-curve-cache-')
    os.environ.setdefault('MPLCONFIGDIR',cache);os.environ.setdefault('XDG_CACHE_HOME',cache)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.9),layout='constrained')
    for ax,t,title in zip(axes,TARGETS,['TPOT','TTFT','Dynamic energy / output token']):
        for family,color in [('transformer','#2563eb'),('xgboost','#d97706')]:
            rs=[r for r in summary if r['model']==family and r['target']==t and r['partition']=='test']
            means=np.array([r['mape_mean'] for r in rs]);sd=np.array([r['mape_split_sd'] for r in rs])
            ax.plot(sizes,means,'o-',color=color,label=family);ax.fill_between(sizes,means-sd,means+sd,color=color,alpha=.15)
        ax.set(title=title,xlabel='Training architectures',ylabel='Held-out MAPE (%)');ax.grid(alpha=.2);ax.legend()
    fig.suptitle('5 random splits × 2 seeds; mean ± split SD; separate early stopping',fontsize=12)
    fig.savefig(out/'learning_curve.png',dpi=160);plt.close(fig)
    lines=['# Transformer data-scaling experiment','',
        'Architecture-only 936-row dataset. Each split has a pool of 598 training rows, 150 separate early-stopping rows, and 188 test rows. Nested training subsets: 120, 240, 360, 480, 598. Five random splits and two training/subset seeds, paired with XGBoost (50 Transformer fits and 150 XGBoost regressors).',
        'Transformer: original two-layer, four-head, width-32 network, dropout 0.2, AdamW learning rate 0.0003. Up to 500 epochs with patience 40. Train-only feature and log-target scaling at each size. XGBoost: original depth-2 baseline. No per-size tuning or test-driven selection.',
        '', '| Model | Training rows | TPOT MAPE | TTFT MAPE | Energy MAPE |','|---|---:|---:|---:|---:|']
    for family in ['transformer','xgboost']:
        for n in sizes:
            values=[next(r for r in summary if r['model']==family and r['target']==t and r['n_train']==n and r['partition']=='test')['mape_mean'] for t in TARGETS]
            lines.append(f'| {family} | {n} | '+ ' | '.join(f'{v:.2f}%' for v in values)+' |')
    lines+=['','![Learning curves](learning_curve.png)','',
        '## Limits','',
        '- Previously chosen hyperparameters and an already-explored dataset are reused. These curves are exploratory and not an independent performance claim.',
        '- The five test splits overlap. Shaded split SD is descriptive, not a confidence interval or five independent prospective trials.',
        '- The fixed 150-row stopping set provides supervision even at the smallest training size. Training-size labels exclude these rows.',
        '- Training subsets are nested within each split/seed; changing the seed changes both the subset ordering and model initialization.',
        '- No temperature, voltage, measured performance, config ID or run order appears among inputs. No batch-2 measurements were merged.',
        '- Learning curves stop at 598 fitting rows and cannot establish behavior at several thousand rows. Energy labels remain prefill-inclusive.',
        f"- Transformer fits reaching the 500-epoch cap: {sum(r['hit_cap'] for r in training if r['model']=='transformer')}/50. Inspect training.csv for best and stopping epochs.",
        '', '## Reproduce','', '```bash','conda activate nanollmforge',
        'python -m scripts.prediction transformer-learning-curve --output scripts/prediction/outputs/transformer_learning_curve_repeat','```']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print('COMPLETE',out,flush=True)


if __name__=='__main__':main()
