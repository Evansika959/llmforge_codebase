"""Reproducible architecture-only surrogate comparison; never tune on test data."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr
import torch
from torch import nn
import xgboost as xgb

from ..config import ROOT, FEATURES, TARGETS
from ..models.neural import TinyTransformer
from ..data.io import write_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='scripts/prediction/outputs/surrogate_comparison_936')
    args = parser.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    src = ROOT/'scripts/sweep/outputs/watch5_random_50M_150M_40C_decode32_results_clean.csv'
    with src.open() as f:
        rows = list(csv.DictReader(f))
    with (ROOT/'scripts/sweep/configs/watch5_random_1000_50M_150M_sweep.csv').open() as f:
        configs = {r['config_id']: r for r in csv.DictReader(f)}
    assert len(rows) == 936 and len({r['config_id'] for r in rows}) == 936
    features = []
    for r in rows:
        c = {k: float(v) for k, v in configs[r['config_id']].items() if k in FEATURES}
        assert all(float(r[k]) == c[k] for k in FEATURES[:7])
        c['kv_bytes_per_token'] = 4*c['n_layer']*c['n_kv']*(c['d_qk']+c['d_v'])
        c['attention_width'] = c['n_h']*c['d_qk']
        features.append([c[k] for k in FEATURES])
    X = np.asarray(features, dtype=np.float32)
    Y = np.asarray([[float(r[k]) for k in TARGETS] for r in rows], dtype=np.float32)
    assert np.isfinite(X).all() and np.isfinite(Y).all() and (Y > 0).all()
    assert len(np.unique(X[:, :7], axis=0)) == len(X), 'Duplicate architectures require group splitting'
    strata = X[:, FEATURES.index('q8_group_size')].astype(int)
    trainval, test = train_test_split(np.arange(len(X)), test_size=.2, random_state=20260909, stratify=strata)
    train, val = train_test_split(trainval, test_size=.2, random_state=20260909, stratify=strata[trainval])
    xm, xs = X[train].mean(0), X[train].std(0).clip(1e-6)
    logy = np.log(Y)
    ym, ys = logy[train].mean(0), logy[train].std(0).clip(1e-6)
    xn = torch.tensor((X-xm)/xs)
    yn = torch.tensor((logy-ym)/ys)
    meta = dict(source=str(src), source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
                features=FEATURES, targets=TARGETS, split_seed=20260909,
                splits={k:[rows[i]['config_id'] for i in ids] for k, ids in [('train',train),('validation',val),('test',test)]},
                x_mean=xm.tolist(), x_std=xs.tolist(), log_y_mean=ym.tolist(), log_y_std=ys.tolist(),
                versions=dict(torch=torch.__version__, sklearn=sklearn.__version__, xgboost=xgb.__version__),
                notes='Architecture-only features; shared fixed test; log targets; validation-selected tuning. Transformer predicts three targets jointly; XGBoost uses one regressor per target.')
    (out/'metadata.json').write_text(json.dumps(meta, indent=2))
    print('Split sizes:', len(train), len(val), len(test), flush=True)
    tuning = []

    def fit_xgb(hp, seed):
        models = []
        for j in range(len(TARGETS)):
            model = xgb.XGBRegressor(n_estimators=1500, learning_rate=.03, tree_method='hist',
                n_jobs=4, objective='reg:squarederror', early_stopping_rounds=50,
                subsample=.85, colsample_bytree=.9, random_state=seed, **hp)
            model.fit(X[train], logy[train,j], eval_set=[(X[val],logy[val,j])], verbose=False)
            models.append(model)
        pred = np.column_stack([m.predict(X[val]) for m in models])
        score = float(np.mean(((pred-logy[val])/ys)**2))
        return models, score

    def fit_tf(hp, seed):
        torch.manual_seed(seed)
        model = TinyTransformer(len(FEATURES), hp['width'], hp['dropout'])
        opt = torch.optim.AdamW(model.parameters(), lr=hp['lr'], weight_decay=.01)
        best, state, best_epoch = float('inf'), None, 0
        for epoch in range(1, 251):
            model.train()
            order = torch.tensor(train)[torch.randperm(len(train))]
            for batch in order.split(128):
                opt.zero_grad()
                loss = ((model(xn[batch])-yn[batch])**2).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                opt.step()
            model.eval()
            with torch.no_grad():
                score = float(((model(xn[val])-yn[val])**2).mean())
            if score < best - 1e-5:
                best, best_epoch = score, epoch
                state = {k:v.detach().clone() for k,v in model.state_dict().items()}
            if epoch-best_epoch >= 30:
                break
        model.load_state_dict(state)
        return model, best, best_epoch

    grids = {
        'xgboost': [dict(max_depth=d, min_child_weight=w, reg_lambda=l) for d,w,l in [(2,3,5),(3,3,5),(4,5,10),(6,5,10)]],
        'transformer': [dict(width=w, dropout=d, lr=l) for w,d,l in [(32,.1,.001),(32,.2,.0003),(64,.1,.001),(64,.2,.0003)]]}
    chosen = {}
    for family, grid in grids.items():
        for hp in grid:
            start = time.monotonic()
            result = fit_xgb(hp, 42) if family == 'xgboost' else fit_tf(hp,42)
            score = result[1]
            tuning.append(dict(model=family, params=json.dumps(hp), validation_standardized_log_mse=score,
                               seconds=time.monotonic()-start))
            print('TUNE', family, hp, round(score,5), 'seconds', round(time.monotonic()-start), flush=True)
            if family not in chosen or score < chosen[family][0]:
                chosen[family] = (score,hp)
        write_csv(out/'tuning.csv', tuning)
    metrics, predictions = [], []
    for family in grids:
        hp = chosen[family][1]
        for seed in [42,123,2026]:
            if family == 'xgboost':
                models, _ = fit_xgb(hp,seed)
                pred = np.exp(np.column_stack([m.predict(X[test]) for m in models]))
                for target, model in zip(TARGETS,models):
                    model.save_model(out/f'xgboost_seed{seed}_{target}.json')
            else:
                model, _, epoch = fit_tf(hp,seed)
                model.eval()
                with torch.no_grad():
                    pred = np.exp(model(xn[test]).numpy()*ys+ym)
                torch.save(dict(state_dict=model.state_dict(), hyperparameters=hp, best_epoch=epoch,
                                metadata=meta), out/f'transformer_seed{seed}.pt')
            assert np.isfinite(pred).all()
            for j,target in enumerate(TARGETS):
                actual, estimated = Y[test,j], pred[:,j]
                metrics.append(dict(model=family,seed=seed,target=target,
                    mae=float(mean_absolute_error(actual,estimated)),
                    mape_percent=float(np.mean(np.abs(estimated-actual)/actual)*100),
                    r2=float(r2_score(actual,estimated)),spearman=float(spearmanr(actual,estimated).statistic)))
                for i,a,p in zip(test,actual,estimated):
                    predictions.append(dict(model=family,seed=seed,config_id=rows[i]['config_id'],target=target,actual=float(a),prediction=float(p)))
            print('EVALUATED',family,seed,flush=True)
            write_csv(out/'metrics.csv',metrics)
            write_csv(out/'test_predictions.csv', predictions)
    summary = []
    for family in grids:
        for target in TARGETS:
            ms=[m for m in metrics if m['model']==family and m['target']==target]
            entry=dict(model=family,target=target)
            for k in ['mae','mape_percent','r2','spearman']:
                entry[k+'_mean']=float(np.mean([m[k] for m in ms]))
                entry[k+'_std']=float(np.std([m[k] for m in ms],ddof=1))
            summary.append(entry)
    write_csv(out/'summary.csv',summary)
    (out/'selected_hyperparameters.json').write_text(json.dumps({k:v[1] for k,v in chosen.items()},indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()
