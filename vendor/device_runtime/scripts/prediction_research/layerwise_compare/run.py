"""Freeze the current data and train matched tree/Transformer predictors."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time
import joblib
import numpy as np
import sklearn
import torch
import xgboost
from scripts.prediction_research.layerwise_refit.data import snapshot, TARGETS
from scripts.prediction_research.layerwise_refit.features import matrix
from .training import fit_tree, fit_transformer, predict
from .report import evaluate, render


def write(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--snapshot',type=Path,help='Reuse an existing frozen dataset instead of live registry rows')
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    root=Path(__file__).resolve().parents[3]
    base=root/'scripts/sweep/outputs'
    folders=[base/'watch5_layerwise_500_v1',base/'watch5_layerwise_500_variable_kv_v2']
    folders += [base/f'watch5_layerwise_1000_variable_kv_v3/part{i}' for i in (1,2)]
    data=json.loads(args.snapshot.read_text()) if args.snapshot else snapshot(folders)
    rows=data['rows']
    # Verify the labels stored in SQLite against their raw result artifacts.
    for r in rows:
        raw=json.loads(Path(r['artifact_path']).read_text())
        for t in TARGETS:
            if raw[t]!=r['metrics'][t]: raise ValueError('Artifact/database label mismatch')
    x,names=matrix([r['architecture'] for r in rows])
    y=np.array([[r['metrics'][t] if r['metrics'][t] is not None else np.nan for t in TARGETS] for r in rows])
    y[~np.isfinite(y) | (y<=0)]=np.nan
    split=np.array([r['split'] for r in rows])
    tr,va,te=[split==s for s in ('train','validation','test')]
    ids=np.array([r['candidate_id'] for r in rows])
    groups=np.array([r['permutation_group'] for r in rows])
    for a,b in ((tr,va),(tr,te),(va,te)):
        assert not set(groups[a]) & set(groups[b])
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    write(out/'dataset_snapshot.json',data)
    sha=hashlib.sha256((out/'dataset_snapshot.json').read_bytes()).hexdigest()
    seeds=[42,123,2026]
    configs={'XGBoost':dict(depth=3),'Transformer-1x32':dict(width=32,layers=1),
             'Transformer-2x64':dict(width=64,layers=2)}
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),dataset_sha256=sha,
        counts=dict(Counter(split)),features=names,targets=TARGETS,seeds=seeds,configs=configs,
        versions=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,
                      sklearn=sklearn.__version__,xgboost=xgboost.__version__),
        target_counts={s:{t:int(np.isfinite(y[split==s,j]).sum()) for j,t in enumerate(TARGETS)}
                       for s in ('train','validation','test')},
        energy_warnings=dict(Counter(str(r['energy_warning']) for r in rows)),
        policy='Identical 110 architecture features, same grouped registry splits, three seeds; all current validation rows used for early stopping; test never passed to training.',
        limitations=['Partial sequential sample; previously reported test rows reused, so exploratory rather than a new confirmatory test.',
                     'XGBoost uses independent log-MSE regressors/validation log-RMSE stopping; Transformer uses joint standardized-log Smooth L1/mean validation MAPE stopping.',
                     'Transformer configurations are fixed before scoring; no exhaustive hyperparameter search.',
                     'Aggregate and quarter features do not uniquely encode every layer sequence.',
                     'Old homogeneous AL measurements are excluded because their measurement protocol differs.',
                     'All baseline-drift warnings retained; only invalid/missing target labels masked.',
                     'Dynamic energy includes prefill and decode divided by 32, not decode-only energy.',
                     'No model deployment or collection-state mutation.'])
    sources=list(Path(__file__).parent.glob('*.py'))
    sources += [root/p for p in ['scripts/prediction_research/layerwise_refit/data.py',
        'scripts/prediction_research/layerwise_refit/features.py','scripts/prediction_research/ranking_metrics.py',
        'scripts/prediction/models/trees.py','scripts/sweep/layerwise/candidates.py']]
    manifest['source_hashes']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    for source in sources:
        destination=out/'source_snapshot'/source.relative_to(root)
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,destination)
    write(out/'manifest.json',manifest)
    print('Frozen snapshot:',len(rows),manifest['counts'],manifest['target_counts'],flush=True)
    predictions={};details={};validation={}
    started=time.monotonic()
    for name,config in configs.items():
        packs=[];details[name]=[];validation[name]=[]
        for seed in seeds:
            if name=='XGBoost': pack=fit_tree(x[tr],y[tr],x[va],y[va],seed)
            else: pack=fit_transformer(x[tr],y[tr],x[va],y[va],names,seed,**config)
            pack.update(features=names,targets=TARGETS,dataset_sha256=sha,protocol=data['protocol'])
            path=out/f'{name}_seed{seed}.joblib';joblib.dump(pack,path)
            pv=predict(pack,x[va])
            np.testing.assert_array_equal(pv,predict(joblib.load(path),x[va]))
            val=np.nanmean(abs(pv-y[va])/y[va],axis=0)*100
            validation[name].append(val.tolist())
            details[name].append({k:pack[k] for k in ('seed','train_seconds','best_epoch','epochs','parameter_count','counts') if k in pack})
            if 'history' in pack: write(out/f'{name}_seed{seed}_history.json',pack['history'])
            packs.append(pack)
            print(name,seed,'validation MAPE',val,'seconds',pack['train_seconds'],flush=True)
            write(out/'training_progress.json',dict(details=details,validation=validation,elapsed_seconds=time.monotonic()-started))
        # Save validation predictions; defer ALL test evaluation until all models are fitted.
    selected_transformer=min((n for n in configs if n!='XGBoost'),key=lambda n:np.mean(validation[n]))
    for name in configs:
        predictions[name]=np.stack([predict(joblib.load(out/f'{name}_seed{s}.joblib'),x[te]) for s in seeds])
    cohort=np.array([r['cohort'] for r in rows])[te]
    family=np.array([r['architecture']['family'] for r in rows])[te]
    cohorts={'all':np.ones(int(te.sum()),dtype=bool)}
    cohorts.update({c:cohort==c for c in np.unique(cohort)})
    cohorts.update({f:family==f for f in np.unique(family)})
    result=evaluate(y[te],predictions,ids[te],groups[te],TARGETS,cohorts)
    result.update(validation=validation,selected_transformer_by_validation=selected_transformer,
                  training_details=details,elapsed_seconds=time.monotonic()-started)
    np.savez(out/'test_predictions.npz',actual=y[te],ids=ids[te],groups=groups[te],**predictions)
    write(out/'metrics.json',result)
    checkpoints={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.joblib')}
    write(out/'checkpoint_hashes.json',checkpoints)
    (out/'README.md').write_text(render(result))
    print(render(result),flush=True)
    print('Validation-selected Transformer:',selected_transformer,flush=True)


if __name__=='__main__': main()
