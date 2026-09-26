"""Fit baseline/refit with a shared fixed stopping set; never mutate sweep state."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from scripts.prediction.models.trees import make_xgboost
from .data import snapshot, TARGETS
from .features import matrix, predict
from .report import score, markdown


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    base = root/'sweep/outputs'
    folders = [base/'watch5_layerwise_500_v1', base/'watch5_layerwise_500_variable_kv_v2']
    folders += [base/f'watch5_layerwise_1000_variable_kv_v3/part{i}' for i in (1,2)]
    data = snapshot(folders)
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    serialized = json.dumps(data, sort_keys=True, allow_nan=False)
    (out/'dataset_snapshot.json').write_text(serialized+'\n')
    sha = hashlib.sha256(serialized.encode()).hexdigest()
    rows = data['rows']
    x, names = matrix([r['architecture'] for r in rows])
    y = np.array([[r['metrics'][t] if r['metrics'][t] is not None else np.nan for t in TARGETS] for r in rows])
    split = np.array([r['split'] for r in rows])
    old = np.array([r['cohort']=='initial1000' for r in rows])
    valid = np.isfinite(y) & (y>0)
    counts = {cohort:dict(Counter(r['split'] for r in rows if cohort=='all' or r['cohort']==cohort))
              for cohort in ('initial1000','new1000','all')}
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), dataset_sha256=sha,
        counts=counts, features=names, targets=TARGETS, seeds=[42,123,2026],
        energy_warnings=dict(Counter(str(r['energy_warning']) for r in rows)),
        missing_or_nonpositive_labels={t:int((~valid[:,j]).sum()) for j,t in enumerate(TARGETS)},
        policy='Architecture-only; same initial1000 validation stopping set for baseline and refit; fixed registry test sets; no tuning on test; no imputation or outlier deletion.',
        limitations=['Seed SD does not quantify split uncertainty.',
                     'New cohort is a partial chronological snapshot, not a random complete batch.',
                     'Baseline here is a newly fitted heterogeneous model, not the historical homogeneous checkpoint.',
                     'No model deployment or active-learning state changes.',
                     'Dynamic energy includes prefill plus decode, divided by 32 output tokens.',
                     'Layer statistics/quarter features do not encode every possible layer permutation uniquely.'])
    manifest['source_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    results, predictions = [], {}
    for variant in ['initial1000','all_current']:
        train = (split=='train') & (old if variant=='initial1000' else True)
        stop = (split=='validation') & old
        for seed in manifest['seeds']:
            models, fitted = [], {}
            for j,t in enumerate(TARGETS):
                tr, va = train & valid[:,j], stop & valid[:,j]
                model = make_xgboost(seed, 3)
                model.set_params(n_jobs=2)
                model.fit(x[tr], np.log(y[tr,j]), eval_set=[(x[va],np.log(y[va,j]))], verbose=False)
                models.append(model)
                fitted[t] = dict(train=int(tr.sum()), validation=int(va.sum()), best_iteration=model.best_iteration)
            bundle = dict(models=models,features=names,targets=TARGETS,seed=seed,
                          dataset_sha256=sha,counts=fitted,feature_version='layerwise_stats_v1',
                          protocol=data['protocol'],diagnostic_only=True)
            path=out/f'{variant}_seed{seed}.joblib'
            joblib.dump(bundle,path)
            p=predict(bundle,[r['architecture'] for r in rows])
            np.testing.assert_array_equal(p,predict(joblib.load(path),[r['architecture'] for r in rows]))
            assert np.isfinite(p).all() and (p>0).all()
            predictions[f'{variant}_{seed}']=p
            for cohort,mask in [('initial1000',old),('new1000',~old),('all',np.ones(len(rows),dtype=bool))]:
                for j,t in enumerate(TARGETS):
                    ix=(split=='test') & mask & valid[:,j]
                    if ix.sum()<3: continue
                    results.append(dict(model=variant,seed=seed,cohort=cohort,target=t,**score(y[ix,j],p[ix,j])))
            print(variant,seed,fitted,flush=True)
    summary=[]
    for model,cohort,target in sorted({(r['model'],r['cohort'],r['target']) for r in results}):
        pack=[r for r in results if (r['model'],r['cohort'],r['target'])==(model,cohort,target)]
        summary.append(dict(model=model,cohort=cohort,target=target,n=pack[0]['n'],
            **{k:float(np.mean([r[k] for r in pack])) for k in ['mape','mae','spearman','kendall_tau_b']},
            mape_seed_sd=float(np.std([r['mape'] for r in pack],ddof=1))))
    np.savez(out/'predictions.npz',actual=y,ids=np.array([r['candidate_id'] for r in rows]),split=split,**predictions)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'metrics.json').write_text(json.dumps(dict(per_seed=results,summary=summary),indent=2)+'\n')
    (out/'README.md').write_text(markdown(summary))
    print(json.dumps(counts),flush=True)
    print(markdown(summary),flush=True)


if __name__=='__main__': main()
