"""Paired gross-versus-dynamic energy experiment on the frozen 1564-row cohort.

Architecture-only inputs, unchanged historical split/hyperparameters. Writes a
new experiment directory, never changes measurements or deployed predictors.
"""
import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import joblib
from ..models.serialization import load_bundle
import numpy as np
import sklearn
import torch
import xgboost
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

from ..config import OUTPUTS, PREDICTION_ROOT, SWEEP_CONFIGS, SWEEP_OUTPUTS

from ..config import FEATURES
from ..features.analytic import physical_features
from ..training.trees import fit_xgb
from ..inference.trees import tree_predict
from ..features.physics import HardwareProfile
from ..features.physics import FEATURES32
from ..training.neural import fit_neural
from ..inference.neural import predict_neural

SEEDS = [42, 123, 2026]
TARGETS = ['decode_tok_s', 'ttft_ms', 'gross_energy_per_token_mj']


from ..data.io import write_json
from ..data.legacy import load_cohort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, default=OUTPUTS/'batch2_progress_632')
    parser.add_argument('--output', type=Path, default=OUTPUTS/'gross_energy_comparison_1564')
    args = parser.parse_args()
    snapshot, oldmeta, rows, configs, ix, dynamic, gross, audit = load_cohort(args.previous)
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    train = np.concatenate([ix['old_train'], ix['batch2_train']])
    val = ix['fixed_validation']
    tests = {k: ix[k] for k in ['old_batch_test', 'new_batch_test', 'pooled_test']}
    stages = {'old_only': ix['old_train'], 'plus_half': np.concatenate([ix['old_train'], ix['batch2_half']]), 'plus_all': train}
    x14 = np.array([[physical_features(c)[0][f] for f in FEATURES] for c in configs], dtype=np.float32)
    profile = HardwareProfile().fit([configs[i] for i in train], dynamic[train])
    x32 = profile.transform(configs)
    y = dynamic.copy()
    y[:, 2] = gross
    source_paths = [args.previous/'input_snapshot.json', args.previous/'metadata.json',
                    args.previous/'summary.json', Path(__file__),
                    PREDICTION_ROOT/'features/physics.py',
                    PREDICTION_ROOT/'training/trees.py',
                    PREDICTION_ROOT/'features/analytic.py',
                    PREDICTION_ROOT/'models/neural.py',
                    PREDICTION_ROOT/'training/neural.py',
                    PREDICTION_ROOT/'inference/neural.py',
                    PREDICTION_ROOT/'inference/trees.py',
                    PREDICTION_ROOT/'data/legacy.py']
    source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    manifest = dict(source_hashes=source_hashes, measurement_source_hashes=snapshot['hashes'],
        splits=oldmeta['splits'], seeds=SEEDS, train_n=len(train), validation_n=len(val), test_n=len(ix['pooled_test']),
        targets=TARGETS, gross_formula='total_energy_j * 1000 / 32',
        workload='Nominal48 prompt (49 actual inferred), 32 outputs. Total energy includes prefill and device background.',
        features={'xgboost14': FEATURES, 'xgboost32_and_transformer': FEATURES32},
        hardware_profile=asdict(profile), architecture_only=True,
        protocol='Same frozen cohort/split as prior batch2 experiment. XGBoost14 models log TPOT and converts to throughput; other models predict log throughput. Same hyperparameters. Validation-only early stopping. Frozen dynamic checkpoints are independently checked against old metrics. Gross models newly trained. XGBoost14 gross learning curve also fitted at 598/849/1100 rows.',
        limitations=['Different physical target; lower percentage error is not itself lower absolute noise.',
          'Same previously inspected test set: exploratory, not a fresh prospective validation.',
          'Three seeds on one split, not three independent test sets.',
          'Keeps prior filtering including dynamic-positive selection and stall exclusion for paired comparison.',
          'Old measurement window/rounding limitations remain; removing baseline does not fix them.',
          'No stacking retraining in this focused XGBoost/Transformer comparison.'],
        versions={'torch': torch.__version__, 'numpy': np.__version__, 'sklearn': sklearn.__version__, 'xgboost': xgboost.__version__})
    write_json(out/'metadata.json', manifest)
    write_json(out/'input_snapshot.json', snapshot)
    write_json(out/'labels.json', [dict(config_id=r['config_id'], dynamic_energy_per_token_mj=float(d[2]),
                                     gross_energy_per_token_mj=float(g)) for r,d,g in zip(rows,dynamic,gross)])
    metrics, predictions, training = [], [], []
    historical = json.loads((args.previous/'metrics.json').read_text())
    check_deltas = []

    def evaluate(name, kind, seed, pred, stage='plus_all', fit_n=len(train)):
        actual = y if kind == 'gross' else dynamic
        assert pred.shape == actual.shape and np.isfinite(pred).all() and (pred > 0).all()
        for test, ids in tests.items():
            for j, metric in enumerate(['throughput', 'ttft', 'energy']):
                a, p = actual[ids,j], pred[ids,j]
                error = abs(p-a)
                item = dict(model=name, energy_target=kind, seed=seed, stage=stage, train_n=fit_n,
                            test=test, metric=metric, n=len(ids), mape=float(np.mean(error/a)*100),
                            mae=float(np.mean(error)), r2=float(r2_score(a,p)),
                            spearman=float(spearmanr(a,p).statistic),
                            within10_pct=float(np.mean(error/a <= .1)*100),
                            rms_log_error=float(np.sqrt(np.mean(np.log(p/a)**2))))
                metrics.append(item)
                if kind == 'dynamic':
                    target = ['decode_tok_s','ttft_ms','dynamic_energy_per_token_mj'][j]
                    old = next(r for r in historical if (r['model'],r['stage'],r['test'],r['target'],r['seed']) ==
                               (name,stage,test,target,seed))
                    check_deltas.append(abs(item['mape'] - old['mape']))
                    assert abs(item['mape']-old['mape']) < .0002, (item, old)
                for idx, a_i, p_i in zip(ids, a, p):
                    predictions.append(dict(model=name,energy_target=kind,seed=seed,stage=stage,test=test,
                        metric=metric,config_id=rows[idx]['config_id'],actual=float(a_i),prediction=float(p_i)))
        write_json(out/'metrics.json', metrics)
        write_json(out/'predictions.json', predictions)
        print(name, kind, stage, seed, 'pooled MAPE',
              [round(r['mape'],3) for r in metrics[-3:]], flush=True)

    print('START', 'fit',len(train),'validation',len(val),'test',len(ix['pooled_test']),flush=True)
    for seed in SEEDS:
        models = load_bundle(args.previous/f'xgboost14_seed{seed}.joblib')['models']
        evaluate('xgboost14', 'dynamic', seed, tree_predict(models, x14, True))
        stack = load_bundle(args.previous/f'stacked_ensemble_seed{seed}.joblib')
        base = stack['base']
        evaluate('xgboost32', 'dynamic', seed, tree_predict(base['xgboost'], base['profile'].transform(configs)))
        pack = load_bundle(args.previous/f'grouped_transformer_seed{seed}.joblib')
        evaluate('transformer32', 'dynamic', seed, np.exp(predict_neural(pack,pack['profile'].transform(configs))))
        for stage, ids in stages.items():
            z14 = np.log(np.column_stack([[float(r['tpot_ms']) for r in rows], y[:,1], y[:,2]]).astype(np.float32))
            models = fit_xgb(x14,z14,ids,val,seed,2)
            evaluate('xgboost14','gross',seed,tree_predict(models,x14,True),stage,len(ids))
            joblib.dump(dict(models=models,features=FEATURES,targets=TARGETS,inverse_tpot=True),out/f'xgboost14_{stage}_seed{seed}.joblib')
        models = fit_xgb(x32,np.log(y),train,val,seed,3)
        evaluate('xgboost32','gross',seed,tree_predict(models,x32))
        joblib.dump(dict(models=models,profile=profile,targets=TARGETS,inverse_tpot=False),out/f'xgboost32_seed{seed}.joblib')
        start = time.monotonic()
        print('Training grouped Transformer',seed,flush=True)
        pack = fit_neural('transformer',x32[train],np.log(y[train]),x32[val],np.log(y[val]),seed)
        pack.update(profile=profile,targets=TARGETS)
        path = out/f'grouped_transformer_seed{seed}.joblib'
        joblib.dump(pack,path)
        predicted = np.exp(predict_neural(pack,x32))
        np.testing.assert_allclose(predicted,np.exp(predict_neural(load_bundle(path),x32)),rtol=1e-6)
        evaluate('transformer32','gross',seed,predicted)
        training.append(dict(seed=seed,seconds=time.monotonic()-start,**{k:pack[k] for k in ['best_epoch','total_epochs','parameter_count']}))
        write_json(out/'training.json',training)

    summary = []
    keys = list(dict.fromkeys((r['model'],r['energy_target'],r['stage'],r['test'],r['metric']) for r in metrics))
    for name,kind,stage,test,metric in keys:
        group = [r for r in metrics if (r['model'],r['energy_target'],r['stage'],r['test'],r['metric']) == (name,kind,stage,test,metric)]
        assert len(group) == 3
        summary.append(dict(model=name,energy_target=kind,stage=stage,test=test,metric=metric,train_n=group[0]['train_n'],n=group[0]['n'],
            **{k:float(np.mean([r[k] for r in group])) for k in ['mape','mae','r2','spearman','within10_pct','rms_log_error']},
            mape_seed_sd=float(np.std([r['mape'] for r in group],ddof=1))))
    write_json(out/'summary.json',summary)
    # Paired bootstrap of architecture-level errors averaged across training seeds.
    boot = []
    for test, ids in tests.items():
        draws = np.random.default_rng(20260912).integers(0,len(ids),size=(10000,len(ids)))
        for name in ['xgboost14','xgboost32','transformer32']:
            by_kind = {}
            for kind in ['dynamic','gross']:
                lookup = {}
                for r in predictions:
                    if (r['model'],r['energy_target'],r['stage'],r['test'],r['metric']) == (name,kind,'plus_all',test,'energy'):
                        err = abs(r['prediction']-r['actual'])
                        lookup.setdefault(r['config_id'],[]).append([err/r['actual']*100,err])
                by_kind[kind] = np.array([np.mean(lookup[rows[i]['config_id']],axis=0) for i in ids])
            delta = by_kind['dynamic'] - by_kind['gross']
            for j, unit in enumerate(['mape_pp','mae_mj_per_token']):
                low,high = np.quantile(delta[draws,j].mean(1),[.025,.975])
                boot.append(dict(model=name,test=test,measure=unit,dynamic_minus_gross=float(delta[:,j].mean()),ci95=[float(low),float(high)]))
    write_json(out/'paired_bootstrap.json',boot)
    audit.update(max_frozen_baseline_mape_difference_pp=max(check_deltas),
                 frozen_metric_checks=len(check_deltas),source_hashes_unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in source_hashes.items()))
    assert audit['source_hashes_unchanged']
    write_json(out/'audit.json',audit)
    lines = ['# Gross energy retraining on 1564 architectures','',
             'Same 1100 fitting / 150 early-stopping / 314 test architectures. Three seeds; mean of per-seed metrics, not ensemble predictions.',
             'Gross energy = recorded total_energy_j × 1000 / 32. Includes prefill and device background. No runtime-state inputs.', '',
             '| Test | Model | Target | Energy MAPE | MAE (mJ/token) | R² | Spearman | Within 10% |',
             '|---|---|---|---:|---:|---:|---:|---:|']
    for test in tests:
        for name in ['xgboost14','xgboost32','transformer32']:
            for kind in ['dynamic','gross']:
                r = next(r for r in summary if (r['model'],r['energy_target'],r['stage'],r['test'],r['metric']) == (name,kind,'plus_all',test,'energy'))
                lines.append(f"| {test} | {name} | {kind} | {r['mape']:.2f}% | {r['mae']:.2f} | {r['r2']:.3f} | {r['spearman']:.3f} | {r['within10_pct']:.1f}% |")
    lines += ['', '## Gross-target model metrics (pooled test)', '',
              '| Model | Throughput MAPE | TTFT MAPE | Gross energy MAPE |','|---|---:|---:|---:|']
    for name in ['xgboost14','xgboost32','transformer32']:
        vals = [next(r['mape'] for r in summary if (r['model'],r['energy_target'],r['stage'],r['test'],r['metric']) == (name,'gross','plus_all','pooled_test',m)) for m in ['throughput','ttft','energy']]
        lines.append(f"| {name} | " + ' | '.join(f'{v:.2f}%' for v in vals) + ' |')
    lines += ['', '## XGBoost14 gross-energy learning curve', '',
              '| Fit rows | Old-test MAPE | New-test MAPE | Pooled MAPE |','|---:|---:|---:|---:|']
    for stage, ids in stages.items():
        vals = [next(r['mape'] for r in summary if (r['model'],r['energy_target'],r['stage'],r['test'],r['metric']) == ('xgboost14','gross',stage,t,'energy')) for t in tests]
        lines.append(f'| {len(ids)} | ' + ' | '.join(f'{v:.2f}%' for v in vals) + ' |')
    lines += ['', '## Interpretation and checks', ''] + ['- '+s for s in manifest['limitations']]
    lines += ['- Bootstrap compares errors on different physical energy targets. Positive delta favors gross. Resampling assumes independent architectures and does not account for temporal correlation.',
              f"- Reproduced {len(check_deltas)} frozen baseline metrics; maximum MAPE discrepancy {max(check_deltas):.8f} percentage points. Gross and dynamic formulas reconcile within CSV rounding tolerance.",
              '- Models are evaluation checkpoints, not automatically deployed. Original data, runq_llmforge.c, and measurement scripts are unchanged.', '',
              '## Reproduce', '', '```bash',
              'python -m llmforge.hw.device.prediction evaluate-gross-energy --output runs/device/gross_energy_comparison_1564_rerun', '```']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print('COMPLETE',out,flush=True)


if __name__ == '__main__':
    main()
