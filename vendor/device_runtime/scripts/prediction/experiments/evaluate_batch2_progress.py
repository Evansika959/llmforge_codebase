"""Fixed-test, architecture-only data-growth experiment using a frozen batch-2 snapshot.

Never edits measurements or deploys a model. The 20 s stall cutoff is an explicit
sensitivity convention, not proof of a hardware failure. Full raw snapshot is kept.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from datetime import datetime, timezone

import joblib
from ..models.serialization import load_bundle
import numpy as np
import torch
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import Ridge

from ..config import ROOT

from ..config import FEATURES
from ..features.analytic import physical_features
from ..features.physics import HardwareProfile
from ..training.neural import fit_neural
from ..inference.neural import predict_neural
from ..config import PHYSICS_TARGETS as TARGETS
from ..training.ensemble import BASE_NAMES
from ..training.ensemble import fit_bases
from ..inference.ensemble import predict_bases

SEEDS = [42, 123, 2026]


def read_snapshot(path):
    raw = path.read_bytes()
    return list(csv.DictReader(io.StringIO(raw.decode()))), hashlib.sha256(raw).hexdigest()


from ..training.trees import fit_xgb
from ..inference.trees import tree_predict
from ..inference.ensemble import stack_predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'scripts/prediction/outputs/batch2_progress_632')
    parser.add_argument('--snapshot', type=Path, help='Reuse input_snapshot.json from a previous run.')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    olddir = ROOT / 'scripts/prediction/outputs/surrogate_comparison_936'
    proposed = ROOT / 'scripts/prediction/outputs/proposed_physics_surrogate_936'
    oldmeta = json.loads((olddir / 'metadata.json').read_text())
    if args.snapshot:
        snapshot = json.loads(args.snapshot.read_text())
    else:
        paths = dict(batch1=Path(oldmeta['source']),
            batch2=ROOT / 'scripts/sweep/outputs/watch5_random_50M_150M_40C_decode32_batch2_results.csv',
            config1=ROOT / 'scripts/sweep/configs/watch5_random_1000_50M_150M_sweep.csv',
            config2=ROOT / 'scripts/sweep/configs/watch5_random_1000_50M_150M_sweep_batch2.csv')
        snapshot = dict(captured_utc=datetime.now(timezone.utc).isoformat(), paths={}, hashes={}, records={})
        for key, path in paths.items():
            snapshot['records'][key], snapshot['hashes'][key] = read_snapshot(path)
            snapshot['paths'][key] = str(path)
    (out / 'input_snapshot.json').write_text(json.dumps(snapshot, indent=2))
    assert snapshot['hashes']['batch1'] == oldmeta['source_sha256']
    oldrows = snapshot['records']['batch1']
    raw = snapshot['records']['batch2']
    cfg = {c['config_id']: c for k in ['config1', 'config2'] for c in snapshot['records'][k]}
    accepted, invalid, stalls = [], [], []
    for r in raw:
        try:
            assert None not in r
            assert all(np.isfinite(float(r[t])) and float(r[t]) > 0 for t in TARGETS + ['tpot_ms', 'duration_s'])
            assert all(int(r[k]) == int(cfg[r['config_id']][k]) for k in FEATURES[:7])
            assert not r['notes']
        except (AssertionError, KeyError, TypeError, ValueError):
            invalid.append(r)
            continue
        if float(r['duration_s']) > 20:
            stalls.append(r)
        else:
            accepted.append(r)
    rows = oldrows + accepted + stalls
    configs = [cfg[r['config_id']] for r in rows]
    index = {r['config_id']: i for i, r in enumerate(rows)}
    assert len(index) == len(rows)
    arch_keys = [tuple(int(c[k]) for k in FEATURES[:7]) for c in configs]
    assert len(set(arch_keys)) == len(rows), 'Repeated architectures require grouped partitioning.'
    tr, va, oldtest = (np.array([index[k] for k in oldmeta['splits'][s]]) for s in ['train', 'validation', 'test'])
    b2 = np.arange(len(oldrows), len(oldrows) + len(accepted))
    anomaly = np.arange(len(oldrows) + len(accepted), len(rows))
    gs = np.array([int(c['q8_group_size']) for c in configs])
    # These early batch-2 rows were inspected in a previous experiment. Do not
    # call them fresh test data. They remain eligible for training only.
    seen = set(json.loads((ROOT / 'scripts/prediction/outputs/state_signal_investigation_936/metadata.json').read_text())['batch2_ids'])
    eligible = np.array([i for i in b2 if rows[i]['config_id'] not in seen])
    _, newtest = train_test_split(eligible, test_size=math.ceil(.2 * len(b2)), random_state=20260912, stratify=gs[eligible])
    newtrain = np.array([i for i in b2 if i not in set(newtest)])
    half, _ = train_test_split(newtrain, train_size=len(newtrain)//2, random_state=20260912, stratify=gs[newtrain])
    fulltrain = np.concatenate([tr, newtrain])
    tests = dict(old_batch_test=oldtest, new_batch_test=newtest, pooled_test=np.concatenate([oldtest, newtest]))
    if len(anomaly):
        tests['new_test_plus_flagged_stalls'] = np.concatenate([newtest, anomaly])
    alltest = set(oldtest) | set(newtest) | set(anomaly)
    assert not set(fulltrain) & alltest and not set(va) & alltest and not set(fulltrain) & set(va)
    x14 = np.array([[physical_features(c)[0][f] for f in FEATURES] for c in configs], dtype=np.float32)
    y = np.array([[float(r[t]) for t in TARGETS] for r in rows], dtype=np.float32)
    z = np.log(y)
    legacy_z = np.log(np.array([[float(r[t]) for t in ['tpot_ms', 'ttft_ms', 'dynamic_energy_per_token_mj']] for r in rows], dtype=np.float32))
    states = {}
    for name, rs in [('batch1', oldrows), ('batch2', accepted)]:
        states[name] = {k:dict(zip(['min','p05','median','p95','max'], np.quantile([float(r[k]) for r in rs], [0,.05,.5,.95,1]).tolist()))
            for k in ['temp_cpu_start_c','temp_cpu_end_c','voltage_v','decode_tok_s','ttft_ms','dynamic_energy_per_token_mj']}
    manifest = dict(snapshot_hashes=snapshot['hashes'], snapshot_utc=snapshot['captured_utc'], seeds=SEEDS,
        batch2_raw_count=len(raw), batch2_accepted_count=len(accepted), invalid_ids=[r['config_id'] for r in invalid],
        flagged_stall_ids=[r['config_id'] for r in stalls], filtering='Positive finite targets, empty notes, matched architecture; duration >20 s excluded from main comparison and included in separate test sensitivity. No residual-based filtering.',
        splits={k:[rows[i]['config_id'] for i in ids] for k, ids in dict(old_train=tr, fixed_validation=va, batch2_train=newtrain, batch2_half=half, **tests).items()},
        state_quantiles=states, architecture_only=True, stop_set='Same original 150 rows at every data size; no new test labels used for fitting, stopping, calibration or tuning.',
        limitations='Exploratory old test reuse; single split and three training seeds; batch2 differs in device state; 20s filter is a sensitivity convention. Nominal48 prompt implies49 actual; energy includes prefill.')
    (out / 'metadata.json').write_text(json.dumps(manifest, indent=2))
    metrics, pred_records, training, fold_records, weights = [], [], [], [], []

    def evaluate(model, stage, seed, train_n, pred):
        assert pred.shape == y.shape and np.isfinite(pred).all() and (pred > 0).all()
        for testname, ids in tests.items():
            for j, target in enumerate(TARGETS):
                a, p = y[ids, j], pred[ids, j]
                metrics.append(dict(model=model, stage=stage, seed=seed, train_n=train_n, test=testname, target=target, n=len(ids),
                    mape=float(np.mean(abs(p-a)/a)*100), mae=float(np.mean(abs(p-a))), r2=float(r2_score(a,p)), spearman=float(spearmanr(a,p).statistic)))
                for i, actual, prediction in zip(ids, a, p):
                    pred_records.append(dict(model=model, stage=stage, seed=seed, test=testname, target=target,
                        config_id=rows[i]['config_id'], actual=float(actual), prediction=float(prediction)))
        (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))

    print('SNAPSHOT', len(raw), 'accepted', len(accepted), 'train', len(tr), 'to', len(fulltrain), 'new test', len(newtest), flush=True)
    for seed in SEEDS:
        # Frozen old models ensure the 0-new-rows point reproduces prior results.
        oldmodels = []
        for target in ['tpot_ms', 'ttft_ms', 'dynamic_energy_per_token_mj']:
            m = xgb.XGBRegressor(); m.load_model(olddir / f'xgboost_seed{seed}_{target}.json'); oldmodels.append(m)
        evaluate('xgboost14', 'old_only', seed, len(tr), tree_predict(oldmodels, x14, True))
        pack = load_bundle(proposed / f'grouped_transformer_seed{seed}.joblib')
        evaluate('transformer32', 'old_only', seed, len(tr), np.exp(predict_neural(pack, pack['profile'].transform(configs))))
        stack = load_bundle(proposed / f'stacked_ensemble_seed{seed}.joblib')
        evaluate('xgboost32', 'old_only', seed, len(tr), tree_predict(stack['base']['xgboost'], stack['base']['profile'].transform(configs)))
        evaluate('stack', 'old_only', seed, len(tr), stack_predict(stack, configs))
        for stage, ids in [('plus_half', np.concatenate([tr, half])), ('plus_all', fulltrain)]:
            rawmodels = fit_xgb(x14, legacy_z, ids, va, seed, 2)
            evaluate('xgboost14', stage, seed, len(ids), tree_predict(rawmodels, x14, True))
            profile = HardwareProfile().fit([configs[i] for i in ids], y[ids])
            x = profile.transform(configs)
            models = fit_xgb(x, z, ids, va, seed, 3)
            evaluate('xgboost32', stage, seed, len(ids), tree_predict(models, x))
            pack = fit_neural('transformer', x[ids], z[ids], x[va], z[va], seed)
            pack['profile'] = profile
            evaluate('transformer32', stage, seed, len(ids), np.exp(predict_neural(pack, x)))
            training.append(dict(seed=seed, stage=stage, n=len(ids), best_epoch=pack['best_epoch'], total_epochs=pack['total_epochs']))
            if stage == 'plus_all':
                joblib.dump(pack, out / f'grouped_transformer_seed{seed}.joblib')
                joblib.dump(dict(models=rawmodels, features=FEATURES, target0='tpot_ms'), out / f'xgboost14_seed{seed}.joblib')
            print('FITTED', seed, stage, len(ids), 'Transformer epoch', pack['best_epoch'], flush=True)
        # Similar-sized batch2-only XGB control helps interpret state adaptation.
        models = fit_xgb(x14, legacy_z, newtrain, va, seed, 2)
        evaluate('xgboost14', 'batch2_only', seed, len(newtrain), tree_predict(models, x14, True))
        oof = np.full((len(fulltrain), 3, 4), np.nan)
        assigned = np.zeros(len(fulltrain), dtype=int)
        # Stratify on batch and GS, neither is a predictor input.
        strata = np.array([f'{int(i >= len(oldrows))}_{gs[i]}' for i in fulltrain])
        for fold, (fitloc, holdloc) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(fulltrain, strata)):
            fit, stop = train_test_split(fulltrain[fitloc], test_size=.15, random_state=seed, stratify=strata[fitloc])
            held = fulltrain[holdloc]
            assert not set(held) & (set(fit) | set(stop)) and not (set(fit) | set(stop) | set(held)) & alltest
            base = fit_bases(configs, y, fit, stop, seed, 524288.)
            oof[holdloc] = predict_bases(base, [configs[i] for i in held]); assigned[holdloc] += 1
            fold_records.append(dict(seed=seed, fold=fold, fitting=[rows[i]['config_id'] for i in fit],
                stopping=[rows[i]['config_id'] for i in stop], held_out=[rows[i]['config_id'] for i in held]))
            print('OOF', seed, fold, flush=True)
        assert np.isfinite(oof).all() and (assigned == 1).all()
        combiners = []
        for j, target in enumerate(TARGETS):
            ridge = Ridge(alpha=.1, positive=True, solver='lbfgs', tol=1e-7, max_iter=10000).fit(oof[:,j,:], z[fulltrain,j])
            assert (ridge.coef_ >= 0).all()
            combiners.append(ridge)
            weights.append(dict(seed=seed, target=target, intercept=float(ridge.intercept_), **dict(zip(BASE_NAMES, ridge.coef_.tolist()))))
        base = fit_bases(configs, y, fulltrain, va, seed, 524288.)
        pack = dict(base=base, combiners=combiners, targets=TARGETS)
        joblib.dump(pack, out / f'stacked_ensemble_seed{seed}.joblib')
        predicted = stack_predict(pack, configs)
        evaluate('stack', 'plus_all', seed, len(fulltrain), predicted)
        np.testing.assert_allclose(predicted, stack_predict(load_bundle(out / f'stacked_ensemble_seed{seed}.joblib'), configs), rtol=1e-5)
        np.savez(out / f'oof_seed{seed}.npz', predictions=oof, targets=z[fulltrain], config_ids=np.array([rows[i]['config_id'] for i in fulltrain]))
        print('SEED COMPLETE', seed, flush=True)
    for name, obj in [('predictions', pred_records), ('training', training), ('oof_folds', fold_records), ('stacking_weights', weights)]:
        (out / f'{name}.json').write_text(json.dumps(obj, indent=2))
    summary = []
    keys = list(dict.fromkeys((r['model'], r['stage'], r['test'], r['target']) for r in metrics))
    for model, stage, test, target in keys:
        group = [r for r in metrics if (r['model'],r['stage'],r['test'],r['target']) == (model,stage,test,target)]
        summary.append(dict(model=model, stage=stage, test=test, target=target, train_n=group[0]['train_n'], n=group[0]['n'],
            **{k:float(np.mean([r[k] for r in group])) for k in ['mape','mae','r2','spearman']}, mape_seed_sd=float(np.std([r['mape'] for r in group], ddof=1))))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    errors = {}
    for r in pred_records:
        key = r['model'], r['stage'], r['test'], r['target'], r['config_id']
        errors.setdefault(key, []).append(abs(r['actual']-r['prediction'])/r['actual']*100)
    intervals = []
    for test, ids in tests.items():
        draws = np.random.default_rng(20260912).integers(0, len(ids), size=(10000, len(ids)))
        for model in ['xgboost14','xgboost32','transformer32','stack']:
            for target in TARGETS:
                delta = np.array([np.mean(errors[model,'old_only',test,target,rows[i]['config_id']])-np.mean(errors[model,'plus_all',test,target,rows[i]['config_id']]) for i in ids])
                low, high = np.quantile(delta[draws].mean(1), [.025,.975])
                intervals.append(dict(model=model, test=test, target=target, mape_improvement_pp=float(delta.mean()), ci95_lower=float(low), ci95_upper=float(high)))
    (out / 'paired_bootstrap.json').write_text(json.dumps(intervals, indent=2))
    lines = ['# Batch-2 data-growth evaluation', '',
        f"Snapshot: {len(raw)} raw batch-2 rows; {len(accepted)} included; {len(invalid)} invalid/nonpositive and {len(stalls)} duration >20 s rows excluded from the main comparison. Full source snapshot and exact IDs retained. Original936 preserved.", '',
        f"Training grows {len(tr)} → {len(tr)+len(half)} → {len(fulltrain)}. Fixed original150 stopping rows, original188 test rows, and {len(newtest)} new-batch test rows. Three training seeds, one split. Previously inspected early batch2 rows cannot enter the new test set. No temperature/state inputs or test-driven tuning.", '',
        '| Test | Model | Training | N fit | Throughput MAPE | TTFT MAPE | Energy MAPE |',
        '|---|---|---|---:|---:|---:|---:|']
    for test in tests:
        for model in ['xgboost14','xgboost32','transformer32','stack']:
            for stage in ['old_only','plus_half','plus_all','batch2_only']:
                group = [next((r for r in summary if (r['model'],r['stage'],r['test'],r['target']) == (model,stage,test,t)), None) for t in TARGETS]
                if any(r is None for r in group): continue
                lines.append(f"| {test} | {model} | {stage} | {group[0]['train_n']} | " + ' | '.join(f"{r['mape']:.2f}%" for r in group) + ' |')
    lines += ['', '## Interpretation limits', '',
        '- Paired bootstrap intervals in paired_bootstrap.json resample test architectures, using mean error across three seeds per architecture. Positive improvement favors added data. These are unadjusted exploratory intervals, not three independent data splits.',
        '- Main filtering removes gross long-duration records under a declared 20s rule. This extends the previous removal of a 40s stall; it is not a hardware-derived failure criterion. new_test_plus_flagged_stalls reports predictions on excluded stalls, which never enter training.',
        '- The original validation set remains fixed to isolate training-data additions. Its batch1-only distribution may limit adaptation to batch2.',
        '- A batch2-only XGBoost control uses fewer fitting rows than the original model, with the same stopping set. Differences help separate acquisition-distribution effects from a simple sample-count story, but do not establish causality.',
        '- Architecture counts, quantization and fixed workload match previous experiments. Nominal48 prompt implies49 actual tokens; dynamic energy includes prefill. No runtime C or measurement scripts were modified.',
        '- Added-data models are evaluation checkpoints, not automatically deployed search predictors.', '', '## Reproduce the exact snapshot', '', '```bash',
        f'python -m scripts.prediction evaluate-batch2-progress --snapshot {out.relative_to(ROOT)}/input_snapshot.json --output scripts/prediction/outputs/batch2_progress_repeat', '```']
    (out / 'README.md').write_text('\n'.join(lines) + '\n')
    print('COMPLETE', out, flush=True)


if __name__ == '__main__':
    main()
