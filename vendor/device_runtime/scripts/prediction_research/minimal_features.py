"""Offline minimal-input XGBoost ablation on a committed active-learning dataset."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import r2_score

from scripts.prediction.active_learning.engine import verify_workspace
from scripts.prediction.features.physics import FEATURES32, HardwareProfile
from scripts.prediction.inference.predictor import predict_bundle
from scripts.prediction.inference.trees import tree_predict
from scripts.prediction.models.serialization import load_bundle
from scripts.prediction.training.trees import fit_xgb


VARIANTS = {
    'params_only': ['total_params'],
    'params_kv': ['total_params', 'kv_bytes_per_token'],
    'params_group': ['total_params', 'q8_group_size'],
    'params_kv_group': ['total_params', 'kv_bytes_per_token', 'q8_group_size'],
    'physics32': FEATURES32,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve(strict=True)
    _, state, dataset = verify_workspace(workspace)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    rows = dataset.observations
    configs = [r['architecture'] for r in rows]
    tr = np.array([i for i, r in enumerate(rows) if r['split']=='train'])
    va = np.array([i for i, r in enumerate(rows) if r['split']=='validation'])
    y = np.array([[float(r['metrics'][t]) for t in dataset.targets] for r in rows], dtype=np.float32)
    # Match production fit_dataset target precision and training-only profile fit.
    yfit = np.array([[float(r['metrics'][t]) for t in dataset.targets] for r in rows], dtype=np.float64)
    profile = HardwareProfile().fit([configs[i] for i in tr], yfit[tr])
    full = profile.transform(configs)
    seeds = [42, 123, 2026]
    dataset.save(out/'dataset_snapshot.json')
    manifest = dict(workspace=str(workspace), completed_round=state['completed_rounds'],
                    dataset_sha256=dataset.fingerprint, source_model=state['latest_model'],
                    source_model_sha256=hashlib.sha256((workspace/state['latest_model']).read_bytes()).hexdigest(),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    variants=VARIANTS, seeds=seeds, targets=dataset.targets,
                    train_ids=[rows[i]['measurement_id'] for i in tr],
                    validation_ids=[rows[i]['measurement_id'] for i in va],
                    counts={k: sum(r['split']==k for r in rows) for k in ['train','validation','test']},
                    kv_definition='FP32 KV bytes per context token across all layers; fixed workload. No separate allocator-capacity or traffic feature.',
                    protocol=dataset.protocol,
                    limitations=['Same fixed depth=3/hyperparameters; no per-variant tuning.',
                                 'Validation is also used for early stopping; exploratory, not unbiased test performance.',
                                 'Test rows retained in snapshot for provenance but not predicted or scored.',
                                 'Latest AL training distribution differs from the original historical cohort.',
                                 'Diagnostic checkpoints require feature slicing; not production predict_bundle artifacts.',
                                 'No hardware access, production checkpoint replacement or AL state changes.'])
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    predictions = {}
    metrics = []
    for name, names in VARIANTS.items():
        indices = [FEATURES32.index(f) for f in names]
        x = full[:, indices]
        runs = []
        for seed in seeds:
            models = fit_xgb(x, np.log(yfit), tr, va, seed, 3)
            p = tree_predict(models, x[va])
            path = out/f'{name}_seed{seed}.joblib'
            joblib.dump(dict(estimator=models, profile=profile, feature_indices=indices,
                             features=names, targets=dataset.targets, seed=seed,
                             dataset_sha256=dataset.fingerprint, diagnostic_only=True), path)
            restored = joblib.load(path)
            np.testing.assert_array_equal(tree_predict(restored['estimator'], x[va]), p)
            if name=='physics32' and seed==42:
                baseline = predict_bundle(load_bundle(workspace/state['latest_model']), [configs[i] for i in va])
                np.testing.assert_allclose(p, baseline, rtol=1e-6, atol=1e-6)
            assert p.shape==y[va].shape and np.isfinite(p).all() and (p>0).all()
            runs.append(p)
            print(name, seed, 'validation MAPE:', np.mean(abs(p-yfit[va])/yfit[va], axis=0)*100, flush=True)
        predictions[name] = np.stack(runs)
        for j, target in enumerate(dataset.targets):
            a = yfit[va, j]; p = predictions[name][:, :, j]
            errors = np.mean(abs(p-a)/a, axis=1)*100
            metrics.append(dict(model=name, target=target, mape=float(errors.mean()),
                                seed_sd=float(errors.std(ddof=1)), mae=float(np.mean(abs(p-a))),
                                r2=float(np.mean([r2_score(a, q) for q in p]))))
    np.savez(out/'validation_predictions.npz', actual=yfit[va],
             measurement_ids=np.array(manifest['validation_ids']), **predictions)
    (out/'summary.json').write_text(json.dumps(dict(metrics=metrics), indent=2)+'\n')
    lines = ['# Minimal features: active-learning checkpoint cohort', '',
             f'Completed round {state["completed_rounds"]}; {len(tr)} train / {len(va)} fixed validation. '
             'Three seeds; XGBoost depth 3 and production log-target recipe. No test evaluation.', '',
             '| Features | Throughput MAPE | TTFT MAPE | Dynamic energy MAPE |',
             '|---|---:|---:|---:|']
    for name in VARIANTS:
        r = [next(m for m in metrics if m['model']==name and m['target']==t) for t in dataset.targets]
        lines.append('| '+name+' | '+' | '.join(f'{m["mape"]:.2f} ± {m["seed_sd"]:.2f}%' for m in r)+' |')
    lines += ['', '± denotes training-seed SD, not uncertainty over splits. Validation also determines early stopping.', '',
              'KV size is bytes per context token across all layers; scaling by the fixed context length adds no independent information. '
              'Group size is the inferred effective quantization group for the homogeneous kernel. '
              'No device-state or measured-latency inputs. No additional cleaning.', '',
              'All checkpoints were reloaded and prediction equivalence verified. Full-feature seed 42 reproduces the committed AL checkpoint. '
              'Checkpoints are diagnostic: transform configs with stored profile, slice feature_indices, then call tree_predict; do not use production predict_bundle.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines), flush=True)


if __name__=='__main__':
    main()
