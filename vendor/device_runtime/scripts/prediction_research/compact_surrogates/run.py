"""Validation-selected small-network comparison; no hardware or AL modifications."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import joblib
import numpy as np
import torch

from scripts.prediction.config import ROOT, PHYSICS_TARGETS, FEATURES
from scripts.prediction.data.legacy import load_cohort
from scripts.prediction.features.physics import HardwareProfile
from scripts.prediction.features.analytic import physical_features
from scripts.prediction.models.serialization import load_bundle
from scripts.prediction.inference.neural import predict_neural
from scripts.prediction.inference.trees import tree_predict
from .training import fit, predict
from .report import report

SEEDS = [42, 123, 2026]
CONFIGS = ([dict(name=f'mlp_w{w}', family='mlp', width=w, dropout=.1, lr=1e-3)
            for w in [32, 64, 128, 256]] +
           [dict(name=f'transformer_l{l}_w{w}', family='transformer', width=w,
                 layers=l, dropout=.1, lr=3e-4) for w in [32, 64] for l in [1, 2]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-epochs', type=int, default=350)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    source = ROOT/'scripts/prediction/outputs/batch2_progress_632'
    snapshot, meta, rows, configs, ix, y, gross, audit = load_cohort(source)
    tr = np.concatenate([ix['old_train'], ix['batch2_train']])
    va, te = ix['fixed_validation'], ix['pooled_test']
    assert [len(tr), len(va), len(te)] == [1100, 150, 314]
    assert not set(tr)&set(va) and not set(tr)&set(te) and not set(va)&set(te)
    profile = HardwareProfile().fit([configs[i] for i in tr], y[tr])
    x = profile.transform(configs)
    manifest = dict(seeds=SEEDS, configs=CONFIGS, targets=PHYSICS_TARGETS,
        selection='Lowest three-seed mean validation MAPE across three targets; no test-based tuning.',
        stopping='Validation mean physical-unit MAPE; patience 45, max_epochs '+str(args.max_epochs),
        loss='Smooth L1 on standardized natural-log targets; training-only log1p feature scaler.',
        source=str(source), source_hashes={n:hashlib.sha256((source/n).read_bytes()).hexdigest()
            for n in ['input_snapshot.json','metadata.json']},
        splits={k:[rows[i]['config_id'] for i in ids] for k,ids in [('train',tr),('validation',va),('test',te)]},
        audit=audit, features=32, architecture_only=True,
        versions=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__),
        experiment_sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
        limitations=['Previously inspected test set: exploratory comparison, not fresh confirmation.',
          'Same split, three training seeds; not three independent dataset splits.',
          'Historical models retain original tuning/stopping; capacity is not the only difference.',
          'No new layerwise or active-learning rows; no automatic model deployment.'])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    tuning = []
    for config in CONFIGS:
        for seed in SEEDS:
            pack = fit(config, x[tr], y[tr], x[va], y[va], seed, args.max_epochs)
            pack.update(profile=profile, targets=PHYSICS_TARGETS, feature_version='physics32',
                        training_config_ids=manifest['splits']['train'])
            joblib.dump(pack, out/f'{config["name"]}_seed{seed}.joblib')
            record={k:pack[k] for k in ['seed','parameter_count','best_epoch','epochs','validation_mape','train_seconds']}
            record.update(name=config['name'], family=config['family'])
            tuning.append(record)
            (out/'tuning.json').write_text(json.dumps(tuning,indent=2)+'\n')
            print(json.dumps(record),flush=True)
    selected = {}
    for family in ['mlp','transformer']:
        options = [c['name'] for c in CONFIGS if c['family']==family]
        selected[family] = min(options,key=lambda name:np.mean([r['validation_mape'] for r in tuning if r['name']==name]))
    # Persist choices before computing any test predictions.
    (out/'selection.json').write_text(json.dumps(selected,indent=2)+'\n')
    predictions = {}
    for family,name in selected.items():
        predictions[family] = np.stack([predict(joblib.load(out/f'{name}_seed{s}.joblib'),x[te]) for s in SEEDS])
    x14=np.array([[physical_features(c)[0][f] for f in FEATURES] for c in configs],dtype=np.float32)
    predictions['xgboost14'] = np.stack([tree_predict(load_bundle(source/f'xgboost14_seed{s}.joblib')['models'],x14[te],True) for s in SEEDS])
    predictions['transformer128x4'] = np.stack([np.exp(predict_neural(
        load_bundle(source/f'grouped_transformer_seed{s}.joblib'),x[te])) for s in SEEDS])
    # Verify historical predictions reproduce the saved same-cohort baselines.
    historical=json.loads((source/'metrics.json').read_text())
    for label,oldname in [('xgboost14','xgboost14'),('transformer128x4','transformer32')]:
        for k,seed in enumerate(SEEDS):
            for j,target in enumerate(PHYSICS_TARGETS):
                expected=next(r['mape'] for r in historical if r['model']==oldname and r['stage']=='plus_all' and r['test']=='pooled_test' and r['seed']==seed and r['target']==target)
                actual=float(np.mean(abs(predictions[label][k,:,j]-y[te,j])/y[te,j])*100)
                assert abs(actual-expected)<.005,(label,seed,target,actual,expected)
    np.savez(out/'test_predictions.npz',actual=y[te],config_ids=np.array(manifest['splits']['test']),
             old_batch=np.isin(te,ix['old_batch_test']),**predictions)
    report(out)
    print('COMPLETE '+str(out),flush=True)


if __name__ == '__main__':
    main()
