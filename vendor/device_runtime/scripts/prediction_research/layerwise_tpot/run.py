"""Refit XGBoost to TPOT, TTFT and dynamic energy on an audited frozen snapshot."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil

import joblib
import numpy as np
import sklearn
import xgboost

from scripts.prediction_research.layerwise_refit.features import matrix
from scripts.prediction_research.layerwise_compare.training import fit_tree, predict
from .data import TARGETS, UNITS, load_labels, sha
from .report import evaluate, render


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data, y, split = load_labels(args.snapshot)
    rows = data['rows']
    x, names = matrix([r['architecture'] for r in rows])
    tr, va, te = [split == s for s in ('train', 'validation', 'test')]
    ids = np.array([r['candidate_id'] for r in rows])
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'models').mkdir()
    shutil.copyfile(args.snapshot, out / 'dataset_snapshot.json')
    write(out / 'labels.json', dict(targets=TARGETS, units=UNITS, rows=[
        dict(candidate_id=r['candidate_id'], split=r['split'], values=label.tolist())
        for r, label in zip(rows, y)]))
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    dataset_sha256=sha(args.snapshot), labels_sha256=sha(out / 'labels.json'),
                    source_dataset=str(args.snapshot.resolve()), targets=TARGETS, units=UNITS,
                    higher_is_better=[False, False, False], features=names,
                    counts=dict(Counter(split)), seeds=[42, 123, 2026],
                    energy_warnings=dict(Counter(str(r['energy_warning']) for r in rows)),
                    protocol=data['protocol'], versions=dict(python=platform.python_version(),
                    numpy=np.__version__, sklearn=sklearn.__version__, xgboost=xgboost.__version__,
                    joblib=joblib.__version__))
    root = Path(__file__).resolve().parents[3]
    sources = list(Path(__file__).parent.glob('*.py')) + [root / p for p in (
        'scripts/prediction_research/layerwise_refit/features.py',
        'scripts/prediction_research/layerwise_compare/training.py',
        'scripts/prediction_research/layerwise_compare/models.py',
        'scripts/prediction_research/ranking_metrics.py',
        'scripts/prediction/models/trees.py', 'scripts/sweep/layerwise/candidates.py')]
    manifest['source_hashes'] = {str(p.relative_to(root)): sha(p) for p in sources}
    for source in sources:
        dest = out / 'source_snapshot' / source.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    write(out / 'manifest.json', manifest)
    print('Verified labels and frozen dataset:', manifest['counts'], len(names), 'features', flush=True)
    validation, packs, details = [], [], []
    for seed in manifest['seeds']:
        pack = fit_tree(x[tr], y[tr], x[va], y[va], seed)
        pack.update(features=names, targets=TARGETS, units=UNITS,
                    higher_is_better=[False, False, False], dataset_sha256=manifest['dataset_sha256'],
                    labels_sha256=manifest['labels_sha256'], protocol=data['protocol'])
        path = out / 'models' / f'XGBoost_seed{seed}.joblib'
        joblib.dump(pack, path)
        pv = predict(pack, x[va])
        np.testing.assert_array_equal(pv, predict(joblib.load(path), x[va]))
        val = (np.mean(abs(pv-y[va])/y[va], axis=0)*100).tolist()
        validation.append(val)
        packs.append(pack)
        config = {k: str(v) if isinstance(v, float) and not np.isfinite(v) else v
                  for k, v in pack['config'].items()}
        details.append(dict(seed=seed, counts=pack['counts'], config=config,
                            train_seconds=pack['train_seconds']))
        write(out / 'progress.json', dict(validation=validation, training=details))
        print('Seed', seed, 'validation MAPE:', val, flush=True)
    selected = int(np.argmin(np.mean(validation, axis=1)))
    selection = dict(selected_seed=manifest['seeds'][selected], validation_mape=validation,
                     policy='Minimum three-target mean validation MAPE; selection before test prediction.')
    write(out / 'selection.json', selection)
    shutil.copyfile(out / 'models' / f'XGBoost_seed{selection["selected_seed"]}.joblib',
                    out / 'models' / 'predictor_final.joblib')
    preds = np.stack([predict(pack, x[te]) for pack in packs])
    np.testing.assert_array_equal(preds[selected], predict(joblib.load(out / 'models/predictor_final.joblib'), x[te]))
    np.savez(out / 'test_predictions.npz', actual=y[te], predictions=preds, ids=ids[te],
             targets=np.array(TARGETS), seeds=np.array(manifest['seeds']))
    result = dict(**selection, training=details, test=evaluate(y[te], preds, ids[te]),
                  selected_test=evaluate(y[te], preds[selected:selected+1], ids[te]))
    result['test_by_group_size'] = {}
    test_gs = np.array([r['architecture']['q8_group_size'] for r in rows])[te]
    for gs in sorted(set(test_gs)):
        take = test_gs == gs
        result['test_by_group_size'][str(gs)] = dict(count=int(take.sum()),
            metrics=evaluate(y[te][take], preds[:, take], ids[te][take]))
    write(out / 'metrics.json', result)
    (out / 'README.md').write_text(render(result, manifest))
    write(out / 'verification.json', dict(raw_artifacts_verified=len(rows),
        timing_labels_and_reciprocal_verified=len(rows), all_targets_positive_finite=True,
        grouped_splits_disjoint=True, checkpoint_reload_predictions_exact=True,
        source_dataset_unchanged=sha(args.snapshot)==manifest['dataset_sha256']))
    write(out / 'release_hashes.json', {str(p.relative_to(out)): sha(p)
          for p in sorted(out.rglob('*')) if p.is_file()})
    print(render(result, manifest), flush=True)


if __name__ == '__main__':
    main()
