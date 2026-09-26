"""Train and evaluate a versioned dataset round; never launch hardware measurements."""
import argparse
import hashlib
from importlib.metadata import version
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import torch

from ..data.dataset import MeasurementDataset
from ..data.dataset import architecture_key
from ..config import PREDICTION_ROOT
from ..data.io import write_json
from ..evaluation.metrics import metrics
from ..inference.predictor import predict_bundle
from ..reporting.run_report import write_report
from .pipeline import fit_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--family', choices=['xgboost','transformer'], default='xgboost')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-epochs', type=int, default=250)
    args = parser.parse_args()
    dataset = MeasurementDataset.load(args.dataset)
    args.output.mkdir(parents=True, exist_ok=False)
    dataset.save(args.output/'dataset.json')
    source_hashes = {str(path.relative_to(PREDICTION_ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                     for folder in ['data','features','models','training','inference','evaluation','reporting']
                     for path in sorted((PREDICTION_ROOT/folder).glob('*.py'))}
    torch.set_num_threads(4)
    pack = fit_dataset(dataset, args.family, args.seed, args.max_epochs)
    joblib.dump(pack, args.output/'model.joblib')
    heldout = [r for r in dataset.observations if r['split'] == 'test']
    pred = predict_bundle(pack, [r['architecture'] for r in heldout])
    scores = []
    for j,target in enumerate(dataset.targets):
        actual = np.array([float(r['metrics'][target]) for r in heldout])
        values = metrics(actual,pred[:,j])
        scores.append(dict(target=target, n=len(heldout), **{k:float(v) if np.isfinite(v) else None for k,v in values.items()}))
    write_json(args.output/'metrics.json', scores)
    write_json(args.output/'test_predictions.json', [dict(measurement_id=r['measurement_id'],config_id=r['config_id'],
        actual={t:float(r['metrics'][t]) for t in dataset.targets},prediction=dict(zip(dataset.targets,p.astype(float).tolist())))
        for r,p in zip(heldout,pred)])
    write_json(args.output/'metadata.json', dict(dataset_sha256=dataset.fingerprint,protocol=dataset.protocol,
        family=args.family,seed=args.seed,targets=dataset.targets,hardware_profile=asdict(pack['profile']),
        train_ids=pack['train_ids'],validation_ids=pack['validation_ids'],test_ids=[r['measurement_id'] for r in heldout],
        unique_architectures={split:len({architecture_key(r['architecture']) for r in dataset.observations if r['split']==split})
                              for split in ['train','validation','test']},
        source_hashes=source_hashes,
        versions={name:version(name) for name in ['numpy','torch','scikit-learn','xgboost','joblib']},
        notes='Architecture-only inputs. Train-only calibration/scalers. Repeats stay grouped; each observation has equal fitting weight. Test reuse across rounds is exploratory, not a fresh prospective test.'))
    write_report(args.output,scores)
    print(f'Saved {args.family} predictor and test report to {args.output}')
