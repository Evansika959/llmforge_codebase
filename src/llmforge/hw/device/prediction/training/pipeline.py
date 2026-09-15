"""Reusable training API for any validated measurement round, not a fixed row count."""
import numpy as np

from ..features.physics import HardwareProfile
from .neural import fit_neural
from .trees import fit_xgb


def fit_dataset(dataset, family='xgboost', seed=42, max_epochs=250):
    if family not in ('xgboost', 'transformer'):
        raise ValueError('Unsupported predictor family')
    if max_epochs < 1:
        raise ValueError('max_epochs must be positive')
    rows = dataset.observations
    configs = [r['architecture'] for r in rows]
    y = np.array([[float(r['metrics'][t]) for t in dataset.targets] for r in rows], dtype=np.float32)
    if not np.isfinite(y).all() or not (y > 0).all():
        raise ValueError('Targets exceed the supported float32 log-training range')
    train = np.array([i for i,r in enumerate(rows) if r['split'] == 'train'])
    val = np.array([i for i,r in enumerate(rows) if r['split'] == 'validation'])
    profile = HardwareProfile().fit([configs[i] for i in train], y[train])
    x = profile.transform(configs)
    if family == 'xgboost':
        estimator = fit_xgb(x, np.log(y), train, val, seed, 3)
    else:
        estimator = fit_neural('transformer', x[train], np.log(y[train]), x[val], np.log(y[val]), seed, max_epochs)
    return dict(bundle_version=1, family=family, estimator=estimator, profile=profile,
                targets=dataset.targets, protocol=dataset.protocol, dataset_sha256=dataset.fingerprint,
                feature_version='physics32_fixed49_32_v1', seed=seed,
                train_ids=[rows[i]['measurement_id'] for i in train],
                validation_ids=[rows[i]['measurement_id'] for i in val])
