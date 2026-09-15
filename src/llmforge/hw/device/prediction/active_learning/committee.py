"""Group-bootstrap committee. Disagreement is an acquisition proxy, not a CI."""
import numpy as np

from ..data.dataset import architecture_key
from ..features.physics import HardwareProfile
from ..training.trees import fit_xgb
from ..inference.trees import tree_predict


def fit_committee(dataset, members=5, seed=42):
    if members < 2:
        raise ValueError('A committee needs at least two members')
    rows = dataset.observations
    train = [r for r in rows if r['split']=='train']
    validation = [r for r in rows if r['split']=='validation']
    groups = {}
    for i,row in enumerate(train):
        groups.setdefault(architecture_key(row['architecture']), []).append(i)
    indices = list(groups.values())
    rng = np.random.default_rng(seed)
    packs = []
    for member in range(members):
        sampled = rng.integers(0, len(indices), len(indices))
        selected = [train[i] for g in sampled for i in indices[g]]
        fitting = selected + validation
        configs = [r['architecture'] for r in fitting]
        y = np.array([[r['metrics'][t] for t in dataset.targets] for r in fitting],dtype=np.float32)
        n = len(selected)
        profile = HardwareProfile().fit(configs[:n], y[:n])
        x = profile.transform(configs)
        models = fit_xgb(x, np.log(y), np.arange(n), np.arange(n,len(fitting)), seed+member, 3)
        packs.append(dict(profile=profile,models=models,
                          bootstrap_ids=[r['measurement_id'] for r in selected]))
    return dict(members=packs,targets=dataset.targets,dataset_sha256=dataset.fingerprint,seed=seed)


def committee_predictions(committee, configs):
    predictions = np.stack([tree_predict(m['models'], m['profile'].transform(configs)) for m in committee['members']])
    logs = np.log(predictions)
    return np.exp(logs.mean(0)), logs.std(0,ddof=1)
