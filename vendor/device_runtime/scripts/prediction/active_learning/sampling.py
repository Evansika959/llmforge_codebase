"""Accuracy-oriented acquisition: log disagreement, coverage, random exploration."""
import hashlib
import numpy as np
from sklearn.neighbors import NearestNeighbors

from ..data.dataset import architecture_key
from .committee import committee_predictions


def unseen_candidates(dataset, candidates):
    seen = {architecture_key(r['architecture']) for r in dataset.observations}
    result, identifiers = [], set()
    data_ids = {r['config_id'] for r in dataset.observations}
    for c in candidates:
        key = architecture_key(c)
        if key in seen:
            continue
        if not c.get('config_id') or c['config_id'] in identifiers or c['config_id'] in data_ids:
            raise ValueError('Duplicate/conflicting candidate config_id')
        seen.add(key)
        identifiers.add(c['config_id'])
        result.append(c)
    return result


def select_batch(dataset, candidates, committee, count=20, seed=42, strategy='hybrid', weights=(1.,1.,1.)):
    pool = unseen_candidates(dataset,candidates)
    if strategy not in ('hybrid','random') or count <= 0 or count > len(pool):
        raise ValueError('Invalid strategy/batch size or insufficient unseen candidates')
    weights = np.asarray(weights,dtype=float)
    if weights.shape != (3,) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError('Three nonnegative target weights with positive sum are required')
    mean, dispersion = committee_predictions(committee,pool)
    uncertainty = (dispersion*weights).sum(1)/weights.sum()
    # Scaling and coverage references use training architecture features only.
    train = sorted({architecture_key(r['architecture']) for r in dataset.observations if r['split']=='train'})
    base = np.log1p(np.array(train,dtype=float))
    scale = base.std(0)
    scale[scale < 1e-8] = 1.
    x = (np.log1p(np.array([architecture_key(c) for c in pool]))-base.mean(0))/scale
    reference = (base-base.mean(0))/scale
    distances = NearestNeighbors(n_neighbors=1).fit(reference).kneighbors(x)[0][:,0]
    original_distance = distances.copy()
    rng = np.random.default_rng(seed)
    available = np.ones(len(pool),dtype=bool)
    chosen = []
    random_n = count if strategy=='random' else max(1,round(count*.1))
    uncertainty_n = 0 if strategy=='random' else min(count-random_n,round(count*.6))
    arms = ['uncertainty']*uncertainty_n + ['coverage']*(count-random_n-uncertainty_n) + ['random']*random_n
    # Reserve genuinely random exploration before the adaptive arms narrow the pool.
    arms = arms[-random_n:] + arms[:-random_n]
    for arm in arms:
        ids = np.flatnonzero(available)
        if arm=='random':
            i = int(rng.choice(ids))
        elif arm=='coverage':
            i = int(ids[np.argmax(distances[ids])])
        else:
            shortlist = ids[np.argsort(-uncertainty[ids],kind='stable')[:max(5*count,1)]]
            # Rank balance avoids score-unit domination and redundant near-neighbors.
            ranks = np.argsort(np.argsort(uncertainty[shortlist],kind='stable'),kind='stable')/max(1,len(shortlist)-1)
            diversity = np.argsort(np.argsort(distances[shortlist],kind='stable'),kind='stable')/max(1,len(shortlist)-1)
            i = int(shortlist[np.argmax(.8*ranks+.2*diversity)])
        available[i] = False
        chosen.append(dict(config=pool[i],reason=arm,uncertainty=float(uncertainty[i]),
                           coverage_distance=float(original_distance[i]),
                           prediction=dict(zip(committee['targets'],mean[i].astype(float).tolist())),
                           log_prediction_sd=dict(zip(committee['targets'],dispersion[i].astype(float).tolist()))))
        distances = np.minimum(distances,np.linalg.norm(x-x[i],axis=1))
    rng.shuffle(chosen)  # Measurement order must not be sorted by uncertainty.
    return chosen


def generate_pool(dataset, count=5000, seed=12345, min_params_m=50., max_params_m=150.):
    from scripts.sweep.generate_watch5_random_50m_150m import generate_random_configs
    excluded = {architecture_key(r['architecture'])[:7] for r in dataset.observations}
    configs,_ = generate_random_configs(count,min_params_m,max_params_m,seed,excluded_keys=excluded)
    for c in configs:
        key = architecture_key(c)
        c['config_id'] = 'AL_'+hashlib.sha256(repr(key).encode()).hexdigest()[:16]
        c['category'] = 'ActiveLearningCoverage'
    return configs
