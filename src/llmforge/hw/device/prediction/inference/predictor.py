"""Single inference API for versioned predictors and trusted legacy checkpoints."""
import numpy as np

from ..config import FEATURES, PHYSICS_TARGETS
from ..features.analytic import physical_features
from ..data.dataset import architecture_key, ENERGY_TARGETS
from .neural import predict_neural
from .trees import tree_predict
from .ensemble import stack_predict


def bundle_targets(pack):
    if 'bundle_version' in pack:
        expected = ['decode_tok_s', 'ttft_ms', ENERGY_TARGETS[pack['protocol']['energy_target']]]
        if pack.get('targets') != expected:
            raise ValueError('Bundle targets disagree with the declared energy protocol')
    targets = list(pack.get('targets', PHYSICS_TARGETS))
    if len(targets) != 3 or len(set(targets)) != 3:
        raise ValueError('Invalid model target schema')
    return targets


def predict_bundle(pack, configs):
    if not configs:
        raise ValueError('No architectures supplied')
    for config in configs:
        architecture_key(config)
    bundle_targets(pack)
    if 'bundle_version' in pack:
        if pack['bundle_version'] != 1:
            raise ValueError('Unsupported model bundle version')
        x = pack['profile'].transform(configs)
        if pack['family'] == 'xgboost':
            pred = tree_predict(pack['estimator'], x)
        elif pack['family'] == 'transformer':
            pred = np.exp(predict_neural(pack['estimator'], x))
        else:
            raise ValueError('Unsupported model family')
    elif 'combiners' in pack:
        pred = stack_predict(pack, configs)
    elif 'models' in pack:
        if 'profile' in pack:
            x = pack['profile'].transform(configs)
        elif pack.get('features') == FEATURES:
            x = np.array([[physical_features(c)[0][f] for f in FEATURES] for c in configs],dtype=np.float32)
        else:
            raise ValueError('Unknown legacy tree feature schema')
        inverse = pack.get('inverse_tpot', pack.get('target0') == 'tpot_ms')
        pred = tree_predict(pack['models'], x, inverse)
    elif pack.get('kind') in ('transformer', 'mlp'):
        pred = np.exp(predict_neural(pack, pack['profile'].transform(configs)))
    else:
        raise ValueError('Unrecognized predictor bundle')
    if pred.shape != (len(configs), 3) or not np.isfinite(pred).all() or not (pred > 0).all():
        raise ValueError('Invalid model predictions')
    return pred
