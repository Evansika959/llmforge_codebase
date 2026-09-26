"""Shared independent-target XGBoost fitting with validation-only stopping."""
from ..models.trees import make_xgboost

def fit_xgb(x, z, train, stop, seed, depth):
    models = []
    for j in range(3):
        model = make_xgboost(seed, depth)
        model.fit(x[train], z[train, j], eval_set=[(x[stop], z[stop, j])], verbose=False)
        models.append(model)
    return models
