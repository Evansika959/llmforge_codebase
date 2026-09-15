"""Positive-unit predictions from log-target tree regressors."""
import numpy as np

def tree_predict(models, x, inverse_tpot=False):
    pred = np.exp(np.column_stack([m.predict(x) for m in models]))
    if inverse_tpot:
        pred[:, 0] = 1000 / pred[:, 0]
    return pred
