"""Base and stacked ensemble prediction, without fitting dependencies."""
import numpy as np
from .neural import predict_neural

def predict_bases(pack,configs):
    x=pack['profile'].transform(configs)
    return np.stack([np.column_stack([m.predict(x) for m in pack['xgboost']]),
        pack['extratrees'].predict(x),pack['randomforest'].predict(x),predict_neural(pack['residual_mlp'],x)],axis=2)
def stack_predict(pack, configs):
    bp = predict_bases(pack['base'], configs)
    return np.exp(np.column_stack([m.predict(bp[:, j, :]) for j, m in enumerate(pack['combiners'])]))
