"""Neural inference in log space; no training imports."""
import numpy as np
import torch
from ..models.neural import GroupedTransformer, ResidualMLP

def predict_neural(pack,x):
    model=GroupedTransformer() if pack['kind']=='transformer' else ResidualMLP()
    model.load_state_dict(pack['state_dict']);model.eval()
    xx=torch.tensor(pack['x_scaler'].transform(np.log1p(x)),dtype=torch.float32)
    with torch.no_grad():pred=np.concatenate([model(b).numpy() for b in xx.split(256)])
    return pack['y_scaler'].inverse_transform(pred)
