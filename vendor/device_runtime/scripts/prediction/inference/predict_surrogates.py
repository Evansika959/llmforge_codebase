"""Predict a configuration CSV with saved XGBoost/Transformer ensembles."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb
from ..config import FEATURES
from ..config import TARGETS
from ..models.neural import TinyTransformer
from ..data.io import write_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--family', choices=['xgboost', 'transformer'], default='xgboost')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    meta = json.loads((args.models/'metadata.json').read_text())
    assert meta['features'] == FEATURES and meta['targets'] == TARGETS
    with args.config.open() as f:
        rows = list(csv.DictReader(f))
    features = []
    for r in rows:
        c = {k:float(r[k]) for k in FEATURES[:-2]}
        c['kv_bytes_per_token'] = 4*c['n_layer']*c['n_kv']*(c['d_qk']+c['d_v'])
        c['attention_width'] = c['n_h']*c['d_qk']
        features.append([c[k] for k in FEATURES])
    X = np.asarray(features,dtype=np.float32)
    predictions = []
    torch.set_num_threads(4)
    for seed in [42,123,2026]:
        if args.family == 'xgboost':
            columns=[]
            for target in TARGETS:
                m = xgb.XGBRegressor()
                m.load_model(args.models/f'xgboost_seed{seed}_{target}.json')
                columns.append(np.exp(m.predict(X)))
            predictions.append(np.column_stack(columns))
        else:
            checkpoint = torch.load(args.models/f'transformer_seed{seed}.pt', map_location='cpu', weights_only=False)
            hp = checkpoint['hyperparameters']
            m = TinyTransformer(len(FEATURES),hp['width'],hp['dropout'])
            m.load_state_dict(checkpoint['state_dict'])
            m.eval()
            xn = torch.tensor((X-np.asarray(meta['x_mean'],dtype=np.float32))/np.asarray(meta['x_std'],dtype=np.float32))
            with torch.no_grad():
                p = m(xn).numpy()*np.asarray(meta['log_y_std'])+np.asarray(meta['log_y_mean'])
            predictions.append(np.exp(p))
    mean = np.mean(predictions,axis=0)
    std = np.std(predictions,axis=0,ddof=1)
    result=[]
    for i,r in enumerate(rows):
        entry={'config_id':r['config_id']}
        for j,t in enumerate(TARGETS):
            entry[t+'_pred']=float(mean[i,j])
            entry[t+'_seed_std']=float(std[i,j])
        entry['decode_tok_s_pred']=1000/float(mean[i,0])
        result.append(entry)
    write_csv(args.output,result)
    print(f'Saved {len(result)} predictions. Seed dispersion is not a calibrated uncertainty interval.')


if __name__ == '__main__':
    main()
