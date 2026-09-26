"""Fit using only training and validation arrays; retain timing rows with missing energy."""
import copy
import time
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from scripts.prediction.models.trees import make_xgboost
from .models import Transformer, token_groups


def masked_loss(prediction, truth):
    mask = torch.isfinite(truth)
    clean = torch.where(mask, truth, torch.zeros_like(truth))
    loss = torch.nn.functional.smooth_l1_loss(prediction, clean, reduction='none')
    present = mask.sum(0)>0
    return ((loss*mask).sum(0)/mask.sum(0).clamp(min=1))[present].mean()


def fit_tree(x, y, vx, vy, seed):
    started = time.monotonic()
    models, counts = [], []
    for j in range(3):
        tr, va = np.isfinite(y[:,j]), np.isfinite(vy[:,j])
        m = make_xgboost(seed, 3)
        m.set_params(n_jobs=2)
        m.fit(x[tr], np.log(y[tr,j]), eval_set=[(vx[va], np.log(vy[va,j]))], verbose=False)
        models.append(m)
        counts.append(dict(train=int(tr.sum()), validation=int(va.sum()), best_iteration=m.best_iteration))
    return dict(kind='xgboost', models=models, seed=seed, counts=counts,
                config=models[0].get_params(), train_seconds=time.monotonic()-started)


def fit_transformer(x, y, vx, vy, names, seed, width, layers, epochs=300):
    started=time.monotonic()
    torch.manual_seed(seed)
    groups=token_groups(names)
    config=dict(width=width,layers=layers,dropout=.1)
    net=Transformer(groups, **config)
    xs=StandardScaler().fit(np.log1p(x))
    log_y=np.log(y)
    mean=np.nanmean(log_y,axis=0)
    scale=np.nanstd(log_y,axis=0).clip(1e-8)
    xx=torch.tensor(xs.transform(np.log1p(x)),dtype=torch.float32)
    yy=torch.tensor((log_y-mean)/scale,dtype=torch.float32)
    valid_x=torch.tensor(xs.transform(np.log1p(vx)),dtype=torch.float32)
    optimizer=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs)
    best=float('inf'); best_epoch=0; state=None; history=[]
    for epoch in range(1,epochs+1):
        net.train(); losses=[]
        for ix in torch.randperm(len(xx)).split(128):
            optimizer.zero_grad()
            loss=masked_loss(net(xx[ix]), yy[ix])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
            optimizer.step(); losses.append(float(loss.detach()))
        scheduler.step(); net.eval()
        with torch.no_grad():
            p=np.exp(net(valid_x).numpy()*scale+mean)
        per_target=np.nanmean(abs(p-vy)/vy,axis=0)*100
        val=float(per_target.mean())
        history.append(dict(epoch=epoch,train_loss=float(np.mean(losses)),validation_mape=val,
                            validation_mape_per_target=per_target.tolist()))
        if np.isfinite(val) and val<best-1e-5:
            best=val; best_epoch=epoch; state=copy.deepcopy(net.state_dict())
        if epoch%50==0:
            print(f'Transformer {layers}x{width} seed {seed} epoch {epoch}: validation MAPE {val:.3f}',flush=True)
        if epoch-best_epoch>=45: break
    if state is None: raise RuntimeError('No finite validation checkpoint')
    return dict(kind='transformer',config=config,groups=groups,state_dict=state,x_scaler=xs,
                y_mean=mean,y_scale=scale,seed=seed,best_epoch=best_epoch,epochs=epoch,
                parameter_count=sum(p.numel() for p in net.parameters()),
                validation_mape=best,history=history,train_seconds=time.monotonic()-started)


def predict(pack, x):
    if pack['kind']=='xgboost':
        p=np.column_stack([np.exp(m.predict(x)) for m in pack['models']])
    else:
        net=Transformer(pack['groups'],**pack['config'])
        net.load_state_dict(pack['state_dict']); net.eval()
        xx=torch.tensor(pack['x_scaler'].transform(np.log1p(x)),dtype=torch.float32)
        with torch.no_grad(): z=np.concatenate([net(b).numpy() for b in xx.split(256)])
        p=np.exp(z*pack['y_scale']+pack['y_mean'])
    if not np.isfinite(p).all() or (p<=0).any(): raise ValueError('Invalid physical-unit prediction')
    return p
