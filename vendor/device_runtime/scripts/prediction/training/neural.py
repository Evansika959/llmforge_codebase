"""Shared log-target neural optimization."""
import copy
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from ..models.neural import GroupedTransformer, ResidualMLP

def fit_neural(kind,x,z,stop_x,stop_z,seed,max_epochs=250):
    torch.manual_seed(seed)
    model=GroupedTransformer() if kind=='transformer' else ResidualMLP()
    xs=StandardScaler().fit(np.log1p(x));ys=StandardScaler().fit(z)
    xx=torch.tensor(xs.transform(np.log1p(x)),dtype=torch.float32);yy=torch.tensor(ys.transform(z),dtype=torch.float32)
    vx=torch.tensor(xs.transform(np.log1p(stop_x)),dtype=torch.float32);vy=torch.tensor(ys.transform(stop_z),dtype=torch.float32)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4 if kind=='transformer' else 1e-3,weight_decay=.01)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=max_epochs)
    loss_fn=nn.SmoothL1Loss();best=float('inf');best_epoch=0;state=None
    for epoch in range(1,max_epochs+1):
        model.train()
        for ids in torch.randperm(len(x)).split(128):
            optimizer.zero_grad();loss=loss_fn(model(xx[ids]),yy[ids]);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        schedule.step();model.eval()
        with torch.no_grad():value=float(loss_fn(model(vx),vy))
        if value<best-1e-5:best=value;best_epoch=epoch;state=copy.deepcopy(model.state_dict())
        if epoch-best_epoch>=35:break
    model.load_state_dict(state);model.eval()
    return dict(kind=kind,state_dict=state,x_scaler=xs,y_scaler=ys,best_epoch=best_epoch,
                total_epochs=epoch,best_validation_loss=best,parameter_count=sum(p.numel() for p in model.parameters()))
