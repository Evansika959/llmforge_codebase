import copy
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from .models import build


def fit(config, x, y, validation_x, validation_y, seed, max_epochs=350):
    """Training receives only training/validation arrays, never test labels."""
    torch.manual_seed(seed)
    model = build(config)
    xs = StandardScaler().fit(np.log1p(x))
    ys = StandardScaler().fit(np.log(y))
    tx = torch.tensor(xs.transform(np.log1p(x)), dtype=torch.float32)
    ty = torch.tensor(ys.transform(np.log(y)), dtype=torch.float32)
    vx = torch.tensor(xs.transform(np.log1p(validation_x)), dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epochs)
    best, best_epoch, state = float('inf'), 0, None
    history = []
    started = time.monotonic()
    for epoch in range(1, max_epochs+1):
        model.train()
        losses = []
        for ids in torch.randperm(len(tx)).split(128):
            optimizer.zero_grad()
            loss = nn.functional.smooth_l1_loss(model(tx[ids]), ty[ids])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss))
        schedule.step()
        model.eval()
        with torch.no_grad():
            prediction = np.exp(ys.inverse_transform(model(vx).numpy()))
        score = float(np.mean(np.abs(prediction-validation_y)/validation_y)*100)
        history.append(dict(epoch=epoch, train_loss=float(np.mean(losses)), validation_mape=score))
        if np.isfinite(score) and score < best-1e-5:
            best, best_epoch, state = score, epoch, copy.deepcopy(model.state_dict())
        if epoch-best_epoch >= 45:
            break
    if state is None:
        raise RuntimeError('No finite validation checkpoint')
    return dict(config=config, state_dict=state, x_scaler=xs, y_scaler=ys, seed=seed,
                parameter_count=sum(p.numel() for p in model.parameters()),
                best_epoch=best_epoch, epochs=epoch, validation_mape=best,
                train_seconds=time.monotonic()-started, history=history)


def predict(pack, x):
    model = build(pack['config'])
    model.load_state_dict(pack['state_dict'])
    model.eval()
    xx = torch.tensor(pack['x_scaler'].transform(np.log1p(x)), dtype=torch.float32)
    with torch.no_grad():
        z = np.concatenate([model(b).numpy() for b in xx.split(256)])
    p = np.exp(pack['y_scaler'].inverse_transform(z))
    if not np.isfinite(p).all() or (p <= 0).any():
        raise ValueError('Invalid prediction')
    return p
