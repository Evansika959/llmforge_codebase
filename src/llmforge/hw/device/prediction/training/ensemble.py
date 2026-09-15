"""Base-learner fitting for the legacy stacked ensemble."""
import numpy as np
from ..models.trees import make_xgboost
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from ..features.physics import HardwareProfile
from .neural import fit_neural

BASE_NAMES=['xgboost','extratrees','randomforest','residual_mlp']


def fit_bases(configs,y,train,stop,seed,cache_bytes):
    profile=HardwareProfile(cache_bytes=cache_bytes).fit([configs[i] for i in train],y[train])
    x=profile.transform(configs);z=np.log(y)
    models=[]
    for j in range(3):
        m=make_xgboost(seed,3)
        m.fit(x[train],z[train,j],eval_set=[(x[stop],z[stop,j])],verbose=False);models.append(m)
    extra=ExtraTreesRegressor(n_estimators=300,min_samples_leaf=2,max_features=1.,n_jobs=4,random_state=seed).fit(x[train],z[train])
    forest=RandomForestRegressor(n_estimators=300,min_samples_leaf=2,max_features=.8,n_jobs=4,random_state=seed).fit(x[train],z[train])
    mlp=fit_neural('mlp',x[train],z[train],x[stop],z[stop],seed)
    return dict(profile=profile,xgboost=models,extratrees=extra,randomforest=forest,residual_mlp=mlp)
