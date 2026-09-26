"""Shared scalar regression metrics for the historical reports."""
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

def metrics(y,p):
    return dict(mae=float(np.mean(abs(y-p))),mape=float(np.mean(abs(y-p)/y)*100),
                r2=float(r2_score(y,p)),spearman=float(spearmanr(y,p).statistic))
