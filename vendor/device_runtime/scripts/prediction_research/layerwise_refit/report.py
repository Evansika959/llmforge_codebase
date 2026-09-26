"""Held-out physical-unit error and ranking evaluation."""
import numpy as np
from scipy.stats import spearmanr, kendalltau


def score(actual, prediction):
    return dict(n=len(actual), mape=float(np.mean(np.abs(prediction-actual)/actual)*100),
                mae=float(np.mean(np.abs(prediction-actual))),
                spearman=float(spearmanr(actual, prediction).statistic),
                kendall_tau_b=float(kendalltau(actual, prediction).statistic))


def markdown(summary):
    lines = ['# Layerwise XGBoost refit', '',
             'Three seeds; mean scores on fixed registry test splits. Validation only for early stopping.', '',
             '| Training cohort | Test cohort | Metric | N | MAPE | Spearman | Kendall tau-b |',
             '|---|---|---|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['model']} | {r['cohort']} | {r['target']} | {r['n']} | "
                     f"{r['mape']:.2f}% | {r['spearman']:.4f} | {r['kendall_tau_b']:.4f} |")
    return '\n'.join(lines)+'\n'
