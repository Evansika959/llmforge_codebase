"""Hardware evaluator: a cost model calibrated on measurements of other architectures.

A search can put a target's measured cost in the loop, or it can measure a sample of architectures once,
fit a small model of the cost and search against the model. This backend serves such a model. Its
features are the analytic metrics that llmforge.evaluators.hw_analytic records, recomputed from each
architecture, plus derived indicators. experiments/hw_nas/fit_cost_model.py fits a model to a run that
measured the target and stores it as JSON.

Derived features
    int8_group_64, int8_group_16   indicators of the INT8 group size that the Pixel Watch 5 runtime
                                   derives from an architecture, see int8_group_size

A model is linear in its features. With log_space, the logarithm of the cost is linear in the logarithms
of the analytic metrics and in the raw indicators.

Emitted keys
    fitted_<cost key>   predicted cost, in the units of the measured cost the model was fitted to
    hw_feasible         False only when a linear model predicts a cost at or below zero
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from .hw_analytic import HwAnalytic

FOLDS = 5


def int8_group_size(ind: Dict[str, Any]) -> int:
    """INT8 group size the watch runtime derives from an architecture, as in the device physics features:
    64, halved until it divides the model width and every active layer's attention and MLP widths."""
    g = ind["globals"]
    mask = g.get("layer_mask") or [True] * len(ind["layers"])
    widths = [int(g["n_embd"])]
    for layer, active in zip(ind["layers"], mask):
        if active:
            widths += [int(layer["n_head"]) * int(layer["n_v_head_dim"]), int(layer["mlp_size"])]
    size = 64
    while any(w % size for w in widths):
        size //= 2
    return size


DERIVED = {"int8_group_64": lambda ind: float(int8_group_size(ind) == 64),
           "int8_group_16": lambda ind: float(int8_group_size(ind) == 16)}


def parse_spec(spec: str):
    """'log:a+b' -> (['a', 'b'], True), 'a+b' -> (['a', 'b'], False)."""
    log_space = spec.startswith("log:")
    return (spec[len("log:"):] if log_space else spec).split("+"), log_space


def feature_matrix(inds: Sequence[Dict[str, Any]], names: Sequence[str], seq_len: int, log_space: bool,
                   metrics: Sequence[Dict[str, Any]] = None) -> np.ndarray:
    """Feature rows for `inds`. Analytic metrics are recomputed unless `metrics` supplies them."""
    if metrics is None:
        metrics = HwAnalytic(seq_len=seq_len).evaluate(list(inds))
    rows = []
    for ind, m in zip(inds, metrics):
        row = []
        for name in names:
            if name in DERIVED:
                row.append(DERIVED[name](ind))
            else:
                v = float(m[name])
                row.append(float(np.log(v)) if log_space else v)
        rows.append(row)
    return np.array(rows, dtype=float)


def fit_linear(features: np.ndarray, y: np.ndarray, seed: int = 0) -> Dict[str, Any]:
    """Least-squares fit with intercept. Returns cross-validated predictions and the full-data coefficients."""
    X = np.column_stack([features, np.ones(len(y))])
    folds = np.array_split(np.random.default_rng(seed).permutation(len(y)), FOLDS)
    pred = np.empty(len(y))
    for test in folds:
        train = np.setdiff1d(np.arange(len(y)), test)
        coef, *_ = np.linalg.lstsq(X[train], y[train], rcond=None)
        pred[test] = X[test] @ coef
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return {"cv_pred": pred, "coef": coef[:-1], "intercept": float(coef[-1])}


def fit_cost_model(run_dir: str, spec: str) -> Dict[str, Any]:
    """Fit a cost model to the measured cost of every feasible architecture in a run."""
    path = Path(run_dir)
    run = json.loads((path / "run.json").read_text())
    cost_key = run["args"]["objectives"][1]
    seq_len = int(run["args"].get("analytic_seq_len", 1024))
    names, log_space = parse_spec(spec)
    recs = [json.loads(line) for line in (path / "evals.jsonl").read_text().splitlines() if line.strip()]
    recs = [r for r in recs if r["feasible"]]
    inds = [r["individual"] for r in recs]
    recomputed = HwAnalytic(seq_len=seq_len).evaluate(inds)
    analytic_names = [n for n in names if n not in DERIVED]
    drift = max((abs(a[n] / r["metrics"][n] - 1.0) for a, r in zip(recomputed, recs) for n in analytic_names
                 if r["metrics"].get(n)), default=0.0)
    X = feature_matrix(inds, names, seq_len, log_space, recomputed)
    y = np.array([float(r["metrics"][cost_key]) for r in recs])
    f = fit_linear(X, np.log(y) if log_space else y)
    pred = np.exp(f["cv_pred"]) if log_space else f["cv_pred"]
    err = np.abs(pred / y - 1.0)
    return {"source_run": str(path), "cost_key": cost_key, "spec": spec, "features": names,
            "log_space": log_space, "analytic_seq_len": seq_len, "n": len(y),
            "coef": dict(zip(names, map(float, f["coef"]))), "intercept": f["intercept"],
            "cv_median_abs_rel_err": float(np.median(err)), "cv_p95_abs_rel_err": float(np.percentile(err, 95)),
            "recorded_metric_max_rel_diff": float(drift)}


class HwFitted:
    def __init__(self, model: str):
        text = Path(model).read_text()
        self.model = json.loads(text)
        self.names = list(self.model["features"])
        self.coef = np.array([self.model["coef"][n] for n in self.names])
        self.log_space = bool(self.model["log_space"])
        self.key = f"fitted_{self.model['cost_key']}"
        self.settings = {"backend": "fitted", "model": str(model),
                         "model_sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
                         "cost_key": self.model["cost_key"], "spec": self.model.get("spec"),
                         "source_run": self.model.get("source_run")}

    def evaluate(self, inds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        X = feature_matrix(inds, self.names, int(self.model["analytic_seq_len"]), self.log_space)
        z = X @ self.coef + float(self.model["intercept"])
        cost = np.exp(z) if self.log_space else z
        out = []
        for c in cost:
            if c > 0:
                out.append({self.key: float(c), "hw_feasible": True})
            else:
                out.append({self.key: float("inf"), "hw_feasible": False,
                            "fitted_error": f"model predicts {float(c):.4g}, at or below zero"})
        return out
