"""Test how well analytic cost models stand in for hardware targets, on a fully evaluated space.

Given grid runs that evaluated the same architectures on one or more targets, it reports three tables.

    reliability        with --retest, the Spearman correlation between two independent measurements of a
                       target, which bounds any cost model's correlation with that target
    rank correlation   Spearman correlation of each cost model with each target's cost
    choices            each cost model picks its own front, judged on each target at matched loss:
                       median extra cost, worst-case multiple of the target-optimal cost, share of front
                       points paying more than 5% extra, and share of the target front's hypervolume

Cost models are the analytic metrics recorded in every run plus each target's own cost. A target's cost
is the second objective of its run, so its hypervolume box and its cost share units by construction.

With --fit FEATURES, for example --fit params_M+kv_cache_MB, a linear model on those analytic metrics
is fitted to each target's measured cost and added as a cost model. A log: prefix fits the logarithm
of the cost on the logarithms of the metrics instead. Besides the recorded metrics, FEATURES may name
int8_group_64 and int8_group_16, indicators of the INT8 group size the watch runtime derives from an
architecture. Predictions come from 5-fold cross-validation, so no architecture's predicted cost uses
its own measurement. Such a model stands for a cost model calibrated on measurements of other
architectures.

A measured target picks its front on noisy values, so judging that front on the same values favors it.
With --retest NAME=RUN_DIR, an independent second measurement of target NAME over the same
architectures, every chooser is judged on the second measurement. The target's own chooser and every
fitted model still use the first measurement, and the target-optimal front comes from the second.

    python experiments/hw_nas/proxy_fidelity.py \\
        --target gpu-prefill=runs/search/gpu__smollm2-135m__grid \\
        --target watch=runs/search/watch__smollm2-135m__grid --label smollm2-135m
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from llmforge.evaluators.hw_fitted import DERIVED, fit_linear
from llmforge.paths import RUNS
from llmforge.search.pareto import non_dominated, normalized_hypervolume

PROXIES = ("params_M", "flops_per_token", "decode_macs_M", "kv_cache_MB")


def spearman(a: List[float], b: List[float]) -> float:
    try:
        from scipy.stats import spearmanr

        return float(spearmanr(a, b).correlation)
    except ImportError:
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])


def load_target(path: Path) -> Dict:
    run = json.loads((path / "run.json").read_text())
    cost_key = run["args"]["objectives"][1]
    evals, inds = {}, {}
    for line in (path / "evals.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["feasible"]:
                evals[r["key"]] = r["metrics"]
                inds[r["key"]] = r["individual"]
    return {"cost_key": cost_key, "box": run["hv_box"], "evals": evals, "inds": inds}


def parse_named(items: List[str]) -> Dict[str, Dict]:
    out = {}
    for item in items:
        name, _, path = item.partition("=")
        out[name] = load_target(Path(path))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", action="append", required=True, metavar="NAME=RUN_DIR")
    ap.add_argument("--retest", action="append", default=[], metavar="NAME=RUN_DIR")
    ap.add_argument("--fit", action="append", default=[], metavar="METRIC+METRIC",
                    help="linear cost model on these analytic metrics, cross-validated per target")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=str(RUNS / "analysis"))
    a = ap.parse_args()

    targets, retests = parse_named(a.target), parse_named(a.retest)
    for n, t in retests.items():
        if n not in targets:
            raise SystemExit(f"--retest {n} names no --target")
        if t["cost_key"] != targets[n]["cost_key"]:
            raise SystemExit(f"--retest {n} measures {t['cost_key']}, the target measures {targets[n]['cost_key']}")
    names = list(targets)
    runs = list(targets.values()) + list(retests.values())
    keys = sorted(set.intersection(*(set(t["evals"]) for t in runs)))
    first = targets[names[0]]["evals"]
    loss = {k: first[k]["val_loss"] for k in keys}
    for t in runs[1:]:
        worst = max(abs(t["evals"][k]["val_loss"] - loss[k]) for k in keys)
        if worst > 1e-9:
            raise SystemExit(f"a run scores losses differently from {names[0]}, max gap {worst:.2e}")

    costs = {p: {k: first[k][p] for k in keys} for p in PROXIES}
    for n in names:
        costs[n] = {k: targets[n]["evals"][k][targets[n]["cost_key"]] for k in keys}
    judge = {n: ({k: retests[n]["evals"][k][retests[n]["cost_key"]] for k in keys} if n in retests else costs[n])
             for n in names}

    # A fitted cost model has one cost vector per target, fitted to that target's first measurement.
    fitted, fit_report = {}, {}
    first_inds = targets[names[0]]["inds"]
    for spec in a.fit:
        log_space = spec.startswith("log:")
        metrics = (spec[len("log:"):] if log_space else spec).split("+")

        def feature(k, m):
            if m in DERIVED:
                return DERIVED[m](first_inds[k])
            v = float(first[k][m])
            return float(np.log(v)) if log_space else v

        feats = np.array([[feature(k, m) for m in metrics] for k in keys])
        name = ("log fit " if log_space else "fit ") + " + ".join(metrics)
        fitted[name], fit_report[name] = {}, {}
        for n in names:
            y = np.array([costs[n][k] for k in keys])
            f = fit_linear(feats, np.log(y) if log_space else y)
            pred = np.exp(f["cv_pred"]) if log_space else f["cv_pred"]
            fitted[name][n] = dict(zip(keys, pred))
            err = np.abs(pred / y - 1.0)
            fit_report[name][n] = {"coef": dict(zip(metrics, map(float, f["coef"]))), "intercept": f["intercept"],
                                   "log_space": log_space,
                                   "cv_median_abs_rel_err": round(float(np.median(err)), 4),
                                   "cv_p95_abs_rel_err": round(float(np.percentile(err, 95)), 4)}
    models = list(costs) + list(fitted)

    def model_cost(c, n):
        return fitted[c][n] if c in fitted else costs[c]

    def front(cost):
        return [keys[i] for i in non_dominated([(loss[k], cost[k]) for k in keys])]

    fronts = {c: front(costs[c]) for c in costs}
    for c in fitted:
        for n in names:
            fronts[(c, n)] = front(fitted[c][n])
    optimal = {n: front(judge[n]) for n in names}
    result = {"label": a.label, "architectures": len(keys),
              "targets": {n: {"run_cost_key": targets[n]["cost_key"], "retest": n in retests} for n in names},
              "reliability": {n: round(spearman([costs[n][k] for k in keys], [judge[n][k] for k in keys]), 4)
                              for n in retests},
              "fits": fit_report, "rank_correlation": {}, "choices": {}}
    for c in models:
        result["rank_correlation"][c] = {n: round(spearman([model_cost(c, n)[k] for k in keys],
                                                           [costs[n][k] for k in keys]), 4) for n in names}
    for c in models:
        result["choices"][c] = {}
        for n in names:
            box = targets[n]["box"]
            cost_c = model_cost(c, n)
            front_c = fronts[(c, n)] if c in fitted else fronts[c]
            extra = []
            for k in optimal[n]:
                cand = [x for x in front_c if loss[x] <= loss[k] + 1e-12]
                if cand:
                    pick = min(cand, key=lambda x: cost_c[x])
                    extra.append(judge[n][pick] / judge[n][k] - 1.0)
            extra = np.array(extra) if extra else np.array([np.nan])
            hv_c = normalized_hypervolume([(loss[x], judge[n][x]) for x in front_c], box["lo"], box["hi"])
            hv_t = normalized_hypervolume([(loss[x], judge[n][x]) for x in optimal[n]], box["lo"], box["hi"])
            result["choices"][c][n] = {"median_extra": round(float(np.median(extra)), 4),
                                       "worst_multiple": round(float(1.0 + np.max(extra)), 3),
                                       "share_over_5pct": round(float((extra > 0.05).mean()), 3),
                                       "hv_share": round(hv_c / hv_t, 4), "front_size": len(front_c)}

    lines = [f"# Cost-model fidelity, {a.label or ', '.join(names)}, {len(keys)} architectures", ""]
    if retests:
        lines += ["Choices are judged on the independent re-measurement of: " + ", ".join(retests) + ".", "",
                  "| target | Spearman between two measurements |", "|---|---|"]
        lines += [f"| {n} | {r:.3f} |" for n, r in result["reliability"].items()]
        lines.append("")
    if fitted:
        lines += ["Fitted cost models use 5-fold cross-validated predictions.", "",
                  "| fitted model | " + " | ".join(f"{n}: median error | {n}: 95th percentile error" for n in names) + " |",
                  "|---|" + "---|" * (2 * len(names))]
        for c, row in fit_report.items():
            lines.append(f"| {c} | " + " | ".join(f"{100 * row[n]['cv_median_abs_rel_err']:.1f}% | "
                                                  f"{100 * row[n]['cv_p95_abs_rel_err']:.1f}%" for n in names) + " |")
        lines.append("")
    lines += ["| cost model | " + " | ".join(f"Spearman with {n}" for n in names) + " |",
              "|---|" + "---|" * len(names)]
    for c, row in result["rank_correlation"].items():
        lines.append(f"| {c} | " + " | ".join(f"{row[n]:.3f}" for n in names) + " |")
    lines += ["", "| chooser | " + " | ".join(f"{n}: median extra | {n}: worst | {n}: >5% | {n}: HV share"
                                              for n in names) + " |",
              "|---|" + "---|" * (4 * len(names))]
    for c, row in result["choices"].items():
        cells = []
        for n in names:
            v = row[n]
            cells += [f"{100 * v['median_extra']:.0f}%", f"{v['worst_multiple']:.2f}x",
                      f"{100 * v['share_over_5pct']:.0f}%", f"{v['hv_share']:.3f}"]
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    print(text)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # Labels such as qwen3-1.7b contain dots, so the extension is appended rather than set with with_suffix.
    name = f"proxy_fidelity__{a.label or '__'.join(names)}"
    (out / f"{name}.json").write_text(json.dumps(result, indent=1))
    (out / f"{name}.md").write_text(text)


if __name__ == "__main__":
    main()
