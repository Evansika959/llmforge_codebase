"""Fit a cost model to one target's measurements, for a search against the calibrated model.

    python experiments/hw_nas/fit_cost_model.py --run runs/search/gpu-decode__qwen3-1.7b__grid \\
        --fit params_M+kv_cache_MB --out runs/cost_models/gpu-decode__qwen3-1.7b__params-kv.json

The model is fitted to every feasible architecture of the run, and the script prints its 5-fold
cross-validated error. A search uses it with --hw fitted --hw-arg model=PATH and the objective
fitted_<cost key>. The FEATURES syntax, including the log: prefix and the derived indicators, is described
in llmforge.evaluators.hw_fitted.
"""
import argparse
import json
from pathlib import Path

from llmforge.evaluators.hw_fitted import fit_cost_model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory that measured the target")
    ap.add_argument("--fit", required=True, metavar="FEATURES")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model = fit_cost_model(a.run, a.fit)
    if model["recorded_metric_max_rel_diff"] > 1e-6:
        raise SystemExit(f"recomputed analytic metrics differ from the recorded ones by up to "
                         f"{model['recorded_metric_max_rel_diff']:.2e}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(model, indent=1))
    print(f"{model['cost_key']} ~ {a.fit} on {model['n']} architectures: cross-validated median error "
          f"{100 * model['cv_median_abs_rel_err']:.1f}%, 95th percentile {100 * model['cv_p95_abs_rel_err']:.1f}%")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
