"""Compare two decode grids of the same architectures measured with different schedules.

The decode schedule pilot compared schedules on four architectures. This script compares two full grids:
the interleaved schedule of the protocol pilots and the grouped schedule adopted after the pilot. For
every architecture measured in both it reports the ratio of decode energy, then asks whether the schedule
changes the ranking of architectures, and whether it changes how well parameter count predicts decode cost.

    python experiments/hw_nas/pilots/compare_decode_schedules.py \\
        --a interleaved=runs/search/gpu-decode-interleaved__smollm2-135m__grid \\
        --b grouped=runs/search/gpu-decode__smollm2-135m__grid
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from llmforge.paths import RUNS

KEY = "energy_per_token_uJ"


def load(path: Path) -> dict:
    out = {}
    for line in (path / "evals.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["feasible"]:
                out[r["key"]] = r["metrics"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, metavar="NAME=RUN_DIR")
    ap.add_argument("--b", required=True, metavar="NAME=RUN_DIR")
    ap.add_argument("--out", default=str(RUNS / "analysis" / "decode_schedules.json"))
    args = ap.parse_args()
    (na, pa), (nb, pb) = (s.partition("=")[::2] for s in (args.a, args.b))
    ea, eb = load(Path(pa)), load(Path(pb))
    keys = sorted(set(ea) & set(eb))
    a = np.array([ea[k][KEY] for k in keys])
    b = np.array([eb[k][KEY] for k in keys])
    params = np.array([ea[k]["params_M"] for k in keys])
    prefill_power = np.array([eb[k]["prefill_power_W"] for k in keys])
    ratio = a / b
    res = {
        "architectures": len(keys), "a": na, "b": nb,
        "ratio_a_over_b": {"median": float(np.median(ratio)), "p10": float(np.percentile(ratio, 10)),
                           "p90": float(np.percentile(ratio, 90))},
        "spearman_a_b": float(spearmanr(a, b).correlation),
        "spearman_ratio_prefill_power": float(spearmanr(ratio, prefill_power).correlation),
        "spearman_params": {na: float(spearmanr(params, a).correlation), nb: float(spearmanr(params, b).correlation)},
        "window_range_median": {na: float(np.median([ea[k]["zeus_repeats_energy_cv"] for k in keys])),
                                nb: float(np.median([eb[k]["zeus_repeats_energy_cv"] for k in keys]))},
        "span_full_over_smallest": {na: float(a.max() / a.min()), nb: float(b.max() / b.min())},
    }
    print(json.dumps(res, indent=1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
