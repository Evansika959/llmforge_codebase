"""Screen candidate supernet checkpoints from their uniform-grid runs.

A supernet can only stand in for training when its slice scores respect nesting. A slice that is at
least as large on every knob holds every weight of the smaller one, so it should not score worse.
Earlier analysis tied violations of this rule to under-trained supernets, where larger slices
score worse than the smallest slice that the sandwich rule trains at every step.

For each grid run (cosearch --algo grid) this reports
    full, smallest      loss of the full and the smallest uniform slice
    pairs               ordered pairs (a, b) where b is at least a on every knob and larger on one
    violations          pairs where b scores worse than a by more than --tol nats
    vs_smallest         violations whose smaller member is the smallest slice
    spearman            rank correlation between loss and parameter count, negative when healthy

    python experiments/hw_nas/screen_supernets.py runs/search/screen__* --out runs/analysis/screen.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

KNOBS = ("n_head", "n_qk_head_dim", "n_v_head_dim", "mlp_size")


def _rank(x):
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    return r


def screen(run: Path, tol: float) -> dict:
    recs = [json.loads(l) for l in (run / "evals.jsonl").read_text().splitlines() if l.strip()]
    pts = []
    for r in recs:
        g = r["genome"]
        if any(len(g[k]) != 1 for k in KNOBS):
            continue
        pts.append((tuple(g[k][0] for k in KNOBS), r["metrics"]["val_loss"], r["metrics"]["params_M"]))
    meta = json.loads((run / "run.json").read_text())
    small = min(pts, key=lambda p: p[0])
    full = max(pts, key=lambda p: p[0])
    pairs = viol = vs_small = 0
    worst = 0.0
    for a in pts:
        for b in pts:
            if a is b or not all(y >= x for x, y in zip(a[0], b[0])) or a[0] == b[0]:
                continue
            pairs += 1
            gap = b[1] - a[1]
            if gap > tol:
                viol += 1
                vs_small += a[0] == small[0]
                worst = max(worst, gap)
    loss = np.array([p[1] for p in pts])
    params = np.array([p[2] for p in pts])
    rho = float(np.corrcoef(_rank(loss), _rank(params))[0, 1])
    return {"run": run.name, "supernet": meta["sw"]["ckpt"], "model": meta["sw"]["base_model"],
            "n": len(pts), "full": round(full[1], 4), "smallest": round(small[1], 4),
            "spread": round(small[1] - full[1], 4), "pairs": pairs, "violations": viol,
            "violation_rate": round(viol / max(1, pairs), 4), "vs_smallest": vs_small,
            "worst_gap": round(worst, 4), "spearman_loss_params": round(rho, 4)}


def regret(run: Path, refs_path: Path, min_feasible: int = 6) -> dict:
    """Budget-constrained selection regret against dedicated-training references.

    Every reference architecture defines a budget, its (parameters, KV fraction). Among references
    inside a budget, the supernet picks the one with the lowest slice loss. Regret is that pick's
    trained loss minus the best trained loss inside the budget, averaged over budgets holding at
    least `min_feasible` references. The largest-that-fits rule is the free baseline.
    """
    recs = [json.loads(l) for l in (run / "evals.jsonl").read_text().splitlines() if l.strip()]
    slice_loss = {tuple(r["genome"][k][0] for k in KNOBS): r["metrics"]["val_loss"] for r in recs
                  if all(len(r["genome"][k]) == 1 for k in KNOBS)}
    rows = []
    for ref in json.loads(refs_path.read_text()).values():
        arch = (ref["n_h"], ref["d_qk"], ref.get("d_v", ref["d_qk"]), ref["d_mlp"])
        if arch in slice_loss:
            rows.append({"params": ref["params"], "kv": ref["kv"], "truth": ref["nll"], "slice": slice_loss[arch]})
    picked, largest = [], []
    for b in rows:
        feas = [x for x in rows if x["params"] <= b["params"] and x["kv"] <= b["kv"]]
        if len(feas) < min_feasible:
            continue
        best = min(x["truth"] for x in feas)
        picked.append(min(feas, key=lambda x: x["slice"])["truth"] - best)
        largest.append(max(feas, key=lambda x: x["params"])["truth"] - best)
    truth = np.array([x["truth"] for x in rows])
    sl = np.array([x["slice"] for x in rows])
    return {"refs": refs_path.name, "n_refs": len(rows), "budgets": len(picked),
            "regret_top1": round(float(np.mean(picked)), 4) if picked else None,
            "regret_largest_fits": round(float(np.mean(largest)), 4) if largest else None,
            "spearman_slice_truth": round(float(np.corrcoef(_rank(sl), _rank(truth))[0, 1]), 4) if len(rows) > 2 else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--tol", type=float, default=0.0)
    ap.add_argument("--refs", action="append", default=[], metavar="MODEL=PATH",
                    help="dedicated-training references for a model, for example "
                         "smollm2-135m=assets/groundtruth/smollm2-135m__ctx1024.json, repeatable")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    refs = dict(item.split("=", 1) for item in a.refs)
    rows = []
    for r in a.runs:
        if not (Path(r) / "DONE").exists():
            continue
        row = screen(Path(r), a.tol)
        if row["model"] in refs:
            row.update(regret(Path(r), Path(refs[row["model"]])))
        rows.append(row)
    cols = ["model", "supernet", "n", "full", "smallest", "spread", "pairs", "violations",
            "violation_rate", "vs_smallest", "worst_gap", "spearman_loss_params", "n_refs",
            "budgets", "regret_top1", "regret_largest_fits", "spearman_slice_truth"]
    for row in rows:
        for c in cols:
            row.setdefault(c, None)
    print("  ".join(f"{c:>12s}" for c in cols))
    for r in sorted(rows, key=lambda r: (r["model"], r["violation_rate"])):
        print("  ".join(f"{str(r[c]):>12s}" for c in cols))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
