"""Analyze the hardware-aware search runs of one model on one target.

Runs are found by name under runs/search.

    {target}__{model}__grid                      every uniform architecture on the target
    {target}__{model}__nsga2__s{seed}            NSGA-II searches
    {target}__{model}__random__s{seed}           random sampling at the NSGA-II evaluation count
    {target}__{model}__remeasure__proxy-{name}   a proxy search's front measured on the target

Every method gets the normalized hypervolume of its front over all objectives of the run, on search
documents and on held-out documents. The NSGA-II front is compared with every baseline front at matched
loss and at matched cost, once per cost objective, in the plane of loss and that cost. A run with more than
one cost objective also reports how its costs rank the evaluated architectures, and the per-layer allocation
of each cost's winners, including the winners that no other cost selects. The script writes summary.json,
table.md and front figures to --out.

    python experiments/hw_nas/analyze.py --model smollm2-135m --target gpu
    python experiments/hw_nas/analyze.py --model qwen3-1.7b --target gpu-serve
    python experiments/hw_nas/analyze.py --model smollm2-135m --transfer gpu watch
    python experiments/hw_nas/analyze.py --model smollm2-135m --transfer tl-gemmini rdxe --transfer-costs tpot_ms tpot_ms
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional

import numpy as np

from llmforge.paths import ROOT, RUNS

# The search trees of this project are split. `runs` holds the first round together with the evaluation
# cache and the supernet checkpoints, and `runs_v2` holds the LR-corrected reruns that every number in the
# paper comes from. RUNS points at the first of these because the cache and the checkpoints live there, so
# an analysis that takes RUNS silently reads the wrong tree. SEARCH_RUNS is resolved separately, defaults to
# runs_v2 when it exists, and is recorded in every summary this script writes.
SEARCH_RUNS = ROOT / "runs_v2" if (ROOT / "runs_v2" / "search").is_dir() else RUNS
from llmforge.search.pareto import non_dominated, normalized_hypervolume

KNOBS = ("n_head", "n_qk_head_dim", "n_v_head_dim", "mlp_size")
COST_LABELS = {
    "prefill_energy_per_token_uJ": "prefill energy per prompt token, µJ",
    "energy_per_token_uJ": "energy per generated token, µJ",
    "ttft_ms": "time to first token, ms",
    "tpot_ms": "time per output token, ms",
    "params_M": "parameters, millions",
    "flops_per_token": "FLOPs per token",
    "decode_macs_M": "decode MACs, millions",
    "kv_cache_MB": "KV cache, MB",
}
TARGET_LABELS = {"gpu": "GPU prefill", "gpu-decode": "GPU decode", "gpu-serve": "GPU batched decoding",
                 "watch": "Pixel Watch 5", "tl-gemmini": "Gemmini, Timeloop", "tl-eyeriss": "Eyeriss, Timeloop",
                 "tl-flat": "FLAT, Timeloop", "rdxe": "rDXE ring"}
MODEL_LABELS = {"smollm2-135m": "SmolLM2-135M", "smollm2-360m": "SmolLM2-360M", "qwen3-0.6b": "Qwen3-0.6B",
                "qwen3-1.7b": "Qwen3-1.7B", "qwen3-4b": "Qwen3-4B"}


def method_label(method: str) -> str:
    """Legend name of a run method, for example remeasure__proxy-params -> parameter-proxy front, measured."""
    if method == "grid":
        return "uniform grid"
    for prefix, name in (("nsga2__s", "NSGA-II, seed "), ("random__s", "random, seed ")):
        if method.startswith(prefix):
            return name + method[len(prefix):]
    if method.startswith("remeasure__"):
        source = method[len("remeasure__"):]
        source = {"proxy-params": "parameter-proxy", "proxy-flops": "FLOPs-proxy"}.get(source, source)
        return f"{source} front, measured"
    return method


def load_run(path: Path) -> Optional[dict]:
    if not (path / "evals.jsonl").exists():
        return None
    return {
        "name": path.name,
        "done": (path / "DONE").exists(),
        "run": json.loads((path / "run.json").read_text()),
        "evals": [json.loads(l) for l in (path / "evals.jsonl").read_text().splitlines() if l.strip()],
        "front": json.loads((path / "front.json").read_text()) if (path / "front.json").exists() else None,
        "trace": [json.loads(l) for l in (path / "trace.jsonl").read_text().splitlines() if l.strip()],
    }


def front_of(points: List[tuple]) -> List[tuple]:
    idx = non_dominated([p[0] for p in points])
    return sorted((points[i] for i in idx), key=lambda p: p[0][0])


def search_front(run: dict) -> List[tuple]:
    return front_of([(r["objs"], r) for r in run["evals"] if r["feasible"]])


def heldout_front(run: dict) -> List[tuple]:
    if not run["front"]:
        return []
    pts = [([r["metrics"]["val_loss_heldout"]] + r["objs"][1:], r) for r in run["front"]["front"]
           if "val_loss_heldout" in r["metrics"]]
    return front_of(pts)


def project(front: List[tuple], j: int) -> List[tuple]:
    """Front in the plane of loss and objective j. A front over more objectives contains it."""
    return front_of([((p[0][0], p[0][j]), p[1]) for p in front])


def matched(searched: List[tuple], baseline: List[tuple]) -> dict:
    """Cost saving at equal or lower loss, and loss gain at equal or lower cost, per baseline point."""
    savings, gains = [], []
    for ob, _ in baseline:
        costs = [oa[1] for oa, _ in searched if oa[0] <= ob[0] + 1e-12]
        if costs:
            savings.append(1.0 - min(costs) / ob[1])
        losses = [oa[0] for oa, _ in searched if oa[1] <= ob[1] + 1e-12]
        if losses:
            gains.append(ob[0] - min(losses))

    def stat(xs):
        return {"median": round(median(xs), 4), "max": round(max(xs), 4), "n": len(xs)} if xs else None

    return {"cost_saving_at_matched_loss": stat(savings), "loss_gain_at_matched_cost": stat(gains)}


def allocation(front: List[tuple], grids: Dict[str, List[int]]) -> Dict[str, List[float]]:
    """Mean knob value per layer as a fraction of the grid maximum, over front architectures."""
    if not front:
        return {}
    out = {}
    for k in KNOBS:
        rows = [[li[k] / max(grids[k]) for li in r["individual"]["layers"]] for _, r in front]
        out[k] = [round(float(x), 3) for x in np.mean(rows, axis=0)]
    return out


def spearman(a: List[float], b: List[float]) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def cost_rank_correlation(run: dict, costs: List[str]) -> Dict[str, float]:
    """Spearman correlation between every pair of cost objectives over the evaluated architectures."""
    objs = np.array([r["objs"] for r in run["evals"] if r["feasible"]])
    return {f"{a} vs {b}": round(spearman(objs[:, i + 1], objs[:, j + 1]), 4)
            for (i, a), (j, b) in itertools.combinations(enumerate(costs), 2)}


def winners_by_cost(front: List[tuple], costs: List[str], grids: Dict[str, List[int]]) -> dict:
    """Each cost's front in the plane of loss and that cost, and the architectures only that cost selects."""
    fronts = {c: project(front, j) for j, c in enumerate(costs, start=1)}
    keys = {c: {p[1]["key"] for p in f} for c, f in fronts.items()}
    out = {}
    for c, f in fronts.items():
        others = set().union(*(keys[o] for o in costs if o != c))
        unique = [p for p in f if p[1]["key"] not in others]
        out[c] = {"front_size": len(f), "unique": len(unique), "allocation": allocation(f, grids),
                  "allocation_unique": allocation(unique, grids)}
    return out


def analyze(model: str, target: str, out: Path) -> dict:
    print(f"# reading {SEARCH_RUNS}/search")
    runs = {p.name: load_run(p) for p in sorted(SEARCH_RUNS.glob(f"search/{target}__{model}__*"))}
    runs = {k: v for k, v in runs.items() if v and v["done"]}
    if not runs:
        raise SystemExit(f"no finished runs for {target}__{model}")
    boxes = {json.dumps(v["run"].get("hv_box")) for v in runs.values()}
    first = next(iter(runs.values()))
    box = first["run"]["hv_box"]
    lo, hi = box["lo"], box["hi"]
    grids = first["run"]["space"]["knobs"]
    costs = first["run"]["args"]["objectives"][1:]

    summary = {"model": model, "target": target, "search_runs": str(SEARCH_RUNS),
               "objectives": ["val_loss"] + costs, "hv_box": box,
               "consistent_box": len(boxes) == 1, "methods": {}, "comparisons": {}, "allocation": {}}
    if len(costs) > 1:
        summary["cost_rank_correlation"], summary["winners_by_cost"] = {}, {}
    fronts = {}
    for name, run in runs.items():
        method = name.split(f"{target}__{model}__", 1)[1]
        sf, hf = search_front(run), heldout_front(run)
        fronts[method] = (sf, hf, run)
        summary["methods"][method] = {
            "evals": len(run["evals"]),
            "front_size": len(sf),
            "hv_search": round(normalized_hypervolume([p[0] for p in sf], lo, hi), 4) if sf else 0.0,
            "hv_heldout": round(normalized_hypervolume([p[0] for p in hf], lo, hi), 4) if hf else None,
            "minutes": run["front"]["minutes"] if run["front"] else None,
        }
        summary["allocation"][method] = allocation(sf, grids)
        if len(costs) > 1:
            summary["cost_rank_correlation"][method] = cost_rank_correlation(run, costs)
            summary["winners_by_cost"][method] = winners_by_cost(sf, costs, grids)

    # Every NSGA-II front against every other method, and every other heterogeneous front against the
    # uniform grid, so a scale without a measured search still compares against the uniform baseline.
    pairs = [(s, b) for s in fronts if s.startswith("nsga2") for b in fronts if not b.startswith("nsga2")]
    if "grid" in fronts:
        pairs += [(s, "grid") for s in fronts if s != "grid" and not s.startswith("nsga2")]
    for s, b in pairs:
        by_cost = {}
        for j, c in enumerate(costs, start=1):
            by_cost[c] = {
                "search": matched(project(fronts[s][0], j), project(fronts[b][0], j)),
                "heldout": (matched(project(fronts[s][1], j), project(fronts[b][1], j))
                            if fronts[s][1] and fronts[b][1] else None),
            }
        summary["comparisons"][f"{s} vs {b}"] = {**by_cost[costs[0]], "by_cost": by_cost}

    out.mkdir(parents=True, exist_ok=True)
    tag = f"{target}__{model}"
    (out / f"{tag}__summary.json").write_text(json.dumps(summary, indent=1))
    lines = [f"# {model} on {target}", "", "| method | evals | front | HV search | HV held-out |",
             "|---|---|---|---|---|"]
    for m, s in sorted(summary["methods"].items()):
        lines.append(f"| {m} | {s['evals']} | {s['front_size']} | {s['hv_search']} | {s['hv_heldout']} |")
    lines += ["", "Medians over the points of the baseline front, on search and on held-out documents, in the plane of "
              "loss and each cost.", "",
              "| comparison | cost | cost saving at matched loss, search | loss gain at matched cost, search "
              "| cost saving, held-out | loss gain, held-out |", "|---|---|---|---|---|---|"]

    def med(v, doc, key):
        s = v[doc][key] if v.get(doc) else None
        return s["median"] if s else "-"

    for comp, v in summary["comparisons"].items():
        for c, w in v["by_cost"].items():
            lines.append(f"| {comp} | {c} | {med(w, 'search', 'cost_saving_at_matched_loss')} | "
                         f"{med(w, 'search', 'loss_gain_at_matched_cost')} | "
                         f"{med(w, 'heldout', 'cost_saving_at_matched_loss')} | "
                         f"{med(w, 'heldout', 'loss_gain_at_matched_cost')} |")
    if len(costs) > 1:
        pairs_c = [f"{a} vs {b}" for a, b in itertools.combinations(costs, 2)]
        lines += ["", "Spearman correlation between costs over the evaluated architectures, and winners per cost "
                  "with the number that no other cost selects.", "",
                  "| method | " + " | ".join(pairs_c) + " | " + " | ".join(f"{c} front, unique" for c in costs) + " |",
                  "|---|" + "---|" * (len(pairs_c) + len(costs))]
        for m in sorted(summary["methods"]):
            rho = summary["cost_rank_correlation"][m]
            win = summary["winners_by_cost"][m]
            lines.append(f"| {m} | " + " | ".join(f"{rho[p]:.3f}" for p in pairs_c) + " | "
                         + " | ".join(f"{win[c]['front_size']}, {win[c]['unique']}" for c in costs) + " |")
    (out / f"{tag}__table.md").write_text("\n".join(lines) + "\n")
    plot(fronts, costs, summary, out / f"{tag}__fronts")
    print("\n".join(lines))
    return summary


def plot(fronts, costs: List[str], summary, stem: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(costs), 2, figsize=(10, 4 * len(costs)), sharey=False, squeeze=False)
    grid = fronts.get("grid")
    for j, c in enumerate(costs, start=1):
        for col, label in ((0, "search documents"), (1, "held-out documents")):
            ax = axes[j - 1][col]
            if grid is not None and col == 0:
                pts = np.array([r["objs"] for r in grid[2]["evals"] if r["feasible"]])
                ax.scatter(pts[:, j], pts[:, 0], s=6, c="0.8", label="uniform architectures")
            for method, (sf, hf, _) in sorted(fronts.items()):
                f = project(sf if col == 0 else hf, j)
                if not f:
                    continue
                xy = np.array([p[0] for p in f])
                ax.step(xy[:, 1], xy[:, 0], where="post", marker="o", ms=3, label=method_label(method))
            ax.set_xlabel(COST_LABELS.get(c, c))
            ax.set_ylabel("cross-entropy, nats per token")
            ax.set_title(label)
    axes[0][0].legend(fontsize=7)
    fig.suptitle(f"{MODEL_LABELS.get(summary['model'], summary['model'])}, "
                 f"{TARGET_LABELS.get(summary['target'], summary['target'])}")
    fig.tight_layout()
    fig.savefig(str(stem) + ".png", dpi=150)
    fig.savefig(str(stem) + ".pdf")
    plt.close(fig)


def transfer(model: str, a: str, b: str, out: Path, cost_a: Optional[str] = None,
             cost_b: Optional[str] = None) -> dict:
    """Cross-target transfer on the uniform space from the two grid runs.

    The cost of each target defaults to its run's first cost objective.
    """
    ra, rb = (load_run(SEARCH_RUNS / "search" / f"{a}__{model}__grid"),
              load_run(SEARCH_RUNS / "search" / f"{b}__{model}__grid"))
    if not (ra and rb):
        raise SystemExit("both grid runs are needed")
    ca = cost_a or ra["run"]["args"]["objectives"][1]
    cb = cost_b or rb["run"]["args"]["objectives"][1]
    ea = {r["key"]: r for r in ra["evals"] if r["feasible"]}
    eb = {r["key"]: r for r in rb["evals"] if r["feasible"]}
    keys = sorted(set(ea) & set(eb))
    res = {"model": model, "targets": [a, b], "costs": [ca, cb], "n": len(keys)}
    for src, dst, es, ed, cs, cd in ((a, b, ea, eb, ca, cb), (b, a, eb, ea, cb, ca)):
        f_src = front_of([((es[k]["objs"][0], es[k]["metrics"][cs]), k) for k in keys])
        f_dst = front_of([((ed[k]["objs"][0], ed[k]["metrics"][cd]), k) for k in keys])
        penalties = []
        for (od, _) in f_dst:
            chosen = [k for o, k in f_src if o[0] <= od[0] + 1e-12]
            if chosen:
                k = min(chosen, key=lambda k: es[k]["metrics"][cs])
                penalties.append(ed[k]["metrics"][cd] / od[1] - 1.0)
        res[f"{src}_choice_on_{dst}"] = {"median_extra_cost": round(median(penalties), 4) if penalties else None,
                                         "max_extra_cost": round(max(penalties), 4) if penalties else None,
                                         "n": len(penalties)}
    out.mkdir(parents=True, exist_ok=True)
    suffix = f"__{ca}__{cb}" if (cost_a or cost_b) else ""
    (out / f"transfer__{model}__{a}__{b}{suffix}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--target", default="gpu")
    ap.add_argument("--transfer", nargs=2, metavar=("TARGET_A", "TARGET_B"))
    ap.add_argument("--transfer-costs", nargs=2, metavar=("COST_A", "COST_B"), default=(None, None),
                    help="cost metric of each target, default the first cost objective of its grid run")
    ap.add_argument("--out", default=str(RUNS / "analysis"))
    ap.add_argument("--runs", default=None,
                    help="search tree to read, overriding the runs_v2 default")
    a = ap.parse_args()
    if a.transfer:
        transfer(a.model, a.transfer[0], a.transfer[1], Path(a.out), *a.transfer_costs)
    else:
        analyze(a.model, a.target, Path(a.out))


if __name__ == "__main__":
    main()
