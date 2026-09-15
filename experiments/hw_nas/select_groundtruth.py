"""Select architectures for dedicated-training ground truth from finished search runs.

Cost levels are taken at quantiles of the NSGA-II front's hardware cost. At each level it picks the
lowest-loss architecture whose measured cost is within the level from

    searched   the NSGA-II run
    uniform    the uniform-grid run
    proxy-X    each re-measured proxy front

so every level compares trained loss at matched measured cost. The selection writes a set file in the
format of experiments/supernet_fidelity/groundtruth_het.py and prints queue entries that train every
selected architecture and then score them all.

    python experiments/hw_nas/select_groundtruth.py --model smollm2-135m --target gpu \\
        --levels 0.2 0.4 0.6 0.8 --supernet-key sl135_lr8e-3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from llmforge.paths import ROOT, RUNS
from llmforge.search.elastic_space import ElasticSearchSpace


def feasible(run_dir: Path):
    recs = [json.loads(l) for l in (run_dir / "evals.jsonl").read_text().splitlines() if l.strip()]
    return [r for r in recs if r["feasible"]]


def best_within(recs, cost: float):
    inside = [r for r in recs if r["objs"][1] <= cost + 1e-9]
    return min(inside, key=lambda r: r["objs"][0]) if inside else None


def heterogeneity(layers) -> float:
    cvs = []
    for k in ("n_head", "n_qk_head_dim", "n_v_head_dim", "mlp_size"):
        v = np.array([li[k] for li in layers], float)
        cvs.append(float(v.std() / v.mean()))
    return round(float(np.mean(cvs)), 4)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--target", default="gpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--levels", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8])
    ap.add_argument("--out", default=None, help="default runs/groundtruth/<target>__<model>")
    a = ap.parse_args()

    base = RUNS / "search"
    tag = f"{a.target}__{a.model}"
    nsga = feasible(base / f"{tag}__nsga2__s{a.seed}")
    sources = {"searched": nsga, "uniform": feasible(base / f"{tag}__grid")}
    for p in sorted(base.glob(f"{tag}__remeasure__proxy-*")):
        if (p / "DONE").exists():
            sources[p.name.split("__remeasure__", 1)[1]] = feasible(p)

    from llmforge.search.pareto import non_dominated

    front = [nsga[i] for i in non_dominated([r["objs"] for r in nsga])]
    costs = np.array(sorted(r["objs"][1] for r in front))
    space = ElasticSearchSpace(a.model, partition="per_layer")
    full = space.to_elastic_config(space.full())
    w0, kv0 = full.weight_params(include_embed=False), full.kv_bytes_per_token()

    out = Path(a.out) if a.out else RUNS / "groundtruth" / tag
    out.mkdir(parents=True, exist_ok=True)
    chosen, seen = [], {}
    for li, q in enumerate(a.levels):
        level = float(np.quantile(costs, q))
        for method, recs in sources.items():
            r = best_within(recs, level)
            if r is None:
                continue
            if r["key"] in seen:
                seen[r["key"]]["picked_by"].append(f"L{li}:{method}")
                continue
            layers = r["individual"]["layers"]
            cfg = space.to_elastic_config(r["individual"])
            name = f"L{li}{method[0].upper()}{len(chosen):02d}"
            entry = {"name": name, "d_qk": cfg.d_qk, "d_v": cfg.d_v, "n_h": cfg.n_h, "d_mlp": cfg.d_mlp,
                     "w": round(cfg.weight_params(include_embed=False) / w0, 4),
                     "kv": round(cfg.kv_bytes_per_token() / kv0, 4), "het": heterogeneity(layers),
                     "key": r["key"], "level_quantile": q, "level_cost": level, "method": method,
                     "search_loss": r["objs"][0], "cost": r["objs"][1], "picked_by": [f"L{li}:{method}"]}
            seen[r["key"]] = entry
            chosen.append(entry)
    set_path = out / "ARCHS.json"
    set_path.write_text(json.dumps(chosen, indent=1))
    print(f"{len(chosen)} architectures -> {set_path.relative_to(ROOT)}")
    for e in chosen:
        print(f"  {e['name']:6s} {e['method']:14s} level q={e['level_quantile']:.2f} cost {e['cost']:.1f} "
              f"loss {e['search_loss']:.4f} w {e['w']:.3f} kv {e['kv']:.3f} het {e['het']:.3f}")

    rel = out.relative_to(ROOT)
    print("\n# queue entries")
    for e in chosen:
        print(f"  - name: gt__{tag}__{e['name']}\n"
              f"    command: [python, experiments/supernet_fidelity/groundtruth_het.py, --model, {a.model}, "
              f"--set, {rel}/ARCHS.json, --arch, {e['name']}, --out, {rel}]\n"
              f"    out: {rel}/{e['name']}")
    print(f"  - name: gt__{tag}__score\n"
          f"    command: [python, experiments/supernet_fidelity/eval_het.py, --dir, {rel}, --out, {rel}/nll_ctx1024.json]\n"
          f"    out: {rel}/score")


if __name__ == "__main__":
    main()
