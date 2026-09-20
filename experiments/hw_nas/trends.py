"""Architecture trends of the searched fronts across hardware targets and objectives.

    PYTHONPATH=src .venv/bin/python experiments/hw_nas/trends.py --model smollm2-135m \
        --targets gpu tl-gemmini tl-eyeriss tl-flat rdxe rdxe:tpot_ms proxy-params proxy-flops --bands 0 2 5 10

A target is target[:cost]. The cost defaults to the first cost objective of the run. Its winners are the architectures
of the final front that no other front architecture beats on both held-out loss and that cost. The winners are grouped
into bands of held-out loss above the full-width base, and every band reports:

- the mean width of every knob in every block as a fraction of the base width,
- the parameter, attention, MLP and key-value cache sizes as fractions of the base, and the attention and MLP
  parameter fractions block by block,
- the difference of those numbers from a reference target, parameter count by default.

A run that has not finished is read from its evaluations so far, on search loss, and marked provisional.

The uniform grid of each target, when present, gives the cost structure of the backend: the median fraction of each
cost saved and the median search loss added by lowering one knob one grid step with the others fixed. Proxy targets
read their metric from any grid of the model.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
# queue.py next to this script shadows the standard library module that torch imports, so the script directory joins
# the import path only after llmforge and torch have loaded.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != HERE]
from llmforge.search.elastic_space import ElasticSearchSpace  # noqa: E402

sys.path.append(str(HERE))
from analyze import MODEL_LABELS, TARGET_LABELS  # noqa: E402

ROOT = HERE.parents[1]
KNOBS = ("n_head", "n_kv_group", "n_qk_head_dim", "n_v_head_dim", "mlp_size")
SHORT = {"n_head": "n_h", "n_kv_group": "n_kv", "n_qk_head_dim": "d_qk", "n_v_head_dim": "d_v", "mlp_size": "d_mlp"}
TEX = {"n_head": "$n_h$", "n_kv_group": "$n_{kv}$", "n_qk_head_dim": "$d_{qk}$", "n_v_head_dim": "$d_v$",
       "mlp_size": "$d_{\\mathrm{mlp}}$"}
COST_SHORT = {"energy_per_token_uJ": "E_tok", "prefill_energy_per_token_uJ": "prefill E_tok", "tpot_ms": "TPOT",
              "ttft_ms": "TTFT", "params_M": "params", "flops_per_token": "FLOPs", "kv_cache_MB": "KV cache"}
SIZES = ("params", "attention", "mlp", "kv")


def run_dir(target: str, model: str, kind: str) -> Path:
    return ROOT / "runs" / "search" / f"{target}__{model}__{kind}"


def pareto2(recs: list, loss_key: str, cost: str) -> list:
    """Architectures that no other beats on both loss and cost."""
    out, best = [], float("inf")
    for r in sorted(recs, key=lambda r: (r["metrics"][loss_key], r["metrics"][cost])):
        if r["metrics"][cost] < best:
            out.append(r)
            best = r["metrics"][cost]
    return out


def load(target: str, model: str, seed: int) -> dict:
    d = run_dir(target, model, f"nsga2__s{seed}")
    objectives = json.loads((d / "run.json").read_text())["args"]["objectives"]
    evals = {}
    for line in (d / "evals.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["feasible"]:
                evals[r["key"]] = r
    full = max(evals.values(), key=lambda r: r["metrics"]["params_M"])
    if (d / "DONE").exists():
        front = [r for r in json.loads((d / "front.json").read_text())["front"] if "val_loss_heldout" in r["metrics"]]
        base = next((r for r in front if r["key"] == full["key"]), None)
        if base is None:
            raise SystemExit(f"{d.name}: the full-width architecture is not on the final front")
        return dict(objectives=objectives, pool=front, base=base, loss_key="val_loss_heldout", provisional=False,
                    evals=len(evals))
    return dict(objectives=objectives, pool=list(evals.values()), base=full, loss_key="val_loss", provisional=True,
                evals=len(evals))


def sizes(layers: list, d_model: int) -> dict:
    attn = sum(d_model * (L["n_head"] + L["n_kv_group"]) * (L["n_qk_head_dim"] + L["n_v_head_dim"]) for L in layers)
    mlp = sum(3 * d_model * L["mlp_size"] for L in layers)
    kv = sum(L["n_kv_group"] * (L["n_qk_head_dim"] + L["n_v_head_dim"]) for L in layers)
    return dict(params=attn + mlp, attention=attn, mlp=mlp, kv=kv)


def describe(rec: dict, base: dict, blocks: dict, d_model: int) -> dict:
    layers, ref = rec["individual"]["layers"], base["individual"]["layers"]
    frac = {k: [float(np.mean([layers[i][k] / ref[i][k] for i in range(lo, hi)])) for lo, hi in blocks[k]]
            for k in KNOBS}
    s, sb = sizes(layers, d_model), sizes(ref, d_model)
    depth = {part: [sizes(layers[lo:hi], d_model)[part] / sizes(ref[lo:hi], d_model)[part] for lo, hi in blocks[knob]]
             for part, knob in (("attention", "n_head"), ("mlp", "mlp_size"))}
    return dict(knobs=frac, sizes={k: s[k] / sb[k] for k in SIZES}, depth=depth)


def mean_of(descs: list) -> dict:
    def avg(rows):
        return [float(x) for x in np.mean(rows, axis=0)]
    return dict(knobs={k: avg([d["knobs"][k] for d in descs]) for k in KNOBS},
                sizes={k: float(np.mean([d["sizes"][k] for d in descs])) for k in SIZES},
                depth={p: avg([d["depth"][p] for d in descs]) for p in ("attention", "mlp")})


def band_label(lo: float, hi: float) -> str:
    return f"+{lo:g} to +{hi:g}%"


def trends_for(target: str, cost: str, run: dict, bands: list, blocks: dict, d_model: int) -> dict:
    lk, base = run["loss_key"], run["base"]
    winners = pareto2(run["pool"], lk, cost)
    out = dict(target=target, cost=cost, provisional=run["provisional"], evals=run["evals"], winners=len(winners),
               bands={})
    for lo, hi in bands:
        members = [r for r in winners if lo < 100 * (r["metrics"][lk] / base["metrics"][lk] - 1) <= hi]
        entry = dict(n=len(members))
        if members:
            entry.update(mean_of([describe(r, base, blocks, d_model) for r in members]))
            entry["loss_pct"] = float(np.mean([100 * (r["metrics"][lk] / base["metrics"][lk] - 1) for r in members]))
            entry["cost_pct"] = float(np.mean([100 * (r["metrics"][cost] / base["metrics"][cost] - 1) for r in members]))
        out["bands"][band_label(lo, hi)] = entry
    return out


def knob_mean(entry: dict, knob: str, blocks: dict) -> float:
    return float(np.average(entry["knobs"][knob], weights=[hi - lo for lo, hi in blocks[knob]]))


def cost_structure(target: str, model: str, costs: list) -> dict:
    """Median one-step savings per knob on the target's uniform grid, or any grid of the model for proxies."""
    grid = run_dir(target, model, "grid")
    if not grid.exists():
        grid = next(iter(sorted((ROOT / "runs" / "search").glob(f"*__{model}__grid"))), None)
        if grid is None:
            return {}
    shapes = {}
    for line in (grid / "evals.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["feasible"]:
                shapes[tuple(r["individual"]["layers"][0][k] for k in KNOBS)] = r["metrics"]
    costs = [c for c in costs if c in next(iter(shapes.values()))]
    values = [sorted({s[i] for s in shapes}) for i in range(len(KNOBS))]
    out = dict(grid=grid.name, knobs={})
    for i, k in enumerate(KNOBS):
        if len(values[i]) < 2:
            continue
        saved, added = {c: [] for c in costs}, []
        for s, m in shapes.items():
            j = values[i].index(s[i])
            lower = shapes.get(s[:i] + (values[i][j - 1],) + s[i + 1:]) if j else None
            if lower is None:
                continue
            for c in costs:
                saved[c].append(1 - lower[c] / m[c])
            added.append(lower["val_loss"] - m["val_loss"])
        out["knobs"][k] = dict(loss_added=st.median(added), **{c: st.median(saved[c]) for c in costs})
    return out


def varying(results: list) -> list:
    """Knobs whose width differs from the base somewhere, so knobs the space holds fixed are left out."""
    return [k for k in KNOBS if any(abs(v - 1) > 1e-9 for r in results for e in r["bands"].values() if e["n"]
                                    for v in e["knobs"][k])]


def name(r: dict, first_cost: dict) -> str:
    label = TARGET_LABELS.get(r["target"], r["target"])
    if r["cost"] != first_cost[r["target"]]:
        label += f", {COST_SHORT.get(r['cost'], r['cost'])}"
    return label + (" (provisional)" if r["provisional"] else "")


def fmt(values: list) -> str:
    return " ".join(f"{v:.2f}" for v in values)


def markdown(model: str, results: list, reference: dict, structure: dict, blocks: dict, first_cost: dict) -> str:
    knobs = varying(results)
    lines = [f"# Architecture trends, {MODEL_LABELS.get(model, model)}", "",
             "Winners on held-out loss and each cost. Widths are fractions of the base, averaged over the winners in the "
             "band. The attention and MLP depth columns list the parameter fraction of each block, front to back.", ""]
    for band in results[0]["bands"]:
        lines += [f"## Held-out loss {band} above the base", "",
                  "| target | cost | n | loss % | cost % | params | attn | MLP | KV | "
                  + " | ".join(SHORT[k] for k in knobs) + " | attn depth | MLP depth |",
                  "|---" * (11 + len(knobs)) + "|"]
        for r in results:
            e = r["bands"][band]
            label = f"| {name(r, first_cost)} | {COST_SHORT.get(r['cost'], r['cost'])} | {e['n']} |"
            if not e["n"]:
                lines.append(label + " |" * (8 + len(knobs)))
                continue
            s = e["sizes"]
            lines.append(label + f" {e['loss_pct']:.1f} | {e['cost_pct']:.0f} | {s['params']:.2f} | "
                         f"{s['attention']:.2f} | {s['mlp']:.2f} | {s['kv']:.2f} | "
                         + " | ".join(f"{knob_mean(e, k, blocks):.2f}" for k in knobs)
                         + f" | {fmt(e['depth']['attention'])} | {fmt(e['depth']['mlp'])} |")
        if reference:
            ref = reference["bands"][band]
            lines += ["", f"Difference from {name(reference, first_cost)} in the same band, block by block:", ""]
            for r in results:
                e = r["bands"][band]
                if r is reference or not e["n"] or not ref["n"]:
                    continue
                parts = [f"{SHORT[k]} " + " ".join(f"{a - b:+.2f}" for a, b in zip(e["knobs"][k], ref["knobs"][k]))
                         for k in knobs]
                lines.append(f"- {name(r, first_cost)}: " + " | ".join(parts)
                             + f" | attn {e['sizes']['attention'] - ref['sizes']['attention']:+.2f}, "
                             f"MLP {e['sizes']['mlp'] - ref['sizes']['mlp']:+.2f}, "
                             f"KV {e['sizes']['kv'] - ref['sizes']['kv']:+.2f}")
        lines.append("")
    lines += ["## Cost structure on the uniform grids", "",
              "Median over the grid of the cost saved, and of the search loss added, by lowering one knob one grid step "
              "with the others fixed. The last column divides the saving by the added loss.", "",
              "| target | cost | grid | knob | loss added | saved | saved per nat |", "|---|---|---|---|---|---|---|"]
    for (target, cost), cs in structure.items():
        for k, v in cs.get("knobs", {}).items():
            if cost in v:
                per_nat = f"{100 * v[cost] / v['loss_added']:.0f}%/nat" if v["loss_added"] > 0 else "n/a"
                lines.append(f"| {TARGET_LABELS.get(target, target)} | {COST_SHORT.get(cost, cost)} | {cs['grid']} | "
                             f"{SHORT[k]} | {v['loss_added']:.3f} | {100 * v[cost]:.1f}% | {per_nat} |")
    return "\n".join(lines) + "\n"


def figure(model: str, results: list, blocks: dict, first_cost: dict, stem: Path) -> None:
    knobs = varying(results)
    bands = [b for b in results[0]["bands"] if any(r["bands"][b]["n"] for r in results)]
    starts, c = {}, 0
    for k in knobs:
        starts[k] = c
        c += len(blocks[k]) + 1
    cols = c + 2
    fig, axes = plt.subplots(len(bands), 1, figsize=(0.36 * cols + 3.0, 0.42 * len(results) * len(bands) + 1.2),
                             squeeze=False, constrained_layout=True)
    for ax, band in zip(axes[:, 0], bands):
        grid = np.full((len(results), cols), np.nan)
        for i, r in enumerate(results):
            e = r["bands"][band]
            if not e["n"]:
                continue
            for k in knobs:
                grid[i, starts[k]:starts[k] + len(blocks[k])] = e["knobs"][k]
            grid[i, cols - 2], grid[i, cols - 1] = e["sizes"]["params"], e["sizes"]["kv"]
        im = ax.imshow(np.ma.masked_invalid(grid), cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for (i, j), v in np.ndenumerate(grid):
            if not np.isnan(v):
                ax.text(j, i, "1" if v >= 0.995 else f"{v:.2f}"[1:], ha="center", va="center", fontsize=6.5,
                        color="white" if v > 0.6 else "black")
        ax.set_yticks(range(len(results)))
        ax.set_yticklabels([f"{name(r, first_cost)} ({r['bands'][band]['n']})" for r in results], fontsize=8)
        ax.set_xticks([starts[k] + (len(blocks[k]) - 1) / 2 for k in knobs] + [cols - 2, cols - 1])
        ax.set_xticklabels([TEX[k] + " by block" for k in knobs] + ["params", "KV"], fontsize=8)
        ax.tick_params(length=0)
        ax.set_title(f"{MODEL_LABELS.get(model, model)}: winners {band} held-out loss above the base, width as a "
                     "fraction of the base", fontsize=9, loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
    fig.colorbar(im, ax=axes[:, 0].tolist(), shrink=0.8, label="fraction of base")
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_name(stem.name + ".png"), dpi=200)
    fig.savefig(stem.with_name(stem.name + ".pdf"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Architecture trends of searched fronts across targets.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--targets", nargs="+", required=True, help="target[:cost]")
    ap.add_argument("--bands", nargs="+", type=float, default=[0, 2, 5, 10], help="band edges in percent")
    ap.add_argument("--reference", default="proxy-params", help="target whose winners the others are compared with")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "runs" / "analysis"))
    args = ap.parse_args()
    plt.rcParams.update({"font.family": "serif"})

    space = ElasticSearchSpace.from_yaml(str(ROOT / "configs" / "search_spaces" / f"{args.model}.yaml"))
    # Key and value groups are an attention knob and share the attention blocks.
    blocks = {k: space.knob_blocks("n_head" if k == "n_kv_group" else k) for k in KNOBS}
    d_model = space.spec.hidden
    bands = list(zip(args.bands[:-1], args.bands[1:]))

    runs, results, first_cost, structure = {}, [], {}, {}
    for spec in args.targets:
        target, _, cost = spec.partition(":")
        if target not in runs:
            try:
                runs[target] = load(target, args.model, args.seed)
            except FileNotFoundError:
                print(f"skip {target}: no run")
                continue
        run = runs[target]
        first_cost[target] = run["objectives"][1]
        cost = cost or run["objectives"][1]
        results.append(trends_for(target, cost, run, bands, blocks, d_model))
        structure[(target, cost)] = cost_structure(target, args.model, [cost])
    reference = next((r for r in results if r["target"] == args.reference and r["cost"] == first_cost[r["target"]]),
                     None)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"trends__{args.model}"
    (out / f"{stem}.json").write_text(json.dumps(dict(model=args.model, blocks=blocks, results=results,
                                                      cost_structure={f"{t}:{c}": v for (t, c), v in structure.items()}),
                                                 indent=1))
    md = markdown(args.model, results, reference, structure, blocks, first_cost)
    (out / f"{stem}.md").write_text(md)
    figure(args.model, results, blocks, first_cost, out / "figures" / stem)
    print(md)


if __name__ == "__main__":
    main()
