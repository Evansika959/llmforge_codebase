"""Per-layer drawings of searched architectures: one heatmap row per knob and one column per layer.

    PYTHONPATH=src .venv/bin/python experiments/hw_nas/plot_arch.py \
        --picks tl-gemmini:smollm2-135m:energy_per_token_uJ rdxe:smollm2-135m:tpot_ms --tolerance 0.02

Each pick is target:model:cost. The script reads the final front of runs/search/{target}__{model}__nsga2__s{seed},
keeps the architectures whose held-out loss is within the tolerance of the full-width base, and draws the one with the
lowest cost. Every cell prints the knob's value, and its color normalizes the knob to its own grid, from the smallest
value to the base value, so equal colors mean equal fractions of the base in every row.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
# queue.py next to this script shadows the standard library module that torch imports. Loading the standard module
# while the script directory is off the import path keeps every later import of it correct.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != HERE]
import queue  # noqa: E402,F401

sys.path.insert(0, str(HERE))
from analyze import TARGET_LABELS  # noqa: E402
from plot_fronts import MODEL_LABELS, base_record, load_run  # noqa: E402

from llmforge.search.elastic_space import ElasticSearchSpace  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ROWS = [("n_head", "$n_h$"), ("n_kv_group", "$n_{kv}$"), ("n_qk_head_dim", "$d_{qk}$"),
        ("n_v_head_dim", "$d_v$"), ("mlp_size", "$d_{\\mathrm{mlp}}$")]
COST_LABELS = {"energy_per_token_uJ": "$E_{tok}$", "prefill_energy_per_token_uJ": "prefill $E_{tok}$",
               "tpot_ms": "TPOT", "ttft_ms": "TTFT", "params_M": "parameters", "flops_per_token": "FLOPs"}


def pick_by_cost(run: dict, base: dict, cost: str, tolerance: float) -> dict:
    limit = base["metrics"]["val_loss_heldout"] * (1 + tolerance)
    ok = [r for r in run["front"] if r["metrics"]["val_loss_heldout"] <= limit]
    return min(ok, key=lambda r: r["metrics"][cost])


def grids_for(model: str) -> dict:
    space = ElasticSearchSpace.from_yaml(str(ROOT / "configs" / "search_spaces" / f"{model}.yaml"))
    grids = {k: list(space.grids[k]) for k in ("n_head", "n_qk_head_dim", "n_v_head_dim", "mlp_size")}
    # The key and value group grid is every divisor of the base group count, as in the supernet.
    grids["n_kv_group"] = [g for g in range(1, space.spec.n_kv + 1) if space.spec.n_kv % g == 0]
    return dict(grids=grids, d_model=space.spec.hidden)


def energy_text(uJ: float) -> str:
    return f"{uJ / 1e3:.2f} mJ" if uJ >= 1e3 else f"{uJ:.1f} µJ"


def subtitle(m: dict, b: dict, d_model: int, n_layers: int) -> str:
    parts = [f"held-out loss {m['val_loss_heldout']:.3f} ({100 * (m['val_loss_heldout'] / b['val_loss_heldout'] - 1):+.1f}%)"]
    for key, label in (("energy_per_token_uJ", "$E_{tok}$"), ("prefill_energy_per_token_uJ", "prefill $E_{tok}$")):
        if key in m:
            parts.append(f"{label} {energy_text(m[key])} ({100 * (m[key] / b[key] - 1):+.0f}%)")
    for key, label in (("ttft_ms", "TTFT"), ("tpot_ms", "TPOT")):
        if key in m:
            parts.append(f"{label} {m[key]:.3g} ms ({100 * (m[key] / b[key] - 1):+.0f}%)")
    if "n_chips" in m:
        parts.append(f"ring of {m['n_chips']} chips, {m['selected_mac_per_vac']} MACs per core")
    parts.append(f"{m['params_M']:.0f}M params")
    if "kv_cache_MB" in m:
        parts.append(f"KV {100 * m['kv_cache_MB'] / b['kv_cache_MB']:.0f}% of base")
    parts.append(f"$d_{{model}}$={d_model}, $L$={n_layers}")
    return "   ".join(parts).replace("-", "−")


def draw(ax, cax, rec: dict, base: dict, grids: dict, d_model: int, title: str):
    layers = rec["individual"]["layers"]
    n = len(layers)
    values = np.array([[layer[key] for layer in layers] for key, _ in ROWS], float)
    norm = np.zeros_like(values)
    for i, (key, _) in enumerate(ROWS):
        lo, hi = min(grids[key]), max(grids[key])
        norm[i] = 1.0 if hi == lo else (values[i] - lo) / (hi - lo)
    im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    for i in range(values.shape[0]):
        for j in range(n):
            ax.text(j, i, f"{int(values[i, j])}", ha="center", va="center", fontsize=5.5,
                    color="white" if norm[i, j] > 0.6 else "black")
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([label for _, label in ROWS], fontsize=9)
    ax.set_xticks(range(0, n, 5))
    ax.set_xlabel("layer (front $\\rightarrow$ back)", fontsize=9)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ROWS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", length=0)
    ax.set_title(title, fontsize=10, fontweight="bold", loc="left", pad=16)
    ax.text(0.0, 1.02, subtitle(rec["metrics"], base["metrics"], d_model, n), transform=ax.transAxes, fontsize=7.5,
            va="bottom")
    cb = plt.colorbar(im, cax=cax)
    cb.set_label("fraction of the knob grid", fontsize=8)


def main():
    ap = argparse.ArgumentParser(description="Draw searched architectures layer by layer.")
    ap.add_argument("--picks", nargs="+", required=True, help="target:model:cost")
    ap.add_argument("--tolerance", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "runs" / "analysis" / "figures" / "architectures.png"))
    args = ap.parse_args()
    plt.rcParams.update({"font.family": "serif"})
    picks = [p.split(":") for p in args.picks]
    fig, axes = plt.subplots(len(picks), 2, figsize=(15, 2.3 * len(picks)), squeeze=False,
                             gridspec_kw=dict(width_ratios=[60, 1], hspace=0.95, wspace=0.02))
    for (ax, cax), (target, model, cost, *tol) in zip(axes, picks):
        tolerance = float(tol[0]) if tol else args.tolerance
        run = load_run(target, model, args.seed)
        base = base_record(run)
        rec = pick_by_cost(run, base, cost, tolerance)
        g = grids_for(model)
        title = (f"{TARGET_LABELS.get(target, target)}, {MODEL_LABELS.get(model, model)}: lowest "
                 f"{COST_LABELS.get(cost, cost)} within +{tolerance:.0%} held-out loss")
        draw(ax, cax, rec, base, g["grids"], g["d_model"], title)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
