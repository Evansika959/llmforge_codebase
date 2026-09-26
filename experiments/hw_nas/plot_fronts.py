"""Figures of the hardware-aware searches: every evaluation by generation, the final front, and the cheapest front
architecture within a loss tolerance of the full-width base.

    PYTHONPATH=src .venv/bin/python experiments/hw_nas/plot_fronts.py \
        --pairs tl-gemmini:smollm2-135m rdxe:smollm2-135m --tolerance 0.02

For every target and model pair the script reads runs/search/{target}__{model}__nsga2__s{seed} and writes two
figures to --out:

    fronts_summary.{pdf,png}  one panel per pair: the final front in held-out loss against energy per token,
                              colored by TTFT and sized by TPOT, with an arrow from the full-width base to the front
                              architecture with the lowest energy per token within the loss tolerance
    search_rows.{pdf,png}     one row per pair: loss against energy per token, TTFT and TPOT for every evaluation,
                              colored by generation, and the final front in energy per token against held-out
                              loss, colored by TTFT

The base is the full-width architecture of the searched supernet, scored like every other candidate, so the arrow
shows what the search saves for a bounded loss increase. Evaluations use search-document loss, while the final front
and the base use held-out loss.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import TARGET_LABELS  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODEL_LABELS = {"smollm2-135m": "SmolLM2-135M", "smollm2-360m": "SmolLM2-360M", "qwen3-0.6b": "Qwen3-0.6B",
                "qwen3-1.7b": "Qwen3-1.7B", "qwen3-4b": "Qwen3-4B"}
MINUS = "−"


def load_run(target: str, model: str, seed: int, runs: Path = None) -> dict:
    d = (runs or ROOT / "runs") / "search" / f"{target}__{model}__nsga2__s{seed}"
    evals = [json.loads(line) for line in (d / "evals.jsonl").read_text().splitlines() if line.strip()]
    front = json.loads((d / "front.json").read_text())["front"]
    # The energy a run optimized is its second objective, which is not the same key on every
    # target: prefill runs record prefill_energy_per_token_uJ where decode runs record
    # energy_per_token_uJ. Reading it from the run keeps one plot honest across both.
    cost_key = json.loads((d / "run.json").read_text())["args"]["objectives"][1]
    return dict(cost_key=cost_key, evals=[r for r in evals if r["feasible"]],
                front=[r for r in front if "val_loss_heldout" in r["metrics"]])


def base_record(run: dict) -> dict:
    """The full-width architecture, taken from the final front for its held-out loss."""
    full = max(run["evals"], key=lambda r: r["metrics"]["params_M"])
    for r in run["front"]:
        if r["key"] == full["key"]:
            return r
    raise SystemExit("the full-width architecture is not on the final front")


def pick_within_tolerance(run: dict, base: dict, tolerance: float) -> dict:
    limit = base["metrics"]["val_loss_heldout"] * (1 + tolerance)
    ok = [r for r in run["front"] if r["metrics"]["val_loss_heldout"] <= limit]
    return min(ok, key=lambda r: r["metrics"][run["cost_key"]])


def energy_unit(values_uJ) -> tuple:
    return (1e-3, "mJ") if np.median(values_uJ) >= 1e3 else (1.0, "µJ")


def time_unit(values_ms) -> tuple:
    """Milliseconds until they reach four digits, then seconds.

    A four-digit tick label does not fit a panel this narrow: the smartwatch's 392 to 3438 ms of
    TTFT renders as 1000, 2000, 3000 running into each other whatever the tick count.
    """
    return (1e-3, "s") if np.median(values_ms) >= 1e3 else (1.0, "ms")


def sizes(values, lo=None, hi=None, smin=15.0, smax=160.0):
    values = np.asarray(values, float)
    lo = values.min() if lo is None else lo
    hi = values.max() if hi is None else hi
    return smin + (smax - smin) * (values - lo) / max(hi - lo, 1e-12)


def pct(new: float, old: float, digits: int = 0) -> str:
    return f"{100 * (new / old - 1):+.{digits}f}%".replace("-", MINUS)


def size_legend(ax, values, title: str, unit_scale: float = 1.0, lo=None, hi=None, loc="upper right",
                ncol: int = 3):
    """Key for the marker sizes.

    `loc` and `ncol` matter once the figure is drawn at column width. These fronts run from the upper
    left to the lower right, so the only reliably empty corner is the lower left, and only a narrow
    box fits it: a three-column legend spans the panel and lies across the front's low-cost tail.
    The frame is opaque, so whatever it does cover reads as covered rather than as faint data.
    """
    ref = np.quantile(np.asarray(values, float), [0.0, 0.5, 1.0])
    handles = [ax.scatter([], [], s=s, facecolors="lightgray", edgecolors="k", linewidths=0.4)
               for s in sizes(ref, lo=min(values), hi=max(values))]
    ax.legend(handles, [f"{v * unit_scale:.3g}" for v in ref], title=title, loc=loc, fontsize=5.5,
              title_fontsize=5.5, frameon=True, framealpha=1.0, borderpad=0.3, labelspacing=0.25,
              handletextpad=0.2, ncol=ncol, columnspacing=0.6).set_zorder(10)


def plot_summary(pairs, runs, tolerance, out: Path):
    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 3.4), squeeze=False, constrained_layout=True)
    for ax, (target, model), run in zip(axes[0], pairs, runs):
        base, pick = base_record(run), pick_within_tolerance(run, base_record(run), tolerance)
        m = lambda r, k: r["metrics"][k]
        ck = run["cost_key"]
        scale, unit = energy_unit([m(r, ck) for r in run["front"]])
        E = np.array([m(r, ck) for r in run["front"]]) * scale
        L = np.array([m(r, "val_loss_heldout") for r in run["front"]])
        ttft = np.array([m(r, "ttft_ms") for r in run["front"]])
        tpot = np.array([m(r, "tpot_ms") for r in run["front"]])
        sc = ax.scatter(E, L, c=ttft, s=sizes(tpot), cmap="plasma", edgecolors="k", linewidths=0.3, zorder=2)
        eb, lb = m(base, ck) * scale, m(base, "val_loss_heldout")
        ep, lp = m(pick, ck) * scale, m(pick, "val_loss_heldout")
        ax.scatter([eb], [lb], marker="*", s=260, facecolors="white", edgecolors="k", linewidths=0.9, zorder=5)
        ax.annotate(f"{MODEL_LABELS.get(model, model)}, full width", (eb, lb), textcoords="offset points",
                    xytext=(-10, 16), ha="right", fontsize=7, zorder=7,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.6", lw=0.5))
        if pick["key"] != base["key"]:
            ax.scatter([ep], [lp], s=sizes([m(pick, "tpot_ms")], lo=tpot.min(), hi=tpot.max()) * 2.6,
                       facecolors="none", edgecolors="tab:red", linewidths=1.6, zorder=6)
            ax.annotate("", xy=(ep, lp), xytext=(eb, lb), zorder=4,
                        arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.3, connectionstyle="arc3,rad=0.25"))
        text = "\n".join([f"loss {pct(lp, lb, 1)}", f"$E_{{tok}}$ {pct(ep, eb)}",
                          f"TTFT {pct(m(pick, 'ttft_ms'), m(base, 'ttft_ms'))}",
                          f"TPOT {pct(m(pick, 'tpot_ms'), m(base, 'tpot_ms'))}"])
        ax.text(0.03, 0.04, text, transform=ax.transAxes, fontsize=7, color="tab:red", fontweight="bold",
                va="bottom", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="tab:red", lw=0.8))
        size_legend(ax, tpot, "TPOT (ms)")
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label("TTFT (ms)", fontsize=8)
        ax.set_title(f"{TARGET_LABELS.get(target, target)}", fontsize=9, fontweight="bold", loc="left")
        ax.set_xlabel(f"$E_{{tok}}$ ({unit})")
        ax.set_ylabel("Held-out loss")
        ax.margins(x=0.06, y=0.12)
        ax.grid(alpha=0.25, lw=0.5)
        print(f"{target} {model}: base loss {lb:.3f}, E {eb:.4g} {unit}; pick within +{tolerance:.0%}: loss {pct(lp, lb, 1)}, "
              f"E {pct(ep, eb)}, TTFT {pct(m(pick, 'ttft_ms'), m(base, 'ttft_ms'))}, "
              f"TPOT {pct(m(pick, 'tpot_ms'), m(base, 'tpot_ms'))}, {m(pick, 'params_M'):.0f}M params".replace(MINUS, "-"))
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fronts_summary.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_rows(pairs, runs, out: Path, width: float = 17.0):
    """Rows of one figure. Author at the width the page gives it.

    Font sizes are absolute on the canvas, so a figure drawn far wider than the text column has its
    type shrunk by whatever the document scales it by. A four-row figure at the default 17 inches
    reduced to a 5.5 inch column prints 8 point type at about 2 point, which no reader can use.
    """
    n = len(pairs)
    fig, axes = plt.subplots(n, 4, figsize=(width, width * (3.4 / 17.0) * n), squeeze=False,
                             constrained_layout=True, gridspec_kw=dict(width_ratios=[1, 1, 1, 1.45]))
    for row, (target, model), run in zip(axes, pairs, runs):
        base = base_record(run)
        m = lambda r, k: r["metrics"][k]
        evals = sorted(run["evals"], key=lambda r: r["gen"])
        ck = run["cost_key"]
        scale, unit = energy_unit([m(r, ck) for r in evals])
        gen = np.array([r["gen"] for r in evals])
        loss = np.array([m(r, "val_loss") for r in evals])
        ttft_all = np.array([m(r, "ttft_ms") for r in evals])
        tpot_all = np.array([m(r, "tpot_ms") for r in evals])
        ts, tu = time_unit(ttft_all)
        ps, pu = time_unit(tpot_all)
        costs = [(np.array([m(r, ck) for r in evals]) * scale, m(base, ck) * scale,
                  f"$E_{{tok}}$ ({unit})"),
                 (ttft_all * ts, m(base, "ttft_ms") * ts, f"TTFT ({tu})"),
                 (tpot_all * ps, m(base, "tpot_ms") * ps, f"TPOT ({pu})")]
        for ax, (x, xb, label) in zip(row[:3], costs):
            sc = ax.scatter(x, loss, c=gen, cmap="viridis", vmin=0, vmax=gen.max(), s=9, edgecolors="k",
                            linewidths=0.15)
            # The base is the full-width architecture, so it sits at an extreme of every cost axis and
            # lands on the spine. Margins hold it off the frame and clip_on draws the whole star even so.
            ax.scatter([xb], [m(base, "val_loss")], marker="*", s=200, facecolors="white", edgecolors="k",
                       linewidths=0.9, zorder=5, clip_on=False)
            ax.margins(x=0.09, y=0.08)
            ax.set_xlabel(label)
            ax.set_ylabel("Search loss")
            ax.grid(alpha=0.25, lw=0.5)
            # At column width the default tick count runs the labels into each other, and four-digit
            # values such as the smartwatch's TTFT in milliseconds still collide at four ticks.
            ax.locator_params(axis="x", nbins=3)
        cb = fig.colorbar(sc, ax=list(row[:3]), pad=0.01, aspect=25)
        cb.set_label("Generation", fontsize=8)
        row[0].set_title(f"{TARGET_LABELS.get(target, target)}, {MODEL_LABELS.get(model, model)}", fontsize=10,
                         fontweight="bold", loc="left")

        # Energy per token is the objective the row is read for, so it holds the y axis rather than the
        # marker area. Time per output token keeps the third panel to its left, over every evaluation.
        ax = row[3]
        L = np.array([m(r, "val_loss_heldout") for r in run["front"]])
        ttft = np.array([m(r, "ttft_ms") for r in run["front"]])
        E = np.array([m(r, ck) for r in run["front"]]) * scale
        sc = ax.scatter(L, E, c=ttft, s=30, cmap="plasma", edgecolors="k", linewidths=0.3, zorder=2)
        ax.scatter([m(base, "val_loss_heldout")], [m(base, ck) * scale], marker="*", s=240,
                   facecolors="white", edgecolors="k", linewidths=0.9, zorder=5, clip_on=False)
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label("TTFT (ms)", fontsize=8)
        ax.set_title("Final Pareto front", fontsize=9)
        ax.locator_params(axis="x", nbins=4)
        ax.set_xlabel("Held-out loss")
        ax.set_ylabel(f"$E_{{tok}}$ ({unit})")
        ax.margins(x=0.08, y=0.12)
        ax.grid(alpha=0.25, lw=0.5)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"search_rows.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Plot the fronts and search histories of NSGA-II runs.")
    ap.add_argument("--pairs", nargs="+", required=True, help="target:model, for example rdxe:smollm2-135m")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tolerance", type=float, default=0.02,
                    help="held-out loss allowed above the full-width base, relative")
    ap.add_argument("--out", default=str(ROOT / "runs" / "analysis" / "figures"))
    ap.add_argument("--runs-root", nargs="+", default=None,
                    help="run root, one value for every pair or one per pair (default: <repo>/runs). "
                         "Rows whose runs were rerun under a different predictor live in another root.")
    ap.add_argument("--width", type=float, default=17.0,
                    help="figure width in inches for the rows figure. Set this to the width the "
                         "document gives the figure, so the type is not scaled down when included.")
    ap.add_argument("--energy-key", nargs="+", default=None,
                    help="metric to plot as energy, one value for every pair or one per pair, with "
                         "- to keep a run's own default. The default is the run's second objective, "
                         "which is the energy it searched on. A serving run records prefill energy "
                         "as well as total energy per token, so a figure of one must say which.")
    args = ap.parse_args()
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelweight": "bold"})
    pairs = [tuple(p.split(":", 1)) for p in args.pairs]
    roots = [Path(r) for r in (args.runs_root or [str(ROOT / "runs")])]
    if len(roots) == 1:
        roots = roots * len(pairs)
    if len(roots) != len(pairs):
        raise SystemExit("--runs-root takes one value or one per pair")
    runs = [load_run(t, m, args.seed, root) for (t, m), root in zip(pairs, roots)]
    if args.energy_key:
        keys = args.energy_key * len(pairs) if len(args.energy_key) == 1 else args.energy_key
        if len(keys) != len(pairs):
            raise SystemExit("--energy-key takes one value or one per pair")
        for run, key in zip(runs, keys):
            if key != "-":
                run["cost_key"] = key
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plot_summary(pairs, runs, args.tolerance, out)
    plot_rows(pairs, runs, out, args.width)
    print(f"wrote {out / 'fronts_summary.png'} and {out / 'search_rows.png'}")


if __name__ == "__main__":
    main()
