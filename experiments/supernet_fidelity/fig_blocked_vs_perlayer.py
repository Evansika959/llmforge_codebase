"""Blocked vs per-layer sampling, on held-out perplexity only.

This figure deliberately does NOT answer whether blocking worked. Perplexity is the metric that
inverted the knob ranking against capability (it says trim attention first; capability says trim
MLP first), so a perplexity verdict here would be exactly the kind of claim this project has
already been burned by. The capability benchmark decides; it has not run yet.
"""
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from llmforge.supernet.paths import ROOT, RUNS

BLUE, ORANGE, GREY = "#1a5f9e", "#8f4a18", "#5b636b"


def ev(p):
    return {int(s): {k: float(v) for k, v in re.findall(r"([\w-]+) ([\d.]+)", r)}
            for s, r in re.findall(r"\[eval\s+(\d+)\]\s+(.*)", open(ROOT / p).read())}


def main():
    new = ev("logs/supernet/uptrain_qwen3-4b_ab_blocks.log")
    old = ev("logs/supernet/uptrain_qwen3-4b_ab.log")
    steps = [s for s in sorted(new) if s in old]
    UNI = ["qk-lo", "v-lo", "h-lo", "mlp-lo"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#4a5058", "axes.grid": True, "grid.color": "#c9cfd5",
        "grid.linewidth": .6, "grid.alpha": .7, "figure.facecolor": "white"})
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.0))

    for p, col, ls in (("min", ORANGE, "-"), ("het1", BLUE, "-"), ("full", GREY, "-")):
        ax[0].plot(steps, [new[s][p] for s in steps], color=col, ls=ls, lw=2.2,
                   label=f"{p} — blocked")
        ax[0].plot(steps, [old[s][p] for s in steps], color=col, ls=(0, (4, 2)), lw=1.6,
                   alpha=.75, label=f"{p} — per-layer")
    ax[0].set_yscale("log"); ax[0].set_yticks([2.5, 3, 4, 6, 10, 15])
    ax[0].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax[0].set_xlabel("training step"); ax[0].set_ylabel("held-out prediction error (log)")
    ax[0].set_title("Both runs, absolute\n(solid = blocked, dashed = per-layer)",
                    fontsize=10, loc="left", fontweight="bold", pad=8)
    ax[0].legend(frameon=False, fontsize=7.6, ncol=1, loc="upper right")

    ax[1].axhline(0, color="#4a5058", lw=1)
    for p in UNI:
        ax[1].plot(steps, [new[s][p] - old[s][p] for s in steps], color=GREY, lw=1.2, alpha=.55)
    ax[1].plot([], [], color=GREY, lw=1.2, alpha=.55, label="4 uniform single-knob probes")
    for p, col in (("min", ORANGE), ("het1", BLUE), ("het2", "#2c6a4d")):
        ax[1].plot(steps, [new[s][p] - old[s][p] for s in steps], color=col, lw=2.2, label=p)
    ax[1].set_xlabel("training step")
    ax[1].set_ylabel("blocked − per-layer   (negative = blocked better)")
    ax[1].set_title("Difference between the runs\nuniform probes at parity; min and het worse",
                    fontsize=10, loc="left", fontweight="bold", pad=8)
    ax[1].legend(frameon=False, fontsize=8, loc="upper left")
    ax[1].annotate("perplexity only —\ncapability benchmark pending",
                   xy=(.97, .06), xycoords="axes fraction", ha="right", fontsize=8.5,
                   color="#8f4a18", fontweight="bold")

    fig.tight_layout()
    out = RUNS / "figures" / "fig_blocked_vs_perlayer.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out}  (matched steps: {steps[0]}..{steps[-1]})")


if __name__ == "__main__":
    main()
