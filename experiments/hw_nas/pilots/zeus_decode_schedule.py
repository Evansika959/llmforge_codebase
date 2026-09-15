"""Pilot that checks whether a decode energy window reads differently right after a prefill window.

In the v4 protocol every repeat measures a prefill window and then a decode window. Device power during
prefill is two to three times the decode power, so a power reading that lags the load would read the
start of each decode window high, by an amount that varies between windows and grows with the prefill
power of the architecture. The pilot measures the same architectures under four schedules, three
independent calls each, run in shuffled rounds.

    V0   interleaved, the v4 protocol
    V1   interleaved, 1.5 s of untimed decode before every decode window
    V2   grouped, all prefill windows, 1.5 s of untimed decode, then all decode windows
    V3   V2 with 6 s decode windows, the reference level

The analysis reports, per architecture and schedule, the decode energy level against V3, the relative
range across the windows of one call, and the relative range across calls. Every call holds the GPU
lock, so the pilot can run next to the experiment queue.

    python experiments/hw_nas/pilots/zeus_decode_schedule.py
    python experiments/hw_nas/pilots/zeus_decode_schedule.py --analyze
"""
import argparse
import json
import os
import random
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zeus_protocol import architectures  # noqa: E402

from llmforge.paths import RUNS  # noqa: E402

VARIANTS = {"V0": dict(schedule="interleaved", settle_s=0.0, decode_window_s=None),
            "V1": dict(schedule="interleaved", settle_s=1.5, decode_window_s=None),
            "V2": dict(schedule="grouped", settle_s=1.5, decode_window_s=None),
            "V3": dict(schedule="grouped", settle_s=1.5, decode_window_s=6.0)}
CELLS = [("smollm2-135m", "full", ("V0", "V1", "V2", "V3")),
         ("smollm2-135m", "smallest", ("V0", "V1", "V2", "V3")),
         ("smollm2-135m", "het0", ("V0", "V2", "V3")),
         ("qwen3-1.7b", "smallest", ("V0", "V2", "V3"))]
KEEP = ("energy_per_token_uJ", "prefill_energy_per_token_uJ", "tpot_ms", "ttft_ms", "power_W",
        "prefill_power_W", "zeus_repeats_energy_cv", "zeus_prefill_energy_cv", "zeus_decode_passes",
        "zeus_decode_window_s", "hw_feasible", "zeus_error", "zeus_windows")


def run(a):
    import torch  # noqa: F401
    from zeus.monitor import ZeusMonitor

    from llmforge.evaluators.cache import gpu_lock
    from llmforge.hw.zeus.measure import measure_one

    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            done.add((r["model"], r["arch"], r["variant"], r["round"]))
    archs = {m: dict(architectures(m)) for m in sorted({c[0] for c in CELLS})}
    monitor = ZeusMonitor(gpu_indices=[0], cpu_indices=[], sync_execution_with="torch",
                          approx_instant_energy=True)
    with open(a.out, "a") as log:
        for rnd in range(a.rounds):
            calls = [(m, arch, v) for m, arch, vs in CELLS for v in vs]
            random.Random(rnd).shuffle(calls)
            for m, arch, v in calls:
                if (m, arch, v, rnd) in done:
                    continue
                t0 = time.time()
                with gpu_lock():
                    t_lock = time.time()
                    r = measure_one(archs[m][arch], prefill_len=512, decode_len=128, n_repeats=3, warmup=1,
                                    dtype="bf16", monitor=monitor, batch_size=64, cuda_graphs=True,
                                    min_window_s=2.0, return_windows=True, **VARIANTS[v])
                rec = {"model": m, "arch": arch, "variant": v, "round": rnd,
                       "time": time.strftime("%Y-%m-%d %H:%M:%S"), "wait_s": round(t_lock - t0, 1),
                       "wall_s": round(time.time() - t_lock, 1), **{k: r.get(k) for k in KEEP}}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(f"[{rec['time']}] r{rnd} {m} {arch} {v}: decode {r.get('energy_per_token_uJ', float('nan')):.0f} uJ, "
                      f"range {r.get('zeus_repeats_energy_cv', float('nan')):.3f}, {rec['wall_s']} s", flush=True)


def spread(xs):
    return (max(xs) - min(xs)) / float(np.median(xs)) if len(xs) > 1 else 0.0


def analyze(a):
    recs = [json.loads(line) for line in open(a.out)]
    recs = [r for r in recs if r.get("hw_feasible")]
    table = {}
    for m, arch, vs in CELLS:
        ref = [r["energy_per_token_uJ"] for r in recs if (r["model"], r["arch"], r["variant"]) == (m, arch, "V3")]
        ref_level = float(np.median(ref)) if ref else float("nan")
        for v in vs:
            rs = [r for r in recs if (r["model"], r["arch"], r["variant"]) == (m, arch, v)]
            if not rs:
                continue
            vals = [r["energy_per_token_uJ"] for r in rs]
            pre = [r["prefill_energy_per_token_uJ"] for r in rs]
            # Window position effect: each window's energy per pass relative to its call's median.
            pos = np.array([[w["e_dec_pass"] / np.median([x["e_dec_pass"] for x in r["zeus_windows"]])
                             for w in r["zeus_windows"]] for r in rs])
            table[(m, arch, v)] = {
                "calls": len(rs), "level_uJ": float(np.median(vals)),
                "vs_V3": float(np.median(vals)) / ref_level - 1.0,
                "within": float(np.median([r["zeus_repeats_energy_cv"] for r in rs])),
                "between": spread(vals),
                "prefill_uJ": float(np.median(pre)), "prefill_between": spread(pre),
                "window_position": [round(float(x), 3) for x in pos.mean(axis=0)],
                "wall_s": float(np.median([r["wall_s"] for r in rs])),
            }
    lines = ["| model | arch | schedule | calls | decode uJ/token | vs V3 | within-call range | between-call range "
             "| window 1/2/3 vs call median | prefill uJ/token | prefill between | wall s |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (m, arch, v), t in table.items():
        lines.append(f"| {m} | {arch} | {v} | {t['calls']} | {t['level_uJ']:.0f} | {100 * t['vs_V3']:+.1f}% | "
                     f"{t['within']:.3f} | {t['between']:.3f} | {'/'.join(f'{x:.3f}' for x in t['window_position'])} | "
                     f"{t['prefill_uJ']:.1f} | {t['prefill_between']:.3f} | {t['wall_s']:.1f} |")
    text = "\n".join(lines)
    print(text)
    out = os.path.splitext(a.out)[0]
    with open(out + "__summary.json", "w") as f:
        json.dump({"|".join(k): v for k, v in table.items()}, f, indent=1)
    with open(out + "__summary.md", "w") as f:
        f.write(text + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(RUNS / "pilots" / "zeus_decode_schedule.jsonl"))
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    analyze(a) if a.analyze else run(a)


if __name__ == "__main__":
    main()
