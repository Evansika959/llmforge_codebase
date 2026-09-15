"""Train the dedicated reference models for a uniform architecture grid, one shape at a time.

A supernet's rank fidelity can only be measured against architectures trained on their own, and
this is what produces them. The 135M grid built this way is the only place in the project where the
predictor's accuracy is actually known; 360M and 0.6B have had the largest token budgets spent on
them and have zero reference models.

--stride/--offset shard the grid across machines without a shared queue: each machine takes every
Nth shape, and each skips whatever is already on disk, so a machine can be added or lost mid-run
without coordination.

The learning rate is NOT inherited across scales. Measured reference-model optima so far: 135M
3e-3, 360M 1e-3, 0.6B 3e-4 -- all bracketed by their own probes. A reference model trained at the
wrong rate measures initialisation, not architecture, and silently invalidates every rank-fidelity
number computed from it.
"""
import argparse, json, os, subprocess, sys


GROUNDTRUTH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "groundtruth.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", required=True, help="JSON list with name/d_qk/n_h/d_mlp per row")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", required=True)
    ap.add_argument("--mix", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--take", default=None, metavar="START:END",
                    help="Python slice applied AFTER sharding, to hand part of one machine's shard\n"
                         "to another. The skip-if-on-disk check is per-machine, so two machines\n"
                         "working the same shard from opposite ends would both train the middle;\n"
                         "an explicit disjoint range is the only safe way to add a machine to a\n"
                         "shard already in progress.")
    a = ap.parse_args()

    rows = json.load(open(a.archs))[a.offset::a.stride]
    if a.take:
        lo, hi = (int(x) if x else None for x in a.take.split(":"))
        rows = rows[lo:hi]
    print(f"[gt_grid] {a.model} lr={a.lr} -- {len(rows)} shapes "
          f"(offset {a.offset} stride {a.stride})", flush=True)
    done = failed = 0
    for i, r in enumerate(rows):
        if os.path.exists(os.path.join(a.out, r["name"], "arch.json")):
            done += 1
            continue
        shape = f"{r['d_qk']},{r['n_h']},{r['d_mlp']}"
        print(f"[gt_grid] {i+1}/{len(rows)} {r['name']} shape={shape}", flush=True)
        rc = subprocess.run([
            sys.executable, GROUNDTRUTH, "--model", a.model,
            "--shape", shape, "--arch", r["name"], "--steps", str(a.steps),
            "--batch", str(a.batch), "--accum", str(a.accum), "--lr", a.lr,
            "--mix", a.mix, "--init", "base", "--out", a.out]).returncode
        if rc:
            failed += 1
            print(f"[gt_grid] FAILED {r['name']} rc={rc}", flush=True)
        else:
            done += 1
    print(f"[gt_grid] DONE {done} trained/present, {failed} failed", flush=True)


if __name__ == "__main__":
    main()
