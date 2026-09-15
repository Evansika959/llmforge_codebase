"""The identity condition: does the ground-truth recipe damage the architecture it cannot improve?

The full-width architecture IS the base model. groundtruth.py initialises the student from the base
and distils toward the base as teacher, so at full width the student starts exactly at the teacher
and has nothing to learn. Any movement away from the base's own held-out CE is pure recipe damage.

It fails badly at 135M. lr 3e-3 leaves q64h9m1536 at 2.8381 against the untouched base's 2.7102 --
0.128 nats WORSE than its own initialisation, about 29x the seed-to-seed sd of 0.0044, and 2.3x the
gap between the best and second-best architectures in the set. Every tau, every top-1 regret and
every within-band pair in this project is measured against that yardstick, so at the top of its
range the yardstick bends further than the differences it is asked to resolve, and a better
predictor is penalised for disagreeing with a damaged reference.

This sweeps the reference-model learning rate against that condition. The check is one-sided and
absolute: the full-width run must not end above the base's own CE by more than the replication
floor. Passing it does not prove the recipe is right for narrow architectures -- it only removes
the one case where the correct answer is known exactly.
"""
import argparse, json, os, subprocess, sys


GROUNDTRUTH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "groundtruth.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="smollm2-135m")
    ap.add_argument("--shape", required=True, help="the FULL-width shape, e.g. 64,9,1536")
    ap.add_argument("--lrs", required=True)
    ap.add_argument("--mix", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    a = ap.parse_args()
    for lr in a.lrs.split(","):
        name = f"ident_lr{lr}"
        if os.path.exists(os.path.join(a.out, name, "arch.json")):
            print(f"[ident] {name} present, skip", flush=True)
            continue
        print(f"[ident] {name}", flush=True)
        subprocess.run([sys.executable, GROUNDTRUTH, "--model", a.model,
                        "--shape", a.shape, "--arch", name, "--steps", str(a.steps),
                        "--batch", str(a.batch), "--accum", str(a.accum), "--lr", lr,
                        "--mix", a.mix, "--init", "base", "--out", a.out])
    print("[ident] DONE", flush=True)


if __name__ == "__main__":
    main()
