"""Is the prefix a GOOD order, and is the predictor's cost->quality curve even monotone?

Three questions, one sweep, no ground truth and no training required.

1. MONOTONICITY. Widen one knob, everything else full, and NLL must not go up. A violation is an
   unambiguous predictor failure: the search would find "free lunch" points where SHRINKING the
   model improves the predicted score, and every Pareto front built on it would be nonsense. This
   has never been checked here.

2. PER-KNOB HEADROOM. At each width, compare the current prefix against the best of T random
   subsets of the SAME width (drawn via `elastic.permute`, an exact full-width symmetry). The gap

       gap_k(w) = NLL_prefix(w) - NLL_best_random(w)

   is how much a better ORDER could buy on knob k, measured directly. It answers whether
   importance-ordered slicing is worth tens of GPU-hours WITHOUT waiting for stand-alone training
   runs. Two readings matter:
     * gaps small and similar across knobs -> ordering is not the bottleneck; drop the idea.
     * one knob far worse than the others   -> that knob's prefix is unnatural, so architectures
       that spend budget there are systematically over-penalised relative to how they would fare
       trained from scratch. That is a rank-fidelity distortion, not merely a quality loss.

   Prior evidence says to expect exactly that asymmetry: truncating the base model's qk prefix
   costs far more than truncating its MLP prefix, i.e. pretraining left the MLP nested and the
   attention head dim not.

3. CURVATURE. NLL vs width per knob is the quantity a search actually consumes when it trades one
   knob against another. A reordering that flattens or steepens one knob's curve relative to the
   others tilts the whole front, which is why (2) is reported per knob rather than pooled.

Note on what a "random subset" means per knob. d_qk is reordered by shuffling its RoPE frequency
PAIRS (`pair_order`), never individual dims -- breaking a pair would destroy the rotation and
measure nothing. d_v, n_h and d_mlp are reordered by permuting weights, with n_h restricted to
permutations within a KV group so every group keeps a reader.

  python experiments/supernet_fidelity/knob_sweep.py --ckpt runs/supernet/qwen3-1.7b_ab/step5000 --model qwen3-1.7b
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from llmforge.supernet.config import SPECS
from llmforge.supernet.elastic import build_pair_order, permuted, random_perms, set_elastic_config
from llmforge.supernet.elastic.sampler import head_grid
from llmforge.supernet.data.heldout import heldout_texts
from llmforge.supernet.paths import RUNS
from llmforge.supernet.search.nsga import load_supernet
from llmforge.supernet.space import ElasticConfig


@torch.no_grad()
def make_nll(model, tok, texts, dev="cuda", max_len=1024):
    """Tokenize once; every later call is forward passes only.

    Unlike `nsga.make_evaluator` the pair order is an ARGUMENT rather than a closure, because
    reordering d_qk is one of the things being measured.
    """
    ids = [tok(t, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(dev)
           for t in texts]
    ids = [i for i in ids if i.shape[1] >= 8]

    def nll(cfg, order):
        set_elastic_config(model, cfg.d_qk, cfg.d_v, order, n_h=cfg.n_h, d_mlp=cfg.d_mlp)
        tot = ntok = 0.0
        for i in ids:
            lg = model(input_ids=i).logits[:, :-1].float()
            t = i[:, 1:]
            tot += F.cross_entropy(lg.reshape(-1, lg.shape[-1]), t.reshape(-1),
                                   reduction="sum").item()
            ntok += t.numel()
        return tot / ntok

    return nll, len(ids)


@torch.no_grad()
def self_test(model, tok, spec, order, texts, dev="cuda"):
    """A permutation must not change the FULL-width model. If it does, every number below is junk.

    Run in fp32. bf16 matmuls reassociate when columns move, and at these logit magnitudes one
    bf16 ULP is ~0.25 -- large enough to swamp a real error and to look alarming when nothing is
    wrong. Casting bf16 weights up to fp32 and back is exact, so this costs only memory.

    Two gates, because they fail differently:
      max|dlogit|  catches a genuine mis-pairing of a unit axis with its consuming axis.
      |dNLL|       is the quantity every number in this experiment is built from, so it is the
                   one that actually has to be zero.
    """
    ids = [tok(t, return_tensors="pt", truncation=True, max_length=512).input_ids.to(dev)
           for t in texts[:2]]
    full = ElasticConfig.full(spec)

    def fwd():
        set_elastic_config(model, full.d_qk, full.d_v, order, n_h=full.n_h, d_mlp=full.d_mlp)
        lg, tot, n = [], 0.0, 0
        for i in ids:
            o = model(input_ids=i).logits.float()
            lg.append(o.clone())
            tot += F.cross_entropy(o[:, :-1].reshape(-1, o.shape[-1]),
                                   i[:, 1:].reshape(-1), reduction="sum").item()
            n += i.shape[1] - 1
        return lg, tot / n

    was = next(model.parameters()).dtype
    model.float()
    try:
        ref_lg, ref_nll = fwd()
        rng = np.random.default_rng(0)
        ok = True
        for knob in ("d_mlp", "d_v", "n_h"):
            with permuted(model, knob, random_perms(model, knob, rng)):
                lg, v = fwd()
            d = max((a - b).abs().max().item() for a, b in zip(lg, ref_lg))
            _, back = fwd()
            good = d < 1e-3 and abs(v - ref_nll) < 1e-5 and back == ref_nll
            ok &= good
            print(f"  [self-test fp32] {knob:6} max|dlogit| {d:.2e}  dNLL {v - ref_nll:+.2e}  "
                  f"restored {back - ref_nll:+.1e}  {'OK' if good else 'FAIL'}", flush=True)
    finally:
        model.to(was)
    if not ok:
        raise SystemExit("permutation is not a full-width symmetry -- fix before trusting results")


def grids(spec):
    """Width grid per knob, widest last so the monotonicity read is left-to-right."""
    return {"d_qk": sorted(spec.qk_grid), "d_v": sorted(spec.qk_grid),
            "n_h": sorted(head_grid(spec)), "d_mlp": sorted(spec.mlp_grid)}


def cfg_at(spec, knob, w):
    """Full width everywhere except `knob`."""
    f = ElasticConfig.full(spec)
    kw = {"d_qk": f.d_qk[0], "d_v": f.d_v[0], "n_h": f.n_h[0], "d_mlp": f.d_mlp[0]}
    kw[knob] = w
    return ElasticConfig.uniform(spec, kw["d_qk"], kw["d_v"], n_h=kw["n_h"], d_mlp=kw["d_mlp"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-1.7b", choices=sorted(SPECS))
    ap.add_argument("--ckpt", default=f"{RUNS}/qwen3-1.7b_ab/step5000")
    ap.add_argument("--n-texts", type=int, default=32)
    ap.add_argument("--trials", type=int, default=8, help="random subsets per width")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    spec = SPECS[a.model]
    model, tok, order = load_supernet(a.ckpt, spec)
    nll, n_used = make_nll(model, tok, heldout_texts(a.n_texts))
    print(f"[setup] {a.model} {a.ckpt}  {n_used} held-out docs  {a.trials} random subsets/width",
          flush=True)
    self_test(model, tok, spec, order, heldout_texts(2))

    n_pairs = spec.head_dim // 2
    rng = np.random.default_rng(a.seed)
    res, t0 = {}, time.time()

    for knob, ws in grids(spec).items():
        full_w = ws[-1]
        rows = []
        for w in ws:
            c = cfg_at(spec, knob, w)
            base = nll(c, order)
            rand = []
            if w < full_w:                       # at full width every order sees the same units
                for _ in range(a.trials):
                    if knob == "d_qk":
                        rand.append(nll(c, list(rng.permutation(n_pairs))))
                    else:
                        with permuted(model, knob, random_perms(model, knob, rng)):
                            rand.append(nll(c, order))
            best = min(rand) if rand else base
            rows.append({"w": w, "prefix": base, "best_random": best,
                         "median_random": float(np.median(rand)) if rand else base,
                         "worst_random": max(rand) if rand else base,
                         "gap": base - best, "trials": rand})
            print(f"  {knob:6} w={w:<5} prefix {base:.4f}  best-of-{len(rand)} {best:.4f}  "
                  f"gap {base - best:+.4f}", flush=True)
        res[knob] = rows

    print(f"\n[{time.time() - t0:.0f}s]  ==== 1. MONOTONICITY (NLL must fall as width grows) ====")
    bad = []
    for knob, rows in res.items():
        seq = [r["prefix"] for r in rows]
        viol = [(rows[i]["w"], rows[i + 1]["w"], seq[i + 1] - seq[i])
                for i in range(len(seq) - 1) if seq[i + 1] > seq[i]]
        bad += [(knob, *v) for v in viol]
        print(f"  {knob:6} " + "  ".join(f"{r['w']}:{r['prefix']:.3f}" for r in rows)
              + ("   OK" if not viol else f"   VIOLATION {viol}"))
    if bad:
        print("  !! predicted score IMPROVES when the model shrinks -- Pareto fronts built on "
              "this predictor are invalid until fixed")

    print("\n==== 2. PER-KNOB HEADROOM  gap = NLL(prefix) - NLL(best random subset) ====")
    print("  positive gap = a better ORDER would help this knob; compare gaps ACROSS knobs")
    for knob, rows in res.items():
        g = [r["gap"] for r in rows if r["w"] < rows[-1]["w"]]
        print(f"  {knob:6} " + "  ".join(f"{r['w']}:{r['gap']:+.3f}" for r in rows[:-1])
              + f"   | max {max(g):+.3f}  mean {float(np.mean(g)):+.3f}")

    print("\n==== 3. CURVATURE  NLL rise vs full width ====")
    for knob, rows in res.items():
        f = rows[-1]["prefix"]
        print(f"  {knob:6} " + "  ".join(f"{r['w']}:{r['prefix'] - f:+.3f}" for r in rows))

    out = a.out or RUNS / f"knob_sweep_{spec.key}.json"
    json.dump({"model": a.model, "ckpt": a.ckpt, "n_docs": n_used, "trials": a.trials,
               "seed": a.seed, "results": res, "monotonicity_violations": bad},
              open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
