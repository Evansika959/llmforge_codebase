"""NSGA-II over the trained A+B supernet: model size vs KV cache vs held-out loss.

Three objectives, all minimized, so there is no budget constraint and no repair operator -- cost
IS an objective. That matters: the repair step in the older 2-knob search could not converge once
a knob moved cost discontinuously, and casting cost as an objective removes the problem rather
than working around it.

Genome: 4 knobs x n_groups, each gene an index into that knob's grid.
    d_qk, d_v  in spec.qk_grid          (cache + weights)
    n_h        in head_grid(spec)       (weights only; n_kv | n_h)
    d_mlp      in spec.mlp_grid         (weights only)
    n_kv       in spec.nkv_grid         (CACHE ONLY -- with --with-nkv)

n_kv scales the KV cache linearly while touching only ~8.6% of parameters (k_proj and v_proj rows;
`attn_params` counts n_kv*d_qk + n_kv*d_v, so it is NOT weight-free -- an earlier version of this
docstring said it was). What it does that the width knobs cannot is lower the CACHE FLOOR at
matched parameter count: over 200k random genomes in the 0.60-0.65 parameter band the width knobs
bottom out at 29.5% cache while n_kv reaches 4.4%, a 6.8x difference. The uniform corner makes the
same point -- n_kv=1 alone gives 12.5% cache at 91.4% of parameters, where the width knobs need to
give up 22.2% of parameters to reach 25%. It is off by default because it needs a supernet trained with
--with-nkv: pooling mixes heads rather than selecting a subset, and on a supernet that never saw
it the damage is ~9.5 nats at 8->4. It is also only a real dimension on Qwen3 -- n_kv must divide
the model's own, and SmolLM2's 3 and 5 are prime.

Every candidate is evaluated by SLICING the one trained supernet -- no retraining -- which is the
whole point of the post-NAS setup.

The uniform sweep is the reference the front must beat. Without it a Pareto front is just a curve;
with it, the question "does per-layer heterogeneity buy anything" has an answer.

  python -m llmforge.supernet.search.nsga --ckpt runs/supernet/qwen3-4b_ab/step8000 --pop 48 --gen 30
"""
import argparse
import glob
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..config import SPECS
from ..elastic import add_elastic_temperature, add_kv_alignment, build_pair_order, pair_order_for, enable_elastic, \
    set_elastic_backend, set_elastic_config
from ..elastic.sampler import head_grid, head_options
from ..paths import RUNS
from ..space import ElasticConfig


# ---------------------------------------------------------------- model + data

def load_supernet(ckpt, spec, dev="cuda"):
    """Base -> elastic patch -> register logit_scale -> load weights. Order matters, and the
    checkpoint may be sharded (4B is 2 shards), which the older single-file loader assumed away."""
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        spec.repo, torch_dtype=torch.bfloat16, attn_implementation="eager").to(dev).eval()
    enable_elastic(model)
    # Without this the checkpoint's alignment tensors load as `unexpected` and are silently
    # dropped, and the first genome with n_kv below full width raises. --with-nkv could not run
    # at all until this line existed.
    add_kv_alignment(model)
    set_elastic_backend("eager")
    add_elastic_temperature(model)
    sd = {}
    for f in sorted(glob.glob(os.path.join(ckpt, "*.safetensors"))):
        sd.update(load_file(f))
    miss, unexp = model.load_state_dict(sd, strict=False)
    ls = sum("logit_scale" in k for k in sd)
    print(f"[loader] {ckpt}: {len(sd)} tensors ({ls} logit_scale) missing={len(miss)} "
          f"unexpected={len(unexp)}", flush=True)
    if ls == 0:
        raise SystemExit("no logit_scale in checkpoint -- wrong dir or untrained model")
    model.config.use_cache = False
    return model, AutoTokenizer.from_pretrained(spec.repo), pair_order_for(spec)


@torch.no_grad()
def make_evaluator(model, tok, order, spec, texts, dev="cuda", max_len=1024):
    """Tokenize once, then each evaluation is pure forward passes."""
    ids = [tok(t, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(dev)
           for t in texts]
    ids = [i for i in ids if i.shape[1] >= 8]

    def loss_of(cfg: ElasticConfig):
        set_elastic_config(model, cfg.d_qk, cfg.d_v, order, n_h=cfg.n_h, d_mlp=cfg.d_mlp,
                           n_kv=cfg.n_kv)
        tot = ntok = 0.0
        for i in ids:
            lg = model(input_ids=i).logits[:, :-1].float()
            t = i[:, 1:]
            tot += F.cross_entropy(lg.reshape(-1, lg.shape[-1]), t.reshape(-1),
                                   reduction="sum").item()
            ntok += t.numel()
        return tot / ntok

    return loss_of, len(ids)


# ---------------------------------------------------------------- NSGA-II

def fronts_of(F_):
    """Fast non-dominated sort. Returns a list of fronts, each a list of indices."""
    n = len(F_)
    dominated = [[] for _ in range(n)]
    count = [0] * n
    fronts = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if all(F_[p] <= F_[q]) and any(F_[p] < F_[q]):
                dominated[p].append(q)
            elif all(F_[q] <= F_[p]) and any(F_[q] < F_[p]):
                count[p] += 1
        if count[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in dominated[p]:
                count[q] -= 1
                if count[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    return fronts[:-1]


def crowding(F_, idx):
    """Crowding distance within one front, per objective, normalized by range."""
    d = {i: 0.0 for i in idx}
    if len(idx) <= 2:
        return {i: float("inf") for i in idx}
    for m in range(F_.shape[1]):
        s = sorted(idx, key=lambda i: F_[i, m])
        d[s[0]] = d[s[-1]] = float("inf")
        span = F_[s[-1], m] - F_[s[0], m]
        if span <= 0:
            continue
        for k in range(1, len(s) - 1):
            d[s[k]] += (F_[s[k + 1], m] - F_[s[k - 1], m]) / span
    return d


def hypervolume(F_, ref):
    """Dominated hypervolume of a minimisation front, w.r.t. a reference point.

    This is the convergence number. Front SIZE is not one -- a front can grow while getting no
    better -- and best-loss alone ignores the whole point of a multi-objective search, which is
    the trade-off curve rather than any single corner. For 2 objectives this is the exact area
    swept by the staircase; for 3 it is a slice-based sum, also exact.
    """
    P = np.array([f for f in F_ if all(f < ref)], float)
    if len(P) == 0:
        return 0.0
    P = P[np.lexsort(tuple(P[:, i] for i in range(P.shape[1] - 1, -1, -1)))]
    if P.shape[1] == 2:
        hv, prev = 0.0, ref[1]
        for x, yv in P:
            if yv < prev:
                hv += (ref[0] - x) * (prev - yv); prev = yv
        return float(hv)
    # exact 3-objective sweep: slab along the third axis, 2-D hypervolume of each slice
    hv, zs = 0.0, sorted({f[2] for f in P})
    for i, z in enumerate(zs):
        nxt = zs[i + 1] if i + 1 < len(zs) else ref[2]
        sl = np.array([f[:2] for f in P if f[2] <= z])
        hv += hypervolume(sl, ref[:2]) * max(0.0, nxt - z)
    return float(hv)


def rank_and_crowd(F_):
    rank = {}
    dist = {}
    for r, fr in enumerate(fronts_of(F_)):
        c = crowding(F_, fr)
        for i in fr:
            rank[i], dist[i] = r, c[i]
    return rank, dist


class Space:
    """The genome, defined over MEASURED layer blocks rather than fixed-size groups.

    Attention and MLP get separate partitions because their depth-sensitivity profiles differ
    (attention peaks at layer 0 and the 22-24 band; MLP peaks at the last two layers). Using the
    same partition the trainer samples from is the point: previously training covered a 144-dim
    space while the search covered 36 dims, so most of the trained capacity was unreachable.
    """

    def __init__(self, spec, per=4, blocks=True, with_nkv=False):
        from ..elastic.sampler import blocks_for
        self.spec = spec
        self.blocks = blocks
        self.with_nkv = with_nkv
        self.names = ["d_qk", "d_v", "n_h", "d_mlp"]
        # With n_kv elastic the n_h grid must be the FULL nh_grid, not head_grid(spec) -- the
        # latter intersects with the model's BASE n_kv and leaves {8,16}, so the finer head counts
        # a smaller n_kv unlocks are unreachable and the repair in to_config can never fire (8 and
        # 16 divide 1, 2, 4 and 8). A search run before this fix produced fronts containing only
        # n_h 8 and 16, i.e. the advertised 13 (n_h, n_kv) combinations were never in play.
        self.grids = [spec.qk_grid, spec.qk_grid,
                      list(spec.nh_grid) if with_nkv else head_grid(spec), spec.mlp_grid]
        if blocks:
            ba, bm = blocks_for(spec, "attn"), blocks_for(spec, "mlp")
            self.parts = [ba, ba, ba, bm]
        else:
            self.per = per
            g = [(i, min(i + per, spec.n_layers)) for i in range(0, spec.n_layers, per)]
            self.parts = [g, g, g, g]
        if with_nkv:
            # n_kv is the knob the hardware actually feels: it scales the KV cache directly, and
            # cache is usually what binds on an edge device. It shares the attention partition
            # because a KV group and the query heads that read it have to move together.
            self.names.append("n_kv")
            self.grids.append(spec.nkv_grid)
            self.parts.append(self.parts[2])
        self.lens = [len(p) for p in self.parts]
        self.G = sum(self.lens)
        self.K = len(self.names)

    def random(self, rng):
        return [[rng.randrange(len(g)) for _ in p] for g, p in zip(self.grids, self.parts)]

    def uniform_gene(self, vals):
        """A uniform config as a genome: the same grid index in every block."""
        return [[g.index(v)] * len(p) for g, v, p in zip(self.grids, vals, self.parts)]

    def to_config(self, gen):
        n = self.spec.n_layers
        out = {}
        for k, name in enumerate(self.names):
            per_layer = [None] * n
            for i, (lo, hi) in enumerate(self.parts[k]):
                for L in range(lo, hi):
                    per_layer[L] = self.grids[k][gen[k][i]]
            out[name] = per_layer
        nkv = out.get("n_kv", [self.spec.n_kv] * n)
        if self.with_nkv:
            # n_kv must divide n_h, and crossover and mutation both move the two genes
            # independently, so illegal pairs are produced constantly rather than rarely.
            # Repairing here rather than rejecting keeps the population size stable and keeps the
            # genome a plain integer vector -- rejection sampling on a coupled constraint biases
            # which n_h values survive, which is exactly the axis under study.
            # Repair n_h against n_kv, not the other way round. n_kv is the gene that moves the
            # cache and therefore the front; snapping it to fit a drawn n_h would quietly delete
            # the cheap-cache region the gene exists to reach. Snapping n_h instead keeps the cache
            # target and costs only head granularity.
            nh = [h if h % nkv[i] == 0 else
                  min(head_options(self.spec, nkv[i]) or [h], key=lambda x: abs(x - h))
                  for i, h in enumerate(out["n_h"])]
            out["n_h"] = nh
        return ElasticConfig(self.spec, out["d_qk"], out["d_v"],
                             nkv, out["n_h"], out["d_mlp"])

    def key(self, gen):
        return tuple(tuple(g) for g in gen)

    def crossover(self, a, b, rng):
        return [[a[k][i] if rng.random() < 0.5 else b[k][i] for i in range(self.lens[k])]
                for k in range(self.K)]

    def mutate(self, gen, rng, rate):
        out = [list(g) for g in gen]
        for k in range(self.K):
            for i in range(self.lens[k]):
                if rng.random() < rate:
                    out[k][i] = rng.randrange(len(self.grids[k]))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-4b", choices=sorted(SPECS))
    ap.add_argument("--ckpt", default=str(RUNS / "qwen3-4b_ab" / "step8000"))
    ap.add_argument("--pop", type=int, default=48)
    ap.add_argument("--gen", type=int, default=30)
    ap.add_argument("--n-texts", type=int, default=32, help="docs the search optimizes")
    ap.add_argument("--val-texts", type=int, default=64,
                    help="DISJOINT docs the final front is re-scored on. The search sees one "
                         "fixed set, so it can fit that set's noise; only a held-out re-score "
                         "shows whether the front is real.")
    ap.add_argument("--mut", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per", type=int, default=0,
                    help="Layers per gene. 0 = use the MEASURED sensitivity blocks (5 for attention "
                         "and 4 for MLP at 4B; a k-way equal split elsewhere, since no other model "
                         "has measured blocks). A small --per makes the space genuinely large -- at "
                         "30 layers, per=4 gives 8 groups x 4 knobs = 32 genes and ~1.8e18 "
                         "configurations, which is the regime where no lookup table can be built "
                         "and the supernet is not substitutable.")
    ap.add_argument("--with-nkv", action="store_true",
                    help="Search n_kv (KV-group count) as a fifth gene. Needs a supernet trained "
                         "with uptrain --with-nkv; on one that never saw it the slice scores are "
                         "meaningless. Qwen3 only -- SmolLM2's n_kv is prime.")
    ap.add_argument("--objectives", type=int, default=3, choices=[2, 3],
                    help="2 = (model size, loss) -- the plain size/quality trade-off. "
                         "3 = (model size, KV cache, loss). KV is a separate axis only because it "
                         "is what edge deployment actually binds on; with 2 it is unconstrained.")
    ap.add_argument("--random", action="store_true",
                    help="Random-search control: draw the SAME number of genomes the GA would "
                         "evaluate and take their non-dominated front. This is the control that "
                         "says whether the genetic operators earn their keep. A GA that converges "
                         "in a few generations is only impressive if random sampling at equal "
                         "budget does worse; if the two fronts coincide, fast convergence means "
                         "the landscape is easy, not that the search is good.")
    ap.add_argument("--skip-uniform", action="store_true")
    ap.add_argument("--seed-uniform", action="store_true",
                    help="initialize the population from the uniform Pareto front")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    spec = SPECS[a.model]
    rng = random.Random(a.seed)
    # last 10BT shard: training read 524M of 10B tokens from the FIRST shard, so this is unseen.
    # The validation slice skips past the search slice so the two never share a document.
    from ..data.heldout import heldout_texts
    texts = heldout_texts(a.n_texts)
    val_texts = heldout_texts(a.val_texts, skip=a.n_texts) if a.val_texts else []
    model, tok, order = load_supernet(a.ckpt, spec)
    loss_of, n_docs = make_evaluator(model, tok, order, spec, texts)
    S = Space(spec, per=a.per, blocks=(a.per == 0), with_nkv=a.with_nkv)
    full = ElasticConfig.full(spec)
    W0 = full.weight_params(include_embed=False)
    K0 = full.kv_bytes_per_token()

    t0 = time.time()
    base_loss = loss_of(full)
    dt = time.time() - t0
    print(f"[eval] {n_docs} docs, {dt:.1f}s per candidate | full-config loss {base_loss:.4f}",
          flush=True)
    budget = a.pop * (a.gen + 1)
    print(f"[plan] pop {a.pop} x gen {a.gen} = ~{budget} evals ~= {budget*dt/60:.0f} min",
          flush=True)

    cache = {}

    def objectives(gen):
        k = S.key(gen)
        if k not in cache:
            c = S.to_config(gen).validate()
            w_, l_ = c.weight_params(include_embed=False) / W0, loss_of(c)
            cache[k] = (w_, l_) if a.objectives == 2 else (w_, c.kv_bytes_per_token() / K0, l_)
        return cache[k]

    # ---- uniform reference sweep: every knob combination applied to every layer
    uniform = []
    if not a.skip_uniform:
        combos = [(q, v, h, m) for q in spec.qk_grid for v in spec.qk_grid
                  for h in head_grid(spec) for m in spec.mlp_grid]
        print(f"[uniform] evaluating {len(combos)} uniform configs "
              f"(~{len(combos)*dt/60:.0f} min)", flush=True)
        for n, (q, v, h, m) in enumerate(combos):
            c = ElasticConfig.uniform(spec, q, v, n_kv=spec.n_kv, n_h=h, d_mlp=m)
            uniform.append({"d_qk": q, "d_v": v, "n_h": h, "d_mlp": m, "n_kv": spec.n_kv,
                            "w": c.weight_params(include_embed=False) / W0,
                            "kv": c.kv_frac(), "loss": loss_of(c)})
            if (n + 1) % 32 == 0:
                print(f"  uniform {n+1}/{len(combos)} [{(time.time()-t0)/60:.1f}m]", flush=True)

    # ---- NSGA-II
    P = []
    if a.seed_uniform and uniform:
        # start from the uniform Pareto front: the GA can then only improve on uniform, which
        # turns "does heterogeneity help" into a question the run answers directly
        Uf = np.array([[u["w"], u["kv"], u["loss"]] for u in uniform])
        pick = fronts_of(Uf)[0][:a.pop]
        P = [S.uniform_gene(tuple(uniform[i][nm] if nm in uniform[i] else spec.n_kv
                                   for nm in S.names)) for i in pick]
        print(f"[seed] initialized with {len(P)} uniform Pareto-front configs", flush=True)
    if a.random:
        budget = a.pop * (a.gen + 1)
        print(f"[random] sampling {budget} genomes, same budget as pop {a.pop} x gen {a.gen}",
              flush=True)
        ref = (1.0, base_loss + 3.0) if a.objectives == 2 else (1.0, 1.0, base_loss + 3.0)
        hv_trace, pool = [], []
        for i in range(budget):
            pool.append(S.random(rng))
            if (i + 1) % a.pop == 0:
                Fp = np.array([objectives(x) for x in pool])
                f0 = fronts_of(Fp)[0]
                hv_trace.append({"gen": (i + 1) // a.pop - 1,
                                 "hv": hypervolume(np.array([Fp[j] for j in f0]), ref),
                                 "front": len(f0), "evals": len(cache),
                                 "best_loss": float(Fp[:, -1].min()),
                                 "minutes": (time.time() - t0) / 60})
                t = hv_trace[-1]
                print(f"  batch {t['gen']:2d}/{a.gen}  front {t['front']:3d}  HV {t['hv']:.5f}  "
                      f"best-loss {t['best_loss']:.4f} (ppl {np.exp(t['best_loss']):.2f})  "
                      f"evals {t['evals']}  [{t['minutes']:.1f}m]", flush=True)
        Fp = np.array([objectives(x) for x in pool])
        P = [pool[j] for j in fronts_of(Fp)[0]]
        front = P
        Fv = Fp
    else:
        P += [S.random(rng) for _ in range(a.pop - len(P))]
    # Reference point for the hypervolume: the worst corner the search could return. Fixed before
    # the run so the trace is comparable across generations AND across runs -- which is the whole
    # point, since the GA and the random control have to be read on the same axis.
    if not a.random:
        ref = (1.0, base_loss + 3.0) if a.objectives == 2 else (1.0, 1.0, base_loss + 3.0)
        hv_trace = []
    for g in ([] if a.random else range(a.gen + 1)):
        Fv = np.array([objectives(x) for x in P])
        rank, dist = rank_and_crowd(Fv)
        if g == a.gen:
            break

        def tourn():
            i, j = rng.randrange(len(P)), rng.randrange(len(P))
            return P[i] if (rank[i], -dist[i]) <= (rank[j], -dist[j]) else P[j]

        Q = [S.mutate(S.crossover(tourn(), tourn(), rng), rng, a.mut) for _ in range(a.pop)]
        R = P + Q
        Fr = np.array([objectives(x) for x in R])
        rk, ds = rank_and_crowd(Fr)
        keep = sorted(range(len(R)), key=lambda i: (rk[i], -ds[i]))[:a.pop]
        P = [R[i] for i in keep]
        f0 = [i for i in range(len(R)) if rk[i] == 0]
        hv = hypervolume(np.array([Fr[i] for i in f0]), ref)
        hv_trace.append({"gen": g, "hv": hv, "front": len(f0), "evals": len(cache),
                         "best_loss": float(Fr[:, -1].min()),
                         "minutes": (time.time() - t0) / 60})
        gain = "" if len(hv_trace) < 2 else f"  (+{100*(hv/max(hv_trace[-2]['hv'],1e-12)-1):5.2f}%)"
        print(f"  gen {g:2d}/{a.gen}  front {len(f0):3d}  HV {hv:.5f}{gain}  "
              f"best-loss {Fr[:, -1].min():.4f} (ppl {np.exp(Fr[:, -1].min()):.2f})  "
              f"evals {len(cache)}  [{(time.time()-t0)/60:.1f}m]", flush=True)

    if not a.random:
        Fv = np.array([objectives(x) for x in P])
        front = [P[i] for i in fronts_of(Fv)[0]]
    res = {"model": spec.key, "ckpt": a.ckpt, "n_objectives": a.objectives,
           "hv_trace": hv_trace, "hv_ref": list(ref), "n_docs": n_docs, "base_loss": base_loss,
           "full_weights": W0, "full_kv": K0, "evals": len(cache),
           "front": [], "uniform": uniform}
    for gen in front:
        c = S.to_config(gen)
        o = objectives(gen)
        w, kv, l = (o[0], None, o[1]) if a.objectives == 2 else o
        res["front"].append({"w": w, "kv": kv, "loss": l, "ppl": float(np.exp(l)), **c.to_dict()})
    res["front"].sort(key=lambda r: r["loss"])

    # ---- re-score on DISJOINT docs. If the front's advantage over uniform survives here it is
    # real; if it collapses, the search fit the noise of its own 32-doc slice.
    if val_texts:
        val_of, n_val = make_evaluator(model, tok, order, spec, val_texts)
        res["n_val"] = n_val
        print(f"\n[val] re-scoring {len(res['front'])} front + {len(uniform)} uniform "
              f"configs on {n_val} disjoint docs", flush=True)
        res["base_loss_val"] = val_of(full)
        for r in res["front"]:
            r["loss_val"] = val_of(ElasticConfig.from_dict(r))
        for r in uniform:
            r["loss_val"] = val_of(ElasticConfig.uniform(
                spec, r["d_qk"], r["d_v"], n_kv=spec.n_kv, n_h=r["n_h"], d_mlp=r["d_mlp"]))

    out = RUNS / (a.out or f"nsga_{spec.key}_ab.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print(f"\nfront size {len(front)} | {len(cache)} unique configs evaluated "
          f"in {(time.time()-t0)/60:.1f} min\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
