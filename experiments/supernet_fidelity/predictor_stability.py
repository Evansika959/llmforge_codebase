"""Is the predictor STABLE? A search calls it thousands of times, so an unstable ranking is useless
even if it is accurate on average.

Three things a usable predictor has to survive, none of which has been tested:

  eval set     Score the same architectures on two DISJOINT sets of held-out documents. If the
               ranking moves, search results are not reproducible.
  checkpoint   Score them with the supernet at step 4000 and at step 5000. If 1000 more steps of
               supernet training reorders the architectures, the predictor is measuring the
               checkpoint rather than the architecture.
  resolution   Count how many of the 120 architecture PAIRS are separated by more than noise.
               Spread across a whole cost level says nothing about whether two neighbours can be
               told apart, and that pair count is the real ceiling on how fine a search space
               this predictor can support.

Standard errors are bootstrapped over DOCUMENTS: tokens within a document are correlated, so a
token-level error bar would be optimistic by roughly sqrt(document length).
"""
import argparse, itertools, json
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr, kendalltau

from llmforge.supernet.config import SPECS
from llmforge.supernet.elastic import build_pair_order, set_elastic_config
from llmforge.supernet.paths import RUNS
from llmforge.supernet.search.nsga import load_supernet
from llmforge.supernet.space import ElasticConfig
from llmforge.supernet.data.heldout import heldout_texts


@torch.no_grad()
def per_doc_nll(model, order, cfg, ids):
    set_elastic_config(model, cfg.d_qk, cfg.d_v, order, n_h=cfg.n_h, d_mlp=cfg.d_mlp)
    out = []
    for i in ids:
        lg = model(input_ids=i).logits[:, :-1].float()
        out.append(F.cross_entropy(lg.reshape(-1, lg.shape[-1]),
                                   i[:, 1:].reshape(-1)).item())
    return np.array(out)


def boot_se(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    return float(np.std([rng.choice(x, len(x), replace=True).mean() for _ in range(n)]))


def score_all(ckpt, spec, order, texts, archs, tok=None):
    model, tok, _ = load_supernet(ckpt, spec)
    model.config.use_cache = False
    ids = [tok(t, return_tensors="pt", truncation=True, max_length=768).input_ids.to("cuda")
           for t in texts]
    out = {}
    for name, A in archs.items():
        c = ElasticConfig.uniform(spec, A["d_qk"], A["d_qk"], n_h=A["n_h"], d_mlp=A["d_mlp"])
        d = per_doc_nll(model, order, c, ids)
        out[name] = {"nll": float(d.mean()), "se": boot_se(d), "w": A["w"]}
    del model
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-1.7b")
    ap.add_argument("--ckpt-a", default=f"{RUNS}/qwen3-1.7b_ab/final")
    ap.add_argument("--ckpt-b", default=f"{RUNS}/qwen3-1.7b_ab/step4000")
    a = ap.parse_args()
    spec = SPECS[a.model]
    order = build_pair_order("hilo")
    import sys
    sys.argv = [sys.argv[0]]
    from llmforge.supernet.eval.archsets import arch_set
    archs = arch_set(spec)
    # Sets A and B are the two disjoint halves of assets/heldout/heldout_ab.json.
    two = {"A": heldout_texts(96), "B": heldout_texts(96, skip=96)}

    print("scoring: ckpt=final set=A"); A = score_all(a.ckpt_a, spec, order, two["A"], archs)
    print("scoring: ckpt=final set=B"); B = score_all(a.ckpt_a, spec, order, two["B"], archs)
    print("scoring: ckpt=step4000 set=A"); C = score_all(a.ckpt_b, spec, order, two["A"], archs)

    K = list(archs)
    va, vb, vc = ([-x[k]["nll"] for k in K] for x in (A, B, C))
    lvl = np.array([int(k[1]) for k in K])

    print(f"\n=== 1. eval set: A vs B (disjoint documents) ===")
    print(f"  Spearman {spearmanr(va, vb).correlation:+.4f}   "
          f"Kendall {kendalltau(va, vb).correlation:+.4f}")
    for L in sorted(set(lvl)):
        m = lvl == L
        print(f"    within L{L}: Spearman {spearmanr(np.array(va)[m], np.array(vb)[m]).correlation:+.3f}")

    print(f"\n=== 2. checkpoint: final vs step4000 ===")
    print(f"  Spearman {spearmanr(va, vc).correlation:+.4f}   "
          f"Kendall {kendalltau(va, vc).correlation:+.4f}")
    for L in sorted(set(lvl)):
        m = lvl == L
        print(f"    within L{L}: Spearman {spearmanr(np.array(va)[m], np.array(vc)[m]).correlation:+.3f}")

    print(f"\n=== 3. pair resolution (how fine a distinction survives noise) ===")
    res = tot = 0
    same_lvl_res = same_lvl_tot = 0
    for i, j in itertools.combinations(range(len(K)), 2):
        d = abs(A[K[i]]["nll"] - A[K[j]]["nll"])
        s = np.hypot(A[K[i]]["se"], A[K[j]]["se"])
        ok = d > 1.96 * s
        tot += 1; res += ok
        if lvl[i] == lvl[j]:
            same_lvl_tot += 1; same_lvl_res += ok
    print(f"  all pairs        {res}/{tot} resolvable at 95%  ({res/tot:.0%})")
    print(f"  same cost level  {same_lvl_res}/{same_lvl_tot}  ({same_lvl_res/same_lvl_tot:.0%})"
          f"   <- the ones a search must tell apart")
    json.dump({"A": A, "B": B, "C": C}, open(RUNS / "predictor_stability.json", "w"), indent=1)


if __name__ == "__main__":
    main()
