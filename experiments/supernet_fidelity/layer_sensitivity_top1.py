"""Per-layer sensitivity measured two ways, and data-driven block boundaries.

The existing per-layer sensitivity is measured in cross-entropy. We have since shown that CE gives
the WRONG knob ranking against capability (it says trim attention first; capability says trim MLP
first), so a block partition derived from CE may be optimising the wrong thing.

top-1 disagreement with the full model is the better candidate: it is temperature-invariant, it is
the quantity that COMPOUNDS over a generated sequence (0.694^200 ~ 1e-32 is why w75 has no
mathematics), and it costs exactly one forward pass -- the same as CE.

Measures both, compares the rankings, then partitions the layers into contiguous blocks by
dynamic programming (minimum within-block variance) so the training and search granularity can be
set from evidence instead of by cutting every 4 layers.
"""
import argparse, json
import numpy as np
import torch

from llmforge.supernet.config import SPECS
from llmforge.supernet.elastic import build_pair_order, set_elastic_config
from llmforge.supernet.data.heldout import heldout_texts
from llmforge.supernet.paths import RUNS
from llmforge.supernet.search.nsga import load_supernet
from llmforge.supernet.space import ElasticConfig


def best_partition(x, K):
    """Optimal contiguous split of x into K blocks minimising total within-block variance (DP)."""
    n = len(x)
    cost = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            seg = x[i:j + 1]
            cost[i, j] = seg.var() * len(seg)
    D = np.full((K + 1, n + 1), np.inf); D[0, 0] = 0
    B = np.zeros((K + 1, n + 1), dtype=int)
    for k in range(1, K + 1):
        for j in range(1, n + 1):
            for i in range(k - 1, j):
                c = D[k - 1, i] + cost[i, j - 1]
                if c < D[k, j]: D[k, j], B[k, j] = c, i
    bounds, j = [], n
    for k in range(K, 0, -1):
        bounds.append((B[k, j], j)); j = B[k, j]
    return list(reversed(bounds))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-4b")
    ap.add_argument("--ckpt", default=f"{RUNS}/qwen3-4b_ab/final")
    ap.add_argument("--n", type=int, default=32)
    a = ap.parse_args()
    spec = SPECS[a.model]
    order = build_pair_order("hilo")
    model, tok, _ = load_supernet(a.ckpt, spec)
    model.config.use_cache = False
    ids = [tok(t, return_tensors="pt", truncation=True, max_length=768).input_ids.to("cuda")
           for t in heldout_texts(a.n)]

    def run(cfg):
        set_elastic_config(model, cfg.d_qk, cfg.d_v, order, n_h=cfg.n_h, d_mlp=cfg.d_mlp)
        ce = ntok = 0.0; am = []
        for i in ids:
            lg = model(input_ids=i).logits[:, :-1].float()
            t = i[:, 1:]
            ce += torch.nn.functional.cross_entropy(
                lg.reshape(-1, lg.shape[-1]), t.reshape(-1), reduction="sum").item()
            ntok += t.numel(); am.append(lg.argmax(-1).cpu())
        return ce / ntok, am

    full = ElasticConfig.full(spec)
    ce0, am0 = run(full)
    print(f"[full] CE {ce0:.4f}\n", flush=True)

    res = {}
    for kind in ("attn", "mlp"):
        dce, dt1 = [], []
        for L in range(spec.n_layers):
            qk = [spec.head_dim] * spec.n_layers; v = list(qk)
            mlp = [spec.d_mlp] * spec.n_layers
            if kind == "attn": qk[L] = v[L] = spec.qk_grid[0]
            else:              mlp[L] = spec.mlp_grid[0]
            c = ElasticConfig(spec, qk, v, [spec.n_kv] * spec.n_layers,
                              [spec.n_q] * spec.n_layers, mlp)
            ce, am = run(c)
            dis = float(np.mean([(x != y).float().mean().item() for x, y in zip(am, am0)]))
            dce.append(ce - ce0); dt1.append(dis)
        from scipy.stats import spearmanr
        rho = spearmanr(dce, dt1).correlation
        res[kind] = {"d_ce": dce, "d_top1": dt1, "spearman": float(rho)}
        print(f"=== {kind} ===  Spearman(CE-rank, top1-rank) = {rho:+.3f}", flush=True)
        for nm, arr in (("CE      ", dce), ("top1-dis", dt1)):
            o = sorted(range(len(arr)), key=lambda i: -arr[i])
            print(f"  {nm} most sensitive {o[:6]}   least {o[-5:]}", flush=True)
        print(f"  CV: CE {np.std(dce)/np.mean(dce):.3f}   "
              f"top1 {np.std(dt1)/np.mean(dt1):.3f}", flush=True)

    print("\n=== data-driven contiguous blocks, from top-1 disagreement (attn) ===", flush=True)
    x = np.array(res["attn"]["d_top1"])
    for K in (3, 4, 5, 6):
        bnd = best_partition(x, K)
        segs = [f"{i}-{j-1}" for i, j in bnd]
        means = [f"{x[i:j].mean():.4f}" for i, j in bnd]
        print(f"  K={K}: " + "  ".join(f"[{s}] {m}" for s, m in zip(segs, means)), flush=True)
        res[f"blocks_{K}"] = [[int(i), int(j)] for i, j in bnd]
    json.dump(res, open(RUNS / "layer_sensitivity_top1.json", "w"), indent=1)
    print(f"\nwrote {RUNS/'layer_sensitivity_top1.json'}")


if __name__ == "__main__":
    main()
