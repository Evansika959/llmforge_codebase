"""Do NLL and target rank separate the architecture set where HellaSwag cannot?

HellaSwag puts all 16 architectures inside 6.6 points (27.1-33.8) against a 0.46-point standard
error, while the within-cost-level difference to resolve is ~1.4 points. The limit is RESOLUTION,
not validity: an accuracy over 10k items is a coarse instrument.

NLL and target rank are computed at every token over tens of thousands of positions, so their
standard errors are far smaller. If their spread across the 16 architectures is large relative to
noise, they can carry the correlation that HellaSwag cannot.

Bootstrap over DOCUMENTS, not tokens -- tokens inside a document are correlated, so a token-level
standard error would be optimistic by roughly the square root of the document length.
"""
import argparse
import pathlib, json
import numpy as np
import torch
import torch.nn.functional as F

from llmforge.supernet.config import SPECS
from llmforge.supernet.elastic import build_pair_order, pair_order_for, set_elastic_config
from llmforge.supernet.paths import RUNS
from llmforge.supernet.search.nsga import load_supernet
from llmforge.supernet.space import ElasticConfig
from llmforge.paths import HELDOUT


@torch.no_grad()
def measure(model, order, cfg, ids):
    set_elastic_config(model, cfg.d_qk, cfg.d_v, order, n_h=cfg.n_h, d_mlp=cfg.d_mlp)
    per_doc_nll, per_doc_rank, per_doc_top1 = [], [], []
    for i in ids:
        lg = model(input_ids=i).logits[:, :-1].float()
        t = i[:, 1:]
        nll = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), t.reshape(-1)).item()
        tl = lg.gather(-1, t.unsqueeze(-1))
        r = (lg > tl).sum(-1) + 1
        per_doc_nll.append(nll)
        per_doc_rank.append(r.float().mean().item())
        per_doc_top1.append((r == 1).float().mean().item())
    return np.array(per_doc_nll), np.array(per_doc_rank), np.array(per_doc_top1)


def boot_se(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    return float(np.std([rng.choice(x, len(x), replace=True).mean() for _ in range(n)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-1.7b")
    ap.add_argument("--ckpt", default=f"{RUNS}/qwen3-1.7b_ab/final")
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--texts", default=f"{HELDOUT}/heldout_texts.json",
                    help="Held-out document file. assets/heldout/heldout_ab.json is the 192-document A+B "
                         "union and gives ~115 windows instead of ~58, cutting the paired "
                         "between-architecture SE about 1.45x. This flag did not exist until "
                         "2026-09-11, so --n above 96 was silently capped and several sweeps were "
                         "scored on 58 windows (57 under Qwen3's tokenizer) while being reported "
                         "as 115. Both sides of any comparison must pass the same file, and CE is "
                         "NOT comparable across tokenizers at all -- never compare a SmolLM2 "
                         "number to a Qwen3 one.")
    ap.add_argument("--out", default=None, help="separate file per supernet version")
    ap.add_argument("--ctx", type=int, default=768,
                    help="evaluation context length. Half the search space is KV cache, which only "
                         "binds at long context, yet everything so far was measured at <=1024 "
                         "tokens -- the regime where attention damage barely shows. Documents are "
                         "concatenated to reach longer lengths, keeping the held-out split.")
    ap.add_argument("--archs", default=None,
                    help="JSON list of {name,d_qk,n_h,d_mlp} to score -- e.g. runs/supernet/gt135/ARCHS.json, "
                         "the frozen 48-shape grid the ground truth trains. Scoring the SAME set "
                         "the truth covers is what makes a rank-fidelity number possible; the six "
                         "corner probes cannot produce one.")
    ap.add_argument("--probes", action="store_true",
                    help="Score the six NAMED probes (full, qk-lo, v-lo, h-lo, mlp-lo, min) instead "
                         "of the architecture set. These are the configs uptrain prints as [eval], "
                         "but that line is scored on two packed sequences drawn from the TRAINING "
                         "pool -- 12 documents, 8192 tokens, one bucket. This path scores the same "
                         "configs on the real held-out split with a document bootstrap.")
    ap.add_argument("--shapes", type=int, default=4,
                    help="shapes per cost band; widening only APPENDS, L*_a..d are unchanged")
    a = ap.parse_args()
    spec = SPECS[a.model]
    order = pair_order_for(spec)
    import sys
    sys.argv = [sys.argv[0]]
    if a.archs:
        archs = {r["name"]: r for r in json.load(open(a.archs))}
    elif a.probes:
        from llmforge.supernet.elastic.sampler import head_grid
        hi, lo = spec.qk_grid[-1], spec.qk_grid[0]
        hg, mg = head_grid(spec), spec.mlp_grid
        archs = {"qk-lo":  dict(d_qk=lo, d_v=hi, n_h=spec.n_q, d_mlp=spec.d_mlp),
                 "v-lo":   dict(d_qk=hi, d_v=lo, n_h=spec.n_q, d_mlp=spec.d_mlp),
                 "h-lo":   dict(d_qk=hi, d_v=hi, n_h=hg[0],    d_mlp=spec.d_mlp),
                 "mlp-lo": dict(d_qk=hi, d_v=hi, n_h=spec.n_q, d_mlp=mg[0]),
                 "min":    dict(d_qk=lo, d_v=lo, n_h=hg[0],    d_mlp=mg[0])}
    else:
        from llmforge.supernet.eval.archsets import arch_set
        archs = arch_set(spec, k=a.shapes)

    def heldout(n):
        import os
        if a.texts and os.path.exists(a.texts):
            return json.load(open(a.texts))[:n]
        from llmforge.supernet.data.heldout import heldout_texts as h
        return h(n)

    model, tok, _ = load_supernet(a.ckpt, spec)
    model.config.use_cache = False
    txts = heldout(a.n)
    if a.ctx <= 768:
        ids = [tok(t, return_tensors="pt", truncation=True, max_length=a.ctx).input_ids.to("cuda")
               for t in txts]
    else:
        # Concatenate held-out documents into full-length windows. Document boundaries inside a
        # window are what makes long context non-trivial for attention -- a single padded document
        # would leave the extra positions empty and measure nothing.
        flat = []
        for t in txts:
            flat += tok(t, add_special_tokens=False).input_ids
        n_win = max(1, len(flat) // a.ctx)
        ids = [torch.tensor(flat[i * a.ctx:(i + 1) * a.ctx], device="cuda").unsqueeze(0)
               for i in range(n_win)]
        print(f"[ctx] {len(flat):,} tokens -> {n_win} windows of {a.ctx}", flush=True)
    ntok = sum(i.shape[1] - 1 for i in ids)
    print(f"{len(ids)} docs, {ntok} tokens\n")

    res, per_window = {}, {}
    print(f"{'arch':6} {'W':>6} {'lvl':>4} {'NLL':>8} {'±se':>6} {'meanRank':>9} {'±se':>7} {'top1':>7}")
    for name, A in [("full", None)] + list(archs.items()):
        c = (ElasticConfig.full(spec) if A is None else
             ElasticConfig.uniform(spec, A["d_qk"], A.get("d_v", A["d_qk"]),
                                   n_h=A["n_h"], d_mlp=A["d_mlp"]))   # probes vary d_v
        nll, rk, t1 = measure(model, order, c, ids)
        per_window[name] = [float(x) for x in nll]   # paired bootstrap needs the units, not just SE
        W = c.weight_params(include_embed=False) / \
            ElasticConfig.full(spec).weight_params(include_embed=False)
        res[name] = {"w": W, "nll": nll.mean(), "nll_se": boot_se(nll),
                     "rank": rk.mean(), "rank_se": boot_se(rk), "top1": t1.mean()}
        L = "-" if A is None else name[1]
        print(f"{name:6} {W:6.3f} {L:>4} {nll.mean():8.4f} {res[name]['nll_se']:6.4f} "
              f"{rk.mean():9.2f} {res[name]['rank_se']:7.2f} {t1.mean():7.4f}", flush=True)

    print("\n--- resolution: within-band spread vs noise ---")
    for met, se_key in (("nll", "nll_se"), ("rank", "rank_se")):
        print(f"\n{met}:")
        tot = []
        for L in "1234":
            v = [res[k][met] for k in res if k != "full" and k[1] == L]
            if len(v) < 2:
                continue
            spread = max(v) - min(v)
            se = np.mean([res[k][se_key] for k in res if k != "full" and k[1] == L])
            tot.append(spread / (se * 1.41) if se else 0)
            print(f"  L{L}  within-band spread {spread:8.4f}   mean se {se:7.4f}   "
                  f"= {spread/(se*1.41):5.1f} resolvable steps")
        print(f"  mean {np.mean(tot):.1f} steps  ->  "
              f"{'usable' if np.mean(tot) > 3 else 'insufficient resolution'}")
    out_path = pathlib.Path(a.out) if a.out else (RUNS / f"slice_nll_{spec.key}.json")
    json.dump({k: {kk: float(vv) for kk, vv in v.items()} for k, v in res.items()},
              open(out_path, "w"), indent=1)
    # Per-window NLLs, so a difference between two runs can be bootstrapped PAIRED. Every run
    # scores the same windows in the same order with the same seed, so the unpaired hypot(se_i,
    # se_j) overstates the uncertainty of a difference -- sometimes by enough to manufacture a tie.
    win_path = out_path.with_name(out_path.stem + "_windows.json")
    json.dump({"ctx": a.ctx, "n_windows": len(ids), "nll": per_window}, open(win_path, "w"))
    print(f"wrote {out_path} and {win_path} ({len(ids)} windows)", flush=True)


if __name__ == "__main__":
    main()
