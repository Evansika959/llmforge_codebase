"""What each conversion rule is worth before any training.

The published checkpoint is the untrained supernet: the attention temperature and the KV alignment angles start
at zero, so its full-width slice is the base model. This script slices it under the conversion rules of
docs/supernet.md and under a naive counterpart of each rule, and scores every slice on the same held-out
documents the searches use for their final fronts.

    rotary order    d_qk keeps rotary pairs in the order that interleaves the highest and lowest frequencies,
                    against a plain prefix of the highest frequencies and against random pair orders
    head selection  n_h keeps n_h / n_kv query heads in every KV group, against a plain prefix of query heads
                    that still read their own groups, which leaves the later groups unread
    KV reduction    n_kv mean-pools bundles of KV groups, against keeping the first group of every bundle

The temperature and the alignment angles cannot be tested this way, because both are the identity until trained.
A last pass scores every uniform shape once under all the rules and once under all the naive counterparts. Below
full n_kv the naive pass keeps balanced head selection, because a head prefix needs every group.

    PYTHONPATH=src python experiments/supernet_fidelity/conversion_ablation.py --model smollm2-135m
"""
import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmforge.supernet.config import SPECS
from llmforge.supernet.data.heldout import heldout_texts
from llmforge.supernet.elastic import (add_elastic_temperature, add_kv_alignment, build_pair_order,
                                       enable_elastic, pair_order_for, set_elastic_backend, set_elastic_config)
from llmforge.supernet.elastic import attention as elastic_attention

ROOT = Path(__file__).resolve().parents[2]


def load_untrained(spec, dev):
    model = AutoModelForCausalLM.from_pretrained(spec.repo, torch_dtype=torch.bfloat16,
                                                 attn_implementation="eager").to(dev).eval()
    enable_elastic(model)
    add_kv_alignment(model, spec=spec)
    set_elastic_backend("eager")
    add_elastic_temperature(model)
    model.config.use_cache = False
    return model, AutoTokenizer.from_pretrained(spec.repo)


def make_scorer(model, tok, texts, dev, max_len):
    ids = [tok(t, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(dev) for t in texts]
    ids = [i for i in ids if i.shape[1] >= 8]

    @torch.no_grad()
    def score(qk, v, order, n_h=None, n_kv=None, d_mlp=None, heads="balanced", kv="pool"):
        elastic_attention.ABLATION.update(heads=heads, kv=kv)
        try:
            set_elastic_config(model, qk, v, order, n_h=n_h, d_mlp=d_mlp, n_kv=n_kv)
            tot = ntok = 0.0
            for i in ids:
                logits = model(input_ids=i).logits[:, :-1].float()
                target = i[:, 1:]
                tot += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
                                       reduction="sum").item()
                ntok += target.numel()
        finally:
            elastic_attention.ABLATION.update(heads="balanced", kv="pool")
        return tot / ntok

    return score, len(ids)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="smollm2-135m")
    ap.add_argument("--n-docs", type=int, default=64)
    ap.add_argument("--skip-docs", type=int, default=32, help="documents 0-31 are the search documents")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--random-orders", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-grid", action="store_true", help="skip the pass over every uniform shape")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    spec = SPECS[args.model]
    dev = "cuda"
    out_path = Path(args.out or ROOT / "runs" / "supernet_fidelity" / f"conversion_ablation_{args.model}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model, tok = load_untrained(spec, dev)
    score, n = make_scorer(model, tok, heldout_texts(args.n_docs, skip=args.skip_docs), dev, args.max_len)
    if n != args.n_docs:
        raise SystemExit(f"expected {args.n_docs} scorable documents, got {n}")

    qk_full, h_full, kv_full = spec.qk_grid[-1], spec.n_q, spec.n_kv
    hilo = pair_order_for(spec)
    natural = build_pair_order("natural", spec.head_dim // 2)
    rng = random.Random(args.seed)
    randoms = [rng.sample(range(spec.head_dim // 2), spec.head_dim // 2) for _ in range(args.random_orders)]
    res = dict(model=args.model, docs=[args.skip_docs, args.skip_docs + args.n_docs], max_len=args.max_len,
               seed=args.seed, grids=dict(qk=spec.qk_grid, n_h=spec.nh_grid, n_kv=spec.nkv_grid, d_mlp=spec.mlp_grid))
    t0 = time.time()

    def save():
        out_path.write_text(json.dumps(res, indent=1))

    res["full"] = score(qk_full, qk_full, hilo)
    print(f"full width {res['full']:.4f}", flush=True)

    res["rotary_order"] = []
    for qk in spec.qk_grid[:-1]:
        rec = dict(d_qk=qk, hilo=score(qk, qk_full, hilo), natural=score(qk, qk_full, natural),
                   random=[score(qk, qk_full, o) for o in randoms])
        res["rotary_order"].append(rec)
        print(f"d_qk {qk}: interleaved {rec['hilo']:.4f}, highest-frequency prefix {rec['natural']:.4f}, random "
              f"median {statistics.median(rec['random']):.4f} best {min(rec['random']):.4f}", flush=True)
    save()

    res["head_selection"] = []
    for h in [x for x in spec.nh_grid if x < h_full and x % kv_full == 0]:
        rec = dict(n_h=h, balanced=score(qk_full, qk_full, hilo, n_h=h),
                   prefix=score(qk_full, qk_full, hilo, n_h=h, heads="prefix"))
        res["head_selection"].append(rec)
        print(f"n_h {h}: balanced {rec['balanced']:.4f}, prefix {rec['prefix']:.4f}", flush=True)
    save()

    res["kv_reduction"] = []
    for k in [x for x in spec.nkv_grid if x < kv_full]:
        for h in [x for x in spec.nh_grid if x % k == 0]:
            rec = dict(n_kv=k, n_h=h, pool=score(qk_full, qk_full, hilo, n_h=h, n_kv=k),
                       select=score(qk_full, qk_full, hilo, n_h=h, n_kv=k, kv="select"))
            res["kv_reduction"].append(rec)
            print(f"n_kv {k}, n_h {h}: mean pooling {rec['pool']:.4f}, first group {rec['select']:.4f}", flush=True)
    save()

    if not args.no_grid:
        res["grid"] = []
        shapes = [(k, h, qk, v, m) for k in spec.nkv_grid for h in spec.nh_grid if h % k == 0
                  for qk in spec.qk_grid for v in spec.qk_grid for m in spec.mlp_grid]
        for i, (k, h, qk, v, m) in enumerate(shapes):
            ours = score(qk, v, hilo, n_h=h, n_kv=k, d_mlp=m)
            naive = score(qk, v, natural, n_h=h, n_kv=k, d_mlp=m,
                          heads="prefix" if (k == kv_full and h < h_full) else "balanced", kv="select")
            res["grid"].append(dict(n_kv=k, n_h=h, d_qk=qk, d_v=v, d_mlp=m, ours=ours, naive=naive))
            if (i + 1) % 48 == 0:
                print(f"grid {i + 1}/{len(shapes)} after {time.time() - t0:.0f} s", flush=True)
                save()
        ours = [r["ours"] for r in res["grid"]]
        naive = [r["naive"] for r in res["grid"]]
        gaps = [b - a for a, b in zip(ours, naive)]
        res["grid_summary"] = dict(shapes=len(ours), median_gap=statistics.median(gaps),
                                   ours_lower=sum(g > 0 for g in gaps), spearman=spearmanr(ours, naive).correlation)
        print(f"grid: naive minus ours median {res['grid_summary']['median_gap']:.4f} nats, ours lower on "
              f"{res['grid_summary']['ours_lower']}/{len(ours)} shapes, Spearman {res['grid_summary']['spearman']:.3f}",
              flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    save()
    print(f"wrote {out_path} in {res['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
