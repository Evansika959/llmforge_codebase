#!/usr/bin/env python
"""The 135M ground-truth architecture set: the ENTIRE uniform grid, 48 shapes.

Training all of it rather than a selection is affordable here -- 2000 steps is 40 min at this scale,
so 48 shapes is 32 GPU-hours -- and it removes the selection question completely. Every downstream
question is then answerable from one artefact: the calibration regression (which wants spread), all
22 parameter-tied pairs, all 6 distinct (d_qk, n_h) contrasts, and any free-rule baseline someone
thinks of later. `arch_set` in groundtruth.py cannot supply this: its heuristic spreads shapes apart
within a cost level, which is precisely why the Qwen3 set contains zero parameter-tied pairs.

Order is deterministic and frozen into runs/supernet/gt135/ARCHS.json on first use, so shards can be split
by index across machines and a rerun cannot silently renumber anything.

  python experiments/supernet_fidelity/gt135_set.py --list
  python experiments/supernet_fidelity/gt135_set.py --freeze
  python experiments/supernet_fidelity/gt135_set.py --shard 0 --of 3     # names this machine should train
"""
import argparse, json, os, sys

from llmforge.supernet.config import SPECS
from llmforge.supernet.elastic.sampler import head_grid
from llmforge.supernet.space import ElasticConfig
from llmforge.supernet.paths import RUNS

OUT = f"{RUNS}/gt135"


def build(model="smollm2-135m"):
    spec = SPECS[model]
    W0 = ElasticConfig.full(spec).weight_params(include_embed=False)
    rows = []
    for q in spec.qk_grid:                       # 16 32 48 64
        for h in head_grid(spec):                # 3 6 9
            for m in spec.mlp_grid:              # 384 768 1152 1536
                c = ElasticConfig.uniform(spec, q, q, n_h=h, d_mlp=m)
                rows.append({"name": f"q{q}h{h}m{m}", "d_qk": q, "n_h": h, "d_mlp": m,
                             "w": c.weight_params(include_embed=False) / W0,
                             "params": c.weight_params(), "kv": c.kv_frac(),
                             "attn_width": h * q})
    return sorted(rows, key=lambda r: (r["w"], r["name"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="smollm2-135m")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--of", type=int, default=1)
    a = ap.parse_args()
    rows = build(a.model)
    if a.freeze:
        os.makedirs(OUT, exist_ok=True)
        json.dump(rows, open(os.path.join(OUT, "ARCHS.json"), "w"), indent=1)
        print(f"froze {len(rows)} architectures -> {OUT}/ARCHS.json")
        return
    if a.shard is not None:
        for i, r in enumerate(rows):
            if i % a.of == a.shard:
                print(f"{r['name']} {r['d_qk']},{r['n_h']},{r['d_mlp']}")
        return
    print(f"{'name':14} {'W':>6} {'params':>10} {'KV':>5} {'n_h*d_qk':>9}")
    for r in rows:
        print(f"{r['name']:14} {r['w']:6.3f} {r['params']/1e6:9.1f}M {r['kv']:5.2f} {r['attn_width']:9d}")
    print(f"\n{len(rows)} architectures")


if __name__ == "__main__":
    main()
