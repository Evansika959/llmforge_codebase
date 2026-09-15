"""Why the per-layer search found nothing: supernet training flattens per-layer sensitivity.

Shrink ONE layer to the floor, leave the other 35 at full width, measure the loss it costs.
Do it on the trained supernet and on the untrained base, and compare the SPREAD.

A per-layer search can only win if layers differ in how much they cost to shrink. On the base
checkpoint they differ enormously (71x). If sandwich training -- which samples per-layer widths
uniformly at random -- teaches every layer to tolerate shrinking equally, that spread collapses
and uniform allocation becomes near-optimal by construction. This measures whether that happened.

Compare CV, not the raw range: the base is not trained for elasticity so its ABSOLUTE damage is
much larger at every layer, and only the relative spread is comparable between the two.

  python experiments/supernet_fidelity/layer_sensitivity.py --ckpt runs/supernet/qwen3-4b_ab/step8000
"""
import argparse
import json
import statistics as st

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmforge.supernet.config import SPECS
from llmforge.supernet.elastic import build_pair_order, enable_elastic, set_elastic_backend
from llmforge.supernet.data.heldout import heldout_texts
from llmforge.supernet.paths import RUNS
from llmforge.supernet.search.nsga import load_supernet, make_evaluator
from llmforge.supernet.space import ElasticConfig


def sensitivity(model, tok, order, spec, texts, tag):
    loss_of, _ = make_evaluator(model, tok, order, spec, texts)
    base = loss_of(ElasticConfig.full(spec))
    out = []
    for L in range(spec.n_layers):
        qk = [spec.head_dim] * spec.n_layers
        v = list(qk)
        qk[L] = v[L] = spec.qk_grid[0]
        c = ElasticConfig(spec, qk, v, [spec.n_kv] * spec.n_layers,
                          [spec.n_q] * spec.n_layers, [spec.d_mlp] * spec.n_layers)
        out.append(loss_of(c) - base)
    lo, hi = min(out), max(out)
    cv = st.pstdev(out) / st.mean(out)
    print(f"{tag:22} base {base:.4f} | damage {lo:.4f}..{hi:.4f} spread {hi/max(lo,1e-9):5.1f}x "
          f"CV {cv:.3f} median {st.median(out):.4f}", flush=True)
    print(f"    worst layers: {sorted(range(len(out)), key=lambda i: -out[i])[:5]}", flush=True)
    return {"tag": tag, "base_loss": base, "damage": out, "spread": hi / max(lo, 1e-9), "cv": cv}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-4b", choices=sorted(SPECS))
    ap.add_argument("--ckpt", default=f"{RUNS}/qwen3-4b_ab/step8000")
    ap.add_argument("--n-texts", type=int, default=24)
    a = ap.parse_args()
    spec = SPECS[a.model]
    texts = heldout_texts(a.n_texts)

    m, t, o = load_supernet(a.ckpt, spec)
    tr = sensitivity(m, t, o, spec, texts, "TRAINED supernet")
    del m
    torch.cuda.empty_cache()

    mb = AutoModelForCausalLM.from_pretrained(
        spec.repo, torch_dtype=torch.bfloat16, attn_implementation="eager").to("cuda").eval()
    enable_elastic(mb)
    set_elastic_backend("eager")
    mb.config.use_cache = False
    ba = sensitivity(mb, AutoTokenizer.from_pretrained(spec.repo), build_pair_order("hilo"),
                     spec, texts, "UNTRAINED base")

    r = np.corrcoef(tr["damage"], ba["damage"])[0, 1]
    print(f"\ncorrelation trained vs base: {r:+.3f}  "
          f"(ordering survives, magnitudes compress {ba['cv']/tr['cv']:.1f}x)")
    out = RUNS / f"layer_sensitivity_{spec.key}.json"
    json.dump({"trained": tr, "base": ba, "corr": r, "ckpt": a.ckpt, "n_texts": a.n_texts},
              open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
