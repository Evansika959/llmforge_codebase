#!/usr/bin/env python
"""Is there any per-layer structure for a heterogeneous search to exploit?

A genetic search over per-layer widths can only beat a uniform grid if layers DIFFER in how much
they cost when narrowed. If sandwich training has flattened that, every layer group is
interchangeable, the nominal 1.8e18 space collapses to roughly one value per knob, and a GA will
converge almost immediately -- correctly, because there is nothing to find. That is a property of
the supernet, not a defect of the search, and the two look identical from a convergence curve.

At 4B this is exactly what happened: sandwich training compressed per-layer sensitivity 6x
(CV 2.580 -> 0.429) and the NSGA front did not beat uniform. This measures it at 135M, for the
trained supernet and the untrained base on the same documents, so the compression itself is visible.

Reports the coefficient of variation of per-group damage. High CV = heterogeneity has purchase.
"""
import json, sys
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file
from llmforge.supernet.config import SPECS
from llmforge.supernet.elastic import (pair_order_for, enable_elastic, set_elastic_backend,
                                set_elastic_config, add_elastic_temperature)
from llmforge.supernet.elastic.sampler import head_grid
from llmforge.supernet.space import ElasticConfig
from llmforge.supernet.paths import RUNS
from llmforge.paths import HELDOUT

spec = SPECS["smollm2-135m"]; order = pair_order_for(spec)
tok = AutoTokenizer.from_pretrained(spec.repo)
texts = json.load(open(f"{HELDOUT}/heldout_texts.json"))[:48]
flat = []
for t in texts:
    flat += tok(t, add_special_tokens=False).input_ids
ids = [torch.tensor(flat[i * 1024:(i + 1) * 1024], device="cuda").unsqueeze(0)
       for i in range(len(flat) // 1024)]
PER = 4
groups = [(i, min(i + PER, spec.n_layers)) for i in range(0, spec.n_layers, PER)]

def run(ckpt, label):
    m = AutoModelForCausalLM.from_pretrained(spec.repo, torch_dtype=torch.bfloat16,
                                             attn_implementation="eager").to("cuda").eval()
    enable_elastic(m); set_elastic_backend("eager"); m.config.use_cache = False
    if ckpt:
        add_elastic_temperature(m)
        m.load_state_dict({k: v.to("cuda") for k, v in load_file(ckpt).items()}, strict=False)
    def loss(cfg):
        set_elastic_config(m, cfg.d_qk, cfg.d_v, order, n_h=cfg.n_h, d_mlp=cfg.d_mlp)
        s = n = 0
        with torch.no_grad():
            for x in ids:
                lg = m(input_ids=x).logits[:, :-1].float()
                s += F.cross_entropy(lg.reshape(-1, lg.shape[-1]), x[:, 1:].reshape(-1),
                                     reduction="sum").item(); n += x.shape[1] - 1
        return s / n
    full = ElasticConfig.full(spec); base = loss(full)
    out = {}
    for knob, lo in (("d_qk", spec.qk_grid[0]), ("n_h", head_grid(spec)[0]),
                     ("d_mlp", spec.mlp_grid[0])):
        dmg = []
        for (a_, b_) in groups:
            kw = {"d_qk": list(full.d_qk), "d_v": list(full.d_v),
                  "n_h": list(full.n_h), "d_mlp": list(full.d_mlp)}
            for L in range(a_, b_):
                kw[knob][L] = lo
                if knob == "d_qk":
                    kw["d_v"][L] = lo
            c = ElasticConfig(spec, kw["d_qk"], kw["d_v"], [spec.n_kv] * spec.n_layers,
                              kw["n_h"], kw["d_mlp"])
            dmg.append(loss(c) - base)
        d = np.array(dmg)
        out[knob] = {"damage": [float(x) for x in d], "cv": float(d.std() / d.mean()),
                     "max_over_min": float(d.max() / max(d.min(), 1e-9))}
        print(f"  [{label}] {knob:6s} base {base:.4f} | CV {out[knob]['cv']:.3f} "
              f"max/min {out[knob]['max_over_min']:6.1f} | " +
              " ".join(f"{x:+.3f}" for x in d), flush=True)
    del m; torch.cuda.empty_cache()
    return {"base_loss": base, **out}

res = {"groups": groups,
       "trained_lr2e-3": run(f"{RUNS}/sl135_lr2e-3/step2500/model.safetensors", "trained"),
       "untrained_base": run(None, "base")}
json.dump(res, open(f"{RUNS}/per_layer_sens_sl135.json", "w"), indent=1)
print("\nwrote runs/supernet/per_layer_sens_sl135.json")
for k in ("d_qk", "n_h", "d_mlp"):
    t, b = res["trained_lr2e-3"][k]["cv"], res["untrained_base"][k]["cv"]
    print(f"  {k:6s} CV  untrained {b:.3f} -> trained {t:.3f}   "
          f"({'flattened ' + format(b/max(t,1e-9), '.1f') + 'x' if t < b else 'sharpened'})")
