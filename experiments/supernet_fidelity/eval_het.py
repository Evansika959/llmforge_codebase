#!/usr/bin/env python
"""Score heterogeneous reference models, each sliced at its OWN configuration.

A heterogeneous architecture has no dense form, so it is stored as a full-size checkpoint whose
active slice is the trained architecture. Reading it means loading the full weights, pinning that
architecture's config, and evaluating -- the inactive weights are never touched. Windowing matches
slice_nll.py exactly (concatenate the held-out documents into 1024-token windows above ctx 768),
because the whole point is to compare these numbers against slice scores measured that way.
"""
import argparse, json, os, sys
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file
from llmforge.supernet.config import SPECS
from llmforge.supernet.elastic import enable_elastic, pair_order_for, set_elastic_backend, set_elastic_config
from llmforge.supernet.space import ElasticConfig
from llmforge.supernet.paths import RUNS
from llmforge.paths import HELDOUT

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=f"{RUNS}/gt135_het")
ap.add_argument("--model", default="smollm2-135m",
                help="supernet spec the reference architectures were sliced from")
ap.add_argument("--ctx", type=int, default=1024)
ap.add_argument("--n-texts", type=int, default=1000)
ap.add_argument("--texts", default=f"{HELDOUT}/heldout_texts.json",
                help="Held-out document file. assets/heldout/heldout_ab.json is the A+B union "
                     "(192 documents, 0 overlap between halves) and gives ~123 windows "
                     "instead of 58, cutting the paired between-architecture SE ~1.46x "
                     "from 0.0033. Every comparison must use the same file on both sides.")
ap.add_argument("--out", default=f"{RUNS}/gt135_het_nll_ctx1024.json")
a = ap.parse_args()

spec = SPECS[a.model]; order = pair_order_for(spec)
tok = AutoTokenizer.from_pretrained(spec.repo)
texts = json.load(open(a.texts))[:a.n_texts]
if a.ctx > 768:
    flat = []
    for t in texts:
        flat += tok(t, add_special_tokens=False).input_ids
    ids = [torch.tensor(flat[i * a.ctx:(i + 1) * a.ctx], device="cuda").unsqueeze(0)
           for i in range(len(flat) // a.ctx)]
else:
    ids = [tok(t, return_tensors="pt", truncation=True, max_length=a.ctx).input_ids.to("cuda")
           for t in texts]
print(f"{len(ids)} windows of {a.ctx}")

res = json.load(open(a.out)) if os.path.exists(a.out) else {}
units = {}
for d in sorted(os.listdir(a.dir)):
    p = os.path.join(a.dir, d)
    if not os.path.isfile(os.path.join(p, "arch.json")) or d in res:
        continue
    A = json.load(open(os.path.join(p, "arch.json")))
    m = AutoModelForCausalLM.from_pretrained(spec.repo, torch_dtype=torch.bfloat16,
                                             attn_implementation="eager").to("cuda").eval()
    enable_elastic(m); set_elastic_backend("eager"); m.config.use_cache = False
    sd = load_file(os.path.join(p, "model.safetensors"))
    miss, unexp = m.load_state_dict({k: v.to("cuda") for k, v in sd.items()}, strict=False)
    assert len(miss) <= 1, f"{d}: {len(miss)} missing tensors"
    set_elastic_config(m, A["d_qk"], A["d_v"], order, n_h=A["n_h"], d_mlp=A["d_mlp"])
    per = []
    with torch.no_grad():
        for x in ids:
            lg = m(input_ids=x).logits[:, :-1].float()
            t = x[:, 1:]
            per.append((F.cross_entropy(lg.reshape(-1, lg.shape[-1]), t.reshape(-1),
                                        reduction="sum").item(), float(t.numel())))
    tot = sum(s for s, _ in per); cnt = sum(c for _, c in per)
    rng = np.random.default_rng(0)
    bs = [sum(per[j][0] for j in k) / sum(per[j][1] for j in k)
          for k in (rng.integers(0, len(per), len(per)) for _ in range(1000))]
    res[d] = {"nll": tot / cnt, "se": float(np.std(bs)), "n_units": len(per),
              **{k: A[k] for k in ("w", "kv", "het", "slice_loss", "slice_loss_val")
                 if k in A}}
    units[d] = [(float(s), float(c)) for s, c in per]
    print(f"  {d:6} w={A['w']:.3f} het={A.get('het',0):.3f} nll={res[d]['nll']:.4f} "
          f"+-{res[d]['se']:.4f}  (slice said {A.get('slice_loss', float('nan')):.4f})", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    json.dump(units, open(a.out.replace(".json", "_units.json"), "w"))
    del m; torch.cuda.empty_cache()
print(f"wrote {a.out} ({len(res)} architectures)")
