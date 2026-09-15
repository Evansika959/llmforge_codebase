"""Selection-effect probe: how much does Pareto-front selection compress the loss spread
at fixed size?  Scores RANDOM heterogeneous genomes with the same supernet slice, on the
same 58 windows the het truth and het predictors use."""
import json, os, random, sys, time
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoTokenizer
from llmforge.supernet.config import SPECS
from llmforge.supernet.elastic import pair_order_for, set_elastic_config
from llmforge.supernet.search.nsga import load_supernet, Space
from llmforge.supernet.space import ElasticConfig
from llmforge.supernet.paths import RUNS
from llmforge.paths import HELDOUT

CKPT = f"{RUNS}/sl135_lr2e-3/step2500"
N_RANDOM = int(os.environ.get("NRAND", "400"))
spec = SPECS["smollm2-135m"]
model, tok, order = load_supernet(CKPT, spec)
texts = json.load(open(f"{HELDOUT}/heldout_texts.json"))[:96]
flat = []
for t in texts:
    flat += tok(t, add_special_tokens=False).input_ids
CTX = 1024
ids = [torch.tensor(flat[i*CTX:(i+1)*CTX], device="cuda").unsqueeze(0) for i in range(len(flat)//CTX)]
print(f"{len(ids)} windows of {CTX}", flush=True)

@torch.no_grad()
def loss_of(cfg):
    set_elastic_config(model, cfg.d_qk, cfg.d_v, order, n_h=cfg.n_h, d_mlp=cfg.d_mlp)
    tot = ntok = 0.0
    for x in ids:
        lg = model(input_ids=x).logits[:, :-1].float(); t = x[:, 1:]
        tot += F.cross_entropy(lg.reshape(-1, lg.shape[-1]), t.reshape(-1), reduction="sum").item()
        ntok += t.numel()
    return tot/ntok

S = Space(spec, per=4, blocks=False)
print("genome group counts", S.lens, "total genes", S.G, flush=True)
full = ElasticConfig.full(spec); W0 = full.weight_params(include_embed=False)

def rec(cfg, tag):
    w = cfg.weight_params(include_embed=False)/W0
    cv = lambda a: float(np.std(a)/np.mean(a))
    het = float(np.mean([cv(cfg.d_qk), cv(cfg.d_v), cv(cfg.n_h), cv(cfg.d_mlp)]))
    return {"tag": tag, "w": w, "het": het, "loss": loss_of(cfg),
            "md_qk": float(np.mean(cfg.d_qk)), "md_v": float(np.mean(cfg.d_v)),
            "mn_h": float(np.mean(cfg.n_h)), "md_mlp": float(np.mean(cfg.d_mlp))}

out, t0 = [], time.time()
# 1) the 48 GA front members, re-scored on the SAME 58 windows
GA = json.load(open(f"{RUNS}/nsga2_sl135_lr2e-3.json"))
for i, f in enumerate(GA["front"]):
    c = ElasticConfig(spec, f["d_qk"], f["d_v"], [spec.n_kv]*spec.n_layers, f["n_h"], f["d_mlp"])
    out.append(rec(c, "front")); 
    if i % 10 == 0: print(f"front {i}/48 [{(time.time()-t0)/60:.1f}m]", flush=True)
json.dump(out, open(f"{RUNS}/sel_probe.json", "w"), indent=1)
# 2) random heterogeneous genomes from the same space
rng = random.Random(1234)
for i in range(N_RANDOM):
    c = S.to_config(S.random(rng)).validate()
    out.append(rec(c, "random"))
    if (i+1) % 25 == 0:
        print(f"random {i+1}/{N_RANDOM} [{(time.time()-t0)/60:.1f}m]", flush=True)
        json.dump(out, open(f"{RUNS}/sel_probe.json", "w"), indent=1)
json.dump(out, open(f"{RUNS}/sel_probe.json", "w"), indent=1)
print("done", len(out), f"{(time.time()-t0)/60:.1f}m", flush=True)
