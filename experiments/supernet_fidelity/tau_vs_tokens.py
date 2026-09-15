"""Measure rank fidelity as a function of the supernet's own training budget.

The predictor is not the deployable model, so the usual token-budget argument does not transfer.
Conventional post-NAS work -- Sheared-LLaMA (50B continued pretraining), Minitron (~94B retraining
after pruning), Nemotron-Elastic (~110B) -- spends those tokens to make the compressed network
GOOD. We spend ours to make the slice ORDER architectures correctly, and those are different
requirements: a slice that is uniformly 0.5 nats too pessimistic is a perfect ranker and a useless
model.

So the question "is 0.164B too few?" has to be answered against rank fidelity, not against loss.
This scores the ground-truth architectures at each intermediate checkpoint and reports Kendall tau
against dedicated training, plus the within-size-band concordance and the top-1 regret. A curve
that is still rising at the last checkpoint says the budget binds; one that plateaued early says it
does not, whatever the absolute loss is doing.
"""
import argparse, itertools, json, os
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmforge.supernet.config import SPECS
from llmforge.supernet.elastic import (add_elastic_temperature, enable_elastic, pair_order_for,
                                set_elastic_backend, set_elastic_config)
from llmforge.supernet.space import ElasticConfig
from llmforge.supernet.paths import RUNS
from llmforge.paths import HELDOUT


def tau(x, y):
    c = d = 0
    for i, j in itertools.combinations(range(len(x)), 2):
        s = (x[i] - x[j]) * (y[i] - y[j])
        c += s > 0
        d += s < 0
    return (c - d) / (c + d) if c + d else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="smollm2-135m")
    ap.add_argument("--arch-dir", default=f"{RUNS}/gt135")
    ap.add_argument("--truth", default=f"{RUNS}/gt135_nll_ctx1024.json")
    ap.add_argument("--ckpts", required=True, help="comma-separated checkpoint dirs, in step order")
    ap.add_argument("--tokens-per-step", type=int, default=65536)
    ap.add_argument("--texts", default=f"{HELDOUT}/heldout_ab.json")
    ap.add_argument("--n", type=int, default=192)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--out", default=f"{RUNS}/tau_vs_tokens.json")
    a = ap.parse_args()

    spec = SPECS[a.model]
    truth = {k: v["nll"] for k, v in json.load(open(a.truth)).items()}
    archs = {r["name"]: r for r in json.load(open(f"{a.arch_dir}/ARCHS.json"))}
    names = sorted(k for k in truth if k in archs)
    y = np.array([truth[k] for k in names])
    print(f"{len(names)} architectures with dedicated-training ground truth")

    tok = AutoTokenizer.from_pretrained(spec.repo)
    flat = []
    for t in json.load(open(a.texts))[:a.n]:
        flat += tok(t, add_special_tokens=False).input_ids
    ids = [torch.tensor(flat[i * a.ctx:(i + 1) * a.ctx], device="cuda").unsqueeze(0)
           for i in range(len(flat) // a.ctx)]
    print(f"{len(ids)} windows of {a.ctx}")

    m = AutoModelForCausalLM.from_pretrained(spec.repo, torch_dtype=torch.bfloat16,
                                             attn_implementation="eager").to("cuda").eval()
    enable_elastic(m); set_elastic_backend("eager"); m.config.use_cache = False
    add_elastic_temperature(m)
    order = pair_order_for(spec)

    # Free baselines, both directions -- the comparison the slice has to beat to be worth anything.
    free = {}
    for lab, v in [("w", [archs[k]["w"] for k in names]),
                   ("d_qk", [archs[k]["d_qk"] for k in names]),
                   ("n_h", [archs[k]["n_h"] for k in names]),
                   ("d_mlp", [archs[k]["d_mlp"] for k in names]),
                   ("attn_width", [archs[k].get("attn_width", archs[k]["d_qk"] * archs[k]["n_h"])
                                   for k in names])]:
        free[lab] = np.array(v, dtype=float)

    best_i = int(np.argmin(y))
    rows = []
    for ck in a.ckpts.split(","):
        step = int("".join(c for c in os.path.basename(ck) if c.isdigit()) or 0)
        m.load_state_dict({k: v.to("cuda") for k, v in
                           load_file(f"{ck}/model.safetensors").items()}, strict=False)
        s_all = []
        for k in names:
            r = archs[k]
            c = ElasticConfig.uniform(spec, r["d_qk"], r["d_qk"], n_h=r["n_h"], d_mlp=r["d_mlp"])
            set_elastic_config(m, c.d_qk, c.d_v, order, n_h=c.n_h, d_mlp=c.d_mlp)
            s = n = 0
            with torch.no_grad():
                for x in ids:
                    lg = m(input_ids=x).logits[:, :-1].float()
                    s += F.cross_entropy(lg.reshape(-1, lg.shape[-1]), x[:, 1:].reshape(-1),
                                         reduction="sum").item()
                    n += x.shape[1] - 1
            s_all.append(s / n)
        s_all = np.array(s_all)
        t = tau(s_all, y)
        # within-size-band: pairs whose parameter counts tie exactly
        P = [(i, j) for i, j in itertools.combinations(range(len(names)), 2)
             if abs(archs[names[i]]["w"] - archs[names[j]]["w"]) < 1e-9]
        band = (sum(1 for i, j in P if (s_all[i] < s_all[j]) == (y[i] < y[j])) / len(P)
                if P else float("nan"))
        regret = y[int(np.argmin(s_all))] - y[best_i]
        rows.append({"ckpt": ck, "step": step, "tokens_M": step * a.tokens_per_step / 1e6,
                     "tau": float(t), "band": float(band), "regret": float(regret),
                     "mean_slice": float(s_all.mean())})
        print(f"  step {step:5d} ({step*a.tokens_per_step/1e6:6.1f}M tok): "
              f"tau {t:+.4f}  within-band {band:.3f} ({len(P)} pairs)  top-1 regret {regret:+.4f}  "
              f"mean slice {s_all.mean():.3f}", flush=True)

    print("\nfree baselines, both directions:")
    for lab, v in free.items():
        for d, sgn in [("max", -1.0), ("min", 1.0)]:
            tt = tau(sgn * v, y)
            rg = y[int(np.argmin(sgn * v))] - y[best_i]
            if tt > 0:
                print(f"  {d} {lab:11s} tau {tt:+.4f}  top-1 regret {rg:+.4f}")
    json.dump({"rows": rows, "n_arch": len(names), "n_windows": len(ids)},
              open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
