"""Is layer 0's attention special in every pretrained LM, or only in Qwen3?

Qwen3-1.7B's layer 0 accounts for 84.8% of all the damage done by narrowing attention one layer at
a time, and it is 82x more sensitive than the second-worst layer. That shows up at step 0, before
any supernet training, so it is inherited from pretraining rather than created by the elastic
objective. Qwen3-4B shows the same ordering. This asks whether it survives a change of family.

The probe deliberately does NOT use the elastic machinery, which is Qwen3-specific and assumes
QK-Norm. Instead it zeroes the tail of each head's q/k/v at the projection output, which is a
forward hook and works on any Llama-style attention. Two consequences worth stating:

  * On Qwen3 the zeros then pass through QK-Norm, which renormalises the surviving dims; on Llama
    there is no such layer. So absolute damage is NOT comparable across families -- but the
    question is which layer inside a model is worst, and that comparison is unaffected because
    every layer of a given model is treated identically.
  * Zeroing keeps the 1/sqrt(head_dim) attention scale rather than switching to 1/sqrt(kept), so
    it isolates information loss from the temperature change that true slicing also introduces.

Pairs, not dims: HF applies RoPE by splitting the head in half and rotating dim i against dim
i + d/2. Keeping a prefix of PAIRS therefore means keeping [0, k) and [d/2, d/2 + k); zeroing a
contiguous block of dims instead would break half of every pair it touched and measure the broken
rotation rather than the narrower head.

  python experiments/supernet_fidelity/layer0_universality.py --models qwen3-1.7b,qwen3-0.6b,smollm-135m
"""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from llmforge.supernet.paths import RUNS

REPOS = {
    "qwen3-0.6b": "Qwen/Qwen3-0.6B-Base",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B-Base",
    "qwen3-4b": "Qwen/Qwen3-4B-Base",
    "smollm-135m": "HuggingFaceTB/SmolLM-135M",
    "smollm-360m": "HuggingFaceTB/SmolLM-360M",
}


def pair_mask(head_dim, keep, device, dtype):
    """Keep `keep` RoPE pairs: dims [0,keep) and [d/2, d/2+keep). Zero the rest.

    Built in the module's own dtype: a float32 mask silently promotes a bf16 activation and the
    next Linear then rejects it.
    """
    half = head_dim // 2
    m = torch.zeros(head_dim, device=device, dtype=dtype)
    m[:keep] = 1
    m[half:half + keep] = 1
    return m


def attach(layer, head_dim, keep, which=("q", "k", "v")):
    """Zero the tail of each head at the projection output. Returns handles to remove."""
    hs = []
    for name in which:
        mod = getattr(layer.self_attn, f"{name}_proj")
        if name == "v":                       # v carries no RoPE, so a plain prefix is correct
            m = torch.zeros(head_dim, device=mod.weight.device, dtype=mod.weight.dtype)
            m[:keep] = 1
        else:
            m = pair_mask(head_dim, keep, mod.weight.device, mod.weight.dtype)

        def hook(_mod, _inp, out, mask=m, hd=head_dim):
            B, S, _ = out.shape
            return (out.view(B, S, -1, hd) * mask).view(B, S, -1)

        hs.append(mod.register_forward_hook(hook))
    return hs


@torch.no_grad()
def nll(model, ids):
    tot = cnt = 0.0
    for i in ids:
        lg = model(input_ids=i).logits[:, :-1].float()
        t = i[:, 1:]
        tot += F.cross_entropy(lg.reshape(-1, lg.shape[-1]), t.reshape(-1),
                               reduction="sum").item()
        cnt += t.numel()
    return tot / cnt


def run(key, texts, keep_frac, dev="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    repo = REPOS[key]
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(
        repo, torch_dtype=torch.bfloat16).to(dev).eval()
    model.config.use_cache = False
    cfg = model.config
    hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    keep = max(2, int(round(hd * keep_frac / 2)) * 2 // 2)      # in PAIRS
    layers = model.model.layers
    ids = [tok(t, return_tensors="pt", truncation=True, max_length=768).input_ids.to(dev)
           for t in texts]
    ids = [i for i in ids if i.shape[1] >= 8]
    base = nll(model, ids)
    print(f"\n=== {key}  ({cfg.model_type}, {len(layers)}L, "
          f"{cfg.num_attention_heads}Q/{getattr(cfg,'num_key_value_heads','-')}KV, hd{hd}) ===")
    print(f"  base NLL {base:.4f}   keeping {keep}/{hd//2} RoPE pairs per head", flush=True)
    dmg = []
    for li, L in enumerate(layers):
        hs = attach(L, hd, keep)
        dmg.append(nll(model, ids) - base)
        for h in hs:
            h.remove()
    d = np.array(dmg)
    rest = d[1:]
    order = list(np.argsort(-d)[:4])
    print(f"  layer 0 {d[0]:8.4f}   others {rest.min():.4f}-{rest.max():.4f} "
          f"(median {np.median(rest):.4f})")
    print(f"  layer 0 share of total {d[0]/d.sum():6.1%}   "
          f"ratio to 2nd worst {d[0]/np.sort(d)[-2]:6.1f}x   worst layers {order}")
    del model
    torch.cuda.empty_cache()
    return {"model": key, "arch": cfg.model_type, "n_layers": len(layers), "head_dim": hd,
            "keep_pairs": keep, "base": base, "damage": [float(x) for x in d],
            "layer0_share": float(d[0] / d.sum()),
            "ratio_2nd": float(d[0] / np.sort(d)[-2])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen3-1.7b,qwen3-0.6b,smollm-135m")
    ap.add_argument("--n-texts", type=int, default=24)
    ap.add_argument("--keep-frac", type=float, default=0.25,
                    help="fraction of each head's RoPE pairs to keep in the probed layer")
    a = ap.parse_args()
    from llmforge.supernet.data.heldout import heldout_texts
    texts = heldout_texts(a.n_texts, skip=8)
    out = [run(k, texts, a.keep_frac) for k in a.models.split(",")]
    p = RUNS / "layer0_universality.json"
    json.dump(out, open(p, "w"), indent=1)
    print(f"\n{'model':14} {'arch':8} {'L':>3} {'layer0 share':>13} {'ratio to 2nd':>13}")
    for r in out:
        print(f"{r['model']:14} {r['arch']:8} {r['n_layers']:>3} "
              f"{r['layer0_share']:12.1%} {r['ratio_2nd']:12.1f}x")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
