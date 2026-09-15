"""Did supernet training actually reorganise the weights so the PREFIX carries the importance?

Slicing keeps the first k dimensions. That is only sensible if training pushed the important
content into those dimensions -- otherwise the sandwich rule taught the model to tolerate
truncation without ever making the prefix special, and every slice is throwing away an arbitrary
subset. Star Elastic describes its children as reusing "the most important slices of every weight
matrix"; this measures whether ours does the same.

For each layer, compare the energy in the kept dimensions against the dropped ones, on the base
checkpoint and on the trained supernet. If training worked, the ratio should rise.
"""
import json
import numpy as np
import torch

from llmforge.supernet.config import SPECS
from llmforge.supernet.elastic import build_pair_order
from llmforge.supernet.elastic.attention import qk_gather_index
from llmforge.supernet.paths import RUNS


def energies(sd, spec, order):
    """Per-layer share of q/k/v energy sitting in the dims a 64-wide slice would keep."""
    idx = qk_gather_index(order, 64).numpy()
    hd, n_q, n_kv = spec.head_dim, spec.n_q, spec.n_kv
    out = {"q": [], "k": [], "v": [], "mlp": []}
    for L in range(spec.n_layers):
        p = f"model.layers.{L}."
        for tag, key, nh in (("q", "self_attn.q_proj.weight", n_q),
                             ("k", "self_attn.k_proj.weight", n_kv)):
            w = sd[p + key].float().view(nh, hd, -1)
            e = w.pow(2).sum(-1).mean(0).numpy()          # energy per head-dim
            out[tag].append(float(e[idx].sum() / e.sum()))
        w = sd[p + "self_attn.v_proj.weight"].float().view(n_kv, hd, -1)
        e = w.pow(2).sum(-1).mean(0).numpy()
        out["v"].append(float(e[:64].sum() / e.sum()))     # v uses a plain prefix
        g = sd[p + "mlp.gate_proj.weight"].float()
        e = g.pow(2).sum(-1).numpy()
        half = spec.d_mlp // 2
        out["mlp"].append(float(e[:half].sum() / e.sum()))
    return out


def main():
    from transformers import AutoModelForCausalLM
    import glob, os
    from safetensors.torch import load_file
    spec = SPECS["qwen3-4b"]
    order = build_pair_order("hilo")

    base = AutoModelForCausalLM.from_pretrained(spec.repo, torch_dtype=torch.float32)
    b = energies(base.state_dict(), spec, order)
    del base

    sd = {}
    for f in sorted(glob.glob(f"{RUNS}/qwen3-4b_ab/final/*.safetensors")):
        sd.update(load_file(f))
    t = energies(sd, spec, order)

    print(f"share of energy in the dimensions a 64-wide slice KEEPS "
          f"(0.50 = no preference, higher = training concentrated importance in the prefix)\n")
    print(f"{'tensor':6} {'base':>18} {'trained supernet':>20} {'change':>10}")
    res = {}
    for k in ("q", "k", "v", "mlp"):
        bm, tm = float(np.mean(b[k])), float(np.mean(t[k]))
        res[k] = {"base": bm, "trained": tm, "base_all": b[k], "trained_all": t[k]}
        print(f"{k:6} {bm:18.4f} {tm:20.4f} {tm-bm:+10.4f}")
    print("\nper-layer, trained supernet (q):")
    print("  " + " ".join(f"{x:.3f}" for x in t["q"][:12]) + " ...")
    json.dump(res, open(RUNS / "prefix_importance.json", "w"), indent=1)
    print(f"\nwrote {RUNS / 'prefix_importance.json'}")


if __name__ == "__main__":
    main()
