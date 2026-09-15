"""Load an M2 checkpoint back into an elastic model for evaluation.

Order matters: build the base, patch in the elastic forward, register the logit_scale params
(add_elastic_temperature) so they exist to receive the trained values, THEN load_state_dict.
Eval uses the EAGER backend + attn_implementation='eager' so HF always materializes an explicit
causal mask (the sdpa path assumes a mask is passed; generate would otherwise be non-causal)."""
import os, sys
import torch
from ..paths import ROOT as _ROOT
ROOT = str(_ROOT)
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file
from ..elastic import (enable_elastic, add_elastic_temperature, set_elastic_backend,
                               build_pair_order, pair_order_for)

BASE = os.environ.get("DSE_BASE", "Qwen/Qwen3-0.6B-Base")  # DSE_BASE=Qwen/Qwen3-1.7B-Base for the M4 supernet


def load_supernet(ckpt_dir, dev="cuda", backend="eager"):
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, attn_implementation="eager").to(dev).eval()
    enable_elastic(model)
    set_elastic_backend(backend)
    add_elastic_temperature(model)
    sd = load_file(os.path.join(ckpt_dir, "model.safetensors"))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    ls = sum("logit_scale" in k for k in sd)
    print(f"[loader] {ckpt_dir}: loaded {len(sd)} tensors ({ls} logit_scale) | "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    tok = AutoTokenizer.from_pretrained(BASE)
    order = pair_order_for(model)
    return model, tok, order
