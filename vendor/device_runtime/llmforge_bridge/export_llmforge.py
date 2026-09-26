"""
Export a LLMForge checkpoint (ckpt.pt) into a llama2.c .bin file.

Two output formats (see doc/llmforge2c.md):
  --version 0  -> legacy v0, read by THIS repo's run.c   (fp32)
  --version 2  -> Q8_0 v2,   read by THIS repo's runq.c  (int8 group-quant)

Only the llama2-compatible "Tier-1" subset is supported:
  causal attention, RoPE (theta 1e4, full head_dim), RMSNorm pre-norm (no peri-LN),
  SwiGLU MLP (gate=c_fc_in1, value=c_fc_in2, down=c_fc_out), no bias, GELU/SiLU activation.
The exporter asserts these so we never silently emit a wrong model.

Usage:
  python export_llmforge.py <ckpt_dir_or_pt> <out.bin> --version 0
"""
import argparse
import os
import struct
import sys

import numpy as np
import torch

# -----------------------------------------------------------------------------
# serialization helpers (mirror export.py)

def serialize_fp32(f, t):
    d = t.detach().cpu().view(-1).to(torch.float32).numpy()
    f.write(struct.pack(f'{len(d)}f', *d))

def serialize_int8(f, t):
    d = t.detach().cpu().view(-1).numpy().astype(np.int8)
    f.write(struct.pack(f'{len(d)}b', *d))

def quantize_q80(w, group_size):
    """symmetric int8, groups of group_size; returns (int8 tensor, fp32 scales, max err)"""
    assert w.numel() % group_size == 0
    ori_shape = w.shape
    w = w.float().reshape(-1, group_size)
    wmax = torch.abs(w).max(dim=1).values
    scale = wmax / 127.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)  # guard all-zero groups
    quant = w / scale[:, None]
    int8val = torch.round(quant).to(torch.int8)
    fp32val = (int8val.float() * scale[:, None]).reshape(ori_shape)
    err = torch.abs(fp32val - w.reshape(ori_shape)).max().item()
    return int8val.reshape(-1), scale, err

# -----------------------------------------------------------------------------
# checkpoint loading + arch validation

REQUIRED = {
    "attention_variant": "causal",
    "mlp_variant": "swiglu",
    "norm_variant_attn": "rmsnorm",
    "norm_variant_output": "rmsnorm",
    "use_rotary_embeddings": True,
    "bias": False,
}
# use_peri_ln is ALLOWED (Tier-2): if True, extra peri norms are exported and the C engine
# must be built with -DPERI_LN (make run_periln / runq_periln). use_post_ln is NOT supported.

def load_ckpt(path):
    if os.path.isdir(path):
        path = os.path.join(path, "ckpt.pt")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"]
    # strip torch.compile prefix
    sd = { (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in sd.items() }
    ma = ck["model_args"]
    return sd, ma

def validate(ma, sd):
    problems = []
    for k, want in REQUIRED.items():
        got = ma.get(k)
        if got != want:
            problems.append(f"  model_args[{k}]={got!r}, require {want!r}")
    if ma.get("activation_variant") not in ("gelu", "silu"):
        problems.append(f"  activation_variant={ma.get('activation_variant')!r}, require gelu or silu")
    if ma.get("rope_length") not in (None, ma.get("n_embd", 0) // max(ma.get("n_head", 1), 1)):
        problems.append(f"  rope_length={ma.get('rope_length')!r} (partial RoPE unsupported)")
    if ma.get("use_post_ln"):
        problems.append("  use_post_ln=True (post-LN unsupported)")
    # offsets must be zero (we drop them)
    for k, v in sd.items():
        if k.endswith("activation_x_offset") or k.endswith("activation_y_offset"):
            if float(v) != 0.0:
                problems.append(f"  nonzero {k}={float(v)} (offsets unsupported)")
    if problems:
        raise SystemExit("INCOMPATIBLE checkpoint for llama2.c export:\n" + "\n".join(problems))

# -----------------------------------------------------------------------------
# weight gathering in canonical (per-layer) order

def gather(sd, ma):
    L = ma["n_layer"]
    g = lambda name: sd[name]
    tok = g("transformer.wte.weight")
    attn_norm = [g(f"transformer.h.{i}.pre_ln_attn.gain") for i in range(L)]
    wq = [g(f"transformer.h.{i}.attn.c_attn_q.weight") for i in range(L)]
    wk = [g(f"transformer.h.{i}.attn.c_attn_k.weight") for i in range(L)]
    wv = [g(f"transformer.h.{i}.attn.c_attn_v.weight") for i in range(L)]
    wo = [g(f"transformer.h.{i}.attn.c_proj.weight") for i in range(L)]
    ffn_norm = [g(f"transformer.h.{i}.pre_ln_mlp.gain") for i in range(L)]
    w1 = [g(f"transformer.h.{i}.mlp.c_fc_in1.weight") for i in range(L)]  # gate
    w3 = [g(f"transformer.h.{i}.mlp.c_fc_in2.weight") for i in range(L)]  # value
    w2 = [g(f"transformer.h.{i}.mlp.c_fc_out.weight") for i in range(L)]  # down
    final_norm = g("transformer.ln_f.gain")
    lm_head = sd.get("lm_head.weight")
    shared = (lm_head is None) or torch.equal(lm_head, tok)
    peri = bool(ma.get("use_peri_ln", False))
    peri_attn = peri_mlp = None
    if peri:
        peri_attn = [g(f"transformer.h.{i}.peri_ln_attn.gain") for i in range(L)]
        peri_mlp  = [g(f"transformer.h.{i}.peri_ln_mlp.gain") for i in range(L)]
    return dict(tok=tok, attn_norm=attn_norm, wq=wq, wk=wk, wv=wv, wo=wo,
                ffn_norm=ffn_norm, w1=w1, w2=w2, w3=w3, final_norm=final_norm,
                lm_head=lm_head, shared=shared,
                peri=peri, peri_attn=peri_attn, peri_mlp=peri_mlp)

def header_ints(ma, hidden_dim, vocab_signed):
    n_kv = ma.get("n_kv_group")
    if n_kv is None:            # None => plain MHA => kv heads == n_head
        n_kv = ma["n_head"]
    return struct.pack('iiiiiii',
        ma["n_embd"], hidden_dim, ma["n_layer"], ma["n_head"],
        n_kv, vocab_signed, ma["block_size"])

# -----------------------------------------------------------------------------
# v0 export (fp32, run.c)

def export_v0(W, ma, out):
    dim = ma["n_embd"]; n_heads = ma["n_head"]; head_size = dim // n_heads
    hidden_dim = W["w1"][0].shape[0]
    vocab = ma["vocab_size"]
    vocab_signed = vocab if W["shared"] else -vocab
    with open(out, "wb") as f:
        f.write(header_ints(ma, hidden_dim, vocab_signed))
        serialize_fp32(f, W["tok"])
        for t in W["attn_norm"]: serialize_fp32(f, t)
        for t in W["wq"]: serialize_fp32(f, t)
        for t in W["wk"]: serialize_fp32(f, t)
        for t in W["wv"]: serialize_fp32(f, t)
        for t in W["wo"]: serialize_fp32(f, t)
        for t in W["ffn_norm"]: serialize_fp32(f, t)
        for t in W["w1"]: serialize_fp32(f, t)
        for t in W["w2"]: serialize_fp32(f, t)
        for t in W["w3"]: serialize_fp32(f, t)
        serialize_fp32(f, W["final_norm"])
        if W["peri"]:   # peri-LN norms (run.c reads these after final_norm, before freq_cis, under -DPERI_LN)
            for t in W["peri_attn"]: serialize_fp32(f, t)
            for t in W["peri_mlp"]: serialize_fp32(f, t)
        # freq_cis placeholders (run.c ignores contents, advances pointer)
        nfreq = ma["block_size"] * head_size // 2
        zeros = torch.zeros(nfreq, dtype=torch.float32)
        serialize_fp32(f, zeros)  # cos
        serialize_fp32(f, zeros)  # sin
        if not W["shared"]:
            serialize_fp32(f, W["lm_head"])
    print(f"wrote v0 fp32 -> {out}  (shared_classifier={W['shared']})")

# -----------------------------------------------------------------------------
# v2 export (Q8_0, runq.c)

def export_v2(W, ma, out, group_size=64):
    dim = ma["n_embd"]
    while dim % group_size != 0:
        group_size //= 2
        print(f"BACKOFF group_size -> {group_size}")
    hidden_dim = W["w1"][0].shape[0]
    # quantized weight list, in v2 order: tok, wq, wk, wv, wo, w1, w2, w3, [lm_head]
    qweights = [W["tok"], *W["wq"], *W["wk"], *W["wv"], *W["wo"], *W["w1"], *W["w2"], *W["w3"]]
    if not W["shared"]:
        qweights.append(W["lm_head"])
    for w in qweights:
        assert w.numel() % group_size == 0, f"numel {w.numel()} not multiple of {group_size}"
    with open(out, "wb") as f:
        f.write(struct.pack('I', 0x616b3432))   # magic "ak42"
        f.write(struct.pack('i', 2))            # version
        f.write(header_ints(ma, hidden_dim, ma["vocab_size"]))
        f.write(struct.pack('B', int(W["shared"])))
        f.write(struct.pack('i', group_size))
        pad = 256 - f.tell()
        assert pad >= 0, f"header too big: {f.tell()}"
        f.write(b'\0' * pad)
        # fp32 norms: attn (all), ffn (all), final, [peri_attn (all), peri_mlp (all)]
        for t in W["attn_norm"]: serialize_fp32(f, t)
        for t in W["ffn_norm"]: serialize_fp32(f, t)
        serialize_fp32(f, W["final_norm"])
        if W["peri"]:   # peri-LN norms (runq.c reads these after final_norm, before quantized weights, under -DPERI_LN)
            for t in W["peri_attn"]: serialize_fp32(f, t)
            for t in W["peri_mlp"]: serialize_fp32(f, t)
        # quantized matmuls
        maxerr = 0.0
        for i, w in enumerate(qweights):
            q, s, err = quantize_q80(w, group_size)
            serialize_int8(f, q)
            serialize_fp32(f, s)
            maxerr = max(maxerr, err)
        print(f"wrote v2 Q8_0 -> {out}  (shared={W['shared']}, group={group_size}, max quant err={maxerr:.5f})")

# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="path to ckpt.pt or its directory")
    ap.add_argument("out", help="output .bin path")
    ap.add_argument("--version", type=int, choices=[0, 2], default=0)
    ap.add_argument("--group-size", type=int, default=64)
    args = ap.parse_args()

    sd, ma = load_ckpt(args.ckpt)
    validate(ma, sd)
    W = gather(sd, ma)
    print(f"arch: dim={ma['n_embd']} layers={ma['n_layer']} heads={ma['n_head']} "
          f"kv={ma['n_kv_group']} hidden={W['w1'][0].shape[0]} vocab={ma['vocab_size']} "
          f"seq={ma['block_size']} act={ma['activation_variant']} peri_ln={W['peri']}")
    if args.version == 0:
        export_v0(W, ma, args.out)
    else:
        export_v2(W, ma, args.out, args.group_size)
    if W["peri"]:
        print("NOTE: peri-LN model — build the engine with -DPERI_LN "
              "(make run_periln / runq_periln), else output will be WRONG.")

if __name__ == "__main__":
    main()
