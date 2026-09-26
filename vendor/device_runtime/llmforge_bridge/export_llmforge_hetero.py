"""
Export a HETEROGENEOUS LLMForge checkpoint (per-layer dims, infinite-head attention,
identity attention layers) into the self-describing `.rlm` format read by run_llmforge.c.

This exists because NSGA-searched models (e.g. nsga_best3_rotary_periln_105M) violate every
uniform-dim assumption of the llama2.c .bin formats: each layer has its own n_head / n_kv /
qk_dim / v_dim / mlp_hidden, "infinite" attention uses a non-square c_proj (n_head*v_dim ->
n_embd), and some layers have no attention at all ("identity").

Supported subset (asserted so we never silently emit a wrong model):
  attention_variant_layerlist entries in {infinite, identity}
  use_concat_heads=True, bias=False, no qk/v-norm, no flash-lobo, no MoE,
  mlp_variant=swiglu, activation=gelu (exact erf), norm=rmsnorm (NO eps), peri-LN,
  RoPE (rope variant, theta 1e4, full qk_dim), tied or separate classifier.

See doc/llmforge_hetero_infinite.md for the format spec.

Usage:
  python llmforge_bridge/export_llmforge_hetero.py <ckpt_dir_or_pt> <out.rlm>
"""
import argparse
import os
import struct
import sys

import torch

MAGIC = 0x726C6D31  # "rlm1"
VERSION = 1
ATTN_INFINITE = 0
ATTN_IDENTITY = 1
HEADER_BYTES = 256


def serialize_fp32(f, t):
    d = t.detach().cpu().contiguous().view(-1).to(torch.float32).numpy()
    f.write(d.tobytes())


def serialize_int8(f, t):
    import numpy as np
    d = t.detach().cpu().contiguous().view(-1).numpy().astype(np.int8)
    f.write(d.tobytes())


def quantize_q80(w, group_size):
    """symmetric int8, groups of group_size (flattened row-major); returns (int8, fp32 scales, maxerr)"""
    assert w.numel() % group_size == 0, f"numel {w.numel()} not a multiple of group_size {group_size}"
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


def write_q8(f, w, group_size, errbox):
    """serialize one matmul weight as Q8_0: int8 values then fp32 group scales."""
    q, s, err = quantize_q80(w, group_size)
    serialize_int8(f, q)
    serialize_fp32(f, s)
    if err > errbox[0]:
        errbox[0] = err


def load_ckpt(path):
    if os.path.isdir(path):
        path = os.path.join(path, "ckpt.pt")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
          for k, v in ck["model"].items()}
    ma = ck["model_args"]
    return sd, ma


def _list(ma, key, L):
    """Read a *_layerlist as ints; None entries -> None."""
    v = ma.get(key)
    if v is None:
        return [None] * L
    out = []
    for x in v:
        if x is None or (isinstance(x, str) and x.lower() == "none"):
            out.append(None)
        else:
            out.append(int(x))
    assert len(out) == L, f"{key} length {len(out)} != n_layer {L}"
    return out


def validate(ma, sd):
    problems = []
    def need(k, want):
        if ma.get(k) != want:
            problems.append(f"  {k}={ma.get(k)!r}, require {want!r}")
    need("mlp_variant", "swiglu")
    need("norm_variant_attn", "rmsnorm")
    need("norm_variant_output", "rmsnorm")
    need("use_rotary_embeddings", True)
    need("bias", False)
    need("use_peri_ln", True)
    need("use_pre_ln", True)
    need("use_concat_heads", True)
    if ma.get("activation_variant") != "gelu":
        problems.append(f"  activation_variant={ma.get('activation_variant')!r}, require gelu")
    if ma.get("use_post_ln"):
        problems.append("  use_post_ln=True unsupported")
    for k in ("use_qk_norm", "use_v_norm", "use_qk_norm_scale", "use_flash_lobo",
              "use_moe", "use_lsv", "use_abs_pos_embeddings", "use_embedding_scale",
              "use_parallel_mlp", "use_ln_f_input_mixer"):
        if ma.get(k):
            problems.append(f"  {k}=True unsupported")
    for k in ("attn_logit_softcapping", "final_logit_softcapping",
              "apply_vector_at_layer_idx", "obtain_vector_at_layer_idx",
              "n_embd_wte", "norm_variant_wte"):
        if ma.get(k) is not None:
            problems.append(f"  {k}={ma.get(k)!r} unsupported")
    if ma.get("rope_variant") not in ("rope", None):
        problems.append(f"  rope_variant={ma.get('rope_variant')!r}, require 'rope'")
    if ma.get("rope_length") is not None:
        problems.append(f"  rope_length={ma.get('rope_length')!r} (partial RoPE unsupported)")
    for k in ("attn_residual_combination", "mlp_residual_combination"):
        if ma.get(k, "add") != "add":
            problems.append(f"  {k}={ma.get(k)!r}, only 'add' supported")
    # Fields the C runner assumes are identity/off. They are 0/1.0/False in every model tested,
    # but a future NSGA model could set them and would then SILENTLY diverge — so guard them.
    # (getattr defaults mirror the LLMForge source.)
    def want(k, val, default=None):
        if ma.get(k, default) != val:
            problems.append(f"  {k}={ma.get(k, default)!r}, require {val!r} (unsupported by run_llmforge.c)")
    want("mlp_x_offset", 0.0, 0.0)
    want("mlp_y_offset", 0.0, 0.0)
    want("learn_mlp_x_offset", False, False)
    want("learn_mlp_y_offset", False, False)
    want("mlp_cproj_scale", 1.0, 1.0)
    want("attn_cproj_scale", 1.0, 1.0)
    want("mlp_post_act_l2_norm", False, False)
    want("attn_post_act_l2_norm", False, False)
    want("l2_norm_mlp_up", False, False)
    want("l2_norm_mlp_down", False, False)
    want("l2_norm_attn_q", False, False)
    want("l2_norm_attn_k", False, False)
    want("l2_norm_attn_v", False, False)
    want("l2_norm_attn_cproj", False, False)
    want("mlp_down_projs", 1, 1)
    want("softmax_variant_attn", "softmax", "softmax")
    # NB: n_cproj is intentionally NOT checked — use_concat_heads=True (asserted above) makes the
    # concat branch take priority over the n_cproj sum branch, so n_cproj is irrelevant here.
    # single global activation/mlp variant (no per-layer activation/mlp/scale overrides)
    for k in ("activation_variant_layerlist", "mlp_variant_layerlist",
              "mlp_cproj_scale_layerlist"):
        if ma.get(k) is not None:
            problems.append(f"  {k}={ma.get(k)!r} present (per-layer override unsupported)")
    # per-layer attention variants
    L = ma["n_layer"]
    av = ma.get("attention_variant_layerlist") or [ma.get("attention_variant", "causal")] * L
    for i, a in enumerate(av):
        if a not in ("infinite", "identity"):
            problems.append(f"  attention_variant_layerlist[{i}]={a!r}, only infinite/identity supported")
    # ground-truth: the activation offset BUFFERS in the checkpoint must be zero (the C drops them)
    for k, v in sd.items():
        if k.endswith("activation_x_offset") or k.endswith("activation_y_offset"):
            try:
                fv = float(v)
            except (TypeError, ValueError):
                fv = float(v.reshape(-1)[0]) if v.numel() else 0.0
            if fv != 0.0:
                problems.append(f"  nonzero {k}={fv} (activation offsets unsupported)")
    if problems:
        raise SystemExit("INCOMPATIBLE checkpoint for .rlm export:\n" + "\n".join(problems))


def build_layers(sd, ma):
    """Return per-layer descriptor + weight tensors, verified against tensor shapes."""
    L = ma["n_layer"]
    n_embd = ma["n_embd"]
    av = ma.get("attention_variant_layerlist") or [ma.get("attention_variant")] * L
    n_head_l = _list(ma, "n_head_layerlist", L)
    n_kv_l = _list(ma, "n_kv_group_layerlist", L)

    layers = []
    for i in range(L):
        p = f"transformer.h.{i}."
        rec = dict(
            pre_ln_attn=sd[p + "pre_ln_attn.gain"],
            peri_ln_attn=sd[p + "peri_ln_attn.gain"],
            pre_ln_mlp=sd[p + "pre_ln_mlp.gain"],
            peri_ln_mlp=sd[p + "peri_ln_mlp.gain"],
            w1=sd[p + "mlp.c_fc_in1.weight"],   # gate
            w3=sd[p + "mlp.c_fc_in2.weight"],   # value
            w2=sd[p + "mlp.c_fc_out.weight"],   # down
        )
        for g in ("pre_ln_attn", "peri_ln_attn", "pre_ln_mlp", "peri_ln_mlp"):
            assert rec[g].shape == (n_embd,), f"layer {i} {g} shape {rec[g].shape}"
        mlp_hidden = rec["w1"].shape[0]
        assert rec["w1"].shape == (mlp_hidden, n_embd)
        assert rec["w3"].shape == (mlp_hidden, n_embd)
        assert rec["w2"].shape == (n_embd, mlp_hidden)

        if av[i] == "identity":
            assert (p + "attn.c_attn_q.weight") not in sd, f"layer {i} identity but has attn weights"
            rec.update(attn_type=ATTN_IDENTITY, n_head=0, n_kv=0, qk=0, vd=0, mlp_hidden=mlp_hidden)
        else:  # infinite
            wq = sd[p + "attn.c_attn_q.weight"]
            wk = sd[p + "attn.c_attn_k.weight"]
            wv = sd[p + "attn.c_attn_v.weight"]
            wo = sd[p + "attn.c_proj.weight"]
            n_head = n_head_l[i]
            n_kv = n_kv_l[i] if n_kv_l[i] is not None else n_head
            assert n_head and wq.shape[0] % n_head == 0, f"layer {i} wq {wq.shape} n_head {n_head}"
            qk = wq.shape[0] // n_head
            vd = wo.shape[1] // n_head
            # verify every shape derives consistently
            assert wq.shape == (n_head * qk, n_embd), f"L{i} wq {wq.shape} != {(n_head*qk, n_embd)}"
            assert wk.shape == (n_kv * qk, n_embd), f"L{i} wk {wk.shape} != {(n_kv*qk, n_embd)}"
            assert wv.shape == (n_kv * vd, n_embd), f"L{i} wv {wv.shape} != {(n_kv*vd, n_embd)}"
            assert wo.shape == (n_embd, n_head * vd), f"L{i} wo {wo.shape} != {(n_embd, n_head*vd)}"
            assert qk % 2 == 0, f"L{i} qk_dim {qk} must be even for RoPE"
            rec.update(attn_type=ATTN_INFINITE, n_head=n_head, n_kv=n_kv, qk=qk, vd=vd,
                       mlp_hidden=mlp_hidden, wq=wq, wk=wk, wv=wv, wo=wo)
        layers.append(rec)
    return layers


def resolve_rope_theta(ma):
    """RoPE base for the header.

    LLMForge's RotaryEmbedding hardcodes base 10000 and exposes no config knob today
    (variations/position_encoding_variations.py; nothing in gpt_conf.py / train_args.py). We still
    read it from model_args first so that if a future version adds a base/theta field the exporter
    honors it automatically; otherwise we fall back to the source constant 10000.0. A wrong value
    here would be caught immediately by the ref_dump parity check (which runs the real model)."""
    for k in ("rope_theta", "rope_base", "rotary_base", "rotary_emb_base", "rope_freq_base"):
        v = ma.get(k)
        if v is not None:
            return float(v)
    return 10000.0


def choose_group_size(n_embd, layers, want):
    """GS must divide every matmul in-dim: n_embd, each infinite layer's n_head*vd, each mlp_hidden."""
    dims = {n_embd}
    for r in layers:
        dims.add(r["mlp_hidden"])
        if r["attn_type"] == ATTN_INFINITE:
            dims.add(r["n_head"] * r["vd"])
    gs = want
    while gs > 1 and any(d % gs != 0 for d in dims):
        gs //= 2
        print(f"BACKOFF group_size -> {gs}")
    if any(d % gs != 0 for d in dims):
        raise SystemExit(f"cannot find a group size dividing all matmul in-dims {sorted(dims)}")
    return gs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="path to ckpt.pt or its directory")
    ap.add_argument("out", help="output .rlm path")
    ap.add_argument("--version", type=int, choices=[1, 2], default=1,
                    help="1 = fp32 (run_llmforge), 2 = Q8_0 (runq_llmforge)")
    ap.add_argument("--group-size", type=int, default=64)
    args = ap.parse_args()

    sd, ma = load_ckpt(args.ckpt)
    validate(ma, sd)

    L = ma["n_layer"]
    n_embd = ma["n_embd"]
    vocab = ma["vocab_size"]
    seq_len = ma["block_size"]
    wte = sd["transformer.wte.weight"]
    ln_f = sd["transformer.ln_f.gain"]
    lm_head = sd.get("lm_head.weight")
    shared = (lm_head is None) or torch.equal(lm_head, wte)
    assert wte.shape == (vocab, n_embd)
    assert ln_f.shape == (n_embd,)
    rope_theta = resolve_rope_theta(ma)

    layers = build_layers(sd, ma)
    ver = args.version
    gs = choose_group_size(n_embd, layers, args.group_size) if ver == 2 else 0

    print(f"arch: layers={L} n_embd={n_embd} vocab={vocab} seq={seq_len} shared={shared} "
          f"rope_theta={rope_theta:g} version={ver}{f' group_size={gs}' if ver == 2 else ''}")
    for i, r in enumerate(layers):
        kind = "identity" if r["attn_type"] == ATTN_IDENTITY else "infinite"
        print(f"  L{i}: {kind:8} n_head={r['n_head']} n_kv={r['n_kv']} "
              f"qk={r['qk']} vd={r['vd']} mlp={r['mlp_hidden']}")

    errbox = [0.0]
    emit_mm = (lambda f, w: write_q8(f, w, gs, errbox)) if ver == 2 else serialize_fp32

    with open(args.out, "wb") as f:
        # global header (magic, version, dims, shared, act, rope_theta, group_size)
        act = 0  # 0 = gelu (exact erf)
        f.write(struct.pack("<Iiiiiiiifi",
                            MAGIC, ver, L, n_embd, vocab, seq_len,
                            int(shared), act, rope_theta, gs))
        pad = HEADER_BYTES - f.tell()
        assert pad >= 0, f"header too big: {f.tell()}"
        f.write(b"\0" * pad)
        # per-layer descriptor table
        for r in layers:
            f.write(struct.pack("<8i", r["attn_type"], r["n_head"], r["n_kv"],
                                r["qk"], r["vd"], r["mlp_hidden"], 0, 0))
        # weights, per layer in canonical order (norms fp32; matmuls fp32 or Q8_0)
        for r in layers:
            serialize_fp32(f, r["pre_ln_attn"])
            serialize_fp32(f, r["peri_ln_attn"])
            if r["attn_type"] == ATTN_INFINITE:
                emit_mm(f, r["wq"])
                emit_mm(f, r["wk"])
                emit_mm(f, r["wv"])
                emit_mm(f, r["wo"])
            serialize_fp32(f, r["pre_ln_mlp"])
            serialize_fp32(f, r["peri_ln_mlp"])
            emit_mm(f, r["w1"])
            emit_mm(f, r["w3"])
            emit_mm(f, r["w2"])
        # final
        serialize_fp32(f, ln_f)
        emit_mm(f, wte)
        if not shared:
            emit_mm(f, lm_head)
    sz = os.path.getsize(args.out)
    extra = f", max quant err={errbox[0]:.5f}" if ver == 2 else ""
    print(f"wrote {args.out}  ({sz:,} bytes, shared_classifier={shared}{extra})")


if __name__ == "__main__":
    main()
