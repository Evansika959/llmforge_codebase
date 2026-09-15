"""Extract one elastic configuration into a standalone dense model (Qwen3 or Llama).

Verified equivalent to slicing the supernet: fp32 max|dlogit| 3.4e-05 with 100% argmax agreement
(experiments/validate_slicing.py, T2). Two details that are easy to get wrong and silently produce
a model that merely looks plausible:

  RoPE   a native head_dim=D model derives its own frequencies for D dims, but our slice keeps D of
         the 128-dim model's PAIRS -- a different frequency set. The reference's inv_freq must be
         replaced with the selected subset. qk_gather_index emits [first-halves | second-halves],
         the convention HF rotate_half expects, so a pair's frequency index IS its first-half entry.

  n_kv   asymmetric between the two sides, and the asymmetry is forced by RMSNorm's order of
         operations. v has no norm, so its alignment rotation folds straight into v_proj and the
         un-rotation into o_proj: the v side exports to a stock model. k does not. RMSNorm
         normalises BEFORE applying gamma, so the post-norm rotation gives
         (1/rms(y)) * R @ diag(gamma) @ y, and R @ diag(gamma) is not diagonal -- there is no
         k_norm.weight to fold it into. Folding it into k_proj instead changes y and therefore
         rms(y), which is not the same function.

         Consequence: on a QK-Norm family (Qwen3) a merged model must keep all G k projections and
         run norm -> rotate -> pool at inference. The k-side PARAMETER saving is lost; the KV CACHE
         saving, which is the actual deployment target and the reason the axis exists, is
         untouched. k_proj+v_proj is ~8% of parameters at 4B.

         On a family without QK-Norm (Llama, SmolLM2) the k side folds as cleanly as v -- but n_kv
         must divide the model's own, and SmolLM2's 3 and 5 are prime, so the axis is single-valued
         exactly where the export would have been clean.

  temp   the learned elastic temperature must be folded into q_norm.weight, NOT q_proj. q_norm is
         RMSNorm and therefore scale-invariant, so a factor on q_proj is normalised straight out.
         A family WITHOUT QK-Norm (Llama/SmolLM2) has no q_norm to fold into, and there the same
         reasoning inverts: nothing normalises q, so q_proj is the correct destination. Getting
         this wrong is silent -- the temperature simply vanishes and the dense model scores worse
         than the slice it was extracted from.
"""
import torch

from .elastic.attention import grid_index, head_index, qk_gather_index
from .elastic.families import family


def extract_dense(spec, state_dict, cfg, pair_order, device="cuda", dtype=torch.bfloat16):
    """Build a dense model implementing `cfg`, with weights copied from `state_dict`.

    cfg must be uniform across layers in d_qk/d_v/n_h (a dense model has one shape); d_mlp may
    also only take one value. Returns (model, config).
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    dqk, dv = cfg.d_qk[0], cfg.d_v[0]
    nh, dm = cfg.n_h[0], cfg.d_mlp[0]
    assert len(set(cfg.d_qk)) == 1 and len(set(cfg.d_v)) == 1, "dense model needs uniform widths"
    assert len(set(cfg.n_h)) == 1 and len(set(cfg.d_mlp)) == 1
    assert dqk == dv, f"dense model has one head_dim, got d_qk={dqk} d_v={dv}"

    n_q, n_kv, hd = spec.n_q, spec.n_kv, spec.head_dim
    nkv_t = cfg.n_kv[0] if hasattr(cfg, "n_kv") else n_kv
    if any(x != nkv_t for x in getattr(cfg, "n_kv", [nkv_t])):
        raise NotImplementedError("extract_dense needs a uniform n_kv across layers")
    # A reference model at reduced n_kv is initialised by mean-pooling k_proj/v_proj rows and is
    # then trained on its own. That is deliberately NOT the supernet's post-norm aligned path: a
    # ground-truth model exists to be an independent answer, so it must not inherit the machinery
    # whose fidelity it is being used to judge. Pooling rows is not equivalent to pooling after
    # k_norm (RMSNorm is non-linear), but it does not need to be -- it is a starting point for
    # 2000 steps of training, not a claim of equivalence.
    #
    # For DEPLOYING a supernet slice at reduced n_kv the story is different and is in this module's
    # docstring: the k side cannot fold into a stock checkpoint, so load the supernet and call
    # set_elastic_config(..., n_kv=...). That realises the whole KV-cache saving and gives up only
    # the k-side parameter saving (~8% at 4B).
    # A wrong-length pair_order does not raise here -- a 32-pair order on a head_dim=128 parent
    # yields indices that are all in range but pairs the second half at +32 instead of +64, so
    # every extracted head is a scrambled non-pair AND base_inv[sel] picks the wrong frequencies.
    # set_elastic_config already refuses this; extract bakes it into a saved checkpoint, so it
    # matters more here.
    if 2 * len(pair_order) != hd:
        raise ValueError(f"pair_order has {len(pair_order)} pairs but {spec.key} head_dim is {hd}; "
                         f"use pair_order_for(spec).")
    # Ask the family, not the tensor names: a filtered or renamed state_dict would otherwise send
    # a Qwen3 extraction down the Llama branch, folding the temperature into q_proj where RMSNorm
    # divides it straight back out.
    has_qk_norm = family(spec.family).qk_norm
    def _temp(L):
        """exp(logit_scale) for layer L at this granularity, or None if untrained/absent."""
        ls = state_dict.get(f"model.layers.{L}.self_attn.logit_scale")
        return None if ls is None else torch.exp(ls[grid_index(dqk, hd)].float())
    idx = qk_gather_index(pair_order, dqk)
    qi = head_index(n_q, nkv_t, nh) if nh != n_q else torch.arange(n_q)
    pool = n_kv // nkv_t                      # KV heads merged into each surviving group

    cf = AutoConfig.from_pretrained(spec.repo)
    cf.head_dim, cf.num_attention_heads, cf.intermediate_size = dqk, nh, dm
    cf.num_key_value_heads = nkv_t
    cf._attn_implementation = "sdpa"
    model = AutoModelForCausalLM.from_config(cf, torch_dtype=dtype).to(device)

    new = {}
    for k, v in state_dict.items():
        if "logit_scale" in k:
            continue
        if k.endswith("q_proj.weight"):
            w = v.view(n_q, hd, -1)[qi][:, idx, :].reshape(nh * dqk, -1)
            if not has_qk_norm:                       # no q_norm to absorb it; see `temp` above
                t = _temp(int(k.split(".")[2]))
                if t is not None:
                    w = w * t.to(w.dtype)
            new[k] = w
        elif k.endswith("k_proj.weight"):
            w = v.view(n_kv, hd, -1)[:, idx, :]
            if pool > 1:
                w = w.view(nkv_t, pool, dqk, -1).mean(1)
            new[k] = w.reshape(nkv_t * dqk, -1)
        elif k.endswith("v_proj.weight"):
            w = v.view(n_kv, hd, -1)[:, :dv, :]
            if pool > 1:
                w = w.view(nkv_t, pool, dv, -1).mean(1)
            new[k] = w.reshape(nkv_t * dv, -1)
        elif k.endswith("o_proj.weight"):
            new[k] = v.view(-1, n_q, hd)[:, qi, :dv].reshape(-1, nh * dv)
        elif k.endswith("q_norm.weight"):
            w = v[idx]
            t = _temp(int(k.split(".")[2]))
            if t is not None:
                w = w * t.to(w.dtype)
            new[k] = w
        elif k.endswith("k_norm.weight"):
            new[k] = v[idx]
        elif k.endswith("gate_proj.weight") or k.endswith("up_proj.weight"):
            new[k] = v[:dm]
        elif k.endswith("down_proj.weight"):
            new[k] = v[:, :dm]
        else:
            new[k] = v
    miss, unexp = model.load_state_dict({k: t.to(dtype) for k, t in new.items()}, strict=False)
    if len(miss) > 1:      # tied lm_head is the one expected miss
        raise RuntimeError(f"extract: {len(miss)} missing tensors -- shape mismatch?")

    # RoPE frequencies must follow the SELECTED pairs, not a native head_dim=dqk schedule
    inv = model.model.rotary_emb.inv_freq
    theta = getattr(cf, "rope_theta", 10000.0)
    base_inv = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
    sel = idx[: dqk // 2]
    model.model.rotary_emb.inv_freq = base_inv[sel].to(inv.device, inv.dtype)
    if hasattr(model.model.rotary_emb, "original_inv_freq"):
        model.model.rotary_emb.original_inv_freq = model.model.rotary_emb.inv_freq
    return model, cf
