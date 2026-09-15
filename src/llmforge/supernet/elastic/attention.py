"""Elastic attention: per-layer qk/v dims and query-head count, for Qwen3 and Llama alike.

The MLP knob lives in mlp.py; the model-level API that drives both lives in patch.py.

Design (matches the HF attention forward; Qwen3 and Llama differ only in QK-Norm):
  * HF uses `rotate_half` (NeoX): head dim j is paired with j+64, and
    cos/sin = cat([freqs, freqs]). A valid rotary slice must keep WHOLE pairs.
  * We pick a pair ordering (interleaved hi/lo freq by default), select the first
    `qk_active//2` pairs, and gather their dims into the [first-halves | second-halves]
    layout that `rotate_half` expects. QK-Norm is recomputed over the active dims
    with the gathered gamma; RoPE reuses HF's exact `apply_rotary_pos_emb`.
  * At qk_active == 128 the gather is a permutation of all dims, so q.k (and thus the
    whole attention output) is identical to stock HF up to float reassociation
    (verified by m0/parity_test.py -> max|dlogit| ~ 5e-5).
  * v is a plain contiguous prefix (no RoPE). Output zero-pads each head's v_active
    dims back to head_dim and applies the full o_proj == column-slice.

Scaling uses 1/sqrt(qk_active). NOTE (review #5): with QK-Norm the max logit ~ sqrt(qk_active),
so small slices are structurally softer; a learnable per-granularity temperature belongs in
the *trained* module (M2), not the untrained M0 smoke test.

set_elastic_config accepts either an int (global) or a length-n_layers list (per-layer),
so this same module serves M0 (global) and M2/M4 (per-layer).
"""
import torch
import torch.nn.functional as F
from .families import all_attn_classes, family

# Qwen3's head_dim, kept as the default so the 25 existing Qwen3 call sites need no argument.
# Anything head_dim-dependent below derives the value from its arguments instead of this constant,
# so a 64-dim family (SmolLM2/Llama) works by passing n_pairs=head_dim//2 to build_pair_order.
HEAD_DIM = 128
HALF = HEAD_DIM // 2  # 64 pairs

# --- M2 additions: attention backend + per-granularity temperature ---
import torch.nn as nn

ELASTIC_BACKEND = "eager"           # "eager" (M0 parity) or "sdpa" (M2 training; handles qk!=v)
N_GRID = 4          # ModelSpec.qk_grid is always four quarter-steps of head_dim


def grid_index(qk_active: int, head_dim: int) -> int:
    """Which of the four quarter-step rungs `qk_active` sits on, for the per-granularity temperature.

    This used to be the table {32:0, 64:1, 96:2, 128:3}, which is the correct answer only at
    head_dim 128; at head_dim 64 the grid is {16,32,48,64} and the lookup raised KeyError -- which
    is how it was found. The rung is a property of head_dim, so compute it.
    """
    q = head_dim // 4
    r = qk_active // q
    if qk_active % q or not 1 <= r <= N_GRID:
        raise ValueError(f"d_qk={qk_active} is not a quarter-step of head_dim={head_dim}")
    return r - 1


def set_elastic_backend(name):
    global ELASTIC_BACKEND
    assert name in ("eager", "sdpa")
    ELASTIC_BACKEND = name


def add_elastic_temperature(model):
    """Register a learnable per-granularity logit scale on each attention module (review #5:
    max logit ~ sqrt(qk), so small slices are structurally softer). Init 0 -> exp(0)=1 ->
    identical to plain 1/sqrt(qk) at start (parity-preserving); folded into q so it is
    differentiable through SDPA (whose `scale` arg is a non-differentiable float)."""
    for m in model.modules():
        if isinstance(m, all_attn_classes()):
            m.register_parameter("logit_scale", nn.Parameter(torch.zeros(N_GRID)))
    return model


def pair_order_for(spec_or_model, mode: str = "hilo"):
    """build_pair_order for whatever head_dim this model/spec has. Prefer this in new code."""
    hd = getattr(spec_or_model, "head_dim", None) \
        or getattr(getattr(spec_or_model, "config", None), "head_dim", None)
    if hd is None:
        raise ValueError(f"no head_dim on {spec_or_model!r}")
    return build_pair_order(mode, int(hd) // 2)


def build_pair_order(mode: str = "hilo", n_pairs: int = HALF):
    """Order the head_dim/2 rotary pairs. Pair j has the j-th RoPE frequency: j=0 highest
    (local/short-range), j=63 lowest (long-range). 'hilo' interleaves both ends so any
    prefix keeps a mix of local and long-range pairs (review default for 4k training)."""
    if mode == "natural":
        return list(range(n_pairs))
    if mode == "hilo":
        order, lo, hi = [], 0, n_pairs - 1
        while lo <= hi:
            order.append(lo)
            if hi != lo:
                order.append(hi)
            lo += 1
            hi -= 1
        return order
    raise ValueError(mode)


def qk_gather_index(pair_order, qk_active: int) -> torch.Tensor:
    """Length-qk_active gather index into head_dim, laid out as
    [first-halves of selected pairs | second-halves], so HF rotate_half stays valid."""
    half = len(pair_order)          # = head_dim // 2, so this is family-agnostic
    assert qk_active % 2 == 0 and 0 < qk_active <= 2 * half
    npa = qk_active // 2
    sel = pair_order[:npa]
    return torch.tensor(sel + [p + half for p in sel], dtype=torch.long)


def _elastic_rmsnorm(x, weight, eps):
    dt = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (weight * x).to(dt)



def kv_rotate(x, ang):
    """Rotate each RoPE pair of `x` by `ang`, in the [first-halves | second-halves] layout.

    This is the alignment step that makes KV-group pooling invariance-preserving. It must run
    AFTER q_norm/k_norm and BEFORE RoPE, and the reason is measured: Qwen3's QK-Norm applies a
    learnable per-dim gain between projection and RoPE, and a 2D rotation inside a rotary pair
    commutes with diag(gamma) only when the pair's two gains are equal -- p90 asymmetry is 0.686,
    so it does not. Rotating the weights instead costs +0.2335 nats on the k side with no merging
    at all, while the same rotation applied post-norm is exact (identity check: k -0.00034,
    v -0.00078, kv -0.00017 at 1.7B). The v side has no norm and is invariant either way; the
    whole problem is localised to k.

    Post-norm the only surviving constraint is commuting with RoPE, which a block-diagonal
    per-pair rotation satisfies by construction: q^T U^T R_delta U k = q^T R_delta k exactly.

    x    [B, S, H, D] in gathered layout, so dim j and j+D/2 are one rotary pair
    ang  [H, D/2] rotation angle per (head, pair); zeros is the identity
    """
    half = x.shape[-1] // 2
    a, b = x[..., :half], x[..., half:]
    # cos/sin are computed in fp32 and the result cast back, for two separate reasons. Precision:
    # a bf16 sine loses ~3 decimal digits and the angles start at exactly zero, where that error is
    # the entire signal. Dtype: under autocast torch.cos returns fp32 regardless of its input, so
    # letting the product decide the output type silently promotes q to fp32 and then SDPA rejects
    # the bf16 attention mask -- which is how this first ran and died 71 minutes into a smoke test.
    c, sn = torch.cos(ang.float()), torch.sin(ang.float())
    af, bf = a.float(), b.float()
    return torch.cat([af * c - bf * sn, af * sn + bf * c], dim=-1).to(x.dtype)


# Ablation switches for experiments/supernet_fidelity/conversion_ablation.py. The defaults are the
# conversion rules, and training and search never change them. "prefix" keeps the first n_h query
# heads, each still reading its own KV group, instead of balancing heads across groups. "select" keeps
# the first KV group of every pooled bundle instead of their mean.
ABLATION = {"heads": "balanced", "kv": "pool"}


def head_index(n_q, n_kv, n_h):
    """Which query heads to keep, balanced across KV groups.

    Taking a plain prefix of query heads would empty whole KV groups of readers: with 32 heads
    over 8 groups, the first 8 all belong to groups 0 and 1, leaving six groups with no query at
    all. Keeping n_h/n_kv heads from EACH group avoids that, and it is why n_kv must divide n_h.
    """
    if n_h % n_kv:
        raise ValueError(f"n_kv={n_kv} must divide n_h={n_h}")
    per, nrep = n_h // n_kv, n_q // n_kv
    if per > nrep:
        raise ValueError(f"n_h={n_h} needs {per} heads per group but only {nrep} exist")
    return torch.tensor([g * nrep + j for g in range(n_kv) for j in range(per)], dtype=torch.long)


def _family_of(mod):
    """Resolve the family from the attention module's own class, once per module.

    Cached on the instance because this runs inside the forward: the alternative is a dict lookup
    keyed by type on every layer of every step.
    """
    f = getattr(mod, "_family", None)
    if f is None:
        name = type(mod).__module__.rsplit(".", 1)[-1].replace("modeling_", "")
        f = mod._family = family(name)
    return f


def elastic_attention_forward(self, hidden_states, position_embeddings, attention_mask,
                              past_key_value=None, cache_position=None, **kwargs):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    q = self.q_proj(hidden_states).view(hidden_shape)
    k = self.k_proj(hidden_states).view(hidden_shape)
    v = self.v_proj(hidden_states).view(hidden_shape)

    idx = self._qk_idx.to(q.device)
    if hasattr(self, "q_norm"):
        # Qwen3 applies a learnable per-dim RMSNorm to q/k between projection and RoPE. The gain
        # has to be gathered by the same index as the vector, or a narrowed head is normalised
        # against gains belonging to dimensions it no longer has. Llama has no such layer.
        eps = getattr(self.q_norm, "variance_epsilon", 1e-6)
        q = _elastic_rmsnorm(q.index_select(-1, idx), self.q_norm.weight.index_select(0, idx), eps)
        k = _elastic_rmsnorm(k.index_select(-1, idx), self.k_norm.weight.index_select(0, idx), eps)
    else:
        q = q.index_select(-1, idx)
        k = k.index_select(-1, idx)
    v = v[..., : self.v_active]

    # n_h: drop query heads. Survivors are untouched -- attention heads are independent, so this
    # removes terms from the output sum rather than changing any that remain.
    q_idx = getattr(self, "_q_idx", None)
    if q_idx is not None:
        q = q.index_select(2, q_idx.to(q.device))

    # n_kv: align, then mean-pool KV groups. Untrained pooling alone costs ~6.4 nats at 8->4, so
    # the alignment is not a refinement -- it is what makes the axis usable at all. Skipped
    # entirely at full width so the full configuration is bit-identical to the un-merged forward.
    kv_act = getattr(self, "kv_active", None)
    n_kv_full = k.shape[2]
    if kv_act is not None and kv_act < n_kv_full:
        rung = self._kv_rungs.index(kv_act)
        # The angles are stored over the FULL pair count but q/k carry only qk_active dims and v
        # only v_active, so each is truncated to its own active half. That is correct rather than
        # merely convenient: the gather puts the selected pairs first, so a prefix of the angle
        # array lines up with the prefix of pairs a narrowed head actually has -- the same nested
        # structure the width knobs rely on.
        hq = q.shape[-1] // 2
        hv = v.shape[-1] // 2
        # k/q: the rotary gather puts the selected pairs first, so angle index p always means
        # pair_order[p] whatever d_qk is. A prefix of the angles is genuinely nested.
        ak = self.kv_ang_k[rung][:, :hq].to(q.device)              # [n_kv_full, hq]
        # v: NOT nested, and this cost a measured regression before it was caught. v is a plain
        # contiguous prefix, so kv_rotate pairs dim p with p + d_v/2 -- the PLANE ITSELF moves when
        # d_v changes. Angle 0 rotates (0,64) at d_v=128 and (0,32) at d_v=64, so one angle set
        # cannot serve both. Fitted at full width and applied at d_qk=64,d_v=64,n_kv=4 the result
        # was +0.039 nats WORSE than plain pooling, against -0.253 at the width it was fitted for.
        # A separate angle set per d_v rung is the fix: the arrays are indexed by rung already, so
        # this adds one axis and keeps every plane fixed within its own rung.
        vi = self._v_rungs.index(v.shape[-1])
        av = self.kv_ang_v[rung][vi][:, :hv].to(q.device)
        nrep_orig = q.shape[2] // n_kv_full if q.shape[2] >= n_kv_full else 1
        q = kv_rotate(q, ak.repeat_interleave(nrep_orig, dim=0)[: q.shape[2]])
        k = kv_rotate(k, ak)
        v = kv_rotate(v, av)
        g = n_kv_full // kv_act
        B, S = k.shape[:2]
        if ABLATION["kv"] == "select":
            k = k.view(B, S, kv_act, g, k.shape[-1])[:, :, :, 0]
            v = v.view(B, S, kv_act, g, v.shape[-1])[:, :, :, 0]
        else:
            k = k.view(B, S, kv_act, g, k.shape[-1]).mean(3)
            v = v.view(B, S, kv_act, g, v.shape[-1]).mean(3)
        self._kv_unrot = av
    else:
        self._kv_unrot = None

    if ABLATION["heads"] == "prefix" and q_idx is not None:
        # A plain head prefix leaves some groups unread, so each surviving query head gathers the K
        # and V of the group it belonged to in the base model.
        grp = q_idx.to(q.device) // (self.config.num_attention_heads // k.shape[2])
        k = k.index_select(2, grp)
        v = v.index_select(2, grp)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    cos, sin = position_embeddings
    cos = cos.index_select(-1, idx)
    sin = sin.index_select(-1, idx)
    fam = _family_of(self)
    q, k = fam.apply_rope(q, k, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        k, v = past_key_value.update(k, v, self.layer_idx, cache_kwargs)

    # After pooling there are fewer KV heads than num_key_value_groups was computed for, so the
    # repeat factor has to come from the tensors rather than the cached attribute.
    nrep = q.shape[1] // k.shape[1]
    k = fam.repeat_kv(k, nrep)
    v = fam.repeat_kv(v, nrep)

    if hasattr(self, "logit_scale"):  # per-granularity temperature, folded into q (differentiable)
        q = q * torch.exp(self.logit_scale[grid_index(self.qk_active, self.head_dim)])

    # transformers drops the mask entirely when it is purely causal and lets the attention op
    # recover causality via is_causal. Replacing the forward means we must reproduce that, or the
    # model silently attends BIDIRECTIONALLY whenever a caller does not pass an explicit mask.
    # eval/loader.py forces attn_implementation="eager" to avoid exactly this; relying on that
    # convention is fragile, so handle it here.
    causal = attention_mask is None and q.shape[-2] > 1

    if ELASTIC_BACKEND == "sdpa":
        # SDPA supports qk_active != v_active on math/mem-efficient backends; default scale = 1/sqrt(qk_active)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, is_causal=causal)
        out = out.transpose(1, 2).contiguous()
    else:
        scaling = q.shape[-1] ** -0.5
        attn = torch.matmul(q, k.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn = attn + attention_mask[:, :, :, : k.shape[-2]]
        elif causal:
            qlen, klen = q.shape[-2], k.shape[-2]
            band = torch.ones(qlen, klen, dtype=torch.bool, device=q.device).tril(klen - qlen)
            attn = attn.masked_fill(~band, torch.finfo(attn.dtype).min)
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous()

    # Each query head read a v that had been rotated into its KV group's aligned basis, so the
    # output has to come back before o_proj -- o_proj's columns are defined in the original basis.
    # Rotating by -angle is the inverse because the per-pair map is a 2D rotation.
    unrot = getattr(self, "_kv_unrot", None)
    if unrot is not None:
        dv_half = out.shape[-1] // 2
        n_kv_full = unrot.shape[0]
        nrep_orig = out.shape[2] // n_kv_full if out.shape[2] >= n_kv_full else 1
        aq = unrot[:, :dv_half].repeat_interleave(nrep_orig, dim=0)[: out.shape[2]]
        out = kv_rotate(out, -aq.to(out.device))

    b, s, h, dv = out.shape
    n_q_full = self.config.num_attention_heads
    if h == n_q_full:
        padded = out.new_zeros(b, s, n_q_full, self.head_dim)
        padded[..., :dv] = out
    else:
        # scatter the surviving heads back into their own o_proj slots; dropped heads contribute
        # zero, which is exactly equivalent to deleting their o_proj columns but keeps o_proj
        # whole so one weight set serves every n_h.
        wide = out.new_zeros(b, s, n_q_full, dv)
        wide.index_copy_(2, self._q_idx.to(out.device), out)
        padded = out.new_zeros(b, s, n_q_full, self.head_dim)
        padded[..., :dv] = wide
    padded = padded.reshape(b, s, n_q_full * self.head_dim)
    return self.o_proj(padded), None
