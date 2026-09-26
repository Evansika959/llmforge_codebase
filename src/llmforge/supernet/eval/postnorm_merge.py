"""Post-norm rotation-aligned KV merging: make the alignment exactly invariance-preserving.

Weight-level (pre-norm) rotation is NOT invariant on Qwen3: measured +0.2335 nats for the k side
with no merging at all, because QK-Norm applies a learnable per-dim gain gamma between projection
and RoPE, and a 2D rotation inside a RoPE pair only commutes with diag(gamma) when the pair's two
gains are equal (measured p90 asymmetry 0.686). The v side has no norm and is exactly invariant
(+0.0001), which localises the whole problem to k.

Fix: apply the rotation AFTER q_norm/k_norm and BEFORE RoPE. There gamma is already folded into
the vector, so the only constraint left is commuting with RoPE -- satisfied by construction for
block-diagonal per-pair 2D rotations. Score becomes q^T U^T R_delta U k = q^T R_delta k exactly.

Cost of moving post-norm: the deployed model must still compute all G k projections before pooling,
so the k-side parameter saving is lost. The KV *cache* saving -- the actual target -- is unaffected,
and k_proj+v_proj is only ~8% of parameters at 4B.

  python -m llmforge.supernet.eval.postnorm_merge --model qwen3-4b
"""
import argparse
import json

import torch
import torch.nn.functional as F
from transformers.models.qwen3 import modeling_qwen3 as mq

from ..config import SPECS
from ..paths import RUNS
from .pilot_rotation_merge import (collect_kv, groups_for, heldout_texts, nll, orthogonal_align,
                                   pair_rotations)


def rot_last(x, ang, half):
    """Rotate the RoPE pairs of x [..., H, hd] by per-head angles ang [H, half]."""
    c, s = torch.cos(ang).to(x.dtype), torch.sin(ang).to(x.dtype)
    a, b = x[..., :half], x[..., half:]
    return torch.cat([c * a - s * b, s * a + c * b], dim=-1)


def postnorm_forward(spec, ang_k, U_v, n_kv_active, use_k, use_v):
    """Attention forward with post-norm alignment then pooling. ang_k/U_v are per-layer dicts."""
    half = spec.head_dim // 2
    grp = groups_for(spec.n_kv, n_kv_active)
    nrep_orig = spec.n_q // spec.n_kv
    # map each original kv head -> its pooled group index
    g_of = {g: gi for gi, gr in enumerate(grp) for g in gr}

    def fwd(self, hidden_states, position_embeddings, attention_mask,
            past_key_value=None, cache_position=None, **kwargs):
        B, S = hidden_states.shape[:2]
        hd = self.head_dim
        li = self.layer_idx
        q = self.q_proj(hidden_states).view(B, S, -1, hd)
        k = self.k_proj(hidden_states).view(B, S, -1, hd)
        v = self.v_proj(hidden_states).view(B, S, -1, hd)
        q = self.q_norm(q)
        k = self.k_norm(k)                                   # <-- rotation goes AFTER this

        if use_k:
            a = ang_k[li].to(q.device)                       # [n_kv, half]
            aq = a.repeat_interleave(nrep_orig, dim=0)       # q head h uses its own kv head's angle
            q = rot_last(q, aq, half)
            k = rot_last(k, a, half)
        if use_v:
            Uv = U_v[li].to(device=v.device, dtype=v.dtype)  # [n_kv, hd, hd]
            v = torch.einsum("bshd,hde->bshe", v, Uv)

        # pool post-norm within each group
        if n_kv_active < spec.n_kv:
            g = spec.n_kv // n_kv_active
            k = k.view(B, S, n_kv_active, g, hd).mean(3)
            v = v.view(B, S, n_kv_active, g, hd).mean(3)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = mq.apply_rotary_pos_emb(q, k, cos, sin)
        nrep = q.shape[1] // k.shape[1]
        k, v = mq.repeat_kv(k, nrep), mq.repeat_kv(v, nrep)
        # transformers drops the mask when it is purely causal, and recovers causality via
        # is_causal. Replacing the forward means we must reproduce that or attend bidirectionally.
        is_causal = attention_mask is None and q.shape[-2] > 1
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask,
                                             is_causal=is_causal)
        out = out.transpose(1, 2).contiguous()               # [B,S,n_q,hd]

        if use_v:
            # each q head returns to its OWN kv head's basis before o_proj
            Uv = U_v[li].to(device=out.device, dtype=out.dtype)
            Uq = Uv.repeat_interleave(nrep_orig, dim=0)
            out = torch.einsum("bshe,hde->bshd", out, Uq)
        return self.o_proj(out.reshape(B, S, -1)), None

    return fwd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-4b", choices=sorted(SPECS))
    ap.add_argument("--n-texts", type=int, default=48)
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec, dev = SPECS[args.model], "cuda"
    tok = AutoTokenizer.from_pretrained(spec.repo)
    model = AutoModelForCausalLM.from_pretrained(spec.repo, torch_dtype=torch.bfloat16).to(dev).eval()
    model.config.use_cache = False
    cal_texts = heldout_texts(8)
    texts = heldout_texts(args.n_texts, skip=8)   # disjoint from calibration
    orig_fwd = mq.Qwen3Attention.forward

    base = nll(model, tok, texts)
    print(f"base(n_kv={spec.n_kv}) NLL {base:.4f}", flush=True)
    cal = collect_kv(model, tok, cal_texts, spec, n_cal=8)

    half = spec.head_dim // 2

    def fit(n_kv_target):
        """Alignment in the POST-NORM space, fitted to the grouping ACTUALLY being merged.

        Fitting once for a 4-group partition and reusing it at n_kv=2/1 leaves the cross-pair
        phase offset intact, so the average still cancels and the alignment appears not to pay
        off under aggressive pooling -- an artifact of the mis-specified grouping, not a result.
        """
        grp = groups_for(spec.n_kv, n_kv_target)
        a, u = {}, {}
        for li, L in enumerate(model.model.layers):
            gk = L.self_attn.k_norm.weight.detach().float().cpu()
            K = cal[li]["k"]                                    # [n_kv, N, hd] pre-norm
            Khat = K / K.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt() * gk
            a[li] = pair_rotations(Khat, grp, half).float()
            u[li] = orthogonal_align(cal[li]["v"], grp).float()
        return a, u

    ang, Uv = fit(4)          # the invariance check below merges at identity, so any pooled fit works

    res = {"base": base, "rows": {}}
    print("\n=== INVARIANCE at n_kv=8 (rotate post-norm, no merge; delta must be ~0) ===", flush=True)
    for name, uk, uv in [("k", True, False), ("v", False, True), ("kv", True, True)]:
        mq.Qwen3Attention.forward = postnorm_forward(spec, ang, Uv, spec.n_kv, uk, uv)
        d = nll(model, tok, texts)
        print(f"  rotate {name:2}: NLL {d:.4f}  delta {d - base:+.4f}", flush=True)
        res[f"inv_{name}"] = d - base

    print(f"\n=== post-norm aligned merging ===\n{'n_kv':>5} {'plain':>9} {'pn_align_k':>11} "
          f"{'pn_align_v':>11} {'pn_align_kv':>12}", flush=True)
    for nk in [4, 2, 1]:
        row = {}
        ang_nk, Uv_nk = fit(nk)          # refit per level -- see fit() docstring
        for name, uk, uv in [("plain", False, False), ("pn_align_k", True, False),
                             ("pn_align_v", False, True), ("pn_align_kv", True, True)]:
            mq.Qwen3Attention.forward = postnorm_forward(spec, ang_nk, Uv_nk, nk, uk, uv)
            row[name] = nll(model, tok, texts)
        res["rows"][nk] = row
        print(f"{nk:>5} {row['plain']:>9.4f} {row['pn_align_k']:>11.4f} "
              f"{row['pn_align_v']:>11.4f} {row['pn_align_kv']:>12.4f}", flush=True)

    mq.Qwen3Attention.forward = orig_fwd
    out = RUNS / f"postnorm_merge_{spec.key}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
