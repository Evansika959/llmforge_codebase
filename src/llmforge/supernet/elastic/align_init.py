"""Closed-form initialisation of the KV alignment angles (the align=procrustes ablation).

The default trainer starts every alignment angle at zero and learns it. This module instead fits the
angles once, before training, by per-pair Procrustes alignment on calibration activations, so an
ablation can freeze them there and ask whether learning them matters.

The fit is the phase synchronisation in eval/pilot_rotation_merge.pair_rotations: each rotary pair
of each head is a complex number, a per-pair rotation multiplies it by exp(i*phi), and the angles
that maximise the norm of the group sum are the phases of the leading eigenvector of the group's
Gram matrix. Maximising the norm of the sum is the same as minimising each head's distance to the
group mean, i.e. the error that mean-pooling makes.

The angles are fitted in exactly the space attention.kv_rotate rotates:
  k  after the k_norm (Qwen3) and in the rotary gather layout of `pair_order`, so angle p belongs
     to pair pair_order[p], as in the trained parameters;
  v  per d_v rung, on the contiguous prefix v[:, :d_v], whose pairs are (p, p + d_v/2).
"""
import torch

from ..eval.pilot_rotation_merge import pair_rotations
from .attention import qk_gather_index
from .families import all_attn_classes


def _groups(n_kv_full, n_kv_active):
    g = n_kv_full // n_kv_active
    return [list(range(i * g, (i + 1) * g)) for i in range(n_kv_active)]


@torch.no_grad()
def collect_kv(model, batches, pair_order):
    """Run the model over `batches` and return per-layer (k, v), each [n_kv, N, head_dim] fp32.

    k is post-norm and in the rotary gather layout; v is in its natural layout. `batches` yields
    (input_ids, attention_mask, position_ids) triples; the model must be set to full width.
    """
    attns = sorted((m for m in model.modules() if isinstance(m, all_attn_classes())),
                   key=lambda m: m.layer_idx)
    store = {m.layer_idx: {"k": [], "v": []} for m in attns}
    hooks = []

    def mk(m, which):
        def hook(_mod, _inp, out):
            n_kv, hd = m.config.num_key_value_heads, m.head_dim
            x = out.detach().reshape(-1, n_kv, hd)
            if which == "k":
                if hasattr(m, "k_norm"):
                    x = m.k_norm(x)
                x = x.index_select(-1, qk_gather_index(pair_order, hd).to(x.device))
            store[m.layer_idx][which].append(x.float().cpu())
        return hook

    for m in attns:
        hooks.append(m.k_proj.register_forward_hook(mk(m, "k")))
        hooks.append(m.v_proj.register_forward_hook(mk(m, "v")))
    try:
        for ids, mask, pos in batches:
            model.model(input_ids=ids, attention_mask=mask, position_ids=pos)
    finally:
        for h in hooks:
            h.remove()
    return {li: {w: torch.cat(s[w]).permute(1, 0, 2).contiguous() for w in ("k", "v")}
            for li, s in store.items()}


def merge_error(x, groups):
    """Sum over groups of the squared distance of each head to its group mean. x: [n_kv, N, D]."""
    return sum(float(((x[g] - x[g].mean(0, keepdim=True)) ** 2).sum()) for g in groups)


def rotate_pairs(x, ang):
    """Apply per-head per-pair angles to x [n_kv, N, D] (pairs (p, p+D/2)), like kv_rotate."""
    half = x.shape[-1] // 2
    a, b = x[..., :half], x[..., half:]
    c, s = torch.cos(ang)[:, None, :], torch.sin(ang)[:, None, :]
    return torch.cat([a * c - b * s, a * s + b * c], dim=-1)


@torch.no_grad()
def fit_procrustes_alignment(model, kv):
    """Overwrite every kv_ang_k / kv_ang_v with the Procrustes angles for its rung.

    `kv` is the output of collect_kv. Returns a per-layer summary of the merge error before and
    after alignment, summed over rungs, for k and for each v rung.
    """
    attns = sorted((m for m in model.modules() if isinstance(m, all_attn_classes())),
                   key=lambda m: m.layer_idx)
    report = []
    for m in attns:
        if not getattr(m, "_kv_rungs", None):
            continue
        n_kv_full, half = m.config.num_key_value_heads, m.head_dim // 2
        K, V = kv[m.layer_idx]["k"], kv[m.layer_idx]["v"]
        row = {"layer": m.layer_idx, "k_before": 0.0, "k_after": 0.0, "v_before": 0.0,
               "v_after": 0.0}
        for ri, r in enumerate(m._kv_rungs):
            grp = _groups(n_kv_full, r)
            ak = pair_rotations(K, grp, half).float()                  # [n_kv, half]
            m.kv_ang_k.data[ri] = ak.to(m.kv_ang_k.device, m.kv_ang_k.dtype)
            row["k_before"] += merge_error(K, grp)
            row["k_after"] += merge_error(rotate_pairs(K, ak), grp)
            for vi, dv in enumerate(m._v_rungs):
                Vd = V[..., :dv]
                av = pair_rotations(Vd, grp, dv // 2).float()          # [n_kv, dv/2]
                m.kv_ang_v.data[ri, vi].zero_()
                m.kv_ang_v.data[ri, vi, :, : dv // 2] = av.to(m.kv_ang_v.device,
                                                              m.kv_ang_v.dtype)
                row["v_before"] += merge_error(Vd, grp)
                row["v_after"] += merge_error(rotate_pairs(Vd, av), grp)
        report.append(row)
    return report
