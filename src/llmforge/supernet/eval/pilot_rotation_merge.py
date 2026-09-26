"""Pilot: does rotation-ALIGNED KV-head merging beat plain mean-pooling, with zero training?

Ainslie et al. (GQA, 2023) convert MHA->GQA by mean-pooling the K/V projections of each group.
Mean-pooling implicitly assumes the heads in a group already share a basis. They have no reason
to. Since attention only sees inner products, an orthogonal U applied to BOTH a key head and its
query heads leaves the full model bit-identical -- so we may rotate each head into any basis for
free, and choose the basis that makes averaging destroy least (Procrustes alignment before
averaging; cf. Git Re-Basin for models, here for heads).

WHAT ROPE ALLOWS. The score is q^T U^T R_delta U k, so U must commute with every RoPE rotation.
RoPE is block-diagonal 2x2 rotations, and the matrices commuting with all 2D rotations are the
2D rotations themselves => U is block-diagonal with ONE FREE ANGLE PER ROPE PAIR (64 angles per
head at head_dim=128). v carries no RoPE, so v admits a FULL orthogonal rotation, absorbed into
the matching o_proj column block.

THE ASSUMPTION THIS SCRIPT TESTS RATHER THAN ASSUMES. Qwen3 applies QK-Norm -- RMSNorm with a
learnable per-dim gain -- between projection and RoPE. Rotating the *weights* (pre-norm) is only
score-preserving if U commutes with diag(gamma_q * gamma_k), which for a 2D rotation inside a
RoPE pair requires the two gains in that pair to be equal. Step 1 measures that directly; step 2
verifies end-to-end by applying the rotations at n_kv=8, where the merge is the identity and NLL
must therefore not move. If step 2 fails, the pre-norm formulation is wrong and the alignment
belongs post-norm (which still saves cache, but no longer yields a plain merged projection).

  python -m llmforge.supernet.eval.pilot_rotation_merge --model qwen3-4b --n-texts 48
"""
import argparse
import copy
import json
import time

import torch
import torch.nn.functional as F

from ..config import SPECS
from ..paths import RUNS


# ---------------------------------------------------------------- held-out text

def heldout_texts(n, max_chars=4000, skip=0):
    """Docs from the LAST 10BT shard. The proxy/training stream is read in sorted shard order
    from the first shard, so the tail is genuinely unseen -- the existing FineWeb-NLL proxy takes
    the FIRST documents, which are also the first documents of training."""
    import glob
    import os

    import pyarrow.parquet as pq

    from ..paths import RAW
    cand = sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--HuggingFaceFW--fineweb-edu/snapshots/*/sample/10BT/*.parquet")))
    if not cand:
        cand = sorted(glob.glob(str(RAW / "*.parquet")))
    if not cand:
        raise SystemExit("no 10BT parquet found")
    f = pq.ParquetFile(cand[-1])
    out, seen = [], 0
    for rg in range(f.num_row_groups):
        for v in f.read_row_group(rg, columns=["text"])["text"]:
            t = v.as_py()
            if t and len(t) > 500:
                seen += 1
                if seen > skip:              # `skip` keeps calibration disjoint from evaluation
                    out.append(t[:max_chars])
            if len(out) >= n:
                return out
    return out


@torch.no_grad()
def nll(model, tok, texts, max_len=1024, dev="cuda"):
    tot_nll = tot_tok = 0.0
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(dev)
        if ids.shape[1] < 8:
            continue
        logits = model(input_ids=ids).logits[:, :-1].float()
        tgt = ids[:, 1:]
        l = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="sum")
        tot_nll += l.item(); tot_tok += tgt.numel()
    return tot_nll / tot_tok


# ---------------------------------------------------------------- alignment math

def pair_rotations(K, groups, half):
    """Per-RoPE-pair 2D rotations aligning heads within each group.

    K: [n_kv, N, head_dim] pre-norm keys on calibration data.
    Represent pair (j, j+half) as a complex vector per head; a 2D rotation is multiplication by
    e^{i phi}. Maximising ||sum_g e^{i phi_g} z_g||^2 is phase synchronisation: take the leading
    eigenvector of the GxG complex Gram matrix and read off its phases. Exact up to the standard
    relaxation, and each problem is at most 8x8.

    Returns angles [n_kv, half].
    """
    n_kv, _, hd = K.shape
    ang = torch.zeros(n_kv, half, dtype=torch.float64)
    for grp in groups:
        if len(grp) == 1:
            continue
        for j in range(half):
            z = torch.stack([K[g, :, j].double() + 1j * K[g, :, j + half].double() for g in grp])
            C = z @ z.conj().T                      # GxG Hermitian Gram
            w, V = torch.linalg.eigh(C)
            u = V[:, -1]                            # leading eigenvector
            phi = -torch.angle(u)
            phi = phi - phi[0]                      # gauge: first head fixed => G=1 is identity
            for k, g in enumerate(grp):
                ang[g, j] = phi[k]
    return ang


def apply_pair_rotation(W, ang, half):
    """Rotate rows of one head's projection by per-pair angles. W: [head_dim, hidden]."""
    out = W.clone()
    c, s = torch.cos(ang).to(W.dtype), torch.sin(ang).to(W.dtype)
    a, b = W[:half], W[half:]
    out[:half] = c[:, None] * a - s[:, None] * b
    out[half:] = s[:, None] * a + c[:, None] * b
    return out


def orthogonal_align(V, groups):
    """Full orthogonal Procrustes alignment for v heads (no RoPE => unconstrained).

    V: [n_kv, N, head_dim]. Iterate: rotate each head to the current mean, recompute. Returns
    [n_kv, head_dim, head_dim] orthogonal matrices.
    """
    n_kv, _, hd = V.shape
    U = torch.eye(hd, dtype=torch.float64).repeat(n_kv, 1, 1)
    for grp in groups:
        if len(grp) == 1:
            continue
        X = torch.stack([V[g].double() for g in grp])          # [G, N, hd]
        for _ in range(8):
            M = torch.einsum("gnd,gde->gne", X, U[list(grp)]).mean(0)   # current mean
            for k, g in enumerate(grp):
                # argmin_U ||X_g U - M||  =>  U = A B^T from SVD of X_g^T M
                A, _, Bt = torch.linalg.svd(X[k].T @ M)
                R = A @ Bt
                if torch.det(R) < 0:                            # keep it a rotation
                    A[:, -1] *= -1
                    R = A @ Bt
                U[g] = R
    return U


# ---------------------------------------------------------------- model surgery

def collect_kv(model, tok, texts, spec, dev="cuda", n_cal=8, max_len=512):
    """Pre-norm per-head keys and values on calibration text, one tensor per layer."""
    layers = model.model.layers
    store = {i: {"k": [], "v": []} for i in range(len(layers))}
    hooks = []

    def mk(i, which):
        def hook(mod, inp, out):
            store[i][which].append(out.detach().float().reshape(-1, spec.n_kv, spec.head_dim).cpu())
        return hook

    for i, L in enumerate(layers):
        hooks.append(L.self_attn.k_proj.register_forward_hook(mk(i, "k")))
        hooks.append(L.self_attn.v_proj.register_forward_hook(mk(i, "v")))
    with torch.no_grad():
        for t in texts[:n_cal]:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(dev)
            model(input_ids=ids)
    for h in hooks:
        h.remove()
    return {i: {w: torch.cat(store[i][w]).permute(1, 0, 2).contiguous() for w in ("k", "v")}
            for i in store}


def groups_for(n_kv_full, n_kv_active):
    g = n_kv_full // n_kv_active
    return [list(range(i * g, (i + 1) * g)) for i in range(n_kv_active)]


def build_variant(model, orig, spec, n_kv_active, mode, cal, dev="cuda", rot_n_kv=None):
    """Rewrite k/v/q/o projections in place for one (n_kv, mode) variant.

    rot_n_kv lets the rotation be fitted to a DIFFERENT partition than the one merged. It exists
    for the exactness check: fit rotations for a pooled grouping, then merge at n_kv=8 (identity),
    so the measured delta reflects the rotation alone. Leaving it None (the default) always fits
    the rotation to the grouping actually being merged, which is what every sweep must do.
    """
    half = spec.head_dim // 2
    grp = groups_for(spec.n_kv, n_kv_active)
    rgrp = groups_for(spec.n_kv, rot_n_kv if rot_n_kv is not None else n_kv_active)
    nrep = spec.n_q // spec.n_kv                      # q heads per original kv head

    for li, L in enumerate(model.model.layers):
        Wk = orig[li]["k"].clone().to(dev)            # [n_kv*hd, hidden]
        Wv = orig[li]["v"].clone().to(dev)
        Wq = orig[li]["q"].clone().to(dev)            # [n_q*hd, hidden]
        Wo = orig[li]["o"].clone().to(dev)            # [hidden, n_q*hd]
        hd = spec.head_dim
        Kh = [Wk[g * hd:(g + 1) * hd] for g in range(spec.n_kv)]
        Vh = [Wv[g * hd:(g + 1) * hd] for g in range(spec.n_kv)]

        if mode in ("align_k", "align_kv"):
            ang = pair_rotations(cal[li]["k"], rgrp, half)
            for g in range(spec.n_kv):
                Kh[g] = apply_pair_rotation(Kh[g], ang[g].to(dev), half)
                for h in range(g * nrep, (g + 1) * nrep):     # same U on this head's queries
                    Wq[h * hd:(h + 1) * hd] = apply_pair_rotation(
                        Wq[h * hd:(h + 1) * hd], ang[g].to(dev), half)
        if mode in ("align_v", "align_kv"):
            U = orthogonal_align(cal[li]["v"], rgrp)
            for g in range(spec.n_kv):
                Ug = U[g].to(device=dev, dtype=Wv.dtype)
                Vh[g] = Ug.T @ Vh[g]                          # rotate v rows
                for h in range(g * nrep, (g + 1) * nrep):     # o_proj absorbs the inverse
                    Wo[:, h * hd:(h + 1) * hd] = Wo[:, h * hd:(h + 1) * hd] @ Ug

        newK = torch.cat([torch.stack([Kh[g] for g in gr]).mean(0) for gr in grp])
        newV = torch.cat([torch.stack([Vh[g] for g in gr]).mean(0) for gr in grp])

        L.self_attn.k_proj.weight.data = newK
        L.self_attn.v_proj.weight.data = newV
        L.self_attn.q_proj.weight.data = Wq
        L.self_attn.o_proj.weight.data = Wo
        L.self_attn.config.num_key_value_heads = n_kv_active
        L.self_attn.num_key_value_groups = spec.n_q // n_kv_active
    model.config.num_key_value_heads = n_kv_active
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-4b", choices=sorted(SPECS))
    ap.add_argument("--n-texts", type=int, default=48)
    ap.add_argument("--n-cal", type=int, default=8)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    spec = SPECS[args.model]
    dev = "cuda"
    print(f"[pilot] {spec.key}: {spec.n_layers}L {spec.n_q}Q/{spec.n_kv}KV hd{spec.head_dim}",
          flush=True)

    tok = AutoTokenizer.from_pretrained(spec.repo)
    model = AutoModelForCausalLM.from_pretrained(spec.repo, torch_dtype=torch.bfloat16).to(dev)
    model.eval()
    model.config.use_cache = False

    cal_texts = heldout_texts(args.n_cal)                       # first n_cal docs -> calibration
    texts = heldout_texts(args.n_texts, skip=args.n_cal)        # DISJOINT docs -> evaluation
    print(f"[pilot] {len(cal_texts)} calibration + {len(texts)} evaluation docs "
          f"(disjoint) from the LAST 10BT shard", flush=True)

    # ---- step 1: is QK-Norm's gain pair-uniform? (decides if pre-norm rotation is valid)
    half = spec.head_dim // 2
    rel = torch.cat([  # ALL layers -- layer 0 alone understates this by ~5x
        ((m[:half] - m[half:]).abs() / (m[:half].abs() + m[half:].abs() + 1e-9) * 2)
        for L in model.model.layers
        for m in [L.self_attn.q_norm.weight.detach().float().cpu()
                  * L.self_attn.k_norm.weight.detach().float().cpu()]])
    print(f"[step1] gamma_q*gamma_k pair asymmetry over ALL layers: median {rel.median():.4f} "
          f"p90 {rel.quantile(0.9):.4f} max {rel.max():.4f}   (0 = rotation exactly valid)",
          flush=True)

    orig = {i: {"k": L.self_attn.k_proj.weight.data.clone(),
                "v": L.self_attn.v_proj.weight.data.clone(),
                "q": L.self_attn.q_proj.weight.data.clone(),
                "o": L.self_attn.o_proj.weight.data.clone()}
            for i, L in enumerate(model.model.layers)}

    t0 = time.time()
    base = nll(model, tok, texts)
    print(f"[base] unpooled n_kv={spec.n_kv}: NLL {base:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    print("[pilot] collecting calibration k/v ...", flush=True)
    cal = collect_kv(model, tok, cal_texts, spec, n_cal=args.n_cal)

    results = {"model": spec.key, "base_nll": base,
               "gamma_pair_asym_median": float(rel.median()), "rows": []}

    # ---- step 2: exactness check -- rotations at n_kv=8 must not move NLL
    build_variant(model, orig, spec, spec.n_kv, "align_kv", cal, rot_n_kv=4)
    x = nll(model, tok, texts)
    print(f"[step2] rotations applied at n_kv={spec.n_kv} (merge is identity): NLL {x:.4f} "
          f"vs {base:.4f}  delta {x-base:+.4f}   <- must be ~0 for the transform to be valid",
          flush=True)
    results["exactness_nll"] = x
    results["exactness_delta"] = x - base

    # ---- step 3: plain vs aligned at each pooling level
    print(f"\n{'n_kv':>5} {'KV/tok':>9} {'plain':>9} {'align_k':>9} {'align_kv':>9}  {'best gain':>10}")
    for nk in [g for g in spec.nkv_grid if g < spec.n_kv][::-1]:
        row = {"n_kv": nk, "kv_bytes": spec.n_layers * nk * spec.head_dim * 2 * 2}
        for mode in ("plain", "align_k", "align_kv"):
            build_variant(model, orig, spec, nk, mode, cal)
            row[mode] = nll(model, tok, texts)
        best = min(row["align_k"], row["align_kv"])
        row["gain_vs_plain"] = row["plain"] - best
        results["rows"].append(row)
        print(f"{nk:>5} {row['kv_bytes']/1024:>8.0f}K {row['plain']:>9.4f} "
              f"{row['align_k']:>9.4f} {row['align_kv']:>9.4f}  {row['gain_vs_plain']:>+10.4f}",
              flush=True)

    out = RUNS / f"pilot_rotation_merge_{spec.key}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
