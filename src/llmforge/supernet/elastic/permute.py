"""Exact weight permutations that change WHICH units a prefix slice selects.

Every elastic knob except d_qk selects a contiguous PREFIX of units -- MLP rows, value dims,
query heads -- so the ORDER of those units silently decides what every sub-network gets. That
order is an accident of pretraining; nothing in this repo has ever reordered it. Asking "would a
better order help?" therefore needs a way to try a different one.

Permuting a unit axis together with the axis that consumes it is an exact symmetry of the full
model: the same products are summed in a different order. So `permute + prefix` selects an
ARBITRARY subset while leaving the full-width model unchanged. That single property makes this
both the measurement tool (compare today's prefix against random subsets of the same width) and,
should it ever pay off, the implementation of importance-ordered slicing itself.

d_qk is deliberately absent. It is already parameterised by `pair_order`, and its units are RoPE
frequency PAIRS rather than individual dims, so a different order needs no weight surgery -- pass
a different `pair_order` to `set_elastic_config`.

Convention, used identically on both sides of every knob: the permuted model's unit `j` holds what
the original model's unit `perm[j]` held, i.e. every tensor is `index_select`-ed BY `perm`. A
prefix of width `w` on the permuted model is therefore exactly the subset `perm[:w]` of the
original. `torch.argsort(perm)` undoes it.
"""
from contextlib import contextmanager

import torch
from .families import all_attn_classes

KNOBS = ("d_mlp", "d_v", "n_h")


def _attns(model):
    return sorted((m for m in model.modules() if isinstance(m, all_attn_classes())),
                  key=lambda m: m.layer_idx)


def _mlps(model):
    return [L.mlp for L in model.model.layers]


def _check(perm, n, what):
    perm = torch.as_tensor(perm, dtype=torch.long)
    if perm.shape != (n,) or not torch.equal(perm.sort().values, torch.arange(n)):
        raise ValueError(f"{what}: expected a permutation of 0..{n - 1}, got shape {tuple(perm.shape)}")
    return perm


# ------------------------------------------------------------------ per-knob permutations

@torch.no_grad()
def _apply_mlp(model, perms):
    """gate/up rows and down columns. `mlp_active` keeps the first `m` rows of gate/up and the
    first `m` columns of down, so one permutation over the intermediate axis covers the knob."""
    for mlp, p in zip(_mlps(model), perms):
        p = _check(p, mlp.gate_proj.weight.shape[0], "d_mlp perm").to(mlp.gate_proj.weight.device)
        mlp.gate_proj.weight.copy_(mlp.gate_proj.weight.index_select(0, p))
        mlp.up_proj.weight.copy_(mlp.up_proj.weight.index_select(0, p))
        mlp.down_proj.weight.copy_(mlp.down_proj.weight.index_select(1, p))


@torch.no_grad()
def _apply_v(model, perms):
    """v_proj rows within every KV head, and the o_proj columns that read them.

    `v_active` is one scalar per layer, so all KV heads keep the SAME prefix -- one permutation
    per layer, shared across heads, is exactly the comparison class of the current slicing rule.

    The forward writes value dim `d` of query head `i` into o_proj column `i * head_dim + d`
    (`padded[..., :dv] = wide`, scattered by the ORIGINAL query index). Query head `i` reads KV
    head `i // n_rep`, so permuting a KV head's value dims requires the same permutation on the
    o_proj columns of every query head in its group. With one shared permutation per layer that
    collapses to: apply it inside every head block of both tensors.
    """
    for m, p in zip(_attns(model), perms):
        hd = m.head_dim
        p = _check(p, hd, "d_v perm").to(m.v_proj.weight.device)
        n_kv = m.config.num_key_value_heads
        n_q = m.config.num_attention_heads
        vw = m.v_proj.weight.view(n_kv, hd, -1)
        vw.copy_(vw.index_select(1, p))
        ow = m.o_proj.weight.view(-1, n_q, hd)
        ow.copy_(ow.index_select(2, p))


@torch.no_grad()
def _apply_heads(model, perms):
    """Whole query heads: q_proj row blocks and o_proj column blocks.

    Must permute WITHIN each KV group. `head_index` keeps the first `n_h // n_kv` heads of every
    group so that no group is left without a reader; a permutation that moved heads across groups
    would change which KV head each survivor reads and would not be a symmetry.

    q_norm/k_norm act on the head dim and are shared across heads, so they are untouched.
    """
    for m, p in zip(_attns(model), perms):
        hd, n_q = m.head_dim, m.config.num_attention_heads
        n_kv = m.config.num_key_value_heads
        p = _check(p, n_q, "n_h perm").to(m.q_proj.weight.device)
        nrep = n_q // n_kv
        if not all(int(p[g * nrep + j]) // nrep == g for g in range(n_kv) for j in range(nrep)):
            raise ValueError("n_h perm must permute heads WITHIN each KV group")
        qw = m.q_proj.weight.view(n_q, hd, -1)
        qw.copy_(qw.index_select(0, p))
        ow = m.o_proj.weight.view(-1, n_q, hd)
        ow.copy_(ow.index_select(1, p))


_APPLY = {"d_mlp": _apply_mlp, "d_v": _apply_v, "n_h": _apply_heads}


def apply_permutation(model, knob, perms):
    """Permute `knob`'s unit axis in place. Returns the perms that undo it."""
    if knob not in _APPLY:
        raise ValueError(f"knob must be one of {KNOBS}, got {knob!r} "
                         "(d_qk is reordered via pair_order, not weights)")
    perms = [torch.as_tensor(p, dtype=torch.long) for p in perms]
    _APPLY[knob](model, perms)
    return [torch.argsort(p) for p in perms]


@contextmanager
def permuted(model, knob, perms):
    """Apply a permutation for the duration of the block, then restore exactly.

    Restoration is another gather of the same values, so it is bit-exact -- no drift even across
    many rounds, which matters because this is called in a loop.
    """
    inv = apply_permutation(model, knob, perms)
    try:
        yield model
    finally:
        apply_permutation(model, knob, inv)


# ------------------------------------------------------------------ perm generators

def units(model, knob):
    """How many units the knob's prefix is taken from."""
    a, mlp = _attns(model)[0], _mlps(model)[0]
    return {"d_mlp": mlp.gate_proj.weight.shape[0],
            "d_v": a.head_dim,
            "n_h": a.config.num_attention_heads}[knob]


def random_perms(model, knob, rng):
    """One independent random permutation per layer, respecting the knob's structure.

    Per-layer independence is the right null: unit `k` of layer 3 has nothing to do with unit `k`
    of layer 7, so a single shared permutation would test a needlessly narrow alternative.
    """
    n_layers = len(_mlps(model))
    n = units(model, knob)
    if knob != "n_h":
        return [torch.as_tensor(rng.permutation(n), dtype=torch.long) for _ in range(n_layers)]
    n_kv = _attns(model)[0].config.num_key_value_heads
    nrep = n // n_kv
    out = []
    for _ in range(n_layers):
        p = torch.cat([torch.as_tensor(g * nrep + rng.permutation(nrep), dtype=torch.long)
                       for g in range(n_kv)])
        out.append(p)
    return out


def identity_perms(model, knob):
    n_layers = len(_mlps(model))
    return [torch.arange(units(model, knob)) for _ in range(n_layers)]
