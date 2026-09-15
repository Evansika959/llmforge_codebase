"""Sandwich-rule config sampling for supernet uptraining.

Two APIs:
  * the original 2-knob one (`sandwich`, `sample_config`) drives d_qk/d_v only and is kept so the
    0.6B and 1.7B runs stay reproducible;
  * `sandwich_ab` covers the combined A+B space -- d_qk, d_v, n_h, d_mlp -- which are all the same
    class of operation (select a subset of live weights; survivors keep their function).

MIN IS THE TRUE CORNER. An earlier design raised the min config off the extreme AND down-weighted
its loss. Pre-code review pointed out that the sandwich rule's justification is that MIN and MAX
BOUND the interior, so moving MIN up leaves every claim below it an extrapolation. MIN is therefore
the actual minimum here, and if its gradient proves too noisy the remedy is the explicit
`min_weight` knob in the trainer rather than a silently shifted floor.
"""
import random

GRID = [32, 64, 96, 128]
N_LAYERS = 28
# Up-weight the top two rungs of the four quarter-step widths, BY RANK. This used to be the table
# {32:1, 64:1, 96:2, 128:2}, keyed on the absolute dim, which silently did nothing at head_dim 64:
# SmolLM2's grid is {16,32,48,64}, so no key matched except 32 and 64 -- and 64 there is the FULL
# width, which picked up the weight written for Qwen3's second-narrowest rung. Measured over 40k
# draws the distribution came out uniform (.250/.254/.250/.246) instead of 1:1:2:2, i.e. a
# different sampling recipe from the one every recorded Qwen3 result is defined against.
HI_WEIGHTS_BY_RANK = (1, 1, 2, 2)


def set_hi_weights(w):
    """Override the rung sampling weights, to correct the nested-prefix update imbalance.

    Nested prefixes mean a unit is trained whenever ANY sampled config is at least that wide, so
    under (1,1,2,2) the first quarter of every weight matrix is included with probability 1.00 and
    the last quarter with 0.33 -- a 3x imbalance that dedicated training does not have, since there
    every unit gets every update. It shows up as a measured, size-dependent predictor bias: the
    slice over-rates narrow configurations by 0.058-0.112 nats relative to trained models, exactly
    along the axis a size/quality search trades off.

    Tilting toward wide rungs shrinks the imbalance. It needs no architecture prior -- unlike
    importance-ordered masks, which would bake the answer into the training distribution and make
    the predictor reproduce its own prior.
    """
    global HI_WEIGHTS_BY_RANK
    HI_WEIGHTS_BY_RANK = tuple(w)


def full_config(n=N_LAYERS):
    return [128] * n, [128] * n


def min_config(n=N_LAYERS, floor=32):
    return [floor] * n, [floor] * n


def _draw(grid, hi_weight, rng):
    if hi_weight:
        w = HI_WEIGHTS_BY_RANK
        pool = [g for i, g in enumerate(grid) for _ in range(w[i] if i < len(w) else 1)]
        return rng.choice(pool)
    return rng.choice(grid)


def sample_config(n=N_LAYERS, grid=GRID, per_layer=True, hi_weight=True, rng=random):
    if per_layer:
        return ([_draw(grid, hi_weight, rng) for _ in range(n)],
                [_draw(grid, hi_weight, rng) for _ in range(n)])
    qk, v = _draw(grid, hi_weight, rng), _draw(grid, hi_weight, rng)
    return [qk] * n, [v] * n


def sandwich(step, total_steps, n=N_LAYERS, grid=GRID, K=2, anneal_frac=0.3, rng=random):
    per_layer = step >= anneal_frac * total_steps
    configs = [full_config(n), min_config(n, min(grid))]
    for _ in range(K):
        configs.append(sample_config(n, grid, per_layer=per_layer, rng=rng))
    return configs


# ---------------------------------------------------------------- layer blocks

# Contiguous layer blocks fitted to MEASURED per-layer sensitivity, not cut every N layers.
# Source: experiments/layer_sensitivity_top1.py, which shrinks one layer at a time and records
# top-1 disagreement with the full model -- temperature-invariant, and the quantity that compounds
# over a generated sequence, unlike cross-entropy.
#
# Attention and MLP have genuinely different depth profiles (attn peaks at layer 0 and the 22-24
# band; MLP peaks at the last two layers), so they get separate partitions. One partition cannot
# be right for both.
#
# Why blocks at all: training sampled per layer (144 dims, 1.6e82 configs) while the search ran
# over 4-layer groups (36 dims, 3.5e20) -- so capacity was spent on a space 1e61 times larger than
# the one ever searched. Blocks align the two and cut the dimension the supernet must generalise
# over, while leaving ~1e11 configurations for design-space exploration.
MEASURED_BLOCKS = {
    "qwen3-4b": {
        "attn": [(0, 1), (1, 12), (12, 22), (22, 25), (25, 36)],
        "mlp":  [(0, 1), (1, 4), (4, 34), (34, 36)],
    },
}


def blocks_for(spec, kind="attn", k=5):
    """Measured blocks when available, else k contiguous blocks of equal size."""
    m = MEASURED_BLOCKS.get(spec.key, {}).get(kind)
    if m:
        assert m[0][0] == 0 and m[-1][1] == spec.n_layers, f"blocks must tile {spec.n_layers}"
        return m
    n, out, lo = spec.n_layers, [], 0
    for i in range(k):
        hi = spec.n_layers * (i + 1) // k
        out.append((lo, hi)); lo = hi
    return out


def _expand(vals, blocks, n):
    out = [None] * n
    for v, (lo, hi) in zip(vals, blocks):
        for i in range(lo, hi):
            out[i] = v
    assert all(x is not None for x in out), "blocks do not tile the layers"
    return out


# ---------------------------------------------------------------- combined A+B space

def head_grid(spec):
    """Query-head counts reachable while n_kv stays at the base value.

    n_kv must divide n_h so every KV group keeps a whole number of readers, so the usable grid is
    the spec's head grid intersected with the multiples of n_kv: {8,16} at 16Q/8KV, {8,16,32} at
    32Q/8KV.
    """
    return [h for h in spec.nh_grid if h % spec.n_kv == 0]


def nkv_options(spec, n_h):
    """Legal n_kv values given a query-head count. Kept for callers that fix n_h first."""
    return [k for k in spec.nkv_grid if k <= n_h and n_h % k == 0]


def head_options(spec, n_kv):
    """Legal n_h values given the KV-group count already drawn. THIS is the right direction.

    n_kv must divide n_h -- each KV group keeps the same number of query heads, or a group is
    starved of readers entirely. Drawing n_h first and then a legal n_kv wastes most of the space:
    head_grid() intersects nh_grid with multiples of the model's BASE n_kv, which at Qwen3-0.6B
    leaves {8,16} and at 4B leaves {8,16,32}. But once n_kv is itself elastic the constraint is
    against the ACTIVE value, and a smaller n_kv admits finer head counts:

        0.6B  n_kv 8 -> {8,16}   4 -> {4,8,16}   2 -> {2,4,8,16}   1 -> {2,4,8,16}
        4B    n_kv 8 -> {8,16,32}  4/2/1 -> {4,8,16,32}

    so the reachable (n_h, n_kv) space goes from 2 combinations to 13 at 0.6B and 3 to 15 at 4B.
    SmolLM2 is unchanged: its n_kv is prime, so nkv_grid is a single value and the intersection is
    the same set it always was.

    This uses nh_grid as it stands rather than widening it. The halving rule leaves gaps -- 4B has
    no n_h=24 even though 24 is a multiple of 8 -- but changing that rule would move the Qwen3
    grids every recorded result is defined against, which is a separate decision.
    """
    return [h for h in spec.nh_grid if h % n_kv == 0]


def sample_ab(spec, per_layer=True, hi_weight=True, rng=random, blocks=False, with_nkv=False):
    """One draw over (d_qk, d_v, n_h, d_mlp), each a per-layer list; +n_kv when with_nkv.

    blocks=True draws one value per MEASURED layer block instead of per layer, so the sampler
    covers the same space the search explores rather than one 1e61 times larger.

    n_kv is drawn UNIFORMLY over its legal set rather than through the hi_weight tilt the width
    knobs use. Unlike those, n_kv is not a nested selection -- pooling mixes heads, so its damage
    is qualitatively different and there is no measurement yet saying which direction to bias. The
    one time this project tilted rung weights without measuring first, the bias it was meant to fix
    grew (0.112 -> 0.165).
    """
    n, qk_grid, hg, mg = spec.n_layers, spec.qk_grid, head_grid(spec), spec.mlp_grid
    if blocks:
        ba, bm = blocks_for(spec, "attn"), blocks_for(spec, "mlp")
        if with_nkv:
            kv_b = [rng.choice(spec.nkv_grid) for _ in ba]
            nh_b = [rng.choice(head_options(spec, k)) for k in kv_b]
        else:
            kv_b, nh_b = None, [rng.choice(hg) for _ in ba]
        out = (_expand([_draw(qk_grid, hi_weight, rng) for _ in ba], ba, n),
               _expand([_draw(qk_grid, hi_weight, rng) for _ in ba], ba, n),
               _expand(nh_b, ba, n),
               _expand([rng.choice(mg) for _ in bm], bm, n))
        if with_nkv:
            out = out + (_expand(kv_b, ba, n),)
        return out
    if per_layer:
        kv = [rng.choice(spec.nkv_grid) for _ in range(n)] if with_nkv else None
        nh = ([rng.choice(head_options(spec, k)) for k in kv] if with_nkv
              else [rng.choice(hg) for _ in range(n)])
        out = ([_draw(qk_grid, hi_weight, rng) for _ in range(n)],
               [_draw(qk_grid, hi_weight, rng) for _ in range(n)],
               nh,
               [rng.choice(mg) for _ in range(n)])
        return out + (kv,) if with_nkv else out
    kv1 = rng.choice(spec.nkv_grid) if with_nkv else spec.n_kv
    h = rng.choice(head_options(spec, kv1)) if with_nkv else rng.choice(hg)
    qk, v, m = _draw(qk_grid, hi_weight, rng), _draw(qk_grid, hi_weight, rng), rng.choice(mg)
    out = ([qk] * n, [v] * n, [h] * n, [m] * n)
    return out + ([kv1] * n,) if with_nkv else out


def full_ab(spec, with_nkv=False):
    n = spec.n_layers
    out = ([spec.head_dim] * n, [spec.head_dim] * n, [spec.n_q] * n, [spec.d_mlp] * n)
    return out + ([spec.n_kv] * n,) if with_nkv else out


def min_ab(spec, with_nkv=False):
    """The true corner: smallest value of every knob.

    With n_kv elastic the corner moves: the smallest n_kv unlocks head counts the base n_kv
    forbids, so MIN is the smallest legal n_h under the smallest n_kv, not head_grid()[0].
    """
    n = spec.n_layers
    k0 = spec.nkv_grid[0] if with_nkv else spec.n_kv
    h0 = (head_options(spec, k0) or head_grid(spec))[0]
    out = ([spec.qk_grid[0]] * n, [spec.qk_grid[0]] * n, [h0] * n, [spec.mlp_grid[0]] * n)
    return out + ([k0] * n,) if with_nkv else out


def sandwich_ab(step, total_steps, spec, K=2, anneal_frac=0.1, rng=random, blocks=False,
                with_nkv=False):
    """[FULL, MIN] + K random draws, annealed from global to per-layer sampling.

    Coverage anneal: early steps draw one value shared by all layers so the model first learns to
    be robust globally, then per-layer so it learns heterogeneous combinations.

    with_nkv appends a fifth per-layer list. It changes what MIN means: the corner now includes the
    fewest KV groups as well, which at Qwen3 is a 8x cache cut on top of the width cuts, and that
    corner starts ~9.5 nats worse than full width against ~14 for the four-knob corner. Whether
    training closes that gap the way it closes the others is the open question -- the four width
    knobs are pure selection, while pooling MIXES heads, so the four-knob result does not transfer.
    """
    per_layer = step >= anneal_frac * total_steps
    configs = [full_ab(spec, with_nkv), min_ab(spec, with_nkv)]
    for _ in range(K):
        configs.append(sample_ab(spec, per_layer=per_layer, rng=rng,
                                 blocks=blocks and per_layer, with_nkv=with_nkv))
    return configs
