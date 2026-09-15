"""Named uniform architecture sets for rank-fidelity studies.

`arch_set` selects uniform shapes in four cost levels with several shapes per level. Within a level
the weight cost is close but the shape differs, which is the discrimination a search needs and the
easiest thing for a supernet to get wrong. Dense models need d_qk == d_v, so every shape uses one
per-head dimension for both.

Scripts under experiments/supernet_fidelity import it from here rather than from each other.
"""
from ..elastic.sampler import head_grid
from ..space import ElasticConfig


def arch_set(spec, k=4):
    """4 cost levels x k shapes, keyed L<level>_<letter>."""
    hg, g = head_grid(spec), spec.mlp_grid
    # (d_qk=d_v, n_h, d_mlp) -- dense models need d_qk == d_v
    cands = [(q, h, m) for q in spec.qk_grid for h in hg for m in g]
    W0 = ElasticConfig.full(spec).weight_params(include_embed=False)
    rows = []
    for q, h, m in cands:
        c = ElasticConfig.uniform(spec, q, q, n_h=h, d_mlp=m)
        rows.append((c.weight_params(include_embed=False) / W0, c.kv_frac(), q, h, m))
    out, used = {}, set()
    letters = "abcdefgh"
    for li, target in enumerate([0.85, 0.70, 0.55, 0.40]):
        near = sorted(rows, key=lambda r: abs(r[0] - target))
        picked = []
        for r in near:
            if len(picked) == k:
                break
            if abs(r[0] - target) > 0.06 or (r[2], r[3], r[4]) in used:
                continue
            # prefer shapes unlike the ones already picked at this level
            if any(p[2] == r[2] and p[3] == r[3] for p in picked):
                continue
            picked.append(r); used.add((r[2], r[3], r[4]))
        for ai, r in enumerate(picked):
            out[f"L{li+1}_{letters[ai]}"] = dict(w=r[0], kv=r[1], d_qk=r[2], n_h=r[3], d_mlp=r[4])
    return out
