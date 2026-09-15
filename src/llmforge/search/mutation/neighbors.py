"""1-step neighborhood enumeration for sensitivity-driven mutation.

Strict transition rules:
  - attention_variant `infinite → identity` legal iff layer is "thin enough":
      n_head == 1 AND n_qk_head_dim <= THIN_QK AND n_v_head_dim <= THIN_V
  - attention_variant `identity → infinite` is a *coupled* move that lands in
      (n_head=1, n_qk_head_dim=64, n_v_head_dim=64). No other dim is touched.
  - layer_mask `True → False` legal iff layer is at the floor:
      attention_variant == "identity" AND mlp_size <= MIN_MLP
      AND active_count - 1 >= L_min
  - layer_mask `False → True` is a coupled move that lands in
      (attention_variant=identity, n_head=1, n_qk_head_dim=64, n_v_head_dim=64,
       mlp_size=MIN_MLP, n_kv_group=1, n_cproj=1). Other genes default to the
      spec's low value.

Dead-genome bits of identity-attention layers (n_head, n_kv_group,
n_qk_head_dim, n_v_head_dim, n_cproj) are skipped — mutating them has no
effect on val_loss, so they would waste sensitivity-map slots.
"""
from __future__ import annotations

from typing import Any, Dict, List, NamedTuple, Tuple

from llmforge.search.hetero_space import HeteroSearchSpace, Individual


# Thresholds for the coupled variant flip.
THIN_QK = 128
THIN_V = 128

# Variant flip target dims when waking up an identity layer.
WAKE_QK = 64
WAKE_V = 64

# Genes that have no effect on val_loss for an identity-attention layer.
IDENTITY_DEAD_GENES = {"n_head", "n_kv_group", "n_qk_head_dim",
                       "n_v_head_dim", "n_cproj"}


class GeneRef(NamedTuple):
    """Tag for a (scope, index, gene, direction) edge in the gene graph."""
    scope: str      # "global" | "layer" | "layer_variant" | "layer_mask"
    index: int      # layer index or -1 for globals
    gene: str       # gene name (e.g. "mlp_size") or "" for variant/mask flips
    direction: str  # "shrink" | "grow" | "to_identity" | "to_thin_attn"
                    # | "disable" | "enable"


def _clone_individual(x: Individual) -> Dict[str, Any]:
    """Deep-ish copy of an Individual into a plain dict suitable for mutation."""
    g = dict(x["globals"])
    if "layer_mask" in g:
        g["layer_mask"] = list(g["layer_mask"])
    layers = [dict(li) for li in x["layers"]]
    return {"globals": g, "layers": layers}


def _is_thin_attn(layer: Dict[str, Any]) -> bool:
    return (
        int(layer.get("n_head", 1)) == 1
        and int(layer.get("n_qk_head_dim", 64)) <= THIN_QK
        and int(layer.get("n_v_head_dim", 64)) <= THIN_V
    )


def _is_min_mlp(layer: Dict[str, Any], space: HeteroSearchSpace) -> bool:
    mlp_spec = space.layer_spec.get("mlp_size", None)
    if mlp_spec is None:
        return False
    return int(layer.get("mlp_size", 0)) <= int(mlp_spec["low"])


def _spec_low(space: HeteroSearchSpace, gene: str, fallback) -> Any:
    spec = space.layer_spec.get(gene, None)
    if spec is None:
        return fallback
    if spec.get("type") == "cat":
        return spec["choices"][0]
    return spec.get("low", fallback)


def _step_int(layer_or_globals: Dict[str, Any], gene: str,
              spec: Dict[str, Any], direction: str) -> Tuple[bool, Any]:
    """Apply a ±step move on an int gene. Returns (ok, new_value).
    ok=False if the move would leave bounds.

    Honors an optional `exclude` list: stepping continues in the same
    direction until it lands on an allowed value, so the neighbor is a
    real reachable point rather than one repair would snap away."""
    step = int(spec.get("step", 1))
    lo = int(spec["low"])
    hi = int(spec["high"])
    excl = set(spec.get("exclude") or ())
    cur = int(layer_or_globals.get(gene, lo))
    new = cur
    while True:
        new = new - step if direction == "shrink" else new + step
        if new < lo or new > hi:
            return False, cur
        if new not in excl:
            return True, new


def enumerate_neighbors(
    parent: Individual, space: HeteroSearchSpace
) -> List[Tuple[Individual, GeneRef]]:
    """Return all 1-step neighbors of `parent` under the strict rules.

    Each returned (Individual, GeneRef) pair is post-repair, so downstream
    surrogate scoring sees the same form the evaluator would see.
    """
    out: List[Tuple[Individual, GeneRef]] = []

    g_in = parent["globals"]
    mask = list(g_in.get("layer_mask", [True] * space.L_max))
    if len(mask) < space.L_max:
        mask = mask + [False] * (space.L_max - len(mask))
    active_count = sum(1 for m in mask if m)

    # ── globals: numeric ±step ──
    for gname, gspec in space.globals.items():
        if gspec.get("type") != "int":
            # categorical globals (use_concat_heads etc.) are not in the
            # 1-step neighborhood — they're large jumps the surrogate would
            # see as out-of-distribution.
            continue
        for direction in ("shrink", "grow"):
            ok, new_val = _step_int(g_in, gname, gspec, direction)
            if not ok or new_val == int(g_in.get(gname, 0)):
                continue
            cand = _clone_individual(parent)
            cand["globals"][gname] = new_val
            out.append((space.repair(cand),
                        GeneRef("global", -1, gname, direction)))

    # ── per-bundle ──
    # Gene-tie bundling: each bundle is `bsize` (=bundle_size, K) consecutive
    # layers sharing one spec, so we enumerate at the bundle level and
    # broadcast each edit across the bundle's layers. The representative
    # (first) layer carries the gene state; its index is recorded in the
    # GeneRef. bundle_size=1 makes every bundle a single layer, exactly
    # reproducing the per-layer search. In bundle mode the mask is frozen
    # (freeze_layer_mask), so the enable/disable branches below never fire —
    # bundles are never added or removed.
    bsize = max(1, int(getattr(space, "bundle_size", 1) or 1))
    n_bundles = space.L_max // bsize
    n_layers = len(parent["layers"])

    def _bundle_idxs(b: int) -> List[int]:
        start = b * bsize
        return [start + o for o in range(bsize) if start + o < n_layers]

    for b in range(n_bundles):
        idxs = _bundle_idxs(b)
        if not idxs:
            continue
        r = idxs[0]  # representative layer index for this bundle

        if not mask[r]:
            # inactive bundle → only the "enable" coupled flip is available
            if space.freeze_layer_mask:
                continue
            # respect categorical choices that don't include "identity"
            var_spec = space.layer_spec.get("attention_variant", None)
            if var_spec is not None and var_spec.get("type") == "cat":
                if "identity" not in var_spec["choices"]:
                    # search space doesn't allow identity at all — skip
                    continue
            cand = _clone_individual(parent)
            # land every layer in the bundle in the floor state
            floor_layer = {
                "n_head": 1,
                "n_kv_group": 1,
                "mlp_size": int(_spec_low(space, "mlp_size", 512)),
                "n_qk_head_dim": WAKE_QK,
                "n_v_head_dim": WAKE_V,
                "n_cproj": int(_spec_low(space, "n_cproj", 1)),
                "attention_variant": "identity",
            }
            for j in idxs:
                base = dict(cand["layers"][j])  # carry over unlisted keys
                base.update(floor_layer)
                cand["layers"][j] = base
                cand["globals"]["layer_mask"][j] = True
            out.append((space.repair(cand),
                        GeneRef("layer_mask", r, "", "enable")))
            continue

        layer = parent["layers"][r]
        is_identity = layer.get("attention_variant", "infinite") == "identity"

        # per-layer numeric ±step (skip dead bits on identity layers)
        for lname, lspec in space.layer_spec.items():
            if lspec.get("type") != "int":
                continue
            if is_identity and lname in IDENTITY_DEAD_GENES:
                continue
            for direction in ("shrink", "grow"):
                ok, new_val = _step_int(layer, lname, lspec, direction)
                if not ok or new_val == int(layer.get(lname, 0)):
                    continue
                cand = _clone_individual(parent)
                for j in idxs:
                    cand["layers"][j][lname] = new_val
                out.append((space.repair(cand),
                            GeneRef("layer", r, lname, direction)))

        # variant flip — strict rules
        var_spec = space.layer_spec.get("attention_variant", None)
        if var_spec is not None and var_spec.get("type") == "cat":
            allowed = set(var_spec["choices"])
            if is_identity:
                # identity → infinite, coupled with thin dims
                if "infinite" in allowed:
                    cand = _clone_individual(parent)
                    for j in idxs:
                        L = cand["layers"][j]
                        L["attention_variant"] = "infinite"
                        L["n_head"] = 1
                        L["n_qk_head_dim"] = WAKE_QK
                        L["n_v_head_dim"] = WAKE_V
                    out.append((space.repair(cand),
                                GeneRef("layer_variant", r, "",
                                        "to_thin_attn")))
            else:
                if "identity" in allowed and _is_thin_attn(layer):
                    cand = _clone_individual(parent)
                    for j in idxs:
                        cand["layers"][j]["attention_variant"] = "identity"
                    out.append((space.repair(cand),
                                GeneRef("layer_variant", r, "",
                                        "to_identity")))

        # bundle disable True → False — strict rules. Disabling removes the
        # whole bundle (B active layers), so L_min is checked against that.
        if not space.freeze_layer_mask:
            can_disable = (
                is_identity
                and _is_min_mlp(layer, space)
                and (active_count - len(idxs)) >= space.L_min
            )
            if can_disable:
                cand = _clone_individual(parent)
                for j in idxs:
                    cand["globals"]["layer_mask"][j] = False
                out.append((space.repair(cand),
                            GeneRef("layer_mask", r, "", "disable")))

    return out
