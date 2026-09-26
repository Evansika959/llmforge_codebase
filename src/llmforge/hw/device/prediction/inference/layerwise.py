"""Inference for the layerwise delivery bundles, whose feature schema the legacy path cannot read.

A layerwise bundle carries 110 architecture-only feature names and no fitted `profile`, so
`predict_bundle` rejects it at the legacy feature-schema check. The bundle ships the exact feature
code that produced it under `source_snapshot/`, so this module calls that code instead of
reimplementing it. Predictions therefore match the delivered frozen test predictions.

Two bundles of this shape exist and they do not agree:

    layerwise_2000_20260921   targets decode_tok_s, ttft_ms, dynamic_energy_per_token_mj
    layerwise_tpot_2000_20260922   targets tpot_ms, ttft_ms, dynamic_energy_per_token_mj

The first predicts decode throughput and the second predicts time per output token directly, so a
caller must read `pack['targets']` rather than assume a position. Their energies also come from
different measurement revisions, the later one having replaced every group-size-64 row with an
optimized kernel, so values from the two must never be pooled or compared.

The snapshot is imported lazily, on first use. It registers a top-level `scripts` package, and the
parity test for the legacy bundles asserts that loading those leaves no such module behind.
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

LAYER_FIELDS = ("n_h", "n_kv", "d_qk", "d_v", "d_mlp")


def is_layerwise(pack: Dict[str, Any]) -> bool:
    """A delivery bundle with architecture-only layerwise features and no fitted profile."""
    return "profile" not in pack and "models" in pack and len(pack.get("features", ())) > 20


def bundle_root(bundle_path: Path) -> Path:
    """The delivery directory of a bundle stored at <root>/models/<name>.joblib."""
    return Path(bundle_path).resolve().parent.parent


@lru_cache(maxsize=4)
def _snapshot(root: str):
    """Import the bundle's own feature code, adding its snapshot to the import path once."""
    path = str(Path(root) / "source_snapshot")
    if path not in sys.path:
        sys.path.insert(0, path)
    from scripts.prediction_research.layerwise_refit.features import matrix
    from scripts.sweep.layerwise.candidates import architecture_summary, effective_group
    return matrix, architecture_summary, effective_group


@lru_cache(maxsize=4)
def _dataset(root: str) -> Dict[str, Any]:
    """The frozen architectures behind a bundle, grouped by skeleton for domain checks.

    Every row shares one operator profile, which the features need because two of them, the
    parameter count and the quantized file size, depend on the vocabulary the runtime was
    measured with. Feeding the searched model's own vocabulary instead would move the features
    off the protocol the bundle was fitted on.
    """
    rows = json.loads((Path(root) / "dataset_snapshot.json").read_text())["rows"]
    # Compare parsed values rather than serialized text. The delivery writes some profile fields
    # with different JSON types from row to row, which serializes differently while every value the
    # features read stays the same, so a text comparison reports a difference that does not exist.
    profile = rows[0]["architecture"]["operator_profile"]
    if any(r["architecture"]["operator_profile"] != profile for r in rows):
        raise ValueError("the bundle's rows do not share one operator profile")
    _, summary, _ = _snapshot(root)
    groups: Dict[tuple, Dict[str, Any]] = {}
    for r in rows:
        a = r["architecture"]
        g = groups.setdefault((a["n_layer"], a["d_model"]), {k: [] for k in LAYER_FIELDS})
        g.setdefault("total_params_M", []).append(summary(a)["total_params"] / 1e6)
        for layer in a["layers"]:
            for k in LAYER_FIELDS:
                g[k].append(layer[k])
    return {"profile": profile,
            "domains": {key: {k: (min(v), max(v)) for k, v in g.items()} for key, g in groups.items()}}


def uniform_architecture(config: Dict[str, int], root: str) -> Dict[str, Any]:
    """A uniform search config as the ordered-layer architecture the bundle's features expect."""
    _, _, effective_group = _snapshot(root)
    layer = {k: int(config[k]) for k in LAYER_FIELDS}
    layers = [dict(layer) for _ in range(int(config["n_layer"]))]
    dim = int(config["d_model"])
    return dict(n_layer=len(layers), d_model=dim, operator_profile=_dataset(root)["profile"],
                q8_group_size=effective_group(dim, layers), layers=layers)


def architecture_from_individual(ind: Dict[str, Any], root: str) -> Dict[str, Any]:
    """A search individual as an ordered-layer architecture, layer shapes kept as they are.

    The layerwise bundles are fitted overwhelmingly on architectures whose layers differ, so
    collapsing a candidate to one repeated shape throws away the very variation they model. This
    keeps each active layer's own shape.
    """
    _, _, effective_group = _snapshot(root)
    g, layers = ind["globals"], ind["layers"]
    mask = g.get("layer_mask", [True] * len(layers))
    rows = [dict(n_h=int(li["n_head"]), n_kv=int(li["n_kv_group"]), d_qk=int(li["n_qk_head_dim"]),
                 d_v=int(li["n_v_head_dim"]), d_mlp=int(li["mlp_size"]))
            for i, li in enumerate(layers) if i < len(mask) and mask[i]]
    dim = int(g["n_embd"])
    return dict(n_layer=len(rows), d_model=dim, operator_profile=_dataset(root)["profile"],
                q8_group_size=effective_group(dim, rows), layers=rows)


def architecture_in_domain(arch: Dict[str, Any], bundle_path: Path) -> Optional[bool]:
    """Whether every layer of an architecture lies inside the ranges of its skeleton's rows."""
    root = str(bundle_root(bundle_path))
    dom = _dataset(root)["domains"].get((int(arch["n_layer"]), int(arch["d_model"])))
    if dom is None:
        return False
    for layer in arch["layers"]:
        if not all(dom[k][0] <= int(layer[k]) <= dom[k][1] for k in LAYER_FIELDS):
            return False
    _, summary, _ = _snapshot(root)
    params_m = summary(arch)["total_params"] / 1e6
    return bool(dom["total_params_M"][0] <= params_m <= dom["total_params_M"][1])


def predict_architectures(pack: Dict[str, Any], architectures: List[Dict[str, Any]],
                          bundle_path: Path) -> np.ndarray:
    """Predict every target for ordered-layer architectures, in the bundle's own target order."""
    root = str(bundle_root(bundle_path))
    matrix, _, _ = _snapshot(root)
    x, _ = matrix(architectures, pack["features"])
    pred = np.column_stack([np.exp(m.predict(x)) for m in pack["models"]])
    if pred.shape != (len(architectures), 3) or not np.isfinite(pred).all() or not (pred > 0).all():
        raise ValueError("Invalid model predictions")
    return pred


def predict(pack: Dict[str, Any], configs: List[Dict[str, int]], bundle_path: Path) -> np.ndarray:
    """Predict every target for uniform configs, in the bundle's own target order."""
    root = str(bundle_root(bundle_path))
    return predict_architectures(pack, [uniform_architecture(c, root) for c in configs], bundle_path)


def predict_individuals(pack: Dict[str, Any], inds: List[Dict[str, Any]],
                        bundle_path: Path) -> np.ndarray:
    """Predict every target for search individuals, keeping each layer's own shape."""
    root = str(bundle_root(bundle_path))
    return predict_architectures(pack, [architecture_from_individual(i, root) for i in inds],
                                 bundle_path)


def individual_in_domain(ind: Dict[str, Any], bundle_path: Path) -> Optional[bool]:
    """Whether a search individual lies inside the ranges its skeleton's rows cover."""
    root = str(bundle_root(bundle_path))
    return architecture_in_domain(architecture_from_individual(ind, root), bundle_path)


def in_domain(config: Dict[str, int], bundle_path: Path) -> Optional[bool]:
    """Whether a config lies inside the ranges of the rows sharing its depth and residual width.

    The check is per skeleton rather than over the whole dataset. The delivery mixes two base
    models, and the larger one's ranges would admit architectures far outside anything measured
    at the smaller one's depth and width.
    """
    root = str(bundle_root(bundle_path))
    dom = _dataset(root)["domains"].get((int(config["n_layer"]), int(config["d_model"])))
    if dom is None:
        return False
    if not all(dom[k][0] <= int(config[k]) <= dom[k][1] for k in LAYER_FIELDS):
        return False
    _, summary, _ = _snapshot(root)
    params_m = summary(uniform_architecture(config, root))["total_params"] / 1e6
    return bool(dom["total_params_M"][0] <= params_m <= dom["total_params_M"][1])
