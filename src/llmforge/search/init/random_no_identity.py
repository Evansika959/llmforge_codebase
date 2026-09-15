"""Generate N random archs from a search-space yaml with all active layers
forced to attention_variant="infinite".

Usage:
  python -m init.random_no_identity \
      --search_space search_space_def/search_space_200Mv2.yaml \
      --n 24 --max_layers 40 --min_layers 8 --seed 42 \
      --out ckpts/<exp>/init_no_identity.json

Output JSON is a list of {globals, layers} dicts, compatible with
run_cosearch.py --init_individuals.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import yaml

# Adjust path so we can import the search_space module
THIS_DIR = Path(__file__).resolve().parent
PROJECT = THIS_DIR.parent
sys.path.insert(0, str(PROJECT))
from llmforge.search.hetero_space import HeteroSearchSpace  # noqa: E402


def force_no_identity(arch_dict: dict) -> dict:
    """Mutate `attention_variant` to `infinite` on every active layer."""
    g = arch_dict.get("globals", {})
    mask = g.get("layer_mask", [True] * len(arch_dict.get("layers", [])))
    for i, m in enumerate(mask):
        if not m or i >= len(arch_dict["layers"]):
            continue
        arch_dict["layers"][i]["attention_variant"] = "infinite"
    return arch_dict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--search_space", type=str, required=True)
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--max_layers", type=int, default=40)
    p.add_argument("--min_layers", type=int, default=8)
    p.add_argument("--bundled_search", action="store_true",
                   help="Gene-tie bundled search mode (see run_cosearch.py).")
    p.add_argument("--bundle_size", type=int, default=1,
                   help="[--bundled_search only] bundle size K.")
    p.add_argument("--num_bundles", type=int, default=None,
                   help="[--bundled_search only] number of bundles B.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    random.seed(args.seed)

    with open(args.search_space, "r") as f:
        cfg = yaml.safe_load(f)
    space = HeteroSearchSpace.from_dicts(
        cfg["global_spec"], cfg["layer_spec"],
        L_max=args.max_layers, L_min=args.min_layers,
        bundle_size=args.bundle_size, num_bundles=args.num_bundles,
        bundled_search=args.bundled_search,
    )

    archs = []
    for _ in range(args.n):
        ind = space.sample()
        ind = force_no_identity(ind)
        ind = space.repair(ind)
        archs.append(ind.to_dict() if hasattr(ind, "to_dict") else dict(ind))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(archs, f, indent=2)
    # Sanity report
    id_layers = 0
    total_active = 0
    for a in archs:
        mask = a["globals"].get("layer_mask", [])
        for i, m in enumerate(mask):
            if not m: continue
            total_active += 1
            if a["layers"][i].get("attention_variant") == "identity":
                id_layers += 1
    print(f"Wrote {len(archs)} archs to {out}")
    print(f"  active layers total: {total_active}, identity: {id_layers} "
          f"(should be 0)")


if __name__ == "__main__":
    main()
