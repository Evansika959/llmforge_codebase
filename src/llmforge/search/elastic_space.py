"""Search space over the slices of one trained elastic supernet.

A genome assigns one grid value per knob to each layer block. Grids come from the supernet's
ModelSpec, so every point the search visits is a slice the supernet was trained to serve.

    knob   Individual layer key   grid
    n_h    n_head                 head_grid(spec), the multiples of n_kv inside spec.nh_grid, or
                                  all of spec.nh_grid when n_kv is a knob
    n_kv   n_kv_group             spec.nkv_grid, every divisor of the base n_kv, with with_nkv only
    d_qk   n_qk_head_dim          spec.qk_grid, quarter steps of the base head dimension
    d_v    n_v_head_dim           spec.qk_grid
    d_mlp  mlp_size               spec.mlp_grid, quarter steps of the base MLP width

Without with_nkv the number of KV groups stays at the base value, the space of the width-only
supernets. With it, n_kv is an attention knob, and every block keeps n_kv dividing n_h, the
condition under which each KV group keeps the same number of query heads. The operators restore
that condition by lowering n_kv to the largest grid value that divides n_h. Every layer stays
active. The attention knobs share the attention partition and d_mlp uses the MLP partition.

    uniform    one block that covers every layer
    blocks     blocks_for(spec): measured sensitivity blocks where they exist (Qwen3-4B), otherwise
               n_blocks contiguous blocks of equal size
    per_layer  one block per layer

Write the YAML files that document each space with
    python -m llmforge.search.elastic_space --write-configs configs/search_spaces
"""
from __future__ import annotations

import argparse
import itertools
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .individual import Individual

ATTN_KNOBS = ("n_head", "n_qk_head_dim", "n_v_head_dim")
MLP_KNOBS = ("mlp_size",)
KNOBS = ATTN_KNOBS + MLP_KNOBS
KV_KNOB = "n_kv_group"
PARTITIONS = ("uniform", "blocks", "per_layer")
PAPER_NAMES = {"n_head": "n_h", "n_qk_head_dim": "d_qk", "n_v_head_dim": "d_v", "mlp_size": "d_mlp",
               KV_KNOB: "n_kv"}


class ElasticSearchSpace:
    def __init__(self, base_model: str, partition: str = "blocks", n_blocks: int = 5,
                 block_size: int = 1024, mutation_step_prob: float = 0.7,
                 seed: Optional[int] = None, with_nkv: bool = False, gqa_dims: bool = False):
        from ..supernet.config import SPECS
        from ..supernet.elastic.sampler import blocks_for, head_grid

        if base_model not in SPECS:
            raise ValueError(f"unknown base model {base_model!r}, choose from {sorted(SPECS)}")
        if partition not in PARTITIONS:
            raise ValueError(f"partition must be one of {PARTITIONS}")
        self.spec = spec = SPECS[base_model]
        self.base_model = base_model
        self.partition = partition
        self.n_blocks = int(n_blocks)
        self.block_size = int(block_size)
        self.mutation_step_prob = float(mutation_step_prob)
        self.rng = random.Random(seed)
        self.with_nkv = bool(with_nkv)
        # KV groups are an attention knob, so the gene sits with the other attention knobs.
        self.knobs = ATTN_KNOBS + ((KV_KNOB,) if self.with_nkv else ()) + MLP_KNOBS
        # A smaller n_kv makes head counts legal that the base n_kv does not divide, so with n_kv
        # elastic n_h takes the full nh_grid and n_kv | n_h is kept per block instead of by the grid.
        self.grids = {"n_head": [int(h) for h in (spec.nh_grid if self.with_nkv else head_grid(spec))]}
        if self.with_nkv:
            self.grids[KV_KNOB] = [int(k) for k in spec.nkv_grid]
        self.grids.update({
            "n_qk_head_dim": [int(q) for q in spec.qk_grid],
            "n_v_head_dim": [int(q) for q in spec.qk_grid],
            "mlp_size": [int(m) for m in spec.mlp_grid],
        })
        # The grouped-query ablation. Conventional attention gives query, key and value one shared head
        # dimension and does not search it, so pinning both grids to the base head dimension is what a GQA
        # parameterization exposes. The knob tuple is unchanged and every operator already skips a knob whose
        # grid holds one value, so no operator needs to know about this. full() is unchanged as well, which
        # keeps the full-width anchor identical to the unrestricted space.
        self.gqa_dims = bool(gqa_dims)
        if self.gqa_dims:
            self.grids["n_qk_head_dim"] = [int(spec.head_dim)]
            self.grids["n_v_head_dim"] = [int(spec.head_dim)]
        n = spec.n_layers
        if partition == "uniform":
            attn = mlp = [(0, n)]
        elif partition == "per_layer":
            attn = mlp = [(i, i + 1) for i in range(n)]
        else:
            attn = [tuple(b) for b in blocks_for(spec, "attn", k=self.n_blocks)]
            mlp = [tuple(b) for b in blocks_for(spec, "mlp", k=self.n_blocks)]
        self.blocks = {"attn": attn, "mlp": mlp}
        for kind, bl in self.blocks.items():
            tiles = bl[0][0] == 0 and bl[-1][1] == n and all(a[1] == b[0] for a, b in zip(bl, bl[1:]))
            if not tiles:
                raise ValueError(f"{kind} blocks {bl} do not tile {n} layers")

    # ---- construction from and to YAML -----------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str, partition: Optional[str] = None,
                  seed: Optional[int] = None) -> "ElasticSearchSpace":
        cfg = yaml.safe_load(Path(path).read_text())
        declared = cfg.get("partition", "blocks")
        space = cls(cfg["base_model"], partition=partition or declared,
                    n_blocks=cfg.get("n_blocks", 5), block_size=cfg.get("block_size", 1024),
                    mutation_step_prob=cfg.get("mutation_step_prob", 0.7), seed=seed,
                    with_nkv=cfg.get("with_nkv", False), gqa_dims=cfg.get("gqa_dims", False))
        for knob, grid in (cfg.get("knobs") or {}).items():
            if knob not in space.grids:
                raise ValueError(f"{path}: knob {knob} is not in the space, set with_nkv for n_kv_group")
            if [int(x) for x in grid] != space.grids[knob]:
                raise ValueError(f"{path}: grid for {knob} is {grid}, the supernet spec gives "
                                 f"{space.grids[knob]}")
        if space.partition == declared and "blocks" in cfg:
            for kind, bl in cfg["blocks"].items():
                if [tuple(b) for b in bl] != space.blocks[kind]:
                    raise ValueError(f"{path}: {kind} blocks {bl} differ from {space.blocks[kind]}")
        return space

    def to_yaml_dict(self) -> Dict[str, Any]:
        s = self.spec
        fixed = {"n_layer": s.n_layers, "n_embd": s.hidden}
        if not self.with_nkv:
            fixed[KV_KNOB] = s.n_kv
        fixed.update({"vocab_size": s.vocab, "tie_embeddings": s.tied, "family": s.family,
                      "attention_variant": "infinite", "mlp_variant": "swiglu"})
        out = {
            "base_model": self.base_model,
            "hf_repo": s.repo,
            "partition": self.partition,
            "n_blocks": self.n_blocks,
            "block_size": self.block_size,
            "mutation_step_prob": self.mutation_step_prob,
        }
        if self.with_nkv:
            out["with_nkv"] = True
        if self.gqa_dims:
            out["gqa_dims"] = True
        out.update({
            "knobs": {k: list(v) for k, v in self.grids.items()},
            "fixed": fixed,
            "blocks": {k: [list(b) for b in v] for k, v in self.blocks.items()},
        })
        return out

    def describe(self) -> Dict[str, Any]:
        d = self.to_yaml_dict()
        d["n_genes"] = self.n_genes
        d["log10_size"] = round(self.log10_size(), 3)
        d["default_mutation_rate"] = self.default_mutation_rate
        return d

    # ---- genome ------------------------------------------------------------------------------

    def knob_blocks(self, knob: str) -> List[Tuple[int, int]]:
        return self.blocks["mlp" if knob in MLP_KNOBS else "attn"]

    @property
    def n_genes(self) -> int:
        return sum(len(self.knob_blocks(k)) for k in self.knobs)

    @property
    def default_mutation_rate(self) -> float:
        return round(min(0.5, max(0.1, 1.5 / self.n_genes)), 4)

    def head_pairs(self) -> List[Tuple[int, int]]:
        """(n_h, n_kv) pairs a block can take: n_kv divides n_h."""
        kvs = self.grids.get(KV_KNOB, [self.spec.n_kv])
        return [(h, k) for h in self.grids["n_head"] for k in kvs if h % k == 0]

    def log10_size(self) -> float:
        if not self.with_nkv:
            return sum(len(self.knob_blocks(k)) * math.log10(len(self.grids[k])) for k in KNOBS)
        # n_h and n_kv are one joint choice per attention block, over the pairs with n_kv | n_h.
        per_attn = len(self.head_pairs()) * len(self.grids["n_qk_head_dim"]) * len(self.grids["n_v_head_dim"])
        return (len(self.blocks["attn"]) * math.log10(per_attn)
                + len(self.blocks["mlp"]) * math.log10(len(self.grids["mlp_size"])))

    def genome_of(self, ind: Dict[str, Any]) -> Dict[str, List[int]]:
        layers = ind["layers"]
        if len(layers) != self.spec.n_layers:
            raise ValueError(f"expected {self.spec.n_layers} layers, got {len(layers)}")
        out = {}
        for k in self.knobs:
            vals = []
            for lo, hi in self.knob_blocks(k):
                seen = {int(layers[i][k]) for i in range(lo, hi)}
                if len(seen) != 1:
                    raise ValueError(f"{k} is not constant inside block [{lo}, {hi}): {sorted(seen)}")
                vals.append(seen.pop())
            out[k] = vals
        return out

    def individual(self, genome: Dict[str, List[int]]) -> Individual:
        s = self.spec
        layers = [{"n_head": None, "n_kv_group": s.n_kv, "n_qk_head_dim": None, "n_v_head_dim": None,
                   "mlp_size": None, "n_cproj": 1, "attention_variant": "infinite"}
                  for _ in range(s.n_layers)]
        for k in self.knobs:
            blocks = self.knob_blocks(k)
            if len(genome[k]) != len(blocks):
                raise ValueError(f"{k}: {len(genome[k])} values for {len(blocks)} blocks")
            for (lo, hi), v in zip(blocks, genome[k]):
                if int(v) not in self.grids[k]:
                    raise ValueError(f"{k}={v} is off the grid {self.grids[k]}")
                for i in range(lo, hi):
                    layers[i][k] = int(v)
        for i, li in enumerate(layers):
            if li["n_head"] % li[KV_KNOB]:
                raise ValueError(f"layer {i}: n_kv={li[KV_KNOB]} does not divide n_h={li['n_head']}")
        g = {"base_model": self.base_model, "n_embd": s.hidden, "block_size": self.block_size,
             "vocab_size": s.vocab, "tie_embeddings": s.tied, "mlp_variant": "swiglu",
             "use_concat_heads": True, "attention_variant": "infinite",
             "layer_mask": [True] * s.n_layers}
        return Individual(g, layers)

    def _legal(self, genome: Dict[str, List[int]]) -> Dict[str, List[int]]:
        """Lower n_kv to the largest grid value that divides n_h, in every block where it does not."""
        if self.with_nkv:
            kvs = self.grids[KV_KNOB]
            for j, (h, k) in enumerate(zip(genome["n_head"], genome[KV_KNOB])):
                if h % k:
                    genome[KV_KNOB][j] = max(x for x in kvs if h % x == 0)
        return genome

    def key(self, ind: Dict[str, Any]) -> str:
        """Canonical, partition-independent key: run-length encoding of each knob over layers.

        The KV-group part is appended only when n_kv is a knob, so the width-only spaces keep their keys."""
        parts = []
        for k in KNOBS + ((KV_KNOB,) if self.with_nkv else ()):
            vals = [int(li[k]) for li in ind["layers"]]
            runs = [f"{v}x{len(list(g))}" for v, g in itertools.groupby(vals)]
            parts.append(f"{PAPER_NAMES[k]}={','.join(runs)}")
        return f"{self.base_model}|" + ";".join(parts)

    def to_elastic_config(self, ind: Dict[str, Any]):
        from ..supernet.space import ElasticConfig

        s = self.spec
        L = ind["layers"]
        return ElasticConfig(s, d_qk=[int(li["n_qk_head_dim"]) for li in L],
                             d_v=[int(li["n_v_head_dim"]) for li in L],
                             n_kv=[int(li.get(KV_KNOB, s.n_kv)) for li in L],
                             n_h=[int(li["n_head"]) for li in L],
                             d_mlp=[int(li["mlp_size"]) for li in L]).validate()

    # ---- named points --------------------------------------------------------------------

    def uniform(self, n_head: int, n_qk_head_dim: int, n_v_head_dim: int, mlp_size: int,
                n_kv_group: Optional[int] = None) -> Individual:
        vals = {"n_head": n_head, "n_qk_head_dim": n_qk_head_dim, "n_v_head_dim": n_v_head_dim,
                "mlp_size": mlp_size}
        if self.with_nkv:
            vals[KV_KNOB] = self.spec.n_kv if n_kv_group is None else n_kv_group
        elif n_kv_group not in (None, self.spec.n_kv):
            raise ValueError(f"n_kv_group={n_kv_group} needs a space with with_nkv")
        return self.individual({k: [int(vals[k])] * len(self.knob_blocks(k)) for k in self.knobs})

    def full(self) -> Individual:
        return self.uniform(**{k: max(self.grids[k]) for k in self.knobs})

    def smallest(self) -> Individual:
        return self.uniform(**{k: min(self.grids[k]) for k in self.knobs})

    def enumerate_uniform(self) -> List[Individual]:
        out = []
        for vals in itertools.product(*(self.grids[k] for k in self.knobs)):
            v = dict(zip(self.knobs, vals))
            if self.with_nkv and v["n_head"] % v[KV_KNOB]:
                continue
            out.append(self.uniform(**v))
        return out

    # ---- operators used by llmforge.search.nsga2.Population ------------------------------

    def sample(self) -> Individual:
        genome = {k: [self.rng.choice(self.grids[k]) for _ in self.knob_blocks(k)] for k in self.knobs}
        if self.with_nkv:
            # n_kv first, then n_h among the head counts that n_kv divides, as the supernet sampler draws them.
            for j, kv in enumerate(genome[KV_KNOB]):
                genome["n_head"][j] = self.rng.choice([h for h in self.grids["n_head"] if h % kv == 0])
        return self.individual(genome)

    def repair(self, ind: Dict[str, Any]) -> Individual:
        """Snap every value to its grid, make each block constant using its first layer, and keep n_kv | n_h."""
        genome = {}
        for k in self.knobs:
            grid = self.grids[k]
            vals = []
            for lo, _ in self.knob_blocks(k):
                layer = ind["layers"][lo]
                x = int(layer.get(KV_KNOB, self.spec.n_kv)) if k == KV_KNOB else int(layer[k])
                vals.append(min(grid, key=lambda g: abs(g - x)))
            genome[k] = vals
        return self.individual(self._legal(genome))

    def crossover(self, a: Dict[str, Any], b: Dict[str, Any],
                  crossover_rate: float = 0.9) -> Tuple[Individual, Individual]:
        """Uniform crossover over block genes, applied with probability `crossover_rate`."""
        ga, gb = self.genome_of(a), self.genome_of(b)
        if self.rng.random() >= crossover_rate:
            return self.individual(ga), self.individual(gb)
        ca, cb = {}, {}
        for k in self.knobs:
            ca[k], cb[k] = [], []
            for x, y in zip(ga[k], gb[k]):
                if self.rng.random() < 0.5:
                    ca[k].append(x), cb[k].append(y)
                else:
                    ca[k].append(y), cb[k].append(x)
        return self.individual(self._legal(ca)), self.individual(self._legal(cb))

    def _mutate_value(self, grid: List[int], v: int) -> int:
        i = grid.index(v)
        if self.rng.random() < self.mutation_step_prob:
            step = self.rng.choice((-1, 1))
            j = i + step if 0 <= i + step < len(grid) else i - step
            return grid[j]
        return self.rng.choice([x for x in grid if x != v])

    def mutate(self, ind: Dict[str, Any], mutation_rate: Optional[float] = None) -> Individual:
        """Each gene mutates with probability `mutation_rate`: a one-step grid move with
        probability mutation_step_prob, otherwise a uniform redraw. At least one gene changes."""
        rate = self.default_mutation_rate if mutation_rate is None else float(mutation_rate)
        original = self.genome_of(ind)
        g = self.genome_of(ind)
        changed = False
        for k in self.knobs:
            for j, v in enumerate(g[k]):
                if len(self.grids[k]) > 1 and self.rng.random() < rate:
                    g[k][j] = self._mutate_value(self.grids[k], v)
                    changed = True
        if not changed:
            k = self.rng.choice([k for k in self.knobs if len(self.grids[k]) > 1])
            j = self.rng.randrange(len(g[k]))
            g[k][j] = self._mutate_value(self.grids[k], g[k][j])
        g = self._legal(g)
        if g == original:
            # Restoring n_kv | n_h can undo the only change, so a width gene moves instead.
            k = self.rng.choice([k for k in ("n_qk_head_dim", "n_v_head_dim", "mlp_size") if len(self.grids[k]) > 1])
            j = self.rng.randrange(len(g[k]))
            g[k][j] = self._mutate_value(self.grids[k], g[k][j])
        return self.individual(g)


def write_configs(out_dir: str) -> List[Path]:
    from ..supernet.config import SPECS

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for key in SPECS:
        for with_nkv, kv_suffix in ((False, ""), (True, "_nkv")):
            for partition, suffix in (("blocks", ""), ("uniform", "_uniform")):
                # The grouped-query variant exists only for the block partition, which is what the search
                # ablation compares against. A uniform GQA space would leave two knobs and is not used.
                gqa_axis = ((False, ""), (True, "_gqa")) if partition == "blocks" else ((False, ""),)
                for gqa, gqa_suffix in gqa_axis:
                    space = ElasticSearchSpace(key, partition=partition, with_nkv=with_nkv, gqa_dims=gqa)
                    note = ", KV groups elastic" if with_nkv else ""
                    note += ", query/key and value dimensions fixed at the base head dimension" if gqa else ""
                    header = (f"# Elastic search space over the {key} supernet, partition '{partition}'{note}.\n"
                              f"# Generated by `python -m llmforge.search.elastic_space --write-configs`.\n"
                              f"# Loading validates grids and blocks against the supernet ModelSpec.\n"
                              f"# Genes: {space.n_genes}. Size: 10^{space.log10_size():.2f} architectures.\n")
                    path = out / f"{key}{kv_suffix}{suffix}{gqa_suffix}.yaml"
                    path.write_text(header + yaml.safe_dump(space.to_yaml_dict(), sort_keys=False,
                                                            default_flow_style=None))
                    written.append(path)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write-configs", metavar="DIR")
    ap.add_argument("--describe", metavar="YAML")
    a = ap.parse_args()
    if a.write_configs:
        for p in write_configs(a.write_configs):
            print(p)
    if a.describe:
        print(yaml.safe_dump(ElasticSearchSpace.from_yaml(a.describe).describe(), sort_keys=False))


if __name__ == "__main__":
    main()
