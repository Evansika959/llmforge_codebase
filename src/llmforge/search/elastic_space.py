"""Search space over the slices of one trained elastic supernet.

A genome assigns one grid value per knob to each layer block. Grids come from the supernet's
ModelSpec, so every point the search visits is a slice the supernet was trained to serve.

    knob   Individual layer key   grid
    n_h    n_head                 head_grid(spec), the multiples of n_kv inside spec.nh_grid
    d_qk   n_qk_head_dim          spec.qk_grid, quarter steps of the base head dimension
    d_v    n_v_head_dim           spec.qk_grid
    d_mlp  mlp_size               spec.mlp_grid, quarter steps of the base MLP width

The number of KV groups stays at the base value and every layer stays active. The attention knobs
share the attention partition and d_mlp uses the MLP partition.

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
PARTITIONS = ("uniform", "blocks", "per_layer")
PAPER_NAMES = {"n_head": "n_h", "n_qk_head_dim": "d_qk", "n_v_head_dim": "d_v", "mlp_size": "d_mlp"}


class ElasticSearchSpace:
    def __init__(self, base_model: str, partition: str = "blocks", n_blocks: int = 5,
                 block_size: int = 1024, mutation_step_prob: float = 0.7,
                 seed: Optional[int] = None):
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
        self.grids = {
            "n_head": [int(h) for h in head_grid(spec)],
            "n_qk_head_dim": [int(q) for q in spec.qk_grid],
            "n_v_head_dim": [int(q) for q in spec.qk_grid],
            "mlp_size": [int(m) for m in spec.mlp_grid],
        }
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
                    mutation_step_prob=cfg.get("mutation_step_prob", 0.7), seed=seed)
        for knob, grid in (cfg.get("knobs") or {}).items():
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
        return {
            "base_model": self.base_model,
            "hf_repo": s.repo,
            "partition": self.partition,
            "n_blocks": self.n_blocks,
            "block_size": self.block_size,
            "mutation_step_prob": self.mutation_step_prob,
            "knobs": {k: list(v) for k, v in self.grids.items()},
            "fixed": {"n_layer": s.n_layers, "n_embd": s.hidden, "n_kv_group": s.n_kv,
                      "vocab_size": s.vocab, "tie_embeddings": s.tied, "family": s.family,
                      "attention_variant": "infinite", "mlp_variant": "swiglu"},
            "blocks": {k: [list(b) for b in v] for k, v in self.blocks.items()},
        }

    def describe(self) -> Dict[str, Any]:
        d = self.to_yaml_dict()
        d["n_genes"] = self.n_genes
        d["log10_size"] = round(self.log10_size(), 3)
        d["default_mutation_rate"] = self.default_mutation_rate
        return d

    # ---- genome ------------------------------------------------------------------------------

    def knob_blocks(self, knob: str) -> List[Tuple[int, int]]:
        return self.blocks["attn" if knob in ATTN_KNOBS else "mlp"]

    @property
    def n_genes(self) -> int:
        return sum(len(self.knob_blocks(k)) for k in KNOBS)

    @property
    def default_mutation_rate(self) -> float:
        return round(min(0.5, max(0.1, 1.5 / self.n_genes)), 4)

    def log10_size(self) -> float:
        return sum(len(self.knob_blocks(k)) * math.log10(len(self.grids[k])) for k in KNOBS)

    def genome_of(self, ind: Dict[str, Any]) -> Dict[str, List[int]]:
        layers = ind["layers"]
        if len(layers) != self.spec.n_layers:
            raise ValueError(f"expected {self.spec.n_layers} layers, got {len(layers)}")
        out = {}
        for k in KNOBS:
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
        for k in KNOBS:
            blocks = self.knob_blocks(k)
            if len(genome[k]) != len(blocks):
                raise ValueError(f"{k}: {len(genome[k])} values for {len(blocks)} blocks")
            for (lo, hi), v in zip(blocks, genome[k]):
                if int(v) not in self.grids[k]:
                    raise ValueError(f"{k}={v} is off the grid {self.grids[k]}")
                for i in range(lo, hi):
                    layers[i][k] = int(v)
        g = {"base_model": self.base_model, "n_embd": s.hidden, "block_size": self.block_size,
             "vocab_size": s.vocab, "tie_embeddings": s.tied, "mlp_variant": "swiglu",
             "use_concat_heads": True, "attention_variant": "infinite",
             "layer_mask": [True] * s.n_layers}
        return Individual(g, layers)

    def key(self, ind: Dict[str, Any]) -> str:
        """Canonical, partition-independent key: run-length encoding of each knob over layers."""
        parts = []
        for k in KNOBS:
            vals = [int(li[k]) for li in ind["layers"]]
            runs = [f"{v}x{len(list(g))}" for v, g in itertools.groupby(vals)]
            parts.append(f"{PAPER_NAMES[k]}={','.join(runs)}")
        return f"{self.base_model}|" + ";".join(parts)

    def to_elastic_config(self, ind: Dict[str, Any]):
        from ..supernet.space import ElasticConfig

        s = self.spec
        L = ind["layers"]
        return ElasticConfig(s, d_qk=[int(li["n_qk_head_dim"]) for li in L],
                             d_v=[int(li["n_v_head_dim"]) for li in L], n_kv=[s.n_kv] * s.n_layers,
                             n_h=[int(li["n_head"]) for li in L],
                             d_mlp=[int(li["mlp_size"]) for li in L]).validate()

    # ---- named points --------------------------------------------------------------------

    def uniform(self, n_head: int, n_qk_head_dim: int, n_v_head_dim: int, mlp_size: int) -> Individual:
        vals = {"n_head": n_head, "n_qk_head_dim": n_qk_head_dim, "n_v_head_dim": n_v_head_dim,
                "mlp_size": mlp_size}
        return self.individual({k: [int(vals[k])] * len(self.knob_blocks(k)) for k in KNOBS})

    def full(self) -> Individual:
        return self.uniform(**{k: max(self.grids[k]) for k in KNOBS})

    def smallest(self) -> Individual:
        return self.uniform(**{k: min(self.grids[k]) for k in KNOBS})

    def enumerate_uniform(self) -> List[Individual]:
        return [self.uniform(*vals) for vals in itertools.product(*(self.grids[k] for k in KNOBS))]

    # ---- operators used by llmforge.search.nsga2.Population ------------------------------

    def sample(self) -> Individual:
        return self.individual({k: [self.rng.choice(self.grids[k]) for _ in self.knob_blocks(k)]
                                for k in KNOBS})

    def repair(self, ind: Dict[str, Any]) -> Individual:
        """Snap every value to its grid, then make each block constant using its first layer."""
        genome = {}
        for k in KNOBS:
            grid = self.grids[k]
            genome[k] = [min(grid, key=lambda g: abs(g - int(ind["layers"][lo][k])))
                         for lo, _ in self.knob_blocks(k)]
        return self.individual(genome)

    def crossover(self, a: Dict[str, Any], b: Dict[str, Any],
                  crossover_rate: float = 0.9) -> Tuple[Individual, Individual]:
        """Uniform crossover over block genes, applied with probability `crossover_rate`."""
        ga, gb = self.genome_of(a), self.genome_of(b)
        if self.rng.random() >= crossover_rate:
            return self.individual(ga), self.individual(gb)
        ca, cb = {}, {}
        for k in KNOBS:
            ca[k], cb[k] = [], []
            for x, y in zip(ga[k], gb[k]):
                if self.rng.random() < 0.5:
                    ca[k].append(x), cb[k].append(y)
                else:
                    ca[k].append(y), cb[k].append(x)
        return self.individual(ca), self.individual(cb)

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
        g = self.genome_of(ind)
        changed = False
        for k in KNOBS:
            for j, v in enumerate(g[k]):
                if len(self.grids[k]) > 1 and self.rng.random() < rate:
                    g[k][j] = self._mutate_value(self.grids[k], v)
                    changed = True
        if not changed:
            k = self.rng.choice([k for k in KNOBS if len(self.grids[k]) > 1])
            j = self.rng.randrange(len(g[k]))
            g[k][j] = self._mutate_value(self.grids[k], g[k][j])
        return self.individual(g)


def write_configs(out_dir: str) -> List[Path]:
    from ..supernet.config import SPECS

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for key in SPECS:
        for partition, suffix in (("blocks", ""), ("uniform", "_uniform")):
            space = ElasticSearchSpace(key, partition=partition)
            header = (f"# Elastic search space over the {key} supernet, partition '{partition}'.\n"
                      f"# Generated by `python -m llmforge.search.elastic_space --write-configs`.\n"
                      f"# Loading validates grids and blocks against the supernet ModelSpec.\n"
                      f"# Genes: {space.n_genes}. Size: 10^{space.log10_size():.2f} architectures.\n")
            path = out / f"{key}{suffix}.yaml"
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
