# Search space and search method

This document specifies what the hardware-aware search explores and how, precisely enough to
reimplement it. The implementation lives in `src/llmforge/search/`.

## Architecture record

An architecture is an `Individual`, a dict with `globals` and one entry per layer in `layers`
(`src/llmforge/search/individual.py`). Architectures from a supernet carry `base_model`, the model
width `n_embd`, `vocab_size`, `tie_embeddings`, `mlp_variant: swiglu` and `use_concat_heads: true` in
`globals`. Each layer holds `n_head`, `n_kv_group`, `n_qk_head_dim`, `n_v_head_dim`, `mlp_size`,
`n_cproj` and `attention_variant: infinite`. Infinite Head Attention decouples the query and key head
dimension from the value head dimension, so the four searched knobs of a layer are independent.

## Elastic search space

A search runs over the slices of one trained supernet (`src/llmforge/search/elastic_space.py`). A
slice keeps a prefix of every elastic dimension, so every point of the space is served by the
supernet without training.

| knob | layer key | grid |
|---|---|---|
| query heads n_h | `n_head` | multiples of the base KV group count inside the supernet head grid |
| query and key head dimension d_qk | `n_qk_head_dim` | quarter steps of the base head dimension |
| value head dimension d_v | `n_v_head_dim` | quarter steps of the base head dimension |
| MLP width d_mlp | `mlp_size` | quarter steps of the base MLP width |

The KV group count, the model width, the vocabulary and the depth stay at the base values.

| base model | layers | width | KV groups | n_h | d_qk and d_v | d_mlp | parameters, full to smallest |
|---|---|---|---|---|---|---|---|
| SmolLM2-135M | 30 | 576 | 3 | 3, 6, 9 | 16, 32, 48, 64 | 384, 768, 1152, 1536 | 134.5M to 51.5M |
| SmolLM2-360M | 32 | 960 | 5 | 5, 10, 15 | 16, 32, 48, 64 | 640, 1280, 1920, 2560 | 361.8M to 116.0M |
| Qwen3-0.6B | 28 | 1024 | 8 | 8, 16 | 32, 64, 96, 128 | 768, 1536, 2304, 3072 | 596.0M to 251.0M |
| Qwen3-1.7B | 28 | 2048 | 8 | 8, 16 | 32, 64, 96, 128 | 1536, 3072, 4608, 6144 | 1720.5M to 634.1M |
| Qwen3-4B | 36 | 2560 | 8 | 8, 16, 32 | 32, 64, 96, 128 | 2432, 4864, 7296, 9728 | 4022.3M to 1155.7M |

Parameter counts include the tied embedding once.

### Partitions and genome

A genome holds one grid value per knob per block. The three attention knobs share the attention
partition and d_mlp uses the MLP partition.

| partition | blocks | genes | size |
|---|---|---|---|
| `uniform` | one block over all layers | 4 | 128 or 192 architectures |
| `blocks` | five contiguous blocks of equal size, measured blocks for Qwen3-4B | 19 or 20 | 10^10.5 to 10^11.4 |
| `per_layer` | one block per layer | 4 x layers | used to evaluate arbitrary architectures |

Blocks for 30 layers are [0,6), [6,12), [12,18), [18,24), [24,30). For 32 layers they are [0,6),
[6,12), [12,19), [19,25), [25,32). For 28 layers they are [0,5), [5,11), [11,16), [16,22), [22,28).
Qwen3-4B uses its measured sensitivity blocks, [0,1), [1,12), [12,22), [22,25), [25,36) for attention
and [0,1), [1,4), [4,34), [34,36) for the MLP. The YAML files in `configs/search_spaces/` list every
grid and block and are validated against the supernet specification whenever they load.

An architecture's canonical key is the run-length encoding of each knob over layers, for example
`smollm2-135m|n_h=9x30;d_qk=64x30;d_v=64x30;d_mlp=1536x30`. It does not depend on the partition,
so caches are shared between uniform, block and per-layer runs.

## Objectives and constraints

All objectives are minimized. The software objective `val_loss` is the supernet slice loss in nats
(`src/llmforge/evaluators/sw_supernet.py`). Hardware objectives come from one target, for example
`energy_per_token_uJ`, `ttft_ms` or `tpot_ms`. Analytic metrics are merged into every record:
`params_M`, `nonembed_params_M`, `flops_per_token`, `kv_cache_MB` and `decode_macs_M`.

A constraint `key<=value` or `key>=value` adds a violation term. An architecture the target cannot
evaluate is infeasible through an extra violation term of 1.

## NSGA-II

`src/llmforge/search/nsga2.py` follows NSGA-II with constrained domination.

1. **Initial population.** The full and the smallest architecture, then distinct random genomes up to
   the population size.
2. **Selection.** Two binary tournaments on constrained domination. A feasible solution beats an
   infeasible one, two infeasible solutions compare by total violation, and two feasible solutions
   compare by Pareto dominance. Ties are broken by a fair coin.
3. **Crossover.** With probability 0.9, uniform crossover over block genes, each gene taken from
   either parent with probability 0.5.
4. **Mutation.** Each gene mutates with probability min(0.5, max(0.1, 1.5 / genes)), which is 0.1 for
   the block spaces. A mutation moves one grid step up or down with probability 0.7 and redraws a
   different grid value otherwise. At least one gene changes.
5. **Deduplication.** A child that repeats an architecture evaluated earlier is redrawn, up to 50
   times.
6. **Environmental selection.** Parents and children merge, duplicates are removed, fronts are sorted
   by constrained domination, each front is ordered by crowding distance, and the first
   `population` individuals survive.

The search space draws every random number from one `random.Random(seed)`.

## Baselines

- `--algo random` evaluates the same number of distinct random architectures as NSGA-II.
- `--algo grid` evaluates every uniform architecture.
- `--algo list` evaluates given architectures on a target, used to measure a proxy front on real
  hardware.

## Progress measure

The anchors, the full and the smallest architecture, fix a box per objective from their two values.
Hypervolume is computed after mapping the box to [0, 1] with the reference point at 1.1 on every
axis, and divided by the reference box volume. It is exact for two and three objectives
(`src/llmforge/search/pareto.py`). The trace reports it over the archive of all feasible
evaluations after every generation.

## Determinism and caching

Software scores are cached per supernet checkpoint, document slice and length. Hardware results are
cached per target setting, for example GPU model, batch, prompt and decode length. Every run
rebuilds its archive by replaying evaluations through these caches. A rerun with the same arguments
and seed therefore reproduces the original evaluation order, trace and front exactly, which
`tests/search/test_search_engine.py` checks.
