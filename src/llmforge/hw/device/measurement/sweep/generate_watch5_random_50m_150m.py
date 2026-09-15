#!/usr/bin/env python3
"""Generate random Pixel Watch 5 sweep configurations in a parameter-size band.

Each architecture is sampled uniformly from the same discrete design choices used
by ``generate_watch5_1000_search_space.py``. Architectures outside the requested
parameter range are rejected, as are duplicates.
"""

import argparse
import csv
import math
import os
import random


VOCAB_SIZE = 50_257
DEFAULT_NUM_SAMPLES = 1_000
DEFAULT_MIN_PARAMS_M = 50.0
DEFAULT_MAX_PARAMS_M = 150.0
DEFAULT_SEED = 42

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = os.path.join(
    SCRIPT_DIR,
    "configs",
    "watch5_random_1000_50M_150M_sweep.csv",
)

FIELDNAMES = [
    "config_id",
    "suite",
    "category",
    "n_layer",
    "d_model",
    "n_h",
    "n_kv",
    "d_qk",
    "d_v",
    "d_mlp",
    "total_params_M",
    "q8_group_size",
    "layer_size_kb",
    "mlp_ratio",
    "kv_ratio",
    "sample_idx",
]


def calc_model_specs(
    n_layer,
    d_model,
    n_h,
    n_kv,
    d_qk,
    d_v,
    d_mlp,
    vocab_size=VOCAB_SIZE,
):
    """Return exact custom-model parameters and the per-layer Q8 file size."""
    embed_params = vocab_size * d_model
    attn_params = (
        (d_model * n_h * d_qk)
        + (d_model * n_kv * d_qk)
        + (d_model * n_kv * d_v)
        + (n_h * d_v * d_model)
    )
    mlp_params = 3 * d_model * d_mlp
    matmul_params = attn_params + mlp_params
    # Each infinite-attention block has four RMSNorm gain vectors, and the
    # model has one final RMSNorm gain vector. Matmul weights use Q8_0
    # (1 int8 value plus one fp32 scale per 16/32/64-value group); norm gains
    # remain fp32 in the .rlm format.
    layer_norm_params = 4 * d_model
    layer_params = matmul_params + layer_norm_params
    total_params = embed_params + n_layer * layer_params + d_model
    q8_group_size = 64
    q8_input_dims = (d_model, n_h * d_v, d_mlp)
    while q8_group_size > 1 and any(
        dim % q8_group_size != 0 for dim in q8_input_dims
    ):
        q8_group_size //= 2
    q8_bytes_per_param = 1.0 + (4.0 / q8_group_size)
    layer_size_kb = (
        (matmul_params * q8_bytes_per_param) + (layer_norm_params * 4.0)
    ) / 1024.0
    return total_params, layer_params, q8_group_size, layer_size_kb


def architecture_key(config):
    return (
        config["n_layer"],
        config["d_model"],
        config["n_h"],
        config["n_kv"],
        config["d_qk"],
        config["d_v"],
        config["d_mlp"],
    )


def sample_architecture(rng):
    """Draw one architecture uniformly from the discrete search dimensions."""
    n_layer = rng.randint(2, 28)
    d_model = rng.choice(range(64, 641, 32))

    max_heads = max(1, d_model // 16)
    n_h = rng.choice([h for h in (1, 2, 4, 6, 8, 12, 16) if h <= max_heads])
    n_kv = rng.choice([k for k in range(1, n_h + 1) if n_h % k == 0])
    d_qk = rng.choice((16, 32, 48, 64))
    d_v = rng.choice((16, 32, 48, 64))

    min_d_mlp = math.ceil((1.2 * d_model) / 32) * 32
    max_d_mlp = math.floor((5.5 * d_model) / 32) * 32
    d_mlp = rng.choice(range(min_d_mlp, max_d_mlp + 1, 32))

    return {
        "n_layer": n_layer,
        "d_model": d_model,
        "n_h": n_h,
        "n_kv": n_kv,
        "d_qk": d_qk,
        "d_v": d_v,
        "d_mlp": d_mlp,
    }


def generate_random_configs(
    num_samples=DEFAULT_NUM_SAMPLES,
    min_params_m=DEFAULT_MIN_PARAMS_M,
    max_params_m=DEFAULT_MAX_PARAMS_M,
    seed=DEFAULT_SEED,
    max_attempts=None,
    excluded_keys=None,
    start_index=1,
):
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if min_params_m <= 0 or max_params_m <= min_params_m:
        raise ValueError("require 0 < min_params_m < max_params_m")
    if start_index < 1:
        raise ValueError("start_index must be positive")

    rng = random.Random(seed)
    max_attempts = max_attempts or max(100_000, num_samples * 1_000)
    suite_name = f"Random_{min_params_m:g}M_{max_params_m:g}M"
    configs = []
    seen = set(excluded_keys or ())
    attempts = 0

    while len(configs) < num_samples and attempts < max_attempts:
        attempts += 1
        config = sample_architecture(rng)
        key = architecture_key(config)
        if key in seen:
            continue

        total_params, _, q8_group_size, layer_size_kb = calc_model_specs(**config)
        total_params_m = total_params / 1_000_000.0
        if not min_params_m <= total_params_m <= max_params_m:
            continue

        seen.add(key)
        sample_idx = len(configs) + start_index
        config.update(
            {
                "config_id": f"RND_{sample_idx:04d}",
                "suite": suite_name,
                "category": suite_name,
                "total_params_M": round(total_params_m, 3),
                "q8_group_size": q8_group_size,
                "layer_size_kb": round(layer_size_kb, 1),
                "mlp_ratio": round(config["d_mlp"] / config["d_model"], 3),
                "kv_ratio": round(config["n_kv"] / config["n_h"], 3),
                "sample_idx": sample_idx,
            }
        )
        configs.append(config)

    if len(configs) != num_samples:
        raise RuntimeError(
            f"generated only {len(configs)} of {num_samples} requested configurations "
            f"after {attempts} attempts; widen the parameter range or search space"
        )

    return configs, attempts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate unique random LLM architectures within a parameter-size range."
    )
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--min-params-m", type=float, default=DEFAULT_MIN_PARAMS_M)
    parser.add_argument("--max-params-m", type=float, default=DEFAULT_MAX_PARAMS_M)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--exclude-csv", action="append", default=[],
                        help="Exclude architectures and IDs in an existing CSV; repeat for multiple batches")
    parser.add_argument("--start-index", type=int, default=1,
                        help="First RND ID and sample_idx (default: 1)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    excluded_keys = set()
    excluded_ids = set()
    for path in args.exclude_csv:
        if os.path.realpath(path) == os.path.realpath(args.output):
            raise ValueError("output must differ from excluded CSVs")
        with open(path, newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                excluded_keys.add(architecture_key({
                    key: int(row[key]) for key in
                    ("n_layer", "d_model", "n_h", "n_kv", "d_qk", "d_v", "d_mlp")
                }))
                excluded_ids.add(row["config_id"])
    configs, attempts = generate_random_configs(
        num_samples=args.num_samples,
        min_params_m=args.min_params_m,
        max_params_m=args.max_params_m,
        seed=args.seed,
        max_attempts=args.max_attempts,
        excluded_keys=excluded_keys,
        start_index=args.start_index,
    )
    if any(config["config_id"] in excluded_ids for config in configs):
        raise ValueError("config IDs overlap excluded CSVs; increase --start-index")

    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(configs)

    sizes = [config["total_params_M"] for config in configs]
    print(f"Generated {len(configs)} unique random configurations in {attempts} draws")
    print(f"Parameter range: {min(sizes):.3f}M to {max(sizes):.3f}M")
    print(f"Seed: {args.seed}")
    print(f"Excluded existing architectures: {len(excluded_keys)}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
