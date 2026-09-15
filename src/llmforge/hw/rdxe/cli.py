"""CLI entry points for DXE simulator.

Usage:
    # Single-chip MT5-small decode
    python -m llmforge.hw.rdxe.cli eval --decode-length 64 --prefill-length 960

    # Multi-chip ring
    python -m llmforge.hw.rdxe.cli ring --n-chips 8 --decode-length 64

    # Compare Timeloop-only vs DXE-sim corrected
    python -m llmforge.hw.rdxe.cli compare --decode-length 64
"""

import argparse
import os
import time


# MT5-small decoder reference model
MT5_SMALL = {
    'name': 'MT5-small',
    'n_embd': 512,
    'n_layers': 8,
    'layers': [
        {'n_head': 6, 'n_kv_group': 6, 'n_qk_head_dim': 64,
         'n_v_head_dim': 64, 'mlp_size': 1024, 'n_cproj': 1,
         'attention_variant': 'infinite'}
    ] * 8,
}


def get_model_spec(name: str) -> dict:
    """Get a named model specification."""
    models = {
        'mt5_small': MT5_SMALL,
    }
    if name in models:
        return models[name]
    raise ValueError(f"Unknown model '{name}'. Available: {list(models.keys())}")


def cmd_eval(args):
    """Run single-chip inference simulation."""
    from .core.config import DXEConfig
    from .core.op_model import OpModel
    from .simulator.chip_sim import ChipSimulator
    from .reporting.output import (
        save_inference_csv, save_summary_csv, plot_inference,
    )

    config = DXEConfig(
        prefill_length=args.prefill_length,
        decode_length=args.decode_length,
        enable_kv_correction=not args.no_kv_correction,
        enable_vlink_gqa=not args.no_vlink,
        enable_vrc=not args.no_vrc,
    )
    if args.output_dir:
        config.output_dir = args.output_dir

    model = get_model_spec(args.model)

    print("=" * 70)
    print(f"DXE-Sim Single-Chip — {model['name']}")
    print("=" * 70)
    print(f"Model:   {model['name']} ({model['n_layers']}L, d={model['n_embd']})")
    print(f"Prefill: {config.prefill_length} tokens")
    print(f"Decode:  {config.decode_length} tokens")
    print(f"KV correction: {config.enable_kv_correction}")
    print(f"VLINK GQA:     {config.enable_vlink_gqa}")
    print(f"VRC:           {config.enable_vrc}")
    print()

    t0 = time.time()
    op_model = OpModel(config)

    sim = ChipSimulator(op_model, config)
    result = sim.run_inference(
        model, model_name=model['name'],
        prefill_length=config.prefill_length,
        decode_length=config.decode_length)

    elapsed = time.time() - t0

    # Print summary
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Total energy:         {result.total_energy_uJ:>10.4f} uJ")
    print(f"  Energy per token:     {result.energy_per_token_uJ:>10.4f} uJ/tok")
    print(f"  Total latency:        {result.total_latency_ms:>10.4f} ms")
    print(f"  Latency per token:    {result.latency_per_token_us:>10.3f} us/tok")
    print(f"  Throughput:           {result.tokens_per_second:>10.0f} tok/s")
    print()
    print("  Energy breakdown:")
    print(f"    Compute+SRAM:       {result.compute_energy_uJ:>10.4f} uJ")
    print(f"    KV Cache:           {result.kv_cache_energy_uJ:>10.4f} uJ")
    print(f"    DRAM:               {result.dram_energy_uJ:>10.4f} uJ")
    print(f"    VRC:                {result.vrc_energy_uJ:>10.4f} uJ")
    print(f"    WMEM Reload:        {result.wmem_reload_energy_uJ:>10.4f} uJ")
    print()
    print("  Corrections applied:")
    print(f"    KV cache swap:      {result.total_kv_correction_uJ:>10.4f} uJ")
    print(f"    VLINK GQA:          {result.total_vlink_correction_uJ:>10.4f} uJ")
    print()
    print(f"  Simulation time:      {elapsed:>10.1f} s")

    # Save outputs
    os.makedirs(config.output_dir, exist_ok=True)
    trace_path = os.path.join(config.output_dir, f'{model["name"]}_traces.csv')
    summary_path = os.path.join(config.output_dir, f'{model["name"]}_summary.csv')
    save_inference_csv(result, trace_path)
    save_summary_csv(result, summary_path)
    print(f"\n  Traces: {trace_path}")
    print(f"  Summary: {summary_path}")

    if config.plot:
        os.makedirs(config.plot_dir, exist_ok=True)
        plot_inference(result, config.plot_dir)


def cmd_ring(args):
    """Run multi-chip ring simulation."""
    from .core.config import DXEConfig
    from .core.op_model import OpModel
    from .simulator.ring_sim import RingSimulator
    from .reporting.output import plot_ring

    config = DXEConfig(
        prefill_length=args.prefill_length,
        decode_length=args.decode_length,
        n_chips=args.n_chips,
        layer_assignment=args.assignment,
    )
    if args.output_dir:
        config.output_dir = args.output_dir

    model = get_model_spec(args.model)

    print("=" * 70)
    print(f"DXE-Sim Ring — {args.n_chips} Chips — {model['name']}")
    print("=" * 70)

    op_model = OpModel(config)
    ring = RingSimulator(args.n_chips, op_model, config)
    result = ring.run_ring_inference(
        model, model_name=model['name'],
        prefill_length=config.prefill_length,
        decode_length=config.decode_length)

    print(f"  Pipeline depth:       {result.pipeline_depth}")
    print(f"  Steady-state:         {result.steady_state_throughput_tps:.0f} tok/s")
    print(f"  Energy per token:     {result.energy_per_token_uJ:.4f} uJ")
    print(f"  Total latency:        {result.total_latency_ms:.4f} ms")
    print(f"  Inter-chip comm:      {result.inter_chip_comm_energy_uJ:.4f} uJ total")
    print()
    for ci in range(result.n_chips):
        layers = result.layers_per_chip[ci]
        print(f"  Chip {ci}: layers={layers}, "
              f"E={result.per_chip_energy_uJ[ci]:.4f} uJ/tok, "
              f"util={result.per_chip_utilization[ci]*100:.1f}%")

    os.makedirs(config.output_dir, exist_ok=True)
    if config.plot:
        os.makedirs(config.plot_dir, exist_ok=True)
        plot_ring(result, config.plot_dir)


def cmd_compare(args):
    """Compare Timeloop-only vs DXE-sim corrected."""
    from .core.config import DXEConfig
    from .core.op_model import OpModel
    from .simulator.chip_sim import ChipSimulator

    model = get_model_spec(args.model)

    # Run with corrections
    config_on = DXEConfig(
        prefill_length=args.prefill_length,
        decode_length=args.decode_length,
        enable_kv_correction=True,
        enable_vlink_gqa=True,
        enable_vrc=True,
    )
    # Run without corrections (Timeloop-only equivalent)
    config_off = DXEConfig(
        prefill_length=args.prefill_length,
        decode_length=args.decode_length,
        enable_kv_correction=False,
        enable_vlink_gqa=False,
        enable_vrc=False,
    )

    print("=" * 70)
    print(f"Comparison: Timeloop-only vs DXE-Sim — {model['name']}")
    print("=" * 70)

    results = {}
    for label, cfg in [("Timeloop-only", config_off),
                        ("DXE-Sim", config_on)]:
        op = OpModel(cfg)
        sim = ChipSimulator(op, cfg)
        r = sim.run_inference(model, model_name=model['name'])
        results[label] = r

    fmt = "{:25s} {:>15s} {:>15s} {:>10s}"
    print(fmt.format("Metric", "Timeloop-only", "DXE-Sim", "Delta"))
    print("-" * 70)

    r_tl = results["Timeloop-only"]
    r_dx = results["DXE-Sim"]

    for name, attr in [
        ("Energy/token (uJ)", "energy_per_token_uJ"),
        ("Total energy (uJ)", "total_energy_uJ"),
        ("Latency/token (us)", "latency_per_token_us"),
        ("Total latency (ms)", "total_latency_ms"),
        ("Compute energy (uJ)", "compute_energy_uJ"),
        ("KV cache energy (uJ)", "kv_cache_energy_uJ"),
        ("DRAM energy (uJ)", "dram_energy_uJ"),
        ("VRC energy (uJ)", "vrc_energy_uJ"),
    ]:
        v_tl = getattr(r_tl, attr)
        v_dx = getattr(r_dx, attr)
        delta = v_dx - v_tl
        print(fmt.format(name, f"{v_tl:.4f}", f"{v_dx:.4f}",
                          f"{delta:+.4f}"))



def main():
    parser = argparse.ArgumentParser(
        description="DXE Graph-Level Accelerator Simulator")
    sub = parser.add_subparsers(dest="command")

    # eval
    p = sub.add_parser("eval", help="Single-chip inference")
    p.add_argument("--model", default="mt5_small")
    p.add_argument("--prefill-length", type=int, default=0)
    p.add_argument("--decode-length", type=int, default=64)
    p.add_argument("--output-dir", type=str, default="")
    p.add_argument("--no-kv-correction", action="store_true")
    p.add_argument("--no-vlink", action="store_true")
    p.add_argument("--no-vrc", action="store_true")

    # ring
    p = sub.add_parser("ring", help="Multi-chip ring simulation")
    p.add_argument("--model", default="mt5_small")
    p.add_argument("--n-chips", type=int, required=True)
    p.add_argument("--prefill-length", type=int, default=0)
    p.add_argument("--decode-length", type=int, default=64)
    p.add_argument("--assignment", default="round_robin",
                   choices=["round_robin", "balanced", "single_layer"])
    p.add_argument("--output-dir", type=str, default="")

    # compare
    p = sub.add_parser("compare", help="Timeloop-only vs DXE-sim")
    p.add_argument("--model", default="mt5_small")
    p.add_argument("--prefill-length", type=int, default=0)
    p.add_argument("--decode-length", type=int, default=64)

    args = parser.parse_args()
    if args.command == "eval":
        cmd_eval(args)
    elif args.command == "ring":
        cmd_ring(args)
    elif args.command == "compare":
        cmd_compare(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
