"""Hardware-aware architecture search: a supernet software evaluator, one hardware target, NSGA-II.

Example
    python -m llmforge.search.cosearch \\
        --space configs/search_spaces/smollm2-135m.yaml \\
        --supernet runs/supernet/smollm2-135m/step2500 \\
        --hw zeus --objectives val_loss energy_per_token_uJ \\
        --pop 32 --generations 30 --seed 0 --out runs/search/smollm2-135m__zeus__nsga2__s0

Algorithms
    nsga2   NSGA-II (llmforge.search.nsga2)
    random  random sampling with the evaluation count of NSGA-II at the same pop, offspring and
            generations
    grid    every uniform architecture of the space

Hardware targets (--hw)
    analytic  parameter, FLOP, KV-cache and MAC estimates only
    fitted    a cost model fitted to another run's measurements, pass --hw-arg model=PATH
    zeus      measured on the local NVIDIA GPU (llmforge.evaluators.hw_zeus)
    device    Pixel Watch 5 predictor, uniform architectures only (llmforge.evaluators.hw_device)
    timeloop  Timeloop accelerator substrates, pass --hw-arg substrate=NAME
    rdxe      rDXE ring accelerator with the inner chip co-search
Analytic metrics are merged into every record whatever the target.

Every run first evaluates the full and the smallest architecture of the space. Their objective
values fix the box that normalizes the hypervolume trace, so traces of different algorithms on the
same space and target are comparable.

Evaluations are cached across runs (runs/cache), so rerunning an interrupted command replays the
cached evaluations exactly and continues where it stopped. A completed run writes DONE and is
skipped unless --force is given.

Outputs in --out
    run.json      resolved arguments, search space, evaluator settings, software versions
    evals.jsonl   every evaluated architecture with all metrics, in evaluation order
    trace.jsonl   per generation: evaluations so far, archive front size, normalized hypervolume
    checkpoints/  NSGA-II population after each generation
    front.json    non-dominated feasible architectures over all evaluations, with the loss
                  re-scored on disjoint validation documents
    DONE          written last
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .elastic_space import PARTITIONS, ElasticSearchSpace
from .nsga2 import EvaluationResult, Population, cons_value
from .pareto import non_dominated, normalized_hypervolume


def _num(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("inf")
    return v if math.isfinite(v) else float("inf")


def parse_constraint(text: str):
    m = re.fullmatch(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*(<=|>=)\s*([-+0-9.eE]+)\s*", text)
    if not m:
        raise argparse.ArgumentTypeError("a constraint looks like 'params_M<=100' or 'device_decode_tok_s>=10'")
    key, op, val = m.groups()
    return (key if op == "<=" else f"{key}_min", float(val))


def parse_kv(items: List[str]) -> Dict[str, Any]:
    out = {}
    for item in items:
        k, _, v = item.partition("=")
        val: Any = v
        for cast in (int, float):
            try:
                val = cast(v)
                break
            except ValueError:
                continue
        if isinstance(val, str):
            val = {"true": True, "false": False, "none": None}.get(v.lower(), v)
        out[k] = val
    return out


def load_archs(path: Optional[str]) -> List[Dict[str, Any]]:
    """Individuals from a front.json (front then anchors), an evals.jsonl, or a JSON list."""
    if not path:
        raise SystemExit("--algo list needs --archs")
    p = Path(path)
    if p.suffix == ".jsonl":
        recs = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    else:
        data = json.loads(p.read_text())
        recs = data["front"] + data.get("anchors", []) if isinstance(data, dict) else data
    out, seen = [], set()
    for r in recs:
        ind = r.get("individual", r)
        tag = json.dumps(ind["layers"], sort_keys=True)
        if tag not in seen:
            seen.add(tag)
            out.append(ind)
    return out


def build_hw(args):
    kw = parse_kv(args.hw_arg)
    lock = not args.no_gpu_lock
    if args.hw == "analytic":
        return None
    if args.hw == "zeus":
        from ..evaluators.hw_zeus import HwZeus

        return HwZeus(prefill_len=args.prefill_len, decode_len=args.decode_len,
                      n_repeats=args.zeus_repeats, warmup=args.zeus_warmup, dtype=args.zeus_dtype,
                      batch_size=args.zeus_batch, use_gpu_lock=lock,
                      cuda_graphs=args.zeus_cuda_graphs, min_window_s=args.zeus_min_window_s,
                      schedule=args.zeus_schedule, settle_s=args.zeus_settle_s,
                      decode_window_s=args.zeus_decode_window_s, **kw)
    if args.hw == "fitted":
        from ..evaluators.hw_fitted import HwFitted

        return HwFitted(**kw)
    if args.hw == "device":
        from ..evaluators.hw_device import HwDevice

        return HwDevice(bundle=args.device_bundle, **kw)
    if args.hw == "timeloop":
        from ..evaluators import hw_timeloop

        return hw_timeloop.HwTimeloop(prefill_len=args.prefill_len, decode_len=args.decode_len, **kw)
    if args.hw == "rdxe":
        from ..evaluators import hw_rdxe

        cls = getattr(hw_rdxe, "HwRdxe", None) or getattr(hw_rdxe, "HwRdxeInner")
        return cls(prefill_len=args.prefill_len, decode_len=args.decode_len, **kw)
    raise ValueError(f"unknown hardware target {args.hw}")


class RunEvaluator:
    """Joins software, hardware and analytic metrics and keeps the archive of every evaluation."""

    def __init__(self, space, sw, hw, analytic, objectives: List[str], constraints: Dict[str, float],
                 out: Path):
        self.space, self.sw, self.hw, self.analytic = space, sw, hw, analytic
        self.objectives, self.constraints = list(objectives), dict(constraints)
        # Every invocation rebuilds the archive from scratch. Evaluations come back from the shared
        # caches, so a rerun replays an interrupted run exactly instead of inheriting its archive.
        self.path = out / "evals.jsonl"
        self.path.write_text("")
        self.archive: Dict[str, Dict[str, Any]] = {}
        self.timing = {"sw_s": 0.0, "hw_s": 0.0}

    def __call__(self, inds: List[Dict[str, Any]], gen: int, tag: str) -> List[EvaluationResult]:
        t0 = time.time()
        mu, sigma = self.sw.evaluate(inds)
        t1 = time.time()
        hw = self.hw.evaluate(inds) if self.hw is not None else [{} for _ in inds]
        t2 = time.time()
        an = self.analytic.evaluate(inds)
        self.timing["sw_s"] += t1 - t0
        self.timing["hw_s"] += t2 - t1
        results = []
        with open(self.path, "a") as log:
            for ind, m, s, h, a in zip(inds, mu, sigma, hw, an):
                aux = {**a, **(h or {}), "val_loss": float(m), "val_loss_sigma": float(s)}
                aux["hw_feasible"] = bool(aux.get("hw_feasible", True))
                missing = [o for o in self.objectives if o not in aux]
                if missing and aux["hw_feasible"]:
                    raise KeyError(f"objective(s) {missing} not produced by this target; "
                                   f"available metrics: {sorted(aux)}")
                objs = [_num(aux.get(o)) for o in self.objectives]
                ok = aux["hw_feasible"] and all(math.isfinite(o) for o in objs)
                cons = [cons_value(c, thr, aux) for c, thr in self.constraints.items()]
                cons.append(0.0 if ok else 1.0)
                key = self.space.key(ind)
                if key not in self.archive:
                    rec = {"key": key, "gen": gen, "tag": tag, "objs": objs, "cons": cons,
                           "feasible": all(c <= 0 for c in cons), "metrics": aux,
                           "genome": self.space.genome_of(ind), "individual": ind}
                    self.archive[key] = rec
                    log.write(json.dumps(rec) + "\n")
                results.append(EvaluationResult(objs, cons, aux))
        return results

    def front(self) -> List[Dict[str, Any]]:
        feas = [r for r in self.archive.values() if r["feasible"]]
        return [feas[i] for i in non_dominated([r["objs"] for r in feas])]


def _versions() -> Dict[str, Any]:
    out = {"python": platform.python_version()}
    for mod in ("torch", "transformers", "zeus", "xgboost", "numpy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            pass
    try:
        import torch

        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
            out["cuda"] = torch.version.cuda
    except Exception:
        pass
    try:
        from ..paths import ROOT

        out["git_commit"] = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                           capture_output=True, text=True).stdout.strip() or None
    except Exception:
        pass
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m llmforge.search.cosearch", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("search space")
    g.add_argument("--space", required=True, help="search-space YAML (configs/search_spaces)")
    g.add_argument("--partition", choices=PARTITIONS, default=None,
                   help="override the partition declared in the YAML")
    g = ap.add_argument_group("software evaluator")
    g.add_argument("--supernet", required=True, help="trained supernet checkpoint directory")
    g.add_argument("--n-docs", type=int, default=32, help="held-out documents scored during search")
    g.add_argument("--val-docs", type=int, default=64,
                   help="disjoint documents that re-score the final front, 0 to skip")
    g.add_argument("--max-len", type=int, default=1024, help="tokens per scored document")
    g = ap.add_argument_group("hardware evaluator")
    g.add_argument("--hw", choices=["analytic", "fitted", "zeus", "device", "timeloop", "rdxe"], default="analytic")
    g.add_argument("--prefill-len", type=int, default=128)
    g.add_argument("--decode-len", type=int, default=32)
    g.add_argument("--zeus-repeats", type=int, default=3)
    g.add_argument("--zeus-warmup", type=int, default=1)
    g.add_argument("--zeus-batch", type=int, default=1)
    g.add_argument("--zeus-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    g.add_argument("--zeus-cuda-graphs", action="store_true",
                   help="replay decode steps from CUDA graphs, removing per-kernel launch overhead")
    g.add_argument("--zeus-min-window-s", type=float, default=2.0,
                   help="repeat each measured pass until its energy window lasts this long")
    g.add_argument("--zeus-schedule", choices=["interleaved", "grouped"], default="interleaved",
                   help="grouped measures all prefill windows before all decode windows, so no decode "
                        "window starts right after a prefill window")
    g.add_argument("--zeus-settle-s", type=float, default=0.0,
                   help="untimed decode before the decode windows, so they start at decode power")
    g.add_argument("--zeus-decode-window-s", type=float, default=None,
                   help="minimum decode window length, default --zeus-min-window-s")
    from ..evaluators.hw_device import DEFAULT_BUNDLE

    g.add_argument("--device-bundle", default=DEFAULT_BUNDLE,
                   help="a bundle named under assets/device, or best or latest for the earlier uniform "
                        "predictor. The default is the per-layer predictor of Backend D")
    g.add_argument("--hw-arg", action="append", default=[], metavar="KEY=VALUE",
                   help="extra keyword argument for the hardware evaluator, repeatable")
    g.add_argument("--analytic-seq-len", type=int, default=1024)
    g = ap.add_argument_group("objectives and constraints")
    g.add_argument("--objectives", nargs="+", default=["val_loss", "energy_per_token_uJ"])
    g.add_argument("--constraint", action="append", type=parse_constraint, default=[],
                   help="for example 'params_M<=100', repeatable")
    g = ap.add_argument_group("algorithm")
    g.add_argument("--algo", choices=["nsga2", "random", "grid", "list"], default="nsga2")
    g.add_argument("--archs", default=None,
                   help="--algo list: architectures to evaluate, from a front.json, an evals.jsonl, "
                        "or a JSON list of individuals")
    g.add_argument("--pop", type=int, default=32)
    g.add_argument("--offspring", type=int, default=None, help="children per generation, default pop")
    g.add_argument("--generations", type=int, default=30)
    g.add_argument("--crossover-rate", type=float, default=0.9)
    g.add_argument("--mutation-rate", type=float, default=None,
                   help="per-gene rate, default min(0.5, max(0.1, 1.5 / genes))")
    g.add_argument("--tournament-k", type=int, default=2)
    g.add_argument("--seed", type=int, default=0)
    g = ap.add_argument_group("run")
    g.add_argument("--out", required=True)
    g.add_argument("--no-gpu-lock", action="store_true")
    g.add_argument("--force", action="store_true", help="rerun even when DONE exists")
    return ap


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.offspring = args.offspring or args.pop
    out = Path(args.out)
    if (out / "DONE").exists() and not args.force:
        print(f"[skip] {out} is complete")
        return
    out.mkdir(parents=True, exist_ok=True)
    (out / "trace.jsonl").write_text("")
    t_start = time.time()

    from ..evaluators.hw_analytic import HwAnalytic
    from ..evaluators.sw_supernet import SwSupernet

    rng = random.Random(args.seed)
    space = ElasticSearchSpace.from_yaml(args.space, partition=args.partition)
    space.rng = rng
    sw = SwSupernet(space, args.supernet, n_docs=args.n_docs, skip_docs=0, max_len=args.max_len,
                    use_gpu_lock=not args.no_gpu_lock)
    hw = build_hw(args)
    analytic = HwAnalytic(seq_len=args.analytic_seq_len)
    constraints = dict(args.constraint)
    ev = RunEvaluator(space, sw, hw, analytic, args.objectives, constraints, out)
    meta = {"args": vars(args), "command": " ".join([Path(sys.executable).name, "-m",
                                                    "llmforge.search.cosearch"] + (argv or sys.argv[1:])),
            "space": space.describe(), "sw": sw.settings,
            "hw": getattr(hw, "settings", {"backend": args.hw}), "analytic": analytic.settings,
            "versions": _versions(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "run.json").write_text(json.dumps(meta, indent=1))

    anchors = [space.full(), space.smallest()]
    anchor_res = ev(anchors, gen=0, tag="anchor")
    lo = [min(r.objs[j] for r in anchor_res) for j in range(len(args.objectives))]
    hi = [max(r.objs[j] for r in anchor_res) for j in range(len(args.objectives))]
    if not all(math.isfinite(x) for x in lo + hi):
        raise RuntimeError(f"anchor architectures failed to evaluate: {[r.aux for r in anchor_res]}")
    meta["hv_box"] = {"lo": lo, "hi": hi, "margin": 0.1}
    (out / "run.json").write_text(json.dumps(meta, indent=1))

    def trace(gen: int) -> None:
        front = ev.front()
        hv = normalized_hypervolume([r["objs"] for r in front], lo, hi) if front else 0.0
        rec = {"gen": gen, "evals": len(ev.archive), "front": len(front), "hv": round(hv, 6),
               "best": [min(r["objs"][j] for r in front) for j in range(len(lo))] if front else None,
               "sw_scored": sw.n_scored, "minutes": round((time.time() - t_start) / 60, 2)}
        with open(out / "trace.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[gen {gen:3d}] evals {rec['evals']:5d}  front {rec['front']:3d}  HV {hv:.4f}  "
              f"best {rec['best']}  [{rec['minutes']:.1f} min]", flush=True)

    key = space.key
    if args.algo == "nsga2":
        init, keys = list(anchors), {key(a) for a in anchors}
        while len(init) < args.pop:
            ind = space.sample()
            if key(ind) not in keys:
                init.append(ind)
                keys.add(key(ind))
        pop = Population(init, anchor_res + ev(init[2:], gen=0, tag="init"), space, constraints,
                         args.objectives, rng, n_population=args.pop, n_offspring=args.offspring,
                         tournament_k=args.tournament_k, mutation_rate=args.mutation_rate,
                         crossover_rate=args.crossover_rate)
        pop.reorder_by_non_domination()
        trace(0)
        pop.save_checkpoint(str(out / "checkpoints" / "gen_000.json"))
        for g in range(1, args.generations + 1):
            kids = pop.generate_offspring(key_fn=key, seen=ev.archive.keys())
            pop.offspring_evaluations = ev(kids, gen=g, tag="offspring")
            pop.update_elimination(key_fn=key)
            trace(g)
            pop.save_checkpoint(str(out / "checkpoints" / f"gen_{g:03d}.json"))
        pop.write_to_csv(str(out / "population_final.csv"))
    elif args.algo == "random":
        seen = {key(a) for a in anchors}

        def draw(n: int) -> List[Dict[str, Any]]:
            batch = []
            for _ in range(n):
                for _ in range(200):
                    ind = space.sample()
                    if key(ind) not in seen:
                        break
                else:
                    break
                seen.add(key(ind))
                batch.append(ind)
            return batch

        ev(draw(args.pop - len(anchors)), gen=0, tag="random")
        trace(0)
        for g in range(1, args.generations + 1):
            batch = draw(args.offspring)
            if not batch:
                break
            ev(batch, gen=g, tag="random")
            trace(g)
    elif args.algo == "grid":
        grid = space.enumerate_uniform()
        for g, i in enumerate(range(0, len(grid), args.pop)):
            ev(grid[i:i + args.pop], gen=g, tag="grid")
            trace(g)
    else:
        archs = load_archs(args.archs)
        for g, i in enumerate(range(0, len(archs), args.pop)):
            ev(archs[i:i + args.pop], gen=g, tag="list")
            trace(g)

    front = ev.front()
    anchor_keys = {key(a) for a in anchors}
    if args.val_docs > 0:
        val = sw.with_docs(args.val_docs, args.n_docs)
        for r in front + [ev.archive[k] for k in anchor_keys]:
            r["metrics"]["val_loss_heldout"] = val.score(r["individual"])
    hv = normalized_hypervolume([r["objs"] for r in front], lo, hi) if front else 0.0
    summary = {"objectives": args.objectives, "constraints": constraints, "hv_box": meta["hv_box"],
               "hv": hv, "evals": len(ev.archive), "sw_scored": sw.n_scored,
               "hw_measured": getattr(hw, "n_measured", None), "timing": ev.timing,
               "minutes": round((time.time() - t_start) / 60, 2),
               "front": sorted(front, key=lambda r: r["objs"][0]),
               "anchors": [ev.archive[k] for k in anchor_keys]}
    (out / "front.json").write_text(json.dumps(summary, indent=1))
    (out / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    print(f"[done] {len(front)} front architectures, HV {hv:.4f}, {len(ev.archive)} evaluations, "
          f"{summary['minutes']:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
