"""GPU target: latency and energy of one architecture, measured with ZEUS.

The architecture is instantiated as a ReaLLM-Forge GPT with Infinite Head Attention and random
weights. Weights do not change which kernels a forward pass launches, so random weights measure the
same compute as trained ones. Prefill is one forward pass over `prefill_len` tokens per sequence.
Decode generates `decode_len` tokens against a preallocated KV cache (llmforge.hw.zeus.kv_cache),
optionally replayed from CUDA graphs. ZEUS reads device energy through NVML around each window.

NVML refreshes its energy counter slowly compared with a short burst of kernels, so a window of a
few hundred milliseconds reads with large relative error. Each measurement window therefore repeats
its pass until it lasts at least `min_window_s`. The prefill forward pass runs K_p times and, with CUDA
graphs, the decode pass is replayed K_d times. A timed pass after warmup sets the counts, and a window
that still ends short keeps adding passes until it reaches `min_window_s`. A replayed decode pass
rewrites the same cache positions with the same tokens, so every pass does identical work. Per-token
metrics divide by the tokens actually processed in each window.

By default every repeat measures a prefill window and then a decode window. With schedule="grouped"
all prefill windows run first, then settle_s seconds of untimed decode passes, then all decode windows,
so no decode window starts right after the higher prefill power. With the default schedule, settle_s
adds the untimed passes before every decode window. decode_window_s sets a separate minimum length
for decode windows.

The measured graph matches the supernet architecture in every matrix multiplication: per-layer query
heads, KV groups, query/key and value head dimensions, heads concatenated before the output
projection, a SwiGLU MLP, RMSNorm, no biases, and an LM head over the base vocabulary tied to the
embedding. It omits rotary embeddings and QK-Norm, which the cached decode path does not implement,
and adds no positional embedding in their place. Both are elementwise operations without weights.

Reported per architecture, median over n_repeats. With decode_len 0 only the prefill keys appear.
    ttft_ms                        prefill window time divided by K_p
    prefill_energy_per_token_uJ    prefill window energy divided by K_p x batch x prefill_len
    prefill_power_W                mean device power during the prefill window
    tpot_ms                        decode window time divided by K_d x decode_len
    energy_per_token_uJ            decode window energy divided by K_d x batch x decode_len
    session_energy_per_token_uJ    energy of one prefill and one decode pass per processed token
    dynamic_energy_per_token_uJ    decode energy above the idle power, recorded but not used
    power_W                        mean device power during the decode window, or during the
                                   prefill window when there is no decode
    hw_feasible                    False when the build or the measurement failed, for example OOM

ReaLLM-Forge is located through llmforge.paths.REALLM_FORGE (env LLMFORGE_REALLM_FORGE).
"""
from __future__ import annotations

import contextlib
import io
import logging
import math
import sys
import time
from typing import Any, Dict, List, Optional

import torch

from .kv_cache import (UnsupportedKVCache, attach_iha_kv_cache, capture_decode_graphs,
                       detach_iha_kv_cache, run_cached_decode, set_iha_kv_len, set_iha_mode,
                       set_kv_capacity)

log = logging.getLogger(__name__)
_REALLM = None
MAX_PASSES = 400


def reallm_forge():
    """Import GPTConfig and GPT from the ReaLLM-Forge checkout."""
    global _REALLM
    if _REALLM is None:
        from ...paths import REALLM_FORGE

        if not (REALLM_FORGE / "model.py").exists():
            raise FileNotFoundError(
                f"ReaLLM-Forge not found at {REALLM_FORGE}. Run scripts/setup/fetch_third_party.sh "
                f"or point LLMFORGE_REALLM_FORGE at a checkout.")
        root = str(REALLM_FORGE)
        if root not in sys.path:
            sys.path.insert(0, root)
        from gpt_conf import GPTConfig
        from model import GPT

        _REALLM = (GPTConfig, GPT)
    return _REALLM


def _active_layers(ind: Dict[str, Any]) -> List[Dict[str, Any]]:
    g = ind.get("globals", ind)
    layers = ind.get("layers", [])
    mask = g.get("layer_mask")
    return list(layers) if not mask else [L for L, m in zip(layers, mask) if m]


def build_config(ind: Dict[str, Any], block_size: int):
    GPTConfig, _ = reallm_forge()
    g = ind.get("globals", ind)
    active = _active_layers(ind)
    if not active:
        raise ValueError("Individual has no active layers")

    def col(key, default):
        return [L.get(key, default) for L in active]

    d = int(g.get("n_embd", 768))
    kw = dict(
        n_layer=len(active),
        n_embd=d,
        block_size=int(block_size),
        vocab_size=int(g.get("vocab_size", 50304)),
        n_head=int(col("n_head", 8)[0]),
        n_kv_group=int(col("n_kv_group", 8)[0]),
        n_head_layerlist=[int(x) for x in col("n_head", 8)],
        n_kv_group_layerlist=[int(x) for x in col("n_kv_group", 8)],
        mlp_size_layerlist=[int(x) for x in col("mlp_size", 4 * d)],
        n_cproj_layerlist=[int(x) for x in col("n_cproj", 1)],
        attention_variant_layerlist=col("attention_variant", "infinite"),
        mlp_variant=g.get("mlp_variant", "swiglu"),
        use_concat_heads=bool(g.get("use_concat_heads", False)),
        wte_weight_tying=bool(g.get("tie_embeddings", True)),
        use_abs_pos_embeddings=bool(g.get("use_abs_pos_embeddings", False)),
        use_rotary_embeddings=False,
        bias=False,
    )
    qk, v = col("n_qk_head_dim", None), col("n_v_head_dim", None)
    if all(x is not None for x in qk):
        kw["n_qk_head_dim_layerlist"] = [int(x) for x in qk]
    if all(x is not None for x in v):
        kw["n_v_head_dim_layerlist"] = [int(x) for x in v]
    return GPTConfig(**kw)


def build_model_from_individual(ind: Dict[str, Any], block_size: int, device: torch.device,
                                dtype: torch.dtype):
    """Instantiate the architecture with random weights directly on `device`."""
    _, GPT = reallm_forge()
    cfg = build_config(ind, block_size)
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            with torch.device(device):
                model = GPT(cfg)
        except Exception:
            model = GPT(cfg)
    return model.to(device=device, dtype=dtype).eval()


def measure_idle_power(monitor, seconds: float = 5.0) -> float:
    """Mean device power with no GPU work, in watts."""
    torch.cuda.synchronize()
    time.sleep(1.0)
    monitor.begin_window("idle")
    time.sleep(seconds)
    m = monitor.end_window("idle")
    return float(m.total_energy) / max(1e-9, float(m.time))


def _median(xs: List[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return float("nan")
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def _spread(xs: List[float]) -> float:
    return (max(xs) - min(xs)) / max(1e-12, _median(xs)) if len(xs) > 1 else 0.0


def _failed(reason: str) -> Dict[str, Any]:
    inf = float("inf")
    return {"ttft_ms": inf, "tpot_ms": inf, "energy_per_token_uJ": inf,
            "prefill_energy_per_token_uJ": inf, "dynamic_energy_per_token_uJ": inf,
            "session_energy_per_token_uJ": inf, "power_W": inf, "hw_feasible": False,
            "zeus_error": reason}


def _is_cuda_fault(e: BaseException) -> bool:
    """A CUDA fault other than running out of memory leaves the process unable to run GPU work, so it is
    raised instead of being recorded as an infeasible architecture. The queue reruns the job in a new
    process, which replays every cached measurement."""
    accel = getattr(torch, "AcceleratorError", None)
    text = str(e)
    return (accel is not None and isinstance(e, accel)) or "CUDA error" in text or "CUBLAS_STATUS" in text


def _timed(fn) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


@torch.no_grad()
def measure_one(ind: Dict[str, Any], prefill_len: int = 128, decode_len: int = 32,
                n_repeats: int = 3, warmup: int = 1, dtype: str = "bf16", monitor=None,
                use_kv_cache: bool = True, batch_size: int = 1,
                idle_power_W: Optional[float] = None, cuda_graphs: bool = False,
                min_window_s: float = 0.0, schedule: str = "interleaved", settle_s: float = 0.0,
                decode_window_s: Optional[float] = None, return_windows: bool = False) -> Dict[str, Any]:
    if schedule not in ("interleaved", "grouped"):
        raise ValueError(f"unknown schedule {schedule!r}")
    device = torch.device("cuda:0")
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    if monitor is None:
        from zeus.monitor import ZeusMonitor

        monitor = ZeusMonitor(gpu_indices=[0], cpu_indices=[], sync_execution_with="torch",
                              approx_instant_energy=True)
    try:
        model = build_model_from_individual(ind, prefill_len + decode_len + 8, device, torch_dtype)
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return _failed(f"OOM at build: {e}")
    except Exception as e:
        if _is_cuda_fault(e):
            raise
        log.warning(f"model build failed: {e}")
        return _failed(f"build: {e}")

    kv_active, kv_skip = False, None
    if use_kv_cache:
        try:
            attach_iha_kv_cache(model)
            set_kv_capacity(model, prefill_len + decode_len)
            kv_active = True
        except UnsupportedKVCache as e:
            kv_skip = str(e)

    bs = int(batch_size)
    vocab = int(model.config.vocab_size)
    graphs = []
    try:
        tokens = torch.randint(0, vocab, (bs, prefill_len), device=device, dtype=torch.long)

        def prefill():
            if kv_active:
                set_iha_kv_len(model, 0)
                set_iha_mode(model, "capture")
            return model(tokens)

        if kv_active and cuda_graphs and decode_len > 0:
            # Capturing executes every step once, so it also warms kernels and the allocator.
            prefill()
            decode_tokens = torch.randint(0, vocab, (bs, decode_len), device=device, dtype=torch.long)
            graphs, _ = capture_decode_graphs(model, prefill_len, decode_tokens)

        def decode_pass():
            if decode_len <= 0:
                return None
            if graphs:
                for g in graphs:
                    g.replay()
                return None
            if kv_active:
                set_iha_mode(model, "decode")
                out = run_cached_decode(model, prefill_len=prefill_len, decode_len=decode_len,
                                        batch_size=bs, device=device)
                set_iha_mode(model, "off")
                return out
            return model.generate(tokens, max_new_tokens=decode_len, temperature=1.0, top_k=None)

        # Warmup runs the full measured workload, so kernels, the allocator and the cache buffers are
        # warm. The KV cache grows every decode step, so a shorter warmup would leave the allocator
        # cold for larger sizes. One more timed pass of each phase then sets how many passes fill a
        # window. Timing the cold first pass would overestimate it and shorten every window.
        for _ in range(max(1, warmup)):
            prefill()
            decode_pass()
        t_pre = _timed(prefill)
        t_dec = _timed(decode_pass)
        dec_window_s = min_window_s if decode_window_s is None else float(decode_window_s)
        k_pre = min(MAX_PASSES, max(1, math.ceil(min_window_s / max(t_pre, 1e-4))))
        # An eager decode pass cannot repeat without a new prefill, so only replayed graphs repeat.
        if decode_len <= 0:
            k_dec = 0
        elif graphs:
            k_dec = min(MAX_PASSES, max(1, math.ceil(dec_window_s / max(t_dec, 1e-4))))
        else:
            k_dec = 1

        def window(name, fn, planned, extend, min_s):
            """Run `planned` passes inside one energy window, then, when `extend` allows, add passes
            until the window lasts min_s. Returns the ZEUS measurement and the pass count."""
            torch.cuda.synchronize()
            monitor.begin_window(name)
            t0 = time.perf_counter()
            for _ in range(planned):
                fn()
            passes = planned
            torch.cuda.synchronize()
            while extend and passes < MAX_PASSES and time.perf_counter() - t0 < min_s:
                fn()
                passes += 1
                torch.cuda.synchronize()
            return monitor.end_window(name), passes

        def prefill_window():
            m_pre, n_pre = window("prefill", prefill, k_pre, True, min_window_s)
            return {"ttft": float(m_pre.time) / n_pre, "w_pre": float(m_pre.time), "n_pre": n_pre,
                    "e_pre_pass": float(m_pre.total_energy) / n_pre}

        def decode_window():
            m_dec, n_dec = window("decode", decode_pass, k_dec, bool(graphs), dec_window_s)
            return {"tpot": float(m_dec.time) / (n_dec * decode_len), "w_dec": float(m_dec.time),
                    "n_dec": n_dec, "e_dec_pass": float(m_dec.total_energy) / n_dec}

        def settle():
            """Untimed decode passes for settle_s, so the next decode window starts at decode power."""
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < settle_s:
                decode_pass()
                torch.cuda.synchronize()

        rows = []
        if schedule == "grouped" and k_dec:
            if not graphs:
                raise ValueError("the grouped schedule repeats decode windows, which needs CUDA-graph decode")
            rows = [prefill_window() for _ in range(max(1, n_repeats))]
            if settle_s > 0:
                settle()
            for row in rows:
                row.update(decode_window())
        else:
            for _ in range(max(1, n_repeats)):
                row = prefill_window()
                if k_dec:
                    if settle_s > 0:
                        settle()
                    row.update(decode_window())
                rows.append(row)

        def med(key):
            return _median([r[key] for r in rows])

        e_pre_pass = med("e_pre_pass")
        result = {
            "ttft_ms": med("ttft") * 1e3,
            "prefill_energy_per_token_uJ": e_pre_pass / (bs * prefill_len) * 1e6,
            "prefill_power_W": _median([r["e_pre_pass"] * r["n_pre"] / r["w_pre"] for r in rows]),
            "hw_feasible": True,
            "zeus_prefill_energy_per_pass_J": e_pre_pass,
            "zeus_prefill_passes": int(med("n_pre")),
            "zeus_prefill_window_s": med("w_pre"),
            "zeus_prefill_energy_cv": _spread([r["e_pre_pass"] for r in rows]),
            "zeus_kv_cache_used": kv_active,
            "zeus_kv_cache_skip": kv_skip,
            "zeus_cuda_graphs": bool(graphs),
            "zeus_peak_mem_GB": torch.cuda.max_memory_allocated() / 1e9,
        }
        result["power_W"] = result["prefill_power_W"]
        if k_dec:
            e_dec_pass = med("e_dec_pass")
            result.update({
                "tpot_ms": med("tpot") * 1e3,
                "energy_per_token_uJ": e_dec_pass / (bs * decode_len) * 1e6,
                "session_energy_per_token_uJ": (e_pre_pass + e_dec_pass) / (bs * (prefill_len + decode_len)) * 1e6,
                "power_W": _median([r["e_dec_pass"] * r["n_dec"] / r["w_dec"] for r in rows]),
                "zeus_decode_energy_per_pass_J": e_dec_pass,
                "zeus_decode_passes": int(med("n_dec")),
                "zeus_decode_window_s": med("w_dec"),
                "zeus_repeats_energy_cv": _spread([r["e_dec_pass"] for r in rows]),
            })
            if idle_power_W is not None:
                result["dynamic_energy_per_token_uJ"] = _median(
                    [max(0.0, r["e_dec_pass"] * r["n_dec"] - idle_power_W * r["w_dec"])
                     / (r["n_dec"] * bs * decode_len) for r in rows]) * 1e6
        if idle_power_W is not None:
            result["idle_power_W"] = idle_power_W
        if return_windows:
            result["zeus_windows"] = rows
        return result
    except torch.cuda.OutOfMemoryError as e:
        return _failed(f"OOM: {e}")
    except Exception as e:
        if _is_cuda_fault(e):
            raise
        log.warning(f"measurement failed: {e}")
        return _failed(str(e))
    finally:
        graphs = None
        if kv_active:
            try:
                detach_iha_kv_cache(model)
            except Exception:
                pass
        del model
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Measure one Individual JSON on the local GPU.")
    ap.add_argument("individual", help="JSON file holding one Individual")
    ap.add_argument("--prefill-len", type=int, default=128)
    ap.add_argument("--decode-len", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--cuda-graphs", action="store_true")
    ap.add_argument("--min-window-s", type=float, default=2.0)
    ap.add_argument("--schedule", choices=["interleaved", "grouped"], default="interleaved")
    ap.add_argument("--settle-s", type=float, default=0.0)
    ap.add_argument("--decode-window-s", type=float, default=None)
    a = ap.parse_args()
    ind = json.load(open(a.individual))
    print(json.dumps(measure_one(ind, a.prefill_len, a.decode_len, a.repeats, batch_size=a.batch_size,
                                 cuda_graphs=a.cuda_graphs, min_window_s=a.min_window_s,
                                 schedule=a.schedule, settle_s=a.settle_s,
                                 decode_window_s=a.decode_window_s),
                     indent=2, default=str))
