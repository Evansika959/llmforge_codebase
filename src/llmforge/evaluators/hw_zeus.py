"""Hardware evaluator: measured latency and energy on the local NVIDIA GPU through ZEUS.

Each architecture is measured once per setting and cached under runs/cache/hw, keyed by the active
architecture. The measurement method, GPU model, dtype, batch size, prefill and decode lengths, repeat
count, CUDA-graph decode, minimum window length, a non-default decode schedule and an optional device
label are part of the cache setting. Measurements hold
the cross-process GPU lock, so no other GPU work overlaps an energy window.
llmforge.hw.zeus.measure documents what is measured.

Idle power is measured once when the first measurement starts and is used for
dynamic_energy_per_token_uJ.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from ..search.individual import Individual
from .cache import JsonlCache, gpu_lock, setting_tag

# Preallocated KV cache sized prefill plus decode, native grouped-query attention, full-length
# warmup, and windows repeated to a minimum duration. Bump whenever the measured code path changes,
# so the cache never mixes measurements from different implementations.
MEASURE_METHOD = "prealloc-kv-native-gqa-minwindow-v4"


class HwZeus:
    def __init__(self, prefill_len: int = 128, decode_len: int = 32, n_repeats: int = 3,
                 warmup: int = 1, dtype: str = "bf16", batch_size: int = 1,
                 use_kv_cache: bool = True, use_gpu_lock: bool = True, idle_seconds: float = 5.0,
                 cache_dir: Optional[str] = None, cuda_graphs: bool = False,
                 min_window_s: float = 2.0, schedule: str = "interleaved", settle_s: float = 0.0,
                 decode_window_s: Optional[float] = None, device_label: Optional[str] = None):
        import torch

        from ..paths import CACHE

        if not torch.cuda.is_available():
            raise RuntimeError("hw=zeus needs a CUDA GPU")
        self.gpu_name = torch.cuda.get_device_name(0)
        self.prefill_len, self.decode_len = int(prefill_len), int(decode_len)
        self.n_repeats, self.warmup = int(n_repeats), int(warmup)
        self.dtype, self.batch_size, self.use_kv_cache = dtype, int(batch_size), use_kv_cache
        self.use_gpu_lock, self.idle_seconds = use_gpu_lock, float(idle_seconds)
        self.cuda_graphs, self.min_window_s = bool(cuda_graphs), float(min_window_s)
        self.schedule, self.settle_s = schedule, float(settle_s)
        self.decode_window_s = None if decode_window_s is None else float(decode_window_s)
        self.settings = {"backend": "zeus", "method": MEASURE_METHOD, "gpu": self.gpu_name,
                         "prefill_len": self.prefill_len, "decode_len": self.decode_len,
                         "n_repeats": self.n_repeats, "warmup": self.warmup, "dtype": dtype,
                         "batch_size": self.batch_size, "kv_cache": use_kv_cache,
                         "cuda_graphs": self.cuda_graphs, "min_window_s": self.min_window_s}
        # The schedule options only change decode windows. At their defaults they add nothing to the
        # setting, so caches written before the options existed stay valid.
        sched = ""
        if self.decode_len > 0 and (schedule != "interleaved" or self.settle_s > 0 or self.decode_window_s is not None):
            self.settings.update({"schedule": schedule, "settle_s": self.settle_s,
                                  "decode_window_s": self.decode_window_s})
            sched = f"_{schedule}-s{self.settle_s:g}"
            if self.decode_window_s is not None:
                sched += f"-dw{self.decode_window_s:g}"
        # A campaign that measures on several GPUs of the same model labels each device, so their
        # measurements never share a cache. Without a label nothing is added to the setting.
        label = device_label if device_label is not None else os.environ.get("LLMFORGE_DEVICE_LABEL", "")
        dev = ""
        if label:
            if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
                raise ValueError(f"device label {label!r} may use letters, digits, '-' and '_' only")
            self.settings["device_label"] = label
            dev = f"_dev-{label}"
        slug = re.sub(r"[^A-Za-z0-9]+", "-", self.gpu_name).strip("-").lower()
        graphs = "_cg" if self.cuda_graphs else ""
        fname = (f"zeus__{slug}__p{self.prefill_len}_d{self.decode_len}_b{self.batch_size}_{dtype}_"
                 f"r{self.n_repeats}{graphs}_w{self.min_window_s:g}{sched}{dev}__{setting_tag(**self.settings)}.jsonl")
        self.cache = JsonlCache((CACHE / "hw" / fname) if cache_dir is None else f"{cache_dir}/hw/{fname}")
        self.monitor = None
        self.idle_power_W: Optional[float] = None
        self.n_measured = 0

    def _ensure_monitor(self) -> None:
        if self.monitor is not None:
            return
        from zeus.monitor import ZeusMonitor

        from ..hw.zeus.measure import measure_idle_power

        self.monitor = ZeusMonitor(gpu_indices=[0], cpu_indices=[], sync_execution_with="torch",
                                   approx_instant_energy=True)
        with gpu_lock(self.use_gpu_lock):
            self.idle_power_W = measure_idle_power(self.monitor, self.idle_seconds)

    def evaluate(self, inds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        from ..hw.zeus.measure import measure_one

        out = []
        for ind in inds:
            key = Individual.from_dict(ind).arch_key()
            r = self.cache.get(key)
            if r is None:
                self._ensure_monitor()
                with gpu_lock(self.use_gpu_lock):
                    r = measure_one(ind, prefill_len=self.prefill_len, decode_len=self.decode_len,
                                    n_repeats=self.n_repeats, warmup=self.warmup, dtype=self.dtype,
                                    monitor=self.monitor, use_kv_cache=self.use_kv_cache,
                                    batch_size=self.batch_size, idle_power_W=self.idle_power_W,
                                    cuda_graphs=self.cuda_graphs, min_window_s=self.min_window_s,
                                    schedule=self.schedule, settle_s=self.settle_s,
                                    decode_window_s=self.decode_window_s)
                self.n_measured += 1
                self.cache.put(key, r)
            out.append(dict(r))
        return out
