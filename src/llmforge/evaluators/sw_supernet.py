"""Software evaluator: held-out loss of a supernet slice.

Every candidate is scored by slicing one trained elastic supernet, with no per-candidate training.
The score is the mean token cross-entropy in nats over a fixed slice of the frozen held-out
documents (llmforge.supernet.data.heldout), each truncated to `max_len` tokens. A search scores
documents [skip_docs, skip_docs + n_docs). Final fronts are re-scored on a disjoint slice through
`with_docs`, which shares the loaded weights.

Scores are cached per (supernet checkpoint, document slice, max_len) under runs/cache/sw, so a
baseline and a search over the same supernet never score an architecture twice.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .cache import JsonlCache, gpu_lock, setting_tag


class SwSupernet:
    def __init__(self, space, ckpt: str, n_docs: int = 32, skip_docs: int = 0, max_len: int = 1024,
                 device: str = "cuda", use_gpu_lock: bool = True, cache_dir: Optional[str] = None,
                 _holder: Optional[Dict[str, Any]] = None):
        from ..paths import CACHE

        self.space = space
        self.ckpt = str(Path(ckpt).resolve())
        if not any(Path(self.ckpt).glob("*.safetensors")):
            raise FileNotFoundError(f"no safetensors weights under {self.ckpt}")
        self.n_docs, self.skip_docs, self.max_len = int(n_docs), int(skip_docs), int(max_len)
        self.device, self.use_gpu_lock = device, use_gpu_lock
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE
        ck = Path(self.ckpt)
        self.name = f"{ck.parent.name}/{ck.name}"
        self.settings = {"base_model": space.base_model, "ckpt": self.name, "docs": [self.skip_docs,
                         self.skip_docs + self.n_docs], "max_len": self.max_len}
        tag = setting_tag(**self.settings)
        fname = (f"{space.base_model}__{ck.parent.name}_{ck.name}__docs{self.skip_docs}-"
                 f"{self.skip_docs + self.n_docs}__len{self.max_len}__{tag}.jsonl")
        self.cache = JsonlCache(self.cache_dir / "sw" / fname)
        self._holder = _holder if _holder is not None else {}
        self._loss_of = None
        self.n_scored = 0
        self.seconds = 0.0

    def _ensure(self) -> None:
        if self._loss_of is not None:
            return
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from ..supernet.data.heldout import heldout_texts
        from ..supernet.search.nsga import load_supernet, make_evaluator

        if "model" not in self._holder:
            with gpu_lock(self.use_gpu_lock):
                model, tok, order = load_supernet(self.ckpt, self.space.spec, dev=self.device)
            self._holder.update(model=model, tok=tok, order=order)
        texts = heldout_texts(self.n_docs, skip=self.skip_docs)
        h = self._holder
        self._loss_of, n = make_evaluator(h["model"], h["tok"], h["order"], self.space.spec, texts,
                                          dev=self.device, max_len=self.max_len)
        if n != self.n_docs:
            raise RuntimeError(f"expected {self.n_docs} scorable documents, got {n}")

    def score(self, ind: Dict[str, Any]) -> float:
        key = self.space.key(ind)
        val = self.cache.get(key)
        if val is None:
            self._ensure()
            cfg = self.space.to_elastic_config(ind)
            t0 = time.time()
            with gpu_lock(self.use_gpu_lock):
                val = float(self._loss_of(cfg))
            dt = time.time() - t0
            self.seconds += dt
            self.n_scored += 1
            self.cache.put(key, val, seconds=round(dt, 3))
        return float(val)

    def evaluate(self, inds: List[Dict[str, Any]]) -> Tuple[List[float], List[float]]:
        mu = [self.score(ind) for ind in inds]
        return mu, [0.0] * len(mu)

    def with_docs(self, n_docs: int, skip_docs: int) -> "SwSupernet":
        """A sibling evaluator on another document slice that shares the loaded weights."""
        return SwSupernet(self.space, self.ckpt, n_docs=n_docs, skip_docs=skip_docs,
                          max_len=self.max_len, device=self.device, use_gpu_lock=self.use_gpu_lock,
                          cache_dir=str(self.cache_dir), _holder=self._holder)

    def release(self) -> None:
        """Free the supernet weights from GPU memory."""
        self._holder.clear()
        self._loss_of = None
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
