"""Evaluation caches shared across runs, and the GPU lock.

A cache file maps an architecture key to one result under one fixed evaluation setting, and the
file name encodes that setting. Several processes may append to the same file. Every append takes
an exclusive flock, and a lookup miss re-reads the lines other processes appended since the last
read, so parallel runs reuse each other's evaluations.

The GPU lock serializes GPU work across processes. Energy measured through ZEUS covers the whole
device, so no other kernel may run while a measurement window is open.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


class JsonlCache:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: Dict[str, Any] = {}
        self._offset = 0
        self._refresh()

    def _refresh(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, "r") as f:
            f.seek(self._offset)
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                if not line.endswith("\n"):
                    f.seek(pos)
                    break
                try:
                    rec = json.loads(line)
                    self._data[rec["key"]] = rec["value"]
                except (ValueError, KeyError):
                    pass
            self._offset = f.tell()

    def get(self, key: str) -> Optional[Any]:
        if key not in self._data:
            self._refresh()
        return self._data.get(key)

    def put(self, key: str, value: Any, **extra) -> None:
        self._data[key] = value
        line = json.dumps({"key": key, "value": value, "time": round(time.time(), 3), **extra}) + "\n"
        with open(self.path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(line)
                f.flush()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def __len__(self) -> int:
        return len(self._data)


def setting_tag(**settings) -> str:
    """Short digest of an evaluation setting, used in cache file names."""
    return hashlib.sha1(json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()[:10]


@contextlib.contextmanager
def gpu_lock(enabled: bool = True, path: Optional[str] = None):
    """Hold an exclusive cross-process lock on GPU work for the duration of the block."""
    if not enabled:
        yield
        return
    from ..paths import GPU_LOCK

    p = Path(path) if path else GPU_LOCK
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
