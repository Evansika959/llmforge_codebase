"""Filesystem roots for code, assets, and experiment outputs.

Every location a run reads or writes resolves here, so large outputs can live on another volume
without code edits. Each root can be overridden with an environment variable.

    LLMFORGE_ROOT            repository root (default: two levels above this file)
    LLMFORGE_RUNS            run outputs, evaluation caches, checkpoints (default: <root>/runs)
    LLMFORGE_DATA            packed supernet training data (default: <root>/data)
    LLMFORGE_SCRATCH         temporary files (default: <root>/scratch)
    LLMFORGE_REALLM_FORGE    ReaLLM-Forge checkout for the ZEUS GPU target
                             (default: <root>/third_party/ReaLLM-Forge)
    LLMFORGE_DEVICE_RUNTIME  nanollmforge.c checkout for on-device measurement
                             (default: <root>/third_party/nanollmforge.c)
    LLMFORGE_TIMELOOP_WORK   Timeloop working directory (default: <runs>/cache/timeloop)
"""
import os
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


ROOT = _env_path("LLMFORGE_ROOT", Path(__file__).resolve().parents[2])
PACKAGE = Path(__file__).resolve().parent

ASSETS = ROOT / "assets"
CONFIGS = ROOT / "configs"
HELDOUT = ASSETS / "heldout"
DEVICE_ASSETS = ASSETS / "device"

RUNS = _env_path("LLMFORGE_RUNS", ROOT / "runs")
CACHE = RUNS / "cache"
DATA = _env_path("LLMFORGE_DATA", ROOT / "data")
SCRATCH = _env_path("LLMFORGE_SCRATCH", ROOT / "scratch")

REALLM_FORGE = _env_path("LLMFORGE_REALLM_FORGE", ROOT / "third_party" / "ReaLLM-Forge")
DEVICE_RUNTIME = _env_path("LLMFORGE_DEVICE_RUNTIME", ROOT / "third_party" / "nanollmforge.c")

TIMELOOP_SPECS = PACKAGE / "hw" / "timeloop" / "specs"
TIMELOOP_WORK = _env_path("LLMFORGE_TIMELOOP_WORK", CACHE / "timeloop")

# Serializes GPU work so energy measurements never overlap other GPU jobs on the same device.
GPU_LOCK = SCRATCH / "locks" / "gpu.lock"


def describe() -> str:
    rows = [("ROOT", ROOT), ("RUNS", RUNS), ("DATA", DATA), ("SCRATCH", SCRATCH),
            ("REALLM_FORGE", REALLM_FORGE), ("DEVICE_RUNTIME", DEVICE_RUNTIME),
            ("TIMELOOP_WORK", TIMELOOP_WORK)]
    return "\n".join(f"  {n:15} {p}{'' if p.exists() else '   [missing]'}" for n, p in rows)
