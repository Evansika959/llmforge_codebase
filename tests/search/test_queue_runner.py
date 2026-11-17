"""The experiment queue runner: optional per-queue environment and log directory, defaults unchanged."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
QUEUE = ROOT / "experiments" / "hw_nas" / "queue.py"


def _load():
    spec = importlib.util.spec_from_file_location("hw_nas_queue", QUEUE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_env_block_is_optional_and_resolves_relative_paths():
    q = _load()
    base = {"PATH": "/usr/bin"}
    plain = q.job_env({}, base)
    assert plain["PYTHONPATH"].startswith(str(ROOT / "src")) and plain["HF_HUB_OFFLINE"] == "1"
    assert "LLMFORGE_RUNS" not in plain
    env = q.job_env({"env": {"LLMFORGE_RUNS": "runs_x/rt", "OMP_NUM_THREADS": 4}}, base)
    assert env["LLMFORGE_RUNS"] == str(ROOT / "runs_x" / "rt") and env["OMP_NUM_THREADS"] == "4"


def test_render_keeps_explicit_out():
    q = _load()
    cfg = {"defaults": {"space": "smollm2-135m_nkv", "supernet": "sn"}, "supernets": {"sn": "runs/supernet/x"}}
    argv = q.render({"name": "j", "args": {"out": "runs_rebuttal/search/j", "seed": 1}}, cfg)
    assert argv[argv.index("--out") + 1] == "runs_rebuttal/search/j"
    assert argv[argv.index("--space") + 1] == "configs/search_spaces/smollm2-135m_nkv.yaml"


def test_log_dir_and_env_reach_the_job(tmp_path):
    out = tmp_path / "job_out"
    probe = tmp_path / "probe.txt"
    cfg = {"env": {"REBUTTAL_PROBE": "on"},
           "jobs": [{"name": "probe", "out": str(out),
                     "command": ["python", "-c",
                                 f"import os; open({str(probe)!r}, 'w').write(os.environ['REBUTTAL_PROBE'])"]}]}
    qy = tmp_path / "q.yaml"
    qy.write_text(yaml.safe_dump(cfg))
    logs = tmp_path / "logs"
    subprocess.run([sys.executable, str(QUEUE), str(qy), "--log-dir", str(logs)], check=True, cwd=ROOT)
    assert probe.read_text() == "on"
    assert (out / "DONE").exists() and (logs / "probe.log").exists()


def test_gpu_lock_is_on_unless_disabled():
    """Searches lock the GPU by default; --no-gpu-lock is opt-in (used only on a dedicated search host)."""
    sys.path.insert(0, str(ROOT / "src"))
    from llmforge.search.cosearch import build_parser

    base = ["--space", "configs/search_spaces/smollm2-135m_nkv.yaml", "--supernet", "x", "--out", "y"]
    assert build_parser().parse_args(base).no_gpu_lock is False
    assert build_parser().parse_args(base + ["--no-gpu-lock"]).no_gpu_lock is True
    q = _load()
    argv = q.render({"name": "j", "args": {"space": "s", "supernet": "sn", "no_gpu_lock": True}}, {})
    assert "--no-gpu-lock" in argv
    assert "--no-gpu-lock" not in q.render({"name": "j", "args": {"space": "s", "supernet": "sn"}}, {})
