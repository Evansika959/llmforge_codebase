"""Sequential queue for the hardware-aware search experiments.

Jobs run one at a time in the order the YAML lists them, so no two jobs share the GPU. A job is
skipped when its output directory holds DONE. A search job that exits with an error is retried up to
twice, 30 s apart, because it resumes from the evaluation caches and loses only the evaluation in
flight. A command job is retried only when it sets `retries`. A job that still fails gets a FAILED
file with its exit code and the queue moves on. The YAML is re-read after every job, so jobs can be
added while the queue runs. Logs go to runs/logs/hw_nas/<job>.log.

    python experiments/hw_nas/queue.py experiments/hw_nas/queue.yaml             run pending jobs
    python experiments/hw_nas/queue.py experiments/hw_nas/queue.yaml --status    list job status
    python experiments/hw_nas/queue.py experiments/hw_nas/queue.yaml --only REGEX

A runner records the job it is running in runs/logs/hw_nas/<queue>.running. A new runner started
while an earlier job process is still alive waits for that process first, so a runner can be
replaced without two jobs overlapping on the GPU.

YAML schema
    defaults:   cosearch arguments shared by every job, overridden by a job's own args
    supernets:  name -> checkpoint directory, referenced from a job as `supernet: NAME`
    jobs:
      - name:   unique name, also the output directory runs/search/<name>
        args:   cosearch arguments. Keys use underscores. `space: KEY` expands to
                configs/search_spaces/KEY.yaml. True emits a bare flag, false or null omits it.
                A list becomes space-separated values, except constraint and hw_arg, which
                repeat the flag.
      - name:   a job that is not a search
        command: [python, path/to/script.py, --flag, value]
        out:    output directory, default runs/jobs/<name>. The runner writes DONE there when the
                command exits with status 0.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "runs" / "logs" / "hw_nas"
REPEATED = {"constraint", "hw_arg"}
SEARCH_RETRIES = 2
RETRY_DELAY_S = 30


def render(job: dict, cfg: dict) -> list:
    args = {**(cfg.get("defaults") or {}), **(job.get("args") or {})}
    space = str(args.pop("space"))
    args["space"] = space if "/" in space else f"configs/search_spaces/{space}.yaml"
    sn = str(args.pop("supernet"))
    args["supernet"] = (cfg.get("supernets") or {}).get(sn, sn)
    args.setdefault("out", f"runs/search/{job['name']}")
    argv = []
    for k, v in args.items():
        flag = "--" + k.replace("_", "-")
        if v is True:
            argv.append(flag)
        elif v is False or v is None:
            continue
        elif isinstance(v, list) and k in REPEATED:
            for item in v:
                argv += [flag, str(item)]
        elif isinstance(v, list):
            argv += [flag] + [str(x) for x in v]
        else:
            argv += [flag, str(v)]
    return argv


def job_out(job: dict, cfg: dict) -> Path:
    if "command" in job:
        return ROOT / job.get("out", f"runs/jobs/{job['name']}")
    argv = render(job, cfg)
    return ROOT / argv[argv.index("--out") + 1]


def job_cmd(job: dict, cfg: dict) -> list:
    if "command" in job:
        return [sys.executable if c == "python" else str(c) for c in job["command"]]
    return [sys.executable, "-m", "llmforge.search.cosearch", *render(job, cfg)]


def status_of(job: dict, cfg: dict, running: dict) -> str:
    out = job_out(job, cfg)
    if (out / "DONE").exists():
        return "done"
    if (out / "FAILED").exists():
        return "failed"
    if running.get("job") == job["name"] and _alive(running.get("pid")):
        state = "running"
    else:
        state = "partial" if out.exists() and any(out.iterdir()) else "pending"
    trace = out / "trace.jsonl"
    if trace.exists() and trace.stat().st_size:
        return f"{state} {trace.read_text().strip().splitlines()[-1][:90]}"
    return state


def _alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("queue")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--only", default=None, help="run only jobs whose name matches this regex")
    a = ap.parse_args()
    qpath = Path(a.queue).resolve()
    LOGS.mkdir(parents=True, exist_ok=True)
    running_file = LOGS / f"{qpath.stem}.running"

    def running() -> dict:
        try:
            return json.loads(running_file.read_text())
        except (OSError, ValueError):
            return {}

    if a.status:
        cfg = yaml.safe_load(qpath.read_text())
        r = running()
        for job in cfg["jobs"]:
            print(f"{job['name']:55s} {status_of(job, cfg, r)}")
        return

    lock = open(LOGS / f"{qpath.stem}.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another runner holds {LOGS / (qpath.stem + '.lock')}")

    prev = running()
    if _alive(prev.get("pid")):
        print(f"[{time.strftime('%m-%d %H:%M')}] waiting for {prev.get('job')} (pid {prev['pid']})", flush=True)
        while _alive(prev.get("pid")):
            time.sleep(20)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("HF_HUB_OFFLINE", "1")
    attempted = set()
    while True:
        cfg = yaml.safe_load(qpath.read_text())
        todo = [j for j in cfg["jobs"] if j["name"] not in attempted
                and (a.only is None or re.search(a.only, j["name"]))
                and status_of(j, cfg, {}).split()[0] not in ("done", "failed")]
        if not todo:
            break
        job = todo[0]
        attempted.add(job["name"])
        out = job_out(job, cfg)
        out.mkdir(parents=True, exist_ok=True)
        cmd = job_cmd(job, cfg)
        retries = int(job.get("retries", 0 if "command" in job else SEARCH_RETRIES))
        for attempt in range(retries + 1):
            t0 = time.time()
            note = f", attempt {attempt + 1} of {retries + 1}" if attempt else ""
            print(f"[{time.strftime('%m-%d %H:%M')}] start {job['name']}{note}", flush=True)
            with open(LOGS / f"{job['name']}.log", "a") as log:
                log.write(f"\n$ {' '.join(cmd)}\n")
                log.flush()
                proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                running_file.write_text(json.dumps({"job": job["name"], "pid": proc.pid, "started": time.time()}))
                rc = proc.wait()
            running_file.unlink(missing_ok=True)
            mins = (time.time() - t0) / 60
            if rc == 0 or attempt == retries:
                break
            print(f"[{time.strftime('%m-%d %H:%M')}] FAILED ({rc}) {job['name']} in {mins:.1f} min, retrying",
                  flush=True)
            time.sleep(RETRY_DELAY_S)
        if rc != 0:
            (out / "FAILED").write_text(f"exit {rc} after {mins:.1f} min, see {LOGS / (job['name'] + '.log')}\n")
        elif "command" in job:
            (out / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
        print(f"[{time.strftime('%m-%d %H:%M')}] {'done' if rc == 0 else f'FAILED ({rc})'} {job['name']} "
              f"in {mins:.1f} min", flush=True)


if __name__ == "__main__":
    main()
