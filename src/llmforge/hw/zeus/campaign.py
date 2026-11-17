"""A measurement campaign: many architectures measured with ZEUS in one session on one GPU.

A search measures candidates in the order it proposes them, so a later comparison between two sets
of architectures (say a searched front and a uniform-scaling baseline) can confound the sets with
device drift over the hours between them. A campaign removes that confound:

  * Every listed architecture is measured once, in a random order drawn from a recorded seed, so the
    sets are interleaved in time.
  * Two anchor architectures are re-measured at the start, after every `anchor_every` targets and at
    the end, which tracks drift over the session.
  * Before each architecture the device and the host must be idle: no other process on the GPU and
    less than `max_other_cpu` cores of CPU in use by anything but this process. The same checks span
    the whole measurement: the GPU process list is read at every energy window and the host CPU use
    of other processes is averaged over the whole measurement. A measurement that overlapped other
    work is discarded and repeated once the host is idle again. When the host stays busy for
    `max_wait_s` the campaign stops with exit status 3, and a rerun resumes it.
  * Before each architecture nvidia-smi must run and report the same driver as NVML, and the driver
    and NVML library versions must equal those at the start of the session. A change (for example an
    unattended driver upgrade) stops the campaign with exit status 4.
  * Results are appended to a jsonl, one line per measured slot, and a rerun resumes after the last
    recorded slot. The plan (seed, order, anchor slots) is written next to the results once and a
    resumed campaign refuses a different plan.

Each line records the architecture, the slot, the US Eastern timestamp, the protocol, software
versions, the idle checks, the device temperature and clocks before the measurement, and every
metric of llmforge.hw.zeus.measure.measure_one including the per-window values. Nothing that
identifies the machine (host name, device serial or UUID) is recorded.

The protocol is read from the "hw" block of a search's run.json, so the campaign repeats exactly the
measurement that search ran:

    python -m llmforge.hw.zeus.campaign --archs archs.json \\
        --protocol runs_v2/search/<run>/run.json --out zeus_campaign.jsonl --seed 0

The archs file is a JSON list of {"id", "set", "key", "individual"}.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

PROTOCOL_KEYS = ("method", "gpu", "prefill_len", "decode_len", "n_repeats", "warmup", "dtype",
                 "batch_size", "kv_cache", "cuda_graphs", "min_window_s", "schedule", "settle_s",
                 "decode_window_s")
EXIT_HOST_BUSY = 3
EXIT_DRIVER_CHANGED = 4


def now_et() -> str:
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())


def make_plan(ids: List[str], anchors: List[str], seed: int, anchor_every: int) -> List[Dict[str, Any]]:
    """Slots in measurement order: anchors, then blocks of `anchor_every` shuffled targets each followed
    by the anchors. The anchors swap order at every visit, so neither always follows a target."""
    order = list(ids)
    random.Random(seed).shuffle(order)
    plan: List[Dict[str, Any]] = []
    visit = 0

    def add_anchors():
        nonlocal visit
        seq = anchors if visit % 2 == 0 else list(reversed(anchors))
        for a in seq:
            plan.append({"id": a, "role": "anchor", "anchor_visit": visit})
        visit += 1

    add_anchors()
    for i, a in enumerate(order):
        plan.append({"id": a, "role": "target", "order_index": i})
        if (i + 1) % anchor_every == 0 and i + 1 < len(order):
            add_anchors()
    add_anchors()
    for s, p in enumerate(plan):
        p["slot"] = s
    return plan


# ---------------------------------------------------------------- idle checks

def _proc_cpu_jiffies(pid: str = "self") -> int:
    """User plus system jiffies of one process, all threads included."""
    with open(f"/proc/{pid}/stat") as f:
        parts = f.read().rsplit(")", 1)[1].split()
    return int(parts[11]) + int(parts[12])


def _host_busy_jiffies() -> int:
    with open("/proc/stat") as f:
        vals = [int(x) for x in f.readline().split()[1:]]
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
    return sum(vals[:8]) - idle


class HostProbe:
    """Linux /proc and NVML readings for the idle checks. Replaceable in tests."""

    def __init__(self, gpu_index: int = 0):
        self.gpu_index = gpu_index
        self._nvml = None
        self.hz = os.sysconf("SC_CLK_TCK")

    def nvml(self):
        if self._nvml is None:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = (pynvml, pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index))
        return self._nvml

    def cpu_counters(self) -> Dict[str, float]:
        return {"t": time.monotonic(), "host": _host_busy_jiffies(), "self": _proc_cpu_jiffies()}

    def other_cpu_cores(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        """Cores used by processes other than this one between two counter readings."""
        dt = max(1e-6, b["t"] - a["t"])
        return max(0.0, ((b["host"] - a["host"]) - (b["self"] - a["self"])) / self.hz / dt)

    def foreign_gpu_pids(self) -> List[int]:
        nv, h = self.nvml()
        me = os.getpid()
        pids = set()
        for fn in ("nvmlDeviceGetComputeRunningProcesses", "nvmlDeviceGetGraphicsRunningProcesses"):
            try:
                pids |= {int(p.pid) for p in getattr(nv, fn)(h)}
            except Exception:
                pass
        return sorted(p for p in pids if p != me)

    def driver_versions(self) -> Dict[str, Any]:
        """Driver version from nvidia-smi and from NVML, and the NVML library version. nvidia-smi is
        run every time, so a broken or upgraded driver is noticed before the next window."""
        import subprocess

        out: Dict[str, Any] = {}
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=30)
            out["smi_ok"] = r.returncode == 0
            out["smi_driver"] = r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else None
        except Exception:
            out["smi_ok"], out["smi_driver"] = False, None
        nv, _ = self.nvml()
        for name, fn in (("nvml_driver", nv.nvmlSystemGetDriverVersion),
                         ("nvml_library", nv.nvmlSystemGetNVMLVersion)):
            try:
                v = fn()
                out[name] = v.decode() if isinstance(v, bytes) else v
            except Exception:
                out[name] = None
        return out

    def device_state(self) -> Dict[str, Any]:
        nv, h = self.nvml()
        out: Dict[str, Any] = {}
        for name, fn in (("temp_C", lambda: nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)),
                         ("sm_clock_MHz", lambda: nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM)),
                         ("mem_clock_MHz", lambda: nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM)),
                         ("power_W", lambda: nv.nvmlDeviceGetPowerUsage(h) / 1e3),
                         ("power_limit_W", lambda: nv.nvmlDeviceGetEnforcedPowerLimit(h) / 1e3),
                         ("util_pct", lambda: nv.nvmlDeviceGetUtilizationRates(h).gpu),
                         ("pstate", lambda: nv.nvmlDeviceGetPerformanceState(h))):
            try:
                out[name] = fn()
            except Exception:
                out[name] = None
        return out

    def top_other_processes(self, n: int = 5) -> List[Dict[str, Any]]:
        """Busiest other processes over 0.5 s, by command name only (no arguments, no user)."""
        me = os.getpid()

        def snap():
            s = {}
            for p in os.listdir("/proc"):
                if p.isdigit() and int(p) != me:
                    try:
                        s[p] = _proc_cpu_jiffies(p)
                    except Exception:
                        pass
            return s

        a = snap()
        time.sleep(0.5)
        b = snap()
        rows = []
        for p, j in b.items():
            d = j - a.get(p, j)
            if d > 0:
                try:
                    comm = Path(f"/proc/{p}/comm").read_text().strip()
                except Exception:
                    comm = "?"
                rows.append({"pid": int(p), "comm": comm, "cores": round(d / self.hz / 0.5, 3)})
        return sorted(rows, key=lambda r: -r["cores"])[:n]


class GuardedMonitor:
    """Wraps a ZEUS monitor and reads the GPU process list at every window boundary."""

    def __init__(self, monitor, probe: HostProbe):
        self.monitor, self.probe = monitor, probe
        self.foreign: List[int] = []
        self.n_windows = 0

    def reset(self) -> None:
        self.foreign, self.n_windows = [], 0

    def _check(self) -> None:
        f = self.probe.foreign_gpu_pids()
        if f:
            self.foreign = sorted(set(self.foreign) | set(f))

    def begin_window(self, name, *a, **kw):
        self._check()
        self.n_windows += 1
        return self.monitor.begin_window(name, *a, **kw)

    def end_window(self, name, *a, **kw):
        m = self.monitor.end_window(name, *a, **kw)
        self._check()
        return m


# ---------------------------------------------------------------- campaign

class Campaign:
    def __init__(self, archs: List[Dict[str, Any]], protocol: Dict[str, Any], out: str, seed: int = 0,
                 anchors: Optional[List[str]] = None, anchor_every: int = 25,
                 max_other_cpu: float = 0.5, idle_sample_s: float = 2.0, max_wait_s: float = 1800.0,
                 poll_s: float = 30.0, idle_seconds: float = 5.0,
                 measure_fn: Optional[Callable[..., Dict[str, Any]]] = None,
                 monitor=None, probe: Optional[HostProbe] = None,
                 idle_power_fn: Optional[Callable[[Any], float]] = None,
                 versions: Optional[Dict[str, Any]] = None, sleep=time.sleep, log=print):
        missing = [k for k in PROTOCOL_KEYS if k not in protocol]
        if missing:
            raise ValueError(f"protocol lacks {missing}")
        self.protocol = {k: protocol[k] for k in PROTOCOL_KEYS}
        self.by_id = {a["id"]: a for a in archs}
        if len(self.by_id) != len(archs):
            raise ValueError("duplicate ids in the archs list")
        self.anchors = list(anchors) if anchors else default_anchors(archs)
        for a in self.anchors:
            if a not in self.by_id:
                raise ValueError(f"anchor {a} is not in the archs list")
        self.seed, self.anchor_every = int(seed), int(anchor_every)
        self.out = Path(out)
        self.plan_path = self.out.with_suffix(".plan.json")
        self.status_path = self.out.with_suffix(".status.json")
        self.max_other_cpu, self.idle_sample_s = float(max_other_cpu), float(idle_sample_s)
        self.max_wait_s, self.poll_s, self.idle_seconds = float(max_wait_s), float(poll_s), float(idle_seconds)
        self.measure_fn, self.probe, self.idle_power_fn = measure_fn, probe, idle_power_fn
        self.monitor = monitor
        self.versions = versions or {}
        self.sleep, self.log = sleep, log
        self.idle_power_W: Optional[float] = None
        self.plan = self._load_or_write_plan([a["id"] for a in archs])
        self.driver_ref: Optional[Dict[str, Any]] = None

    def _load_or_write_plan(self, ids: List[str]) -> List[Dict[str, Any]]:
        plan = make_plan(ids, self.anchors, self.seed, self.anchor_every)
        spec = {"seed": self.seed, "anchors": self.anchors, "anchor_every": self.anchor_every,
                "protocol": self.protocol, "n_targets": len(ids), "plan": plan}
        if self.plan_path.exists():
            old = json.loads(self.plan_path.read_text())
            if old["plan"] != plan or old["protocol"] != self.protocol:
                raise SystemExit(f"{self.plan_path} holds a different plan or protocol; use a new --out")
        else:
            self.plan_path.parent.mkdir(parents=True, exist_ok=True)
            spec["created_et"] = now_et()
            self.plan_path.write_text(json.dumps(spec, indent=1))
        return plan

    def done_slots(self) -> set:
        done = set()
        if self.out.exists():
            for line in self.out.read_text().splitlines():
                if line.strip():
                    try:
                        done.add(int(json.loads(line)["slot"]))
                    except (ValueError, KeyError):
                        pass
        return done

    def _status(self, **kw) -> None:
        self.status_path.write_text(json.dumps({"updated_et": now_et(), **kw}, indent=1))

    def _idle_now(self) -> Dict[str, Any]:
        a = self.probe.cpu_counters()
        self.sleep(self.idle_sample_s)
        b = self.probe.cpu_counters()
        cpu = self.probe.other_cpu_cores(a, b)
        gpu = self.probe.foreign_gpu_pids()
        return {"ok": cpu < self.max_other_cpu and not gpu, "other_cpu_cores": round(cpu, 3),
                "foreign_gpu_pids": gpu}

    def check_driver(self, slot: int) -> Dict[str, Any]:
        """nvidia-smi must work and the driver and NVML versions must equal those at the campaign start.
        Anything else stops the campaign with exit status 4, since the session would no longer be
        one software stack."""
        v = self.probe.driver_versions()
        if self.driver_ref is None:
            self.driver_ref = dict(v)
            self.versions.setdefault("driver_check", self.driver_ref)
        ok = bool(v.get("smi_ok")) and v.get("smi_driver") == v.get("nvml_driver") and all(
            v.get(k) == self.driver_ref.get(k) for k in ("smi_driver", "nvml_driver", "nvml_library"))
        if not ok:
            self.log(f"[{now_et()}] driver check failed before slot {slot}: {v} vs start {self.driver_ref}")
            self._status(state="stopped_driver_changed", slot=slot, now=v, start=self.driver_ref)
            raise SystemExit(EXIT_DRIVER_CHANGED)
        return v

    def wait_idle(self, slot: int) -> Dict[str, Any]:
        self.check_driver(slot)
        t0 = time.monotonic()
        while True:
            chk = self._idle_now()
            if chk["ok"]:
                return chk
            waited = time.monotonic() - t0
            top = self.probe.top_other_processes()
            self.log(f"[{now_et()}] host busy before slot {slot}: {chk}, top {top}; waited {waited:.0f}s")
            self._status(state="waiting_for_idle", slot=slot, check=chk, top_other=top, waited_s=waited)
            if waited >= self.max_wait_s:
                self._status(state="stopped_host_busy", slot=slot, check=chk, top_other=top)
                raise SystemExit(EXIT_HOST_BUSY)
            self.sleep(self.poll_s)

    def _measure_idle_power(self) -> None:
        if self.idle_power_fn is not None:
            self.wait_idle(-1)
            self.idle_power_W = float(self.idle_power_fn(self.monitor))
            self.log(f"[{now_et()}] idle power {self.idle_power_W:.1f} W")

    def _kwargs(self) -> Dict[str, Any]:
        p = self.protocol
        return dict(prefill_len=p["prefill_len"], decode_len=p["decode_len"], n_repeats=p["n_repeats"],
                    warmup=p["warmup"], dtype=p["dtype"], use_kv_cache=p["kv_cache"],
                    batch_size=p["batch_size"], cuda_graphs=p["cuda_graphs"],
                    min_window_s=p["min_window_s"], schedule=p["schedule"], settle_s=p["settle_s"],
                    decode_window_s=p["decode_window_s"])

    def measure_slot(self, slot: Dict[str, Any]) -> Dict[str, Any]:
        arch = self.by_id[slot["id"]]
        attempts = 0
        while True:
            attempts += 1
            pre = self.wait_idle(slot["slot"])
            state = self.probe.device_state()
            if hasattr(self.monitor, "reset"):
                self.monitor.reset()
            t_start, ts = self.probe.cpu_counters(), now_et()
            res = self.measure_fn(arch["individual"], monitor=self.monitor, idle_power_W=self.idle_power_W,
                                  return_windows=True, **self._kwargs())
            t_end = self.probe.cpu_counters()
            during_cpu = self.probe.other_cpu_cores(t_start, t_end)
            foreign = list(getattr(self.monitor, "foreign", []))
            clean = during_cpu < self.max_other_cpu and not foreign
            if clean:
                break
            self.log(f"[{now_et()}] slot {slot['slot']} {slot['id']} overlapped other work "
                     f"(cpu {during_cpu:.2f} cores, gpu pids {foreign}); discarded, repeating")
        return {"slot": slot["slot"], "id": arch["id"], "set": arch.get("set"), "role": slot["role"],
                "order_index": slot.get("order_index"), "anchor_visit": slot.get("anchor_visit"),
                "key": arch.get("key"), "timestamp_et": ts, "finished_et": now_et(),
                "wall_s": round(t_end["t"] - t_start["t"], 2), "attempts": attempts, "seed": self.seed,
                "protocol": self.protocol, "versions": self.versions,
                "idle_check": {"before": pre, "during_other_cpu_cores": round(during_cpu, 3),
                               "during_foreign_gpu_pids": foreign,
                               "windows_checked": getattr(self.monitor, "n_windows", None),
                               "max_other_cpu": self.max_other_cpu},
                "device_before": state, "idle_power_W": self.idle_power_W,
                "metrics": res, "individual": arch["individual"]}

    def run(self) -> int:
        done = self.done_slots()
        todo = [s for s in self.plan if s["slot"] not in done]
        self.log(f"[{now_et()}] plan {len(self.plan)} slots, {len(done)} done, {len(todo)} to go")
        if not todo:
            self._status(state="complete", n_slots=len(self.plan))
            return 0
        self._measure_idle_power()
        self.out.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.monotonic()
        for i, s in enumerate(todo):
            self._status(state="measuring", slot=s["slot"], id=s["id"], done=len(done) + i,
                         total=len(self.plan))
            rec = self.measure_slot(s)
            with open(self.out, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
            m = rec["metrics"]
            el = time.monotonic() - t0
            self.log(f"[{now_et()}] slot {s['slot']}/{len(self.plan) - 1} {s['role']:6} {s['id']} "
                     f"E={m.get('energy_per_token_uJ', float('nan')):.1f}uJ "
                     f"tpot={m.get('tpot_ms', float('nan')):.3f}ms feasible={m.get('hw_feasible')} "
                     f"eta={el / (i + 1) * (len(todo) - i - 1) / 3600:.2f}h")
            # Idle power is re-measured after every anchor visit, which tracks idle drift as well.
            if s["role"] == "anchor" and (i + 1 < len(todo)) and todo[i + 1]["role"] == "target":
                self._measure_idle_power()
        self._status(state="complete", n_slots=len(self.plan))
        return 0


def default_anchors(archs: List[Dict[str, Any]]) -> List[str]:
    """The largest and the smallest uniform architecture, by total width summed over layers."""
    uni = [a for a in archs if a.get("set") == "uniform"] or archs

    def size(a):
        return sum(sum(int(v) for k, v in L.items() if isinstance(v, (int, float)) and not isinstance(v, bool))
                   for L in a["individual"]["layers"])

    uni = sorted(uni, key=size)
    return [uni[-1]["id"], uni[0]["id"]]


def software_versions() -> Dict[str, Any]:
    out: Dict[str, Any] = {"python": platform.python_version()}
    for mod in ("torch", "transformers", "zeus", "numpy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            pass
    try:
        import torch

        out["cuda"] = torch.version.cuda
        out["cudnn"] = torch.backends.cudnn.version()
        out["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        import pynvml

        pynvml.nvmlInit()
        d = pynvml.nvmlSystemGetDriverVersion()
        out["driver"] = d.decode() if isinstance(d, bytes) else d
    except Exception:
        pass
    return out


def load_protocol(path: str) -> Dict[str, Any]:
    d = json.loads(Path(path).read_text())
    return d.get("hw", d)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--archs", required=True, help="JSON list of {id, set, key, individual}")
    ap.add_argument("--protocol", required=True, help="run.json of a ZEUS search, or its hw block")
    ap.add_argument("--out", required=True, help="results jsonl; plan and status files sit next to it")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--anchors", default=None, help="two comma-separated ids, default largest,smallest")
    ap.add_argument("--anchor-every", type=int, default=25)
    ap.add_argument("--ids", default=None, help="measure only these comma-separated ids (smoke tests)")
    ap.add_argument("--max-other-cpu", type=float, default=0.5, help="cores other processes may use")
    ap.add_argument("--max-wait-s", type=float, default=1800.0)
    ap.add_argument("--allow-gpu-mismatch", action="store_true")
    a = ap.parse_args(argv)

    archs = json.loads(Path(a.archs).read_text())
    if a.ids:
        want = a.ids.split(",")
        archs = [r for r in archs if r["id"] in want]
    protocol = load_protocol(a.protocol)

    import torch
    from zeus.monitor import ZeusMonitor

    from ...evaluators.hw_zeus import MEASURE_METHOD
    from .measure import measure_idle_power, measure_one

    if protocol.get("method") != MEASURE_METHOD:
        raise SystemExit(f"protocol method {protocol.get('method')} != implemented {MEASURE_METHOD}")
    gpu = torch.cuda.get_device_name(0)
    if gpu != protocol.get("gpu") and not a.allow_gpu_mismatch:
        raise SystemExit(f"GPU {gpu} differs from the protocol's {protocol.get('gpu')}")
    probe = HostProbe(0)
    monitor = GuardedMonitor(ZeusMonitor(gpu_indices=[0], cpu_indices=[], sync_execution_with="torch",
                                         approx_instant_energy=True), probe)
    anchors = a.anchors.split(",") if a.anchors else None
    if a.ids and anchors is None:
        anchors = default_anchors(archs) if len(archs) > 1 else [archs[0]["id"]]
    camp = Campaign(archs, protocol, a.out, seed=a.seed, anchors=anchors, anchor_every=a.anchor_every,
                    max_other_cpu=a.max_other_cpu, max_wait_s=a.max_wait_s, measure_fn=measure_one,
                    monitor=monitor, probe=probe,
                    idle_power_fn=lambda m: measure_idle_power(m.monitor, 5.0),
                    versions=software_versions(), log=lambda s: print(s, flush=True))
    return camp.run()


if __name__ == "__main__":
    sys.exit(main())
