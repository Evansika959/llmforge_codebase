"""Measurement campaign driver: order, anchors, resume and idle checks. Runs without a GPU."""
import json

import pytest

from llmforge.hw.zeus.campaign import (EXIT_DRIVER_CHANGED, EXIT_HOST_BUSY, Campaign, default_anchors,
                                       make_plan)

PROTOCOL = {"method": "prealloc-kv-native-gqa-minwindow-v4", "gpu": "NVIDIA H100 80GB HBM3",
            "prefill_len": 512, "decode_len": 128, "n_repeats": 3, "warmup": 1, "dtype": "bf16",
            "batch_size": 64, "kv_cache": True, "cuda_graphs": True, "min_window_s": 2.0,
            "schedule": "grouped", "settle_s": 1.5, "decode_window_s": 6.0}


def arch(i, width, set_="uniform"):
    return {"id": f"A{i:03d}", "set": set_, "key": f"k{i}",
            "individual": {"globals": {}, "layers": [{"n_head": width, "mlp_size": 4 * width}] * 2}}


ARCHS = [arch(i, 2 + i, "front" if i % 4 == 0 else "uniform") for i in range(12)]


class FakeProbe:
    """Other-process CPU use is scripted per reading pair; GPU pids are scripted per call."""

    def __init__(self, cpu=None, gpu=None, drivers=None):
        self.cpu = list(cpu or [])
        self.gpu = list(gpu or [])
        self.drivers = list(drivers or [])
        self.t = 0.0

    def cpu_counters(self):
        self.t += 1.0
        return {"t": self.t}

    def other_cpu_cores(self, a, b):
        return self.cpu.pop(0) if self.cpu else 0.0

    def foreign_gpu_pids(self):
        return self.gpu.pop(0) if self.gpu else []

    def driver_versions(self):
        v = {"smi_ok": True, "smi_driver": "580.0", "nvml_driver": "580.0", "nvml_library": "13.0"}
        if self.drivers:
            v.update(self.drivers.pop(0))
        return v

    def device_state(self):
        return {"temp_C": 40}

    def top_other_processes(self, n=5):
        return []


class FakeMonitor:
    def __init__(self):
        self.foreign, self.n_windows = [], 0

    def reset(self):
        self.foreign, self.n_windows = [], 0


def make(tmp_path, probe=None, measured=None, fail_after=None, **kw):
    measured = measured if measured is not None else []
    monitor = FakeMonitor()

    def measure_fn(ind, monitor=None, idle_power_W=None, return_windows=False, **proto):
        assert proto["schedule"] == "grouped" and proto["batch_size"] == 64 and proto["use_kv_cache"]
        if fail_after is not None and len(measured) >= fail_after:
            raise KeyboardInterrupt
        measured.append(ind["layers"][0]["n_head"])
        return {"energy_per_token_uJ": float(ind["layers"][0]["n_head"]), "tpot_ms": 1.0, "hw_feasible": True}

    return Campaign(ARCHS, PROTOCOL, str(tmp_path / "res.jsonl"), seed=7, anchor_every=5,
                    measure_fn=measure_fn, monitor=monitor, probe=probe or FakeProbe(),
                    idle_power_fn=lambda m: 100.0, sleep=lambda s: None, log=lambda s: None,
                    **kw), measured, monitor


def rows(tmp_path):
    return [json.loads(x) for x in (tmp_path / "res.jsonl").read_text().splitlines()]


def test_plan_is_seeded_interleaved_and_anchored():
    ids = [a["id"] for a in ARCHS]
    p1, p2 = make_plan(ids, ["A011", "A000"], 7, 5), make_plan(ids, ["A011", "A000"], 7, 5)
    assert p1 == p2 and p1 != make_plan(ids, ["A011", "A000"], 8, 5)
    targets = [s["id"] for s in p1 if s["role"] == "target"]
    assert sorted(targets) == sorted(ids) and targets != ids
    anchors = [s["id"] for s in p1 if s["role"] == "anchor"]
    # start, after targets 5 and 10, end: four visits of two anchors, alternating order
    assert anchors == ["A011", "A000", "A000", "A011", "A011", "A000", "A000", "A011"]
    assert [s["slot"] for s in p1] == list(range(len(p1)))


def test_default_anchors_are_largest_and_smallest_uniform():
    assert default_anchors(ARCHS) == ["A011", "A001"]


def test_full_run_records_every_slot(tmp_path):
    camp, measured, _ = make(tmp_path)
    assert camp.run() == 0
    r = rows(tmp_path)
    assert [x["slot"] for x in r] == list(range(len(camp.plan)))
    assert sorted(x["id"] for x in r if x["role"] == "target") == sorted(a["id"] for a in ARCHS)
    assert all(x["protocol"] == PROTOCOL and x["seed"] == 7 and x["idle_power_W"] == 100.0 for x in r)
    assert all("timestamp_et" in x and x["individual"]["layers"] for x in r)


def test_resume_after_interruption(tmp_path):
    camp, measured, _ = make(tmp_path, fail_after=6)
    with pytest.raises(KeyboardInterrupt):
        camp.run()
    assert len(rows(tmp_path)) == 6
    camp2, measured2, _ = make(tmp_path)
    camp2.run()
    r = rows(tmp_path)
    assert [x["slot"] for x in r] == list(range(len(camp2.plan)))
    assert len(measured2) == len(camp2.plan) - 6


def test_resume_refuses_a_different_plan(tmp_path):
    make(tmp_path)[0].run()
    with pytest.raises(SystemExit):
        Campaign(ARCHS, PROTOCOL, str(tmp_path / "res.jsonl"), seed=8, anchor_every=5,
                 measure_fn=None, probe=FakeProbe())


def test_busy_host_waits_then_measures(tmp_path):
    # first two pre-checks see 3 cores of other work, then idle
    probe = FakeProbe(cpu=[3.0, 3.0])
    camp, measured, _ = make(tmp_path, probe=probe)
    camp.run()
    assert len(rows(tmp_path)) == len(camp.plan)


def test_overlapped_measurement_is_discarded_and_repeated(tmp_path):
    # readings: idle-power pre-check, slot-0 pre-check, slot-0 during (busy), slot-0 pre-check, during
    probe = FakeProbe(cpu=[0.0, 0.0, 2.5])
    camp, measured, _ = make(tmp_path, probe=probe)
    camp.run()
    r = rows(tmp_path)
    assert r[0]["attempts"] == 2 and r[0]["idle_check"]["during_other_cpu_cores"] < 0.5
    assert len(measured) == len(camp.plan) + 1


def test_host_busy_too_long_stops_with_status(tmp_path):
    probe = FakeProbe(cpu=[5.0] * 100)
    camp, _, _ = make(tmp_path, probe=probe, max_wait_s=0.0)
    with pytest.raises(SystemExit) as e:
        camp.run()
    assert e.value.code == EXIT_HOST_BUSY
    assert json.loads((tmp_path / "res.status.json").read_text())["state"] == "stopped_host_busy"


def test_foreign_gpu_process_blocks_measurement(tmp_path):
    probe = FakeProbe(gpu=[[4242]] * 100)
    camp, measured, _ = make(tmp_path, probe=probe, max_wait_s=0.0)
    with pytest.raises(SystemExit):
        camp.run()
    assert measured == []


def test_driver_change_stops_campaign(tmp_path):
    # start reading, idle-power check, slot 0 fine, slot 1 sees an upgraded driver
    probe = FakeProbe(drivers=[{}, {}, {}, {"smi_driver": "581.0", "nvml_driver": "581.0"}])
    camp, measured, _ = make(tmp_path, probe=probe)
    with pytest.raises(SystemExit) as e:
        camp.run()
    assert e.value.code == EXIT_DRIVER_CHANGED
    assert len(rows(tmp_path)) == 2


def test_broken_nvidia_smi_stops_campaign(tmp_path):
    probe = FakeProbe(drivers=[{}, {"smi_ok": False, "smi_driver": None}])
    camp, measured, _ = make(tmp_path, probe=probe)
    with pytest.raises(SystemExit) as e:
        camp.run()
    assert e.value.code == EXIT_DRIVER_CHANGED and measured == []
