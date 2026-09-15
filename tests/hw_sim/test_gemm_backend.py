"""Timeloop GEMM backend without the mapper: registry, padding, fusion, layer decomposition."""
import glob
import os
import pathlib

import pytest

from llmforge.hw.timeloop import gemm
from llmforge.hw.timeloop.stats import parse_dram_dataspace_stats, parse_timeloop_stats

FIXTURE = str(pathlib.Path(__file__).parent / "fixtures" / "dxe_gemm_512i_128o_2l.stats.txt")


@pytest.mark.parametrize("name", sorted(gemm.ARCH_CONFIGS))
def test_registry_spec_files_exist(name):
    cfg = gemm.ARCH_CONFIGS[name]
    for p in (cfg.arch_path, cfg.constraints_path, cfg.variables_path, cfg.mapper_path):
        assert os.path.isfile(p), p
    assert glob.glob(cfg.components_path)
    assert os.path.isfile(gemm._PROBLEM_PATH)


def test_runs_dir_follows_work_root(timeloop_work):
    assert gemm.get_arch_config("eyeriss").runs_dir == str(timeloop_work / "eyeriss")


def test_pad_d_axis_only_on_dxe():
    dxe = gemm.get_arch_config("dxe_relaxed")
    assert gemm._pad_D_for_arch(320, dxe) == (384, (320, 384))
    assert gemm._pad_D_for_arch(64, dxe) == (64, None)
    assert gemm._pad_D_for_arch(256, dxe) == (256, None)
    assert gemm._pad_D_for_arch(320, gemm.get_arch_config("eyeriss")) == (320, None)


def test_prepare_gemm_spec_writes_problem(timeloop_work):
    cfg = gemm.get_arch_config("dxe_relaxed")
    out_dir, problem, pad = gemm._prepare_gemm_spec(576, 320, 1, None, cfg)
    assert out_dir == os.path.join(cfg.runs_dir, "gemm_576i_384o_1l")
    text = open(problem).read()
    assert "$IN_CHANNELS" not in text and "$OUT_CHANNELS" not in text and "$OUT_HEIGHT" not in text
    assert pad == (320, 384)
    assert os.path.isfile(os.path.join(out_dir, "padding.json"))


def _op(energy_uJ, out_updates=0, out_energy=0.0, in_reads=0, in_energy=0.0):
    dram = {"Outputs": {"scalar_updates": out_updates, "scalar_reads": 0, "energy_pJ": out_energy},
            "Inputs": {"scalar_reads": in_reads, "energy_pJ": in_energy}}
    return {"energy_uJ": energy_uJ}, dram


def test_fusion_savings_arithmetic():
    ops = [_op(10.0, out_updates=400, out_energy=1e6), _op(10.0, in_reads=400, in_energy=2e6)]
    saved_uJ, saved_cycles = gemm.compute_fusion_savings(ops, [(0, 1)], dram_read_bw=4, dram_write_bw=2)
    assert saved_uJ == pytest.approx(3.0)                 # producer Outputs + consumer Inputs
    assert saved_cycles == pytest.approx(400 / 2 + 400 / 4)


def test_fusion_savings_scale_and_cap():
    ops = [_op(1.0, out_updates=8, out_energy=5e6), _op(1.0, in_reads=8, in_energy=5e6)]
    saved_uJ, _ = gemm.compute_fusion_savings(ops, [(0, 1)], scale_factors={1: 3})
    # 5 uJ from the producer plus 5 uJ x 3 from the consumer exceed 90 percent of the unfused 1 + 1 x 3 = 4 uJ,
    # so the cap applies.
    assert saved_uJ == pytest.approx(0.9 * 4.0)


def test_fusion_savings_scale_each_endpoint_and_count_a_dataspace_once():
    # Op 0 runs once, ops 1 and 2 run three times each, and op 2 consumes from both op 0 and op 1.
    ops = [_op(100.0, out_updates=400, out_energy=1e6),
           _op(100.0, out_updates=40, out_energy=1e5, in_reads=400, in_energy=2e6),
           _op(100.0, in_reads=40, in_energy=3e5)]
    saved_uJ, saved_cycles = gemm.compute_fusion_savings(ops, [(0, 1), (1, 2), (0, 2)], scale_factors={1: 3, 2: 3},
                                                          dram_read_bw=4, dram_write_bw=4)
    # Outputs of op 0 once and of op 1 three times, Inputs of ops 1 and 2 three times, op 2 counted once.
    assert saved_uJ == pytest.approx(1.0 + 3 * 0.1 + 3 * 2.0 + 3 * 0.3)
    assert saved_cycles == pytest.approx(400 / 4 + 3 * 40 / 4 + 3 * 400 / 4 + 3 * 40 / 4)


def test_evaluate_layer_fusion_counts_kv_groups_once(monkeypatch):
    base = parse_timeloop_stats(FIXTURE)

    def fake(in_channel, out_channel, seq_length, work_dir=None, log_path=None, arch=gemm.DEFAULT_ARCH):
        summary = dict(base, energy_uJ=10.0, cycles=1000.0)
        dram = {"Outputs": {"scalar_updates": 400, "scalar_reads": 0, "energy_pJ": 1e6},
                "Inputs": {"scalar_reads": 400, "energy_pJ": 2e6}}
        return summary, dram

    monkeypatch.setattr(gemm, "run_GEMM_evaluation_detailed", fake)
    layer = {"n_head": 9, "n_kv_group": 3, "n_qk_head_dim": 48, "n_v_head_dim": 32,
             "mlp_size": 1152, "n_cproj": 1, "attention_variant": "infinite"}
    fused = gemm.evaluate_layer(layer, 576, 128, None, fused=True, arch="dxe_relaxed", mode="decode")
    # Outputs: ops 0, 1, 4 and 5 once and ops 2 and 3 three times, 1 uJ x 10. Inputs: ops 4, 5 and 6 once
    # and ops 2 and 3 three times, 2 uJ x 9. Each op's cycles likewise, 100 cycles per instance.
    assert fused["fusion_saved_energy_uJ"] == pytest.approx(10.0 + 18.0)
    assert fused["energy_uJ"] == pytest.approx(10.0 * (5 + 2 * 3) - 28.0)
    assert fused["cycles"] == pytest.approx(1000.0 * (5 + 2 * 3) - (100 * 10 + 100 * 9))


def test_evaluate_layer_shapes_and_kv_scaling(monkeypatch):
    summary, dram = parse_timeloop_stats(FIXTURE), parse_dram_dataspace_stats(FIXTURE)
    calls = []

    def fake(in_channel, out_channel, seq_length, work_dir=None, log_path=None, arch=gemm.DEFAULT_ARCH):
        calls.append((in_channel, out_channel, seq_length))
        return dict(summary), {k: dict(v) for k, v in dram.items()}

    monkeypatch.setattr(gemm, "run_GEMM_evaluation_detailed", fake)
    layer = {"n_head": 9, "n_kv_group": 3, "n_qk_head_dim": 48, "n_v_head_dim": 32,
             "mlp_size": 1152, "n_cproj": 1, "attention_variant": "infinite"}
    unfused = gemm.evaluate_layer(layer, 576, 128, None, fused=False, arch="dxe_relaxed", mode="decode")
    assert calls == [(576, 48 * 12, 1), (576, 32 * 3, 1), (48, 128, 3), (128, 32, 3),
                     (32 * 9, 576, 1), (576, 1152, 1), (1152, 576, 1)]
    # Five projection GEMMs once each, and the two attention GEMMs once per KV group.
    assert unfused["energy_uJ"] == pytest.approx(summary["energy_uJ"] * (5 + 2 * 3))
    assert unfused["cycles"] == pytest.approx(summary["cycles"] * (5 + 2 * 3))

    calls.clear()
    gemm.evaluate_layer(layer, 576, 128, None, fused=False, arch="dxe_relaxed", mode="prefill")
    assert calls[0] == (576, 48 * 12, 128) and calls[2] == (48, 128, 3)

    fused = gemm.evaluate_layer(layer, 576, 128, None, fused=True, arch="dxe_relaxed", mode="decode")
    assert 0 < fused["fusion_saved_energy_uJ"]
    assert fused["energy_uJ"] == pytest.approx(unfused["energy_uJ"] - fused["fusion_saved_energy_uJ"])

    calls.clear()
    mlp_only = dict(layer, attention_variant="identity")
    gemm.evaluate_layer(mlp_only, 576, 128, None, fused=False, arch="dxe_relaxed", mode="decode")
    assert calls == [(576, 1152, 1), (1152, 576, 1)]


def test_eval_individual_sums_active_layers(monkeypatch, small_individual):
    per_layer = {"energy_uJ": 2.0, "cycles": 10.0, "total_ops": 1.0, "total_memory_accesses": 1.0,
                 "utilization_pct": 50.0, "gflops": 1.0,
                 "fusion_saved_energy_uJ": 0.0, "fusion_saved_cycles": 0.0}
    monkeypatch.setattr(gemm, "evaluate_layer", lambda *a, **k: dict(per_layer))
    small_individual["globals"]["layer_mask"] = [True, False, True, True]
    out = gemm.eval_individual(small_individual, None, arch="eyeriss")
    assert out["energy_uJ"] == pytest.approx(6.0)
    assert out["energy_per_token_uJ"] == pytest.approx(6.0 / 128)


@pytest.mark.skipif(not gemm.timeloop_available(), reason="timeloopfe or timeloop-mapper not installed")
def test_mapper_smoke(timeloop_work):
    summary, dram = gemm.run_GEMM_evaluation_detailed(64, 128, 1, arch="eyeriss")
    assert summary["energy_uJ"] > 0 and summary["cycles"] > 0
    assert "Inputs" in dram
