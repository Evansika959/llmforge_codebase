"""rDXE cost model: on-chip accounting of Timeloop mappings, chip area and leakage, ring timing."""
import shutil
from pathlib import Path

import pytest

from llmforge.hw.rdxe import cosearch
from llmforge.hw.rdxe import workflow as wf
from llmforge.hw.rdxe.core import constants as C
from llmforge.hw.rdxe.core import timeloop_evaluator as te
from llmforge.hw.rdxe.core.scaled_arch import ScaledChipSpec
from llmforge.hw.rdxe.simulator import layer_eval_timeloop as let
from llmforge.hw.timeloop.stats import parse_buffer_stats, parse_timeloop_stats

FIXTURE = Path(__file__).parent / "fixtures" / "dxe_gemm_512i_128o_2l.stats.txt"
LAYER = {"n_head": 9, "n_kv_group": 3, "n_qk_head_dim": 64, "n_v_head_dim": 64, "mlp_size": 1536,
         "n_cproj": 1, "attention_variant": "infinite"}


@pytest.fixture
def no_mapper(monkeypatch, timeloop_work):
    """Force every GEMM onto the analytical fallback, whether or not Timeloop is installed."""
    def _raise(*args, **kwargs):
        raise RuntimeError("mapper disabled for this test")
    monkeypatch.setattr(te, "_fast_run_mapper", _raise)


def test_component_constants_match_the_dxe_mapper_statistics():
    levels = parse_buffer_stats(str(FIXTURE))
    cycles = parse_timeloop_stats(str(FIXTURE))["cycles"]

    def spec(name, key):
        return levels[name]["_specs"][key]

    def leak_per_instance_cycle(name):
        return spec(name, "leakage_energy_pJ") / cycles / spec(name, "instances")

    mac_area = levels["mac"]["_stats"]["area_total"] / spec("mac", "instances")
    assert mac_area == pytest.approx(C.AREA_MAC_UM2, rel=1e-4)
    assert spec("acc_buffer", "area") == pytest.approx(C.AREA_ACC_UM2)
    assert spec("wmem", "area") / spec("wmem", "size") == pytest.approx(C.AREA_SRAM_UM2_PER_B, rel=1e-4)
    assert spec("head_sram", "area") == pytest.approx(C.AREA_HEAD_SRAM_UM2)
    assert spec("global_sram", "area") == pytest.approx(C.AREA_GLOBAL_SRAM_UM2)
    wmem_leak = leak_per_instance_cycle("wmem") / spec("wmem", "size")
    assert wmem_leak == pytest.approx(C.LEAK_SRAM_PJ_PER_B_CYCLE, rel=2e-3)
    assert leak_per_instance_cycle("acc_buffer") == pytest.approx(C.LEAK_ACC_PJ_PER_CYCLE, rel=2e-3)
    assert leak_per_instance_cycle("head_sram") == pytest.approx(C.LEAK_HEAD_SRAM_PJ_PER_CYCLE, rel=2e-3)
    assert leak_per_instance_cycle("global_sram") == pytest.approx(C.LEAK_GLOBAL_SRAM_PJ_PER_CYCLE, rel=2e-3)


def test_area_and_leakage_follow_cores_macs_and_memory():
    ref = ScaledChipSpec(name="ref")
    assert ref.estimated_area_mm2 == pytest.approx(3.9786, abs=1e-3)
    wide = ScaledChipSpec(name="wide", n_mac_per_vac=32)
    assert wide.estimated_area_mm2 > ref.estimated_area_mm2
    assert wide.leakage_pJ_per_cycle == pytest.approx(ref.leakage_pJ_per_cycle)
    for bigger in (ScaledChipSpec(name="cores", n_dxt=16), ScaledChipSpec(name="wmem", wmem_per_core_B=48 * 1024)):
        assert bigger.estimated_area_mm2 > ref.estimated_area_mm2
        assert bigger.leakage_pJ_per_cycle > ref.leakage_pJ_per_cycle


def _level(cycles, dynamic_pJ, **dataspaces):
    return {"cycles": cycles, "dynamic_pJ": dynamic_pJ, "leakage_pJ": 1e4, "dataspaces": dataspaces}


def _mapping(K=576, N=128, M=1):
    """A mapping shaped like a DXE decode GEMM: DRAM dominates the raw energy and the cycles."""
    weights = {"total_reads": 73728.0, "total_updates": 0.0, "energy_pJ": 37000.0}
    levels = {
        "mac": _level(36.0, 5000.0),
        "acc_buffer": _level(36.0, 50.0),
        "wmem": _level(36.0, 37000.0, Weights=weights),
        "head_sram": _level(37.0, 650.0),
        "global_sram": _level(44.0, 180.0),
        "DRAM": _level(18576.0, 4.76e6),
    }
    return te.GemmResult(M=M, K=K, N=N, energy_pJ=4.85e6, cycles=18576.0, dram_stats={}, levels=levels)


def test_mapped_cost_keeps_on_chip_levels_and_tiles_cycles():
    res = _mapping()
    ref = ScaledChipSpec(name="ref")
    cost = let._mapped_cost(res, ref, "V_gen", share=3, weight_memory="wmem")
    assert cost["gemm_pJ"] == pytest.approx(5000 + 50 + 37000 + 650 + 180)
    assert cost["kv_read_pJ"] == 0.0
    assert cost["cycles"] == 44.0  # the shared global SRAM is the slowest on-chip level
    # Eight cores hold 16 output columns each, against one per core in the 128-core mapping.
    few_cores = ScaledChipSpec(name="few", n_dxt=1, n_vac_per_dxt=8)
    assert let._mapped_cost(res, few_cores, "V_gen", 3, "wmem")["cycles"] == 36.0 * 16
    deep = ScaledChipSpec(name="deep", wmem_per_core_B=48 * 1024)
    assert let._mapped_cost(res, deep, "V_gen", 3, "wmem")["gemm_pJ"] == pytest.approx(
        5000 + 50 + 650 + 180 + 37000 * deep.wmem_energy_scale)
    assert let._mapped_cost(res, ref, "V_gen", 3, "dram") == {"gemm_pJ": 4.85e6, "kv_read_pJ": 0.0,
                                                             "cycles": 18576.0}


def test_mapped_cost_reads_the_kv_cache_once_per_group():
    cost = let._mapped_cost(_mapping(K=2048, N=128, M=3), ScaledChipSpec(name="ref"), "PV_attn", share=3,
                            weight_memory="wmem")
    assert cost["gemm_pJ"] == pytest.approx(5000 + 50 + 650 + 180)
    assert cost["kv_read_pJ"] == pytest.approx(
        73728.0 / 3 * C.KV_CACHE_ENERGY_PER_ACCESS_PJ * let.IWUR_V_PENALTY)


def test_layer_decode_prices_the_value_head_dimension(no_mapper):
    ev = te.TimeloopEvaluator(arch="dxe_relaxed", n_mac_per_vac=16)
    chip = ScaledChipSpec(name="ref")
    narrow = let.timeloop_layer_decode({**LAYER, "n_v_head_dim": 16}, 576, 256, chip, ev)
    wide = let.timeloop_layer_decode(LAYER, 576, 256, chip, ev)
    assert wide["total_energy_pJ"] > narrow["total_energy_pJ"]
    assert set(wide["per_op_sources"].values()) == {"analytical_fallback_mapper_failed"}


def test_run_rdxe_eval_costs_every_active_layer(no_mapper, small_individual):
    narrow_tail = {"globals": dict(small_individual["globals"]),
                   "layers": [dict(small_individual["layers"][0])]
                   + [{**layer, "n_head": 3, "n_qk_head_dim": 16, "n_v_head_dim": 16, "mlp_size": 384}
                      for layer in small_individual["layers"][1:]]}
    full, narrow = cosearch.run_rdxe_eval([small_individual, narrow_tail], prefill_len=64, decode_len=16,
                                          ctx=256, n_workers=1, envelope_filter=False)
    assert narrow["energy_per_token_uJ"] < full["energy_per_token_uJ"]
    assert full["ops_fallback"] == narrow["ops_fallback"] == 7 * 4


def test_simulate_ring_pipelines_the_prompt_over_unequal_stages(monkeypatch):
    def fake_decode(layer, n_embd, ctx, chip, evaluator, n_users=1, weight_memory="wmem"):
        return dict(total_energy_pJ=1000.0, cycles=100.0, gemm_energy_pJ=1000.0, kv_read_energy_pJ=0.0,
                    kv_write_energy_pJ=0.0, vrc_energy_pJ=0.0, per_op_sources={"MLP_FC1": "timeloop"})
    monkeypatch.setattr(wf, "timeloop_layer_decode", fake_decode)
    chip = wf._build_chip(128, 24 * 1024, 8 * 1024)
    packing = wf.Packing(chip=chip, groups=[[0, 1, 2], [3]], group_wmem_B=[0, 0], group_kv_B=[0, 0],
                         util_pct=[50.0, 50.0])
    info = dict(n_layer=4, n_embd=576, layers=[dict(LAYER) for _ in range(4)])
    r = wf.simulate_ring(info, packing, 256, evaluator=None, prefill_length=8, decode_length=16)
    hop_cy = C.INTER_CHIP_LATENCY_CYCLES + 576 / C.INTER_CHIP_BW_BYTES_PER_CYCLE
    step_cy = 400 + hop_cy
    assert r["tpot_ms"] == pytest.approx(step_cy * C.CLOCK_PERIOD_NS / 1e6)
    # The first prompt token passes both chips, and seven more follow the three-layer bottleneck stage.
    assert r["ttft_ms"] == pytest.approx((step_cy + 7 * (300 + hop_cy)) * C.CLOCK_PERIOD_NS / 1e6)
    hop_pJ = 576 * 8 * C.INTER_CHIP_ENERGY_PJ_PER_BIT
    leak_pJ = chip.leakage_pJ_per_cycle * 2 * step_cy
    assert r["per_tok_uJ"] == pytest.approx((4 * 1000.0 + hop_pJ + leak_pJ) / 1e6)
    assert r["ops_timeloop"] == 4 and r["ops_fallback"] == 0


def test_fast_run_mapper_serves_a_final_loose_mapping_from_cache(timeloop_work, monkeypatch):
    from llmforge.hw.timeloop import gemm
    out_dir, _, _ = gemm._prepare_gemm_spec(512, 128, 2, None, gemm.get_arch_config("dxe_relaxed"))
    stats = Path(out_dir) / "timeloop-mapper.stats.txt"
    shutil.copy(FIXTURE, stats)
    calls = []

    def fake_mapper(settings, in_ch, out_ch, seq_len, arch):
        calls.append(settings["victory_condition"])
        return te._read_stats(str(stats))

    monkeypatch.setattr(te, "_run_mapper_with", fake_mapper)
    monkeypatch.setattr(te, "_analytical_baseline_pJ", lambda *args: 1e12)
    assert te._fast_run_mapper(512, 128, 2, "dxe_relaxed")[2]["mac"]["cycles"] == 64.0
    assert calls == []  # a cached schedule within the sanity bound needs no mapper run
    monkeypatch.setattr(te, "_analytical_baseline_pJ", lambda *args: 1.0)
    te._fast_run_mapper(512, 128, 2, "dxe_relaxed")
    assert calls == [te.FAST_MAPPER["victory_condition"], te.LOOSE_MAPPER["victory_condition"]]
    te._fast_run_mapper(512, 128, 2, "dxe_relaxed")
    assert len(calls) == 2  # the loose schedule is final and comes from the cache
