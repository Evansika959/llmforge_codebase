"""rDXE co-search end to end on the analytical GEMM fallback, which needs neither Timeloop nor a GPU."""
import math

import pytest

from llmforge.hw.rdxe import cosearch
from llmforge.hw.rdxe import workflow as wf
from llmforge.hw.rdxe.core import timeloop_evaluator as te


@pytest.fixture
def no_mapper(monkeypatch, timeloop_work):
    """Force every GEMM onto the analytical fallback, whether or not Timeloop is installed."""
    def _raise(*args, **kwargs):
        raise RuntimeError("mapper disabled for this test")
    monkeypatch.setattr(te, "_fast_run_mapper", _raise)


def test_pad_for_dxe():
    assert te.pad_for_dxe(576, 320, 1) == (576, 384, 1)
    assert te.pad_for_dxe(100, 64, 3) == (112, 128, 3)


def test_decode_shapes_quantize_context():
    layer = {"n_head": 9, "n_kv_group": 3, "n_qk_head_dim": 64, "n_v_head_dim": 64, "mlp_size": 1536}
    shapes = {name: (k, n, m) for name, k, n, m in te.enumerate_gemm_shapes_decode(layer, 576, 100)}
    assert shapes["QK_attn"] == (64, 128, 3)
    assert shapes["PV_attn"] == (128, 64, 3)
    assert dict((s[0], s[2]) for s in te.enumerate_gemm_shapes_decode(layer, 576, 300))["QK_attn"] == 384


def test_shipped_mac_width_variants_are_registered():
    assert te._arch_variant_for_mac_width(16) == "dxe_relaxed"
    assert te._arch_variant_for_mac_width(32) == "dxe_relaxed_m32"
    assert te._arch_variant_for_mac_width(64) == "dxe_relaxed_m64"
    assert te._arch_variant_for_mac_width(32, base_arch="eyeriss") == "eyeriss"


def test_new_mac_width_variant_is_cloned_into_work_dir(timeloop_work):
    from llmforge.hw.timeloop import gemm
    name = te._arch_variant_for_mac_width(128)
    cfg = gemm.get_arch_config(name)
    assert cfg.arch_path.startswith(str(timeloop_work))
    assert "spatial: {meshX: 128}" in open(cfg.arch_path).read()
    assert cfg.runs_dir == str(timeloop_work / name)
    del gemm.ARCH_CONFIGS[name]


def test_simulate_ring_uses_analytical_fallback(no_mapper):
    cfg = dict(n_layer=4, n_embd=576, n_head=9, n_kv_group=3, hd=64, mlp=1536)
    profile, info = wf.profile_model(cfg, ctx=256)
    packing = wf.pack_balanced(profile, max_chips=4)
    assert packing is not None
    ev = te.TimeloopEvaluator(arch="dxe_relaxed", n_mac_per_vac=16)
    r = wf.simulate_ring(info, packing, 256, ev, prefill_length=128, decode_length=32)
    # Eight GEMMs in each of four layers: seven for attention and the MLP, and the SwiGLU gate.
    assert r["ops_timeloop"] == 0 and r["ops_fallback"] == 8 * 4
    for key in ("per_tok_uJ", "tpot_ms", "ttft_ms", "total_area_mm2"):
        assert math.isfinite(r[key]) and r[key] > 0


def test_decode_shapes_and_weights_add_the_swiglu_gate():
    layer = {"n_head": 9, "n_kv_group": 3, "n_qk_head_dim": 64, "n_v_head_dim": 64, "mlp_size": 1536}
    plain = {**layer, "mlp_variant": "mlp"}
    gated = te.enumerate_gemm_shapes_decode(layer, 576, 256)
    assert len(gated) == 8 and gated[-1] == ("MLP_gate", 576, 1536, 1)
    assert "MLP_gate" not in [s[0] for s in te.enumerate_gemm_shapes_decode(plain, 576, 256)]
    assert wf.layer_weight_bytes(layer, 576) - wf.layer_weight_bytes(plain, 576) == 576 * 1536


def test_run_rdxe_eval_selects_from_chip_grid(no_mapper, small_individual):
    masked = {"globals": {**small_individual["globals"], "layer_mask": [False] * 4},
              "layers": small_individual["layers"]}
    good, bad = cosearch.run_rdxe_eval([small_individual, masked], prefill_len=64, decode_len=16,
                                       ctx=256, n_workers=1, envelope_filter=False)
    assert math.isinf(bad["energy_per_token_uJ"]) and bad["pareto_points"] == []
    assert math.isfinite(good["energy_per_token_uJ"]) and good["pareto_points"]
    assert good["selected_mac_per_vac"] in cosearch.RDXE_SWEEP_MAC_PER_VAC
    assert good["selected_wmem_KB"] in cosearch.RDXE_SWEEP_WMEM_KB
    assert good["ops_timeloop"] == 0
    # Without the envelope filter the selection is the energy argmin, which lies on the front.
    assert good["energy_per_token_uJ"] == pytest.approx(min(p["per_tok_uJ"] for p in good["pareto_points"]))


def test_envelope_filter_falls_back_to_least_violating(no_mapper, small_individual):
    rec = cosearch.run_rdxe_eval([small_individual], prefill_len=64, decode_len=16, ctx=256,
                                 n_workers=1, area_min_mm2=1e6, area_max_mm2=2e6)[0]
    assert rec["envelope_feasible"] is False
    assert math.isfinite(rec["energy_per_token_uJ"])


def test_hw_rdxe_evaluator_standard_keys(no_mapper, small_individual):
    from llmforge.evaluators.hw_rdxe import HwRdxeInner
    rec = HwRdxeInner(prefill_len=64, decode_len=16, ctx=256, n_workers=1).evaluate([small_individual])[0]
    assert rec["hw_feasible"] is True
    for key in ("energy_per_token_uJ", "ttft_ms", "tpot_ms"):
        assert math.isfinite(rec[key]) and rec[key] > 0
    assert rec["ttft_ms"] == pytest.approx(rec["ttft"] * 1e3)
    assert "chip_pareto" in rec and "pareto_points" not in rec
    assert isinstance(rec["envelope_feasible"], bool)
