"""Two-pass aggregation and failure handling of the Timeloop evaluator, with a fake mapper."""
import math

import pytest

from llmforge.evaluators.hw_timeloop import SUBSTRATE_MAP, HwTimeloop
from llmforge.hw.timeloop import gemm


def _fake_stats(mode):
    base = {"total_ops": 10.0, "total_memory_accesses": 5.0,
            "fusion_saved_energy_uJ": 0.5, "fusion_saved_cycles": 2.0}
    if mode == "prefill":
        return {**base, "energy_uJ": 100.0, "cycles": 4_000_000.0}
    return {**base, "energy_uJ": 3.0, "cycles": 250_000.0}


@pytest.fixture
def fake_gemm(monkeypatch):
    calls = []

    def fake_eval_individual(individual, work_dir, fused=True, arch=gemm.DEFAULT_ARCH, mode="prefill"):
        calls.append((individual["globals"]["block_size"], mode, arch, work_dir))
        return _fake_stats(mode)

    monkeypatch.setattr(gemm, "eval_individual", fake_eval_individual)
    return calls


def test_two_pass_combination(fake_gemm, small_individual, timeloop_work):
    ev = HwTimeloop("eyeriss", prefill_len=128, decode_len=32, require_timeloop=False)
    before = repr(small_individual)
    rec = ev.evaluate([small_individual])[0]
    assert repr(small_individual) == before              # the caller's individual is untouched
    assert [c[:3] for c in fake_gemm] == [(128, "prefill", "eyeriss"), (32, "decode", "eyeriss")]
    assert fake_gemm[0][3] == str(timeloop_work / "eyeriss" / "prefill")
    assert rec["energy_uJ"] == pytest.approx(100.0 + 3.0 * 32)
    assert rec["energy_per_token_uJ"] == pytest.approx(3.0)
    assert rec["session_e_per_tok_uJ"] == pytest.approx((100.0 + 96.0) / 160)
    assert rec["ttft_ms"] == pytest.approx(4.0)          # 4e6 cycles at 1 GHz
    assert rec["tpot_ms"] == pytest.approx(0.25)
    assert rec["ttft"] == pytest.approx(4e-3) and rec["tpot"] == pytest.approx(2.5e-4)
    assert rec["fusion_saved_cycles"] == pytest.approx(2.0 + 2.0 * 32)
    assert rec["hw_feasible"] is True


def test_single_pass_uses_own_block_size(fake_gemm, small_individual):
    rec = HwTimeloop("simba", prefill_len=0, decode_len=32, require_timeloop=False).evaluate([small_individual])[0]
    assert [c[:2] for c in fake_gemm] == [(128, "prefill")]
    assert rec["ttft_ms"] == pytest.approx(4.0) and rec["hw_feasible"] is True


def test_mapper_failure_marks_individual_infeasible(monkeypatch, small_individual):
    def boom(*args, **kwargs):
        raise RuntimeError("no valid mapping")
    monkeypatch.setattr(gemm, "eval_individual", boom)
    rec = HwTimeloop("dxe_relaxed", require_timeloop=False).evaluate([small_individual])[0]
    assert rec["hw_feasible"] is False
    assert math.isinf(rec["energy_per_token_uJ"]) and "no valid mapping" in rec["timeloop_error"]


def test_substrates_match_registry():
    assert set(SUBSTRATE_MAP.values()) <= set(gemm.ARCH_CONFIGS)
    with pytest.raises(ValueError):
        HwTimeloop("not_a_substrate", require_timeloop=False)


@pytest.mark.skipif(gemm.timeloop_available(), reason="Timeloop is installed")
def test_requires_timeloop_by_default():
    with pytest.raises(RuntimeError):
        HwTimeloop("eyeriss")
