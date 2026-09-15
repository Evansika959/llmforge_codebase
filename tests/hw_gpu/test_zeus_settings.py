"""Cache settings of the GPU evaluator. Needs torch but no GPU, since the device queries are patched."""
import os

import pytest

torch = pytest.importorskip("torch")

PREFILL = dict(prefill_len=512, decode_len=0, batch_size=64, n_repeats=3, min_window_s=2.0)
DECODE = dict(prefill_len=512, decode_len=128, batch_size=64, n_repeats=3, min_window_s=2.0, cuda_graphs=True)


@pytest.fixture
def fake_h100(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index=0: "NVIDIA H100 80GB HBM3")
    monkeypatch.delenv("LLMFORGE_DEVICE_LABEL", raising=False)


def cache_name(tmp_path, **kw):
    from llmforge.evaluators.hw_zeus import HwZeus

    cache = HwZeus(cache_dir=str(tmp_path), **kw).cache
    return os.path.basename(str(getattr(cache, "path", None) or getattr(cache, "_path")))


def test_default_schedule_keeps_v4_cache_names(fake_h100, tmp_path):
    # Measurements cached before the schedule options existed must stay reachable.
    assert cache_name(tmp_path, **PREFILL) == "zeus__nvidia-h100-80gb-hbm3__p512_d0_b64_bf16_r3_w2__eb0ddd3cf5.jsonl"
    assert cache_name(tmp_path, **DECODE) == "zeus__nvidia-h100-80gb-hbm3__p512_d128_b64_bf16_r3_cg_w2__49ce4cf04a.jsonl"


def test_schedule_changes_decode_setting_only(fake_h100, tmp_path):
    grouped = dict(schedule="grouped", settle_s=1.5)
    assert cache_name(tmp_path, **PREFILL, **grouped) == cache_name(tmp_path, **PREFILL)
    name = cache_name(tmp_path, **DECODE, **grouped)
    assert name != cache_name(tmp_path, **DECODE)
    assert "_grouped-s1.5__" in name
    assert cache_name(tmp_path, **DECODE, **grouped, decode_window_s=6.0) != name


def test_device_label_separates_caches(fake_h100, tmp_path, monkeypatch):
    base = cache_name(tmp_path, **PREFILL)
    labeled = cache_name(tmp_path, **PREFILL, device_label="gpu-b")
    assert labeled != base and "_dev-gpu-b__" in labeled
    monkeypatch.setenv("LLMFORGE_DEVICE_LABEL", "gpu-b")
    assert cache_name(tmp_path, **PREFILL) == labeled
    with pytest.raises(ValueError):
        cache_name(tmp_path, **PREFILL, device_label="bad/label")
