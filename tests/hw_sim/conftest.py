"""Shared fixtures for the simulator-target tests. Nothing here needs Timeloop or a GPU."""
import pytest

from llmforge import paths


@pytest.fixture
def timeloop_work(tmp_path, monkeypatch):
    """Point every Timeloop cache at a temporary directory."""
    work = tmp_path / "timeloop"
    monkeypatch.setattr(paths, "TIMELOOP_WORK", work)
    return work


@pytest.fixture
def small_individual():
    """A 4-layer IHA individual with SmolLM2-135M layer shapes."""
    layer = {"n_head": 9, "n_kv_group": 3, "n_qk_head_dim": 64, "n_v_head_dim": 64,
             "mlp_size": 1536, "n_cproj": 1, "attention_variant": "infinite"}
    return {"globals": {"n_embd": 576, "block_size": 128, "layer_mask": [True] * 4},
            "layers": [dict(layer) for _ in range(4)]}
