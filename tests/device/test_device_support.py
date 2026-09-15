"""device_on_support separates observed architecture values from values that only lie inside the ranges."""
import json

import pytest

pytest.importorskip("xgboost")

from llmforge.paths import CONFIGS, DEVICE_ASSETS

if not (DEVICE_ASSETS / "pixel_watch5" / "models" / "predictor_best.joblib").exists():
    pytest.skip("device assets not present", allow_module_level=True)


def in_range_uniform(n_head: int) -> dict:
    """A 24-layer SmolLM2-width architecture with the training vocabulary, inside every training range."""
    from llmforge.search.elastic_space import ElasticSearchSpace

    space = ElasticSearchSpace.from_yaml(str(CONFIGS / "search_spaces" / "smollm2-135m_uniform.yaml"))
    ind = json.loads(json.dumps(space.uniform(n_head=n_head, n_qk_head_dim=64, n_v_head_dim=48, mlp_size=1536)))
    ind["layers"] = ind["layers"][:24]
    ind["globals"]["vocab_size"] = 50257
    if ind["globals"].get("layer_mask"):
        ind["globals"]["layer_mask"] = ind["globals"]["layer_mask"][:24]
    return ind


def test_head_count_missing_from_observations_is_off_support():
    from llmforge.evaluators.hw_device import HwDevice

    dev = HwDevice(bundle="best")
    observed, unobserved = dev.evaluate([in_range_uniform(6), in_range_uniform(3)])
    assert observed["device_in_domain"] and observed["device_on_support"]
    assert unobserved["device_in_domain"] and not unobserved["device_on_support"]
