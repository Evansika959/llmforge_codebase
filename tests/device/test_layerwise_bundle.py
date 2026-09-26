"""The shipped per-layer bundle is the Backend D predictor and reproduces its recorded test results."""
import json
import sys

import numpy as np
import pytest

pytest.importorskip("xgboost")

from llmforge.paths import CONFIGS, DEVICE_ASSETS

ROOT = DEVICE_ASSETS / "layerwise_tpot_2000_20260922"
if not (ROOT / "models" / "predictor_final.joblib").exists():
    pytest.skip("per-layer device bundle not present", allow_module_level=True)


@pytest.fixture(autouse=True, scope="module")
def unload_bundle_snapshot():
    """Remove the bundle's snapshot package afterwards, since the legacy parity test asserts none is loaded."""
    yield
    from llmforge.hw.device.prediction.inference import layerwise

    layerwise._snapshot.cache_clear()
    layerwise._dataset.cache_clear()
    for name in [m for m in sys.modules if m == "scripts" or m.startswith("scripts.")]:
        del sys.modules[name]
    snapshot = str(ROOT / "source_snapshot")
    while snapshot in sys.path:
        sys.path.remove(snapshot)


def test_search_defaults_to_the_per_layer_bundle():
    from llmforge.evaluators.hw_device import HwDevice
    from llmforge.search.cosearch import build_parser

    default = next(a.default for a in build_parser()._actions if a.dest == "device_bundle")
    dev = HwDevice()
    assert default == dev.bundle
    assert dev.layerwise
    assert list(dev.pack["targets"]) == ["tpot_ms", "ttft_ms", "dynamic_energy_per_token_mj"]


def test_selected_checkpoint_reproduces_the_recorded_test_predictions():
    from llmforge.hw.device.prediction.inference import layerwise
    from llmforge.hw.device.prediction.models.serialization import load_bundle

    path = ROOT / "models" / "predictor_final.joblib"
    rows = {r["candidate_id"]: r for r in json.loads((ROOT / "dataset_snapshot.json").read_text())["rows"]}
    recorded = np.load(ROOT / "test_predictions.npz")
    pred = layerwise.predict_architectures(load_bundle(path), [rows[i]["architecture"] for i in recorded["ids"]],
                                           path)
    # The recorded predictions were stored as float32, so they agree to float32 precision.
    np.testing.assert_allclose(pred, recorded["predictions"][list(recorded["seeds"]).index(2026)], rtol=1e-6)
    mape = 100 * np.mean(np.abs(pred - recorded["actual"]) / recorded["actual"], axis=0)
    np.testing.assert_allclose(mape, [15.46, 16.49, 13.97], atol=0.005)


def test_heterogeneous_individuals_are_evaluated_with_their_layer_shapes():
    from llmforge.evaluators.hw_device import HwDevice
    from llmforge.search.elastic_space import ElasticSearchSpace

    space = ElasticSearchSpace.from_yaml(str(CONFIGS / "search_spaces" / "smollm2-135m_nkv.yaml"), seed=0)
    inds = [space.sample() for _ in range(8)]
    assert any(len({(l["n_head"], l["mlp_size"]) for l in i["layers"]}) > 1 for i in inds)
    for rec in HwDevice().evaluate(inds):
        assert rec["hw_feasible"]
        assert rec["tpot_ms"] > 0 and rec["ttft_ms"] > 0 and rec["energy_per_token_uJ"] > 0
        assert rec["device_in_domain"] in (True, False)


def test_release_copy_withholds_the_device_serial():
    from llmforge.hw.device.prediction.models.serialization import load_bundle

    for name in ("predictor_final", "XGBoost_seed42", "XGBoost_seed123", "XGBoost_seed2026"):
        protocol = load_bundle(ROOT / "models" / f"{name}.joblib")["protocol"]["common_measurement_protocol"]
        assert protocol["hardware_serial"] == "withheld"
    manifest = json.loads((ROOT / "manifest.json").read_text())
    assert manifest["protocol"]["common_measurement_protocol"]["hardware_serial"] == "withheld"
