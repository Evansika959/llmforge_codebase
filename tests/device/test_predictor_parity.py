"""Regression check of the shipped Pixel Watch 5 predictor through the ported package.

The expected values were produced by the original prediction package of the measurement repository,
loading the original bundle before the release rewrite. The rewrite must not change them.
"""
import sys
import unittest

import numpy as np

from llmforge import paths
from llmforge.hw.device.prediction.features.physics import architecture_stats
from llmforge.hw.device.prediction.inference.predictor import bundle_targets, predict_bundle
from llmforge.hw.device.prediction.models.serialization import load_bundle

MODELS = paths.DEVICE_ASSETS / 'pixel_watch5' / 'models'
TARGETS = ['decode_tok_s', 'ttft_ms', 'dynamic_energy_per_token_mj']
SMOLLM2_135M = dict(n_layer=30, d_model=576, n_h=9, n_kv=3, d_qk=64, d_v=64, d_mlp=1536, vocab_size=49152)
CONFIGS = {
    'smollm2_135m_shape': SMOLLM2_135M,
    'small_mha': dict(n_layer=6, d_model=384, n_h=6, n_kv=2, d_qk=32, d_v=32, d_mlp=1024),
    'gqa_mid': dict(n_layer=12, d_model=512, n_h=8, n_kv=4, d_qk=64, d_v=48, d_mlp=1536),
    'group16': dict(n_layer=20, d_model=528, n_h=8, n_kv=2, d_qk=48, d_v=48, d_mlp=1408),
    'deep_28': dict(n_layer=28, d_model=384, n_h=6, n_kv=3, d_qk=64, d_v=64, d_mlp=1024),
}
EXPECTED_BEST = {
    'smollm2_135m_shape': [5.051512241363525, 1639.2862548828125, 86.97071075439453],
    'small_mha': [12.637975692749023, 410.1184997558594, 20.28606414794922],
    'gqa_mid': [10.701950073242188, 573.5003051757812, 33.65867614746094],
    'group16': [18.827880859375, 1341.8201904296875, 22.14271354675293],
    'deep_28': [9.557928085327148, 743.7391967773438, 42.14485168457031],
}


class PredictorParityTests(unittest.TestCase):
    def test_best_bundle_matches_original_package(self):
        pack = load_bundle(MODELS / 'predictor_best.joblib')
        self.assertEqual(bundle_targets(pack), TARGETS)
        predicted = predict_bundle(pack, list(CONFIGS.values()))
        expected = np.array([EXPECTED_BEST[name] for name in CONFIGS])
        np.testing.assert_allclose(predicted, expected, rtol=1e-6)

    def test_bundles_load_without_legacy_module_names(self):
        for name in ['predictor_best.joblib', 'predictor_latest.joblib']:
            pack = load_bundle(MODELS / name)
            self.assertEqual(type(pack['profile']).__module__, 'llmforge.hw.device.prediction.features.physics')
            predicted = predict_bundle(pack, [SMOLLM2_135M])
            self.assertTrue(np.isfinite(predicted).all() and (predicted > 0).all())
        leftovers = [m for m in sys.modules if m == 'scripts' or m.startswith('scripts.')]
        self.assertEqual(leftovers, [])

    def test_architecture_stats_name_and_parameter_count(self):
        self.assertEqual(architecture_stats(SMOLLM2_135M)['params'], 134_549_568)


if __name__ == '__main__':
    unittest.main()
