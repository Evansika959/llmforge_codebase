"""Read-only checks for the relocated scripts and preserved local checkpoints."""
import hashlib
import json
from pathlib import Path
import unittest

import joblib
from ..models.serialization import load_bundle
import numpy as np
import torch

from ..config import ROOT

from ..config import FEATURES
from ..features.analytic import physical_features
from ..data.legacy import load_cohort
from ..inference.neural import predict_neural
from ..inference.predictor import predict_bundle, bundle_targets


class PredictionLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.outputs = ROOT/'scripts/prediction/outputs'
        cls.snapshot, _, cls.rows, cls.configs, cls.ix, _, cls.gross, _ = load_cohort(cls.outputs/'batch2_progress_632')
        cls.predictions = json.loads((cls.outputs/'gross_energy_comparison_1564/predictions.json').read_text())

    def test_repository_root_and_shared_inputs(self):
        self.assertEqual(ROOT, Path(__file__).resolve().parents[3])
        self.assertTrue((ROOT/'src/runq_llmforge.c').is_file())
        for name, path in self.snapshot['paths'].items():
            self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), self.snapshot['hashes'][name])

    def test_frozen_split(self):
        train = set(self.ix['old_train']) | set(self.ix['batch2_train'])
        val, test = set(self.ix['fixed_validation']), set(self.ix['pooled_test'])
        self.assertEqual((len(train),len(val),len(test)), (1100,150,314))
        self.assertFalse(train & val or train & test or val & test)
        self.assertEqual(len(train | val | test), len(self.rows))

    def check_predictions(self, name, predicted):
        expected = {r['config_id']:r for r in self.predictions
                    if (r['model'],r['energy_target'],r['stage'],r['test'],r['metric'],r['seed']) ==
                    (name,'gross','plus_all','pooled_test','energy',42)}
        self.assertEqual(len(expected),314)
        for i in self.ix['pooled_test']:
            record = expected[self.rows[i]['config_id']]
            np.testing.assert_allclose(predicted[i],record['prediction'],rtol=2e-6)
            np.testing.assert_allclose(self.gross[i],record['actual'],rtol=1e-7)

    def test_xgboost14_checkpoint(self):
        pack = load_bundle(self.outputs/'gross_energy_comparison_1564/xgboost14_plus_all_seed42.joblib')
        x = np.array([[physical_features(c)[0][f] for f in FEATURES] for c in self.configs],dtype=np.float32)
        self.check_predictions('xgboost14',np.exp(pack['models'][2].predict(x)))

    def test_xgboost32_checkpoint(self):
        pack = load_bundle(self.outputs/'gross_energy_comparison_1564/xgboost32_seed42.joblib')
        x = pack['profile'].transform(self.configs)
        self.check_predictions('xgboost32',np.exp(pack['models'][2].predict(x)))

    def test_transformer_checkpoint(self):
        pack = load_bundle(self.outputs/'gross_energy_comparison_1564/grouped_transformer_seed42.joblib')
        x = pack['profile'].transform(self.configs)
        self.check_predictions('transformer32',np.exp(predict_neural(pack,x))[:,2])

    def test_legacy_stack_predictions_unchanged(self):
        records = json.loads((self.outputs/'batch2_progress_632/predictions.json').read_text())
        expected = [r for r in records if (r['model'],r['stage'],r['test'],r['target'],r['seed']) ==
                    ('stack','plus_all','pooled_test','dynamic_energy_per_token_mj',42)]
        configs = {r['config_id']:c for r,c in zip(self.rows,self.configs)}
        pack = load_bundle(self.outputs/'batch2_progress_632/stacked_ensemble_seed42.joblib')
        pred = predict_bundle(pack,[configs[r['config_id']] for r in expected])
        np.testing.assert_allclose(pred[:,2],[r['prediction'] for r in expected],rtol=2e-6)

    def test_gross_inference_uses_saved_label(self):
        pack = load_bundle(self.outputs/'gross_energy_comparison_1564/grouped_transformer_seed42.joblib')
        self.assertEqual(bundle_targets(pack)[2],'gross_energy_per_token_mj')
        self.check_predictions('transformer32',predict_bundle(pack,self.configs)[:,2])


if __name__ == '__main__':
    unittest.main()
