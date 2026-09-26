import copy
import json
from pathlib import Path
import unittest
import numpy as np
from .features import features, matrix
from scripts.sweep.layerwise.candidates import generate


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).resolve().parents[2]/'sweep/configs/watch5_layerwise_500_variable_kv_space.json'
        cls.arch=generate(json.loads(path.read_text()))[0]['architecture']

    def test_architecture_only(self):
        a=copy.deepcopy(self.arch)
        a.update(initial_temperature=99, battery_percent=1, measured_latency=10000)
        self.assertEqual(features(a),features(self.arch))

    def test_kv_accounting(self):
        f=features(self.arch)
        expected=sum(4*r['n_kv']*(r['d_qk']+r['d_v']) for r in self.arch['layers'])
        self.assertEqual(f['kv_bytes_per_token_sum'],expected)
        self.assertEqual(f['kv_cache_bytes'],expected*self.arch['operator_profile']['seq_len'])

    def test_order_sensitive(self):
        a=copy.deepcopy(self.arch)
        a['layers']=sorted(a['layers'],key=lambda r:r['d_mlp'])
        b=copy.deepcopy(a); b['layers'].reverse()
        fa,fb=features(a),features(b)
        self.assertEqual(fa['total_params'],fb['total_params'])
        self.assertNotEqual(fa['d_mlp_quarter0'],fb['d_mlp_quarter0'])
        x,names=matrix([a,b])
        self.assertEqual(x.shape,(2,len(names)))
        self.assertTrue(np.isfinite(x).all())


if __name__=='__main__': unittest.main()
