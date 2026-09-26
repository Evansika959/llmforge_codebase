import json
from pathlib import Path
import tempfile
import unittest

from scripts.sweep.layerwise.measurement import integrate, parse


class MeasurementTests(unittest.TestCase):
    def test_exact_boundary_interpolation(self):
        self.assertAlmostEqual(integrate([(0,0),(.25,1),(.5,2)],.1,.4),.3)

    def test_gap_and_missing_coverage_rejected(self):
        for points,start,end in [([(0,1),(1,1)],0,1), ([(0,1),(.1,1)],0,.2)]:
            with self.assertRaises(ValueError): integrate(points,start,end)

    def fixture(self,root,power):
        timing=dict(idle_start=0,start=4,prefill_end=5,end=8,post_end=12,prefill_tokens=49,output_tokens=32,decode_tokens=31,admission_temperature_c=30)
        (root/'timing.json').write_text(json.dumps(timing))
        with (root/'trace.csv').open('w') as f:
            for n in range(121):
                t=n/10; watts=power(t)
                f.write(f'{t},{-watts*1e6/4},4000000\n')

    def test_known_power_and_independent_latency(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); self.fixture(root,lambda t:2 if 4<=t<=8 else 1)
            result=parse(root)
            self.assertEqual(result['baseline_power_w'],1)
            self.assertAlmostEqual(result['dynamic_energy_per_token_mj'],125)
            self.assertAlmostEqual(result['gross_energy_per_token_mj'],250)
            self.assertAlmostEqual(result['decode_tok_s'],31/3)
            self.assertEqual(result['ttft_ms'],1000)
            self.assertTrue(result['energy_valid'])

    def test_negative_dynamic_retained_without_zero_label(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); self.fixture(root,lambda t:.5 if 4<=t<=8 else 1)
            result=parse(root)
            self.assertFalse(result['energy_valid'])
            self.assertIsNone(result['dynamic_energy_per_token_mj'])
            self.assertLess(result['raw_dynamic_energy_per_token_mj'],0)
            self.assertGreater(result['decode_tok_s'],0)

    def test_incomplete_workload_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); self.fixture(root,lambda t:1)
            timing=json.loads((root/'timing.json').read_text()); timing['decode_tokens']=1
            (root/'timing.json').write_text(json.dumps(timing))
            with self.assertRaisesRegex(ValueError,'Incomplete'): parse(root)


if __name__=='__main__': unittest.main()
