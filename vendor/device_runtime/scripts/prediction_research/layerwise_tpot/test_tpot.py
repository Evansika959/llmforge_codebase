import copy
import unittest
import numpy as np
from .data import label_row
from .report import evaluate


class TpotTests(unittest.TestCase):
    def example(self):
        raw = dict(timing=dict(decode_tokens=31, output_tokens=32, end=4.1, prefill_end=1.),
                   decode_tok_s=10., ttft_ms=100., dynamic_energy_per_token_mj=4.)
        row = dict(derived_metrics=dict(tpot_ms=100.),
                   metrics={k: v for k, v in raw.items() if k != 'timing'})
        return row, raw

    def test_31_not_32_and_no_mutation(self):
        row, raw = self.example()
        before = copy.deepcopy(row)
        np.testing.assert_allclose(label_row(row, raw), [100., 100., 4.])
        self.assertEqual(row, before)

    def test_reject_wrong_count(self):
        row, raw = self.example()
        raw['timing']['decode_tokens'] = 32
        with self.assertRaises(ValueError):
            label_row(row, raw)

    def test_reject_invalid_energy(self):
        row, raw = self.example()
        raw['dynamic_energy_per_token_mj'] = row['metrics']['dynamic_energy_per_token_mj'] = -1
        with self.assertRaises(ValueError):
            label_row(row, raw)

    def test_reject_mismatched_derived_timing(self):
        row, raw = self.example()
        row['derived_metrics']['tpot_ms'] = 99
        with self.assertRaises(AssertionError):
            label_row(row, raw)

    def test_cost_direction(self):
        actual = np.tile(np.arange(1., 41.)[:, None], (1, 3))
        ids = np.array([str(i) for i in range(40)])
        result = evaluate(actual, actual[None], ids)
        for metric in result.values():
            s = metric['per_seed'][0]
            self.assertEqual(s['actual_best_id'], '0')
            self.assertEqual(s['selected_id'], '0')
            self.assertEqual(s['recall_at_k_pct'], 100.)
            self.assertEqual(s['selection_regret_pct'], 0.)


if __name__ == '__main__':
    unittest.main()
