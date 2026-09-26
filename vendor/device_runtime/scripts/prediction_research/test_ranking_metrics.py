import unittest
import numpy as np
from .ranking_metrics import score


class RankingMetricsTests(unittest.TestCase):
    def test_perfect_both_directions(self):
        for higher in [True,False]:
            r=score([1,2,3,4],[1,2,3,4],list('abcd'),higher,2)
            for f in ['kendall_tau_b','spearman','ndcg_at_k']: self.assertAlmostEqual(r[f],1)
            self.assertEqual(r['pairwise_accuracy_pct'],100)
            self.assertEqual(r['recall_at_k_pct'],100)
            self.assertEqual(r['selection_regret_pct'],0)
            self.assertEqual(r['topk_regret_pct'],0)

    def test_reverse_throughput(self):
        r=score([1,2,3,4],[4,3,2,1],list('abcd'),True,2)
        self.assertAlmostEqual(r['kendall_tau_b'],-1)
        self.assertEqual(r['pairwise_accuracy_pct'],0)
        self.assertEqual(r['recall_at_k_pct'],0)
        self.assertEqual(r['selection_regret_pct'],75)
        self.assertEqual(r['topk_regret_pct'],50)

    def test_reverse_cost(self):
        r=score([1,2,3,4],[4,3,2,1],list('abcd'),False,2)
        self.assertEqual(r['selection_regret_pct'],300)
        self.assertEqual(r['topk_regret_pct'],200)

    def test_ties_half_credit(self):
        r=score([1,2,3],[1,1,2],list('abc'),True,2)
        self.assertAlmostEqual(r['pairwise_accuracy_pct'],100*2.5/3)

    def test_scale_invariance_and_all_candidates(self):
        a=score([1,2,3],[2,1,3],list('abc'),False,3)
        b=score([10,20,30],[20,10,30],list('abc'),False,3)
        for f in ['ndcg_at_k','selection_regret_pct','topk_regret_pct','recall_at_k_pct']:
            self.assertAlmostEqual(a[f],b[f])
        self.assertEqual(a['topk_regret_pct'],0)
        self.assertEqual(a['recall_at_k_pct'],100)


if __name__=='__main__':
    unittest.main()
