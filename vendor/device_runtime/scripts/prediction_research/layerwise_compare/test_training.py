import unittest
import numpy as np
import torch
from .models import token_groups, Transformer
from .training import masked_loss


class MissingLabelTests(unittest.TestCase):
    def test_missing_energy_has_zero_gradient_but_timing_is_retained(self):
        p=torch.tensor([[1.,2.,3.],[3.,2.,1.]],requires_grad=True)
        y=torch.tensor([[2.,1.,float('nan')],[2.,1.,2.]])
        loss=masked_loss(p,y);loss.backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertEqual(p.grad[0,2],0)
        self.assertNotEqual(p.grad[0,0],0)
        self.assertNotEqual(p.grad[1,2],0)

    def test_group_partition_and_forward(self):
        names=['total_params']+[p+'mean' for p in __import__(
            'scripts.prediction_research.layerwise_compare.models',fromlist=['PREFIXES']).PREFIXES]
        groups=token_groups(names)
        self.assertEqual(sorted(i for g in groups for i in g),list(range(len(names))))
        net=Transformer(groups,width=32,layers=1)
        self.assertEqual(tuple(net(torch.zeros(3,len(names))).shape),(3,3))


if __name__=='__main__': unittest.main()
