import unittest
import numpy as np
import torch
from .models import build
from .training import fit, predict


class CompactTests(unittest.TestCase):
    def test_shapes(self):
        for family in ['mlp','transformer']:
            for width in [32,64]:
                model=build(dict(family=family,width=width,layers=2))
                self.assertEqual(model(torch.ones(4,32)).shape,(4,3))

    def test_scalers_and_reload(self):
        torch.set_num_threads(2)
        rng=np.random.default_rng(1)
        x=rng.uniform(0,3,(40,32)).astype('float32')
        y=np.exp(rng.normal(size=(40,3))).astype('float32')
        for family in ['mlp','transformer']:
            c=dict(family=family,width=32,layers=1,dropout=.1,lr=.001)
            pack=fit(c,x[:30],y[:30],x[30:]+10,y[30:],42,max_epochs=2)
            np.testing.assert_allclose(pack['x_scaler'].mean_,np.log1p(x[:30]).mean(0),rtol=1e-6)
            p=predict(pack,x[30:])
            self.assertEqual(p.shape,(10,3))
            self.assertTrue(np.isfinite(p).all() and (p>0).all())
            np.testing.assert_array_equal(p,predict(pack,x[30:]))


if __name__ == '__main__':
    unittest.main()
