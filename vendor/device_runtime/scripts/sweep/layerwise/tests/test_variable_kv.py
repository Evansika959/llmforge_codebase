import copy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from scripts.sweep.layerwise.candidates import generate, layer_shapes, validate_architecture
from scripts.sweep.layerwise.database import ROOT, prepare, validate_database
from scripts.sweep.layerwise.tests import test_candidates as fixtures

VARIABLE_SPEC=ROOT/'scripts/sweep/configs/watch5_layerwise_500_variable_kv_space.json'


class VariableKVTests(unittest.TestCase):
    def test_shape_space_and_invalid_divisor(self):
        spec=json.loads(VARIABLE_SPEC.read_text())
        self.assertEqual([len(layer_shapes(f)) for f in spec['families'].values()],[576,640])
        for f in spec['families'].values():
            self.assertTrue(all(r['n_h']%r['n_kv']==0 for r in layer_shapes(f)))
        broken=copy.deepcopy(spec['families']['smollm2-135m'])
        broken['n_kv_by_n_h']['3']=[2]
        with self.assertRaisesRegex(ValueError,'divisors'): layer_shapes(broken)

    def test_next_batch_determinism_exclusions_and_variable_layers(self):
        old=generate(json.loads(fixtures.SPEC.read_text()))
        spec=json.loads(VARIABLE_SPEC.read_text())
        spec['exclusions']=dict(architecture_sha256=[r['architecture_sha256'] for r in old],
                                permutation_groups=[r['permutation_group'] for r in old])
        rows=generate(spec)
        self.assertEqual(rows,generate(spec))
        self.assertEqual(len(rows),500)
        self.assertFalse(set(r['architecture_sha256'] for r in rows)&set(spec['exclusions']['architecture_sha256']))
        self.assertFalse(set(r['permutation_group'] for r in rows)&set(spec['exclusions']['permutation_groups']))
        heterogeneous=[r for r in rows if r['pattern']!='uniform_control']
        self.assertEqual(len(heterogeneous),452)
        for r in heterogeneous:
            self.assertGreater(len({l['n_kv'] for l in r['architecture']['layers']}),1)
        bad=copy.deepcopy(rows[0]['architecture']);bad['layers'][0]['n_kv']=7
        with self.assertRaises(ValueError): validate_architecture(bad,spec)

    def test_registry_and_exclusion_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            old=Path(temp)/'old';new=Path(temp)/'next'
            with self.assertRaisesRegex(ValueError,'exclude-database'): prepare(VARIABLE_SPEC,new)
            prepare(fixtures.SPEC,old)
            manifest=prepare(VARIABLE_SPEC,new,[old/'candidates.sqlite'])
            report=validate_database(new/'candidates.sqlite')
            self.assertEqual(report,manifest['validation'])
            self.assertEqual(report['measurement_count'],0)
            self.assertEqual(report['excluded_architectures'],500)
            self.assertEqual(report['prior_permutation_overlap'],0)
            self.assertEqual(report['variable_kv_candidates'],{'smollm2-135m':226,'smollm2-360m':226})
            self.assertEqual(report['splits'],{'train':400,'validation':50,'test':50})

    @unittest.skipUnless(shutil.which('cc'),'Host compiler unavailable')
    def test_unchanged_kernel_variable_kv_export_and_decode(self):
        import torch
        from scripts.sweep.layerwise.export import export_mock,validate_export
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);binary=root/'runq_llmforge'
            subprocess.run(['cc','-O2','-I',str(ROOT/'src'),str(ROOT/'src/runq_llmforge.c'),'-lm','-o',str(binary)],
                           check=True,capture_output=True,timeout=60)
            ids=root/'prompt.txt';ids.write_text('1 2 3\n')
            for group in [16,32,64]:
                arch=fixtures.ExportTests.architecture(group)
                arch['layers']=[dict(n_h=h,n_kv=k,d_qk=16,d_v=group if i==0 else 64,d_mlp=128)
                                for i,(h,k) in enumerate([(3,1),(6,2),(9,9),(5,1),(10,10),(15,3)])]
                arch['n_layer']=len(arch['layers'])
                path=root/f'variable_{group}.rlm'
                export_mock(arch,path);validate_export(arch,path)
                result=subprocess.run([str(binary),str(path),'-I',str(ids),'-n','8','-t','0'],
                                      check=True,capture_output=True,text=True,timeout=30)
                tokens=[int(t) for t in result.stdout.split()]
                self.assertEqual(len(tokens),6)
                self.assertTrue(all(0<=t<128 for t in tokens))


if __name__=='__main__': unittest.main()
