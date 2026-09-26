import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tempfile
import unittest

from scripts.sweep.layerwise.candidates import effective_group, generate
from scripts.sweep.layerwise.database import ROOT, prepare, read_candidate, validate_database

SPEC = ROOT/'scripts/sweep/configs/watch5_layerwise_500_space.json'


class CandidateTests(unittest.TestCase):
    def test_deterministic_and_unique(self):
        spec = json.loads(SPEC.read_text())
        rows = generate(spec)
        self.assertEqual(rows, generate(spec))
        self.assertEqual(len({r['architecture_sha256'] for r in rows}), 500)
        spec['seed'] += 1
        self.assertNotEqual(rows, generate(spec))

    def test_group_is_global(self):
        r = dict(n_h=3,n_kv=3,d_qk=16,d_v=64,d_mlp=384)
        self.assertEqual(effective_group(576,[r]),64)
        self.assertEqual(effective_group(576,[r,dict(r,d_v=32)]),32)
        self.assertEqual(effective_group(576,[r,dict(r,d_v=16)]),16)

    def test_database_roundtrip_and_tamper(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'pool'
            manifest=prepare(SPEC,output)
            db=output/'candidates.sqlite'
            report=validate_database(db)
            self.assertEqual(report,manifest['validation'])
            self.assertEqual(report['layers'],15500)
            self.assertEqual(report['patterns']['uniform_control'],48)
            self.assertEqual(report['splits'],dict(train=400,validation=50,test=50))
            self.assertEqual(report['measurement_count'],0)
            self.assertEqual(report['jobs_by_status'],{'planned':500})
            self.assertEqual(report['distinct_layer_shapes'],{'smollm2-135m':192,'smollm2-360m':192})
            self.assertFalse(manifest['hardware_enabled'])
            rows=[json.loads(line) for line in (output/'architectures.jsonl').read_text().splitlines()]
            for row in rows:
                self.assertEqual(read_candidate(db,row['candidate_id']),row['architecture'])
            with self.assertRaises(FileExistsError): prepare(SPEC,output)
            with self.assertRaises(ValueError): read_candidate(db,'unknown')
            with sqlite3.connect(db) as con:
                con.execute('UPDATE layers SET d_v=17 WHERE candidate_id=? AND layer_index=0',(rows[0]['candidate_id'],))
            with self.assertRaisesRegex(ValueError,'Ordered layer'): validate_database(db)


class ExportTests(unittest.TestCase):
    @staticmethod
    def architecture(group):
        spec=json.loads(SPEC.read_text())
        p=dict(spec['operator_profile'],vocab_size=128,seq_len=64)
        layers=[dict(n_h=3,n_kv=3,d_qk=16,d_v=group,d_mlp=128),
                dict(n_h=6,n_kv=3,d_qk=32,d_v=64,d_mlp=192)]
        return dict(family='tiny-fixture',n_layer=2,d_model=64,q8_group_size=group,operator_profile=p,layers=layers)

    def test_export_all_groups(self):
        import torch
        from scripts.sweep.layerwise.export import export_mock, validate_export
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temp:
            for group in [16,32,64]:
                a=self.architecture(group); target=Path(temp)/f'{group}.rlm'
                metadata=export_mock(a,target)
                self.assertTrue(metadata['synthetic_weights'])
                validate_export(a,target)
                second=Path(temp)/f'{group}_again.rlm'
                export_mock(a,second)
                self.assertEqual(target.read_bytes(),second.read_bytes())
                with self.assertRaises(ValueError): export_mock(a,target)
                with target.open('r+b') as stream:
                    stream.seek(256); stream.write(b'\xff'*4)
                with self.assertRaisesRegex(ValueError,'descriptor'): validate_export(a,target)

    def test_unsupported_profile_is_rejected_before_writing(self):
        from scripts.sweep.layerwise.export import export_mock
        with tempfile.TemporaryDirectory() as temp:
            a=self.architecture(64); a['operator_profile']['peri_ln']=False
            target=Path(temp)/'invalid.rlm'
            with self.assertRaisesRegex(ValueError,'Operator profile'): export_mock(a,target)
            self.assertFalse(target.exists())

    @unittest.skipUnless(shutil.which('cc'), 'Host C compiler not installed')
    def test_existing_kernel_loads_and_decodes_heterogeneous_exports(self):
        import torch
        from scripts.sweep.layerwise.export import export_mock
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); executable=root/'runq_llmforge'
            subprocess.run(['cc','-O2','-I',str(ROOT/'src'),str(ROOT/'src/runq_llmforge.c'),
                            '-lm','-o',str(executable)],check=True,capture_output=True,timeout=60)
            ids=root/'prompt.txt'; ids.write_text('1 2 3\n')
            for group in [16,32,64]:
                target=root/f'{group}.rlm'
                export_mock(self.architecture(group),target)
                result=subprocess.run([str(executable),str(target),'-I',str(ids),'-n','8','-t','0'],
                                      check=True,capture_output=True,text=True,timeout=30)
                tokens=[int(x) for x in result.stdout.split()]
                self.assertEqual(len(tokens),6)
                self.assertTrue(all(0<=x<128 for x in tokens))


if __name__=='__main__': unittest.main()
