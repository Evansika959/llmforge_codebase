"""Synthetic round/pipeline tests; never write production datasets or checkpoints."""
import copy
import csv
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import joblib
import numpy as np
import torch

from ..__main__ import COMMANDS
from ..data.dataset import MeasurementDataset
from ..data.hardware_csv import normalize_batch
from ..inference.predictor import bundle_targets, predict_bundle
from ..models.serialization import load_bundle
from ..training.pipeline import fit_dataset


def example_document():
    rows = []
    for i in range(14):
        width = 256 + 32*i
        rows.append(dict(measurement_id=f'm{i}',config_id=f'c{i}',round_id='round0',
            split='train' if i < 8 else 'validation' if i < 11 else 'test',
            architecture=dict(n_layer=4+i,d_model=width,n_h=4,n_kv=2,d_qk=32,d_v=32,d_mlp=width*3,vocab_size=50257),
            metrics=dict(decode_tok_s=30-i,ttft_ms=100+20*i,dynamic_energy_per_token_mj=10+2*i,gross_energy_per_token_mj=20+3*i)))
    return dict(schema_version=1,protocol=dict(protocol_id='synthetic-only',device_id='fake-device',kernel_id='fake-kernel',
                                             prompt_tokens=49,output_tokens=32,energy_target='dynamic'),observations=rows)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.dataset = MeasurementDataset(example_document())

    def batch(self, index=0):
        row = self.dataset.observations[index]
        row.update(measurement_id='new-observation',round_id='round1',split='train')
        return dict(protocol=self.dataset.protocol,observations=[row])

    def test_append_preserves_parent_and_holdout(self):
        before = self.dataset.to_dict()
        child = self.dataset.append_training(self.batch())
        self.assertEqual(self.dataset.to_dict(), before)
        self.assertEqual(child.to_dict()['parent_sha256'],self.dataset.fingerprint)
        self.assertEqual(child.observations[:-1],before['observations'])
        self.assertNotEqual(child.fingerprint,self.dataset.fingerprint)

    def test_alias_of_heldout_architecture_rejected(self):
        for index in [8,11]:
            batch = self.batch(index)
            batch['observations'][0]['config_id'] = 'new-name-same-architecture'
            with self.assertRaisesRegex(ValueError,'validation/test'):
                self.dataset.append_training(batch)

    def test_duplicate_measurements_rejected(self):
        batch = self.batch()
        batch['observations'][0]['measurement_id'] = 'm0'
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            self.dataset.append_training(batch)

    def test_protocol_changes_rejected(self):
        for field,value in [('protocol_id','v2'),('energy_target','gross'),('output_tokens',64),('device_id','other')]:
            batch = self.batch()
            batch['protocol'][field] = value
            with self.assertRaisesRegex(ValueError,'protocol'):
                self.dataset.append_training(batch)

    def test_invalid_labels_and_architectures_rejected(self):
        for value in [0,-1,float('nan'),float('inf')]:
            doc = example_document()
            doc['observations'][0]['metrics']['dynamic_energy_per_token_mj'] = value
            with self.assertRaises(ValueError):
                MeasurementDataset(doc)
        doc = example_document()
        doc['observations'][0]['architecture']['d_model'] = 256.5
        with self.assertRaises(ValueError):
            MeasurementDataset(doc)

    def test_direct_split_leakage_rejected(self):
        doc = example_document()
        doc['observations'][11]['architecture'] = copy.deepcopy(doc['observations'][0]['architecture'])
        with self.assertRaisesRegex(ValueError,'leakage'):
            MeasurementDataset(doc)

    def test_save_is_exclusive_and_roundtrips(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'round0.json'
            self.dataset.save(path)
            self.assertEqual(MeasurementDataset.load(path).fingerprint,self.dataset.fingerprint)
            with self.assertRaises(FileExistsError):
                self.dataset.save(path)

    def test_hardware_csv_import_and_source_provenance(self):
        original = self.dataset.observations[0]
        config = dict(config_id=original['config_id'], **original['architecture'])
        measurement = dict(config, decode_tok_s=30, ttft_ms=100,
                           dynamic_energy_per_token_mj=10, total_energy_j=.64, notes='')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name,record in [('configs.csv',config),('measurements.csv',measurement)]:
                with (root/name).open('w',newline='') as stream:
                    writer = csv.DictWriter(stream,fieldnames=list(record))
                    writer.writeheader()
                    writer.writerow(record)
            batch = normalize_batch(root/'configs.csv',root/'measurements.csv',self.dataset.protocol,'round1')
            self.assertEqual(batch['observations'][0]['metrics']['gross_energy_per_token_mj'],20)
            child = self.dataset.append_training(batch)
            self.assertEqual(child.to_dict()['ingestions'][-1]['source_hashes'],batch['source_hashes'])
            with self.assertRaisesRegex(ValueError,'Duplicate'):
                child.append_training(batch)

    def test_bad_csv_architecture_and_flags_rejected(self):
        config = dict(config_id='c0', **self.dataset.observations[0]['architecture'])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root/'configs.csv').open('w',newline='') as stream:
                writer = csv.DictWriter(stream,fieldnames=list(config))
                writer.writeheader()
                writer.writerow(config)
            for changes in [dict(d_model=999),dict(notes='baseline_drift'),dict(output_tokens=64)]:
                measurement = dict(config,decode_tok_s=30,ttft_ms=100,total_energy_j=.64,
                                   dynamic_energy_per_token_mj=10,notes='')
                measurement.update(changes)
                with (root/'measurements.csv').open('w',newline='') as stream:
                    writer = csv.DictWriter(stream,fieldnames=list(measurement))
                    writer.writeheader()
                    writer.writerow(measurement)
                with self.assertRaises(ValueError):
                    normalize_batch(root/'configs.csv',root/'measurements.csv',self.dataset.protocol,'round1')


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)

    def test_test_labels_cannot_affect_fit_or_calibration(self):
        original = example_document()
        changed = copy.deepcopy(original)
        for row in changed['observations']:
            if row['split'] == 'test':
                row['metrics'] = {k: v*100 for k,v in row['metrics'].items()}
        a,b = MeasurementDataset(original),MeasurementDataset(changed)
        configs = [r['architecture'] for r in a.observations]
        for family in ['xgboost','transformer']:
            left = fit_dataset(a,family,42,max_epochs=2)
            right = fit_dataset(b,family,42,max_epochs=2)
            self.assertEqual(asdict(left['profile']),asdict(right['profile']))
            np.testing.assert_allclose(predict_bundle(left,configs),predict_bundle(right,configs),rtol=1e-7)

    def test_gross_target_bundle_and_reload(self):
        doc = example_document()
        doc['protocol']['energy_target'] = 'gross'
        dataset = MeasurementDataset(doc)
        pack = fit_dataset(dataset,'xgboost')
        self.assertEqual(bundle_targets(pack)[2],'gross_energy_per_token_mj')
        configs = [r['architecture'] for r in dataset.observations]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.joblib'
            joblib.dump(pack,path)
            np.testing.assert_allclose(predict_bundle(pack,configs),predict_bundle(load_bundle(path),configs))

    def test_cli_round_training_reporting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = MeasurementDataset(example_document())
            dataset.save(root/'dataset.json')
            result = subprocess.run([sys.executable,'-m','scripts.prediction','train','--dataset',str(root/'dataset.json'),
                                     '--output',str(root/'run')],capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertTrue((root/'run/README.md').is_file())
            self.assertEqual(MeasurementDataset.load(root/'run/dataset.json').fingerprint,dataset.fingerprint)
            scores = json.loads((root/'run/metrics.json').read_text())
            self.assertEqual(len(scores),3)
            self.assertTrue(all(r['n']==3 for r in scores))

    def test_all_cli_modules_import(self):
        for module in COMMANDS.values():
            self.assertTrue(callable(importlib.import_module(module).main))

    def test_core_does_not_import_experiment_scripts(self):
        import ast
        package = Path(__file__).resolve().parents[1]
        for folder in ['data','features','models','training','inference']:
            for path in (package/folder).glob('*.py'):
                for node in ast.walk(ast.parse(path.read_text())):
                    if isinstance(node,ast.ImportFrom):
                        self.assertNotIn('experiments',node.module or '',str(path))


if __name__ == '__main__':
    unittest.main()
