"""Dynamic-energy migration/acquisition/retraining tests; no hardware access."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from .test_rounds import example_document
from .test_active_learning import candidate_pool
from ..active_learning.energy import switch_to_dynamic
from ..active_learning.engine import initialize,advance,verify_workspace
from ..active_learning.monitoring import update_accuracy
from ..active_learning.quality import baseline_energy,result_metrics
from ..active_learning.storage import atomic_json,read_json,write_configs,digest
from ..data.dataset import MeasurementDataset
from ..models.serialization import load_bundle


def raw_row(c, metrics=None):
    metrics = metrics or dict(decode_tok_s=20.,ttft_ms=200.,gross_energy_per_token_mj=31.25,dynamic_energy_per_token_mj=25.)
    gross,dynamic=metrics['gross_energy_per_token_mj'],metrics['dynamic_energy_per_token_mj']
    return dict(c,decode_tok_s=metrics['decode_tok_s'],ttft_ms=metrics['ttft_ms'],duration_s=1.,
                total_energy_j=gross*32/1000,active_power_w=gross*32/1000,
                baseline_power_w=(gross-dynamic)*32/1000,dynamic_energy_per_token_mj=dynamic,notes='')


class DynamicActiveTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'gross'
        self.destination=Path(self.temp.name)/'dynamic'
        doc=example_document()
        doc['protocol']['energy_target']='gross'
        self.dataset=MeasurementDataset(doc)
        self.snapshot=Path(self.temp.name)/'snapshot.json'
        atomic_json(self.snapshot,dict(records=dict(batch1=[raw_row(dict(r['architecture'],config_id=r['config_id']),r['metrics'])
                    for r in doc['observations']],batch2=[])))
        initialize(self.root,self.dataset,candidate_pool(),dict(batch_size=2,members=2,anchors=0,max_rounds=4,min_params_m=.001,max_params_m=1000.))

    def test_baseline_reconciliation_and_no_double_subtraction(self):
        row=raw_row(candidate_pool()[0])
        self.assertEqual(baseline_energy(row),25.)
        protocol=dict(self.dataset.protocol,energy_target='dynamic',energy_label_validation='baseline_reconciled_v1')
        self.assertEqual(result_metrics(row,candidate_pool()[0],protocol)['dynamic_energy_per_token_mj'],25.)
        for key,value in [('baseline_power_w',0),('baseline_power_w',float('nan')),('dynamic_energy_per_token_mj',0),
                          ('dynamic_energy_per_token_mj',99.)]:
            with self.subTest(key=key,value=value):
                bad=dict(row,**{key:value})
                with self.assertRaises(ValueError):
                    result_metrics(bad,candidate_pool()[0],protocol)

    def test_transition_partial_seed_and_dynamic_loop(self):
        def full(rd,proposal,*args):
            write_configs(rd/'measurements.csv',[raw_row(c) for c in proposal['schedule']])
        advance(self.root,full)
        def partial(rd,proposal,*args):
            write_configs(rd/'measurements.csv',[raw_row(proposal['schedule'][0])])
        advance(self.root,partial)
        original={str(p.relative_to(self.root)):digest(p) for p in self.root.rglob('*') if p.is_file() and p.name!='RUNNING.lock'}
        with patch('scripts.prediction.active_learning.hardware.preflight',side_effect=AssertionError('No ADB')):
            switch_to_dynamic(self.root,self.destination,self.snapshot,'Switch target')
            config,state,dataset=verify_workspace(self.destination)
            self.assertEqual(state['completed_rounds'],0)
            self.assertEqual(dataset.protocol['energy_target'],'dynamic')
            self.assertEqual(sum(r['split']=='train' for r in dataset.observations),11)
            self.assertEqual([(r['measurement_id'],r['architecture']) for r in dataset.observations if r['split']!='train'],
                             [(r['measurement_id'],r['architecture']) for r in self.dataset.observations if r['split']!='train'])
            advance(self.destination)
        self.assertEqual(original,{name:digest(self.root/name) for name in original})
        rd=self.destination/'rounds/0001'
        proposal=read_json(rd/'proposal.json')
        committee=load_bundle(rd/'committee.joblib')
        self.assertEqual(committee['targets'][-1],'dynamic_energy_per_token_mj')
        for selection in proposal['selected']:
            self.assertIn('dynamic_energy_per_token_mj',selection['prediction'])
            self.assertNotIn('gross_energy_per_token_mj',selection['predictor_prediction'])
        baseline=update_accuracy(self.destination)
        self.assertEqual(baseline['energy_target'],'dynamic')
        def bad(rd,proposal,*args):
            rows=[raw_row(c) for c in proposal['schedule']]
            rows[0]['baseline_power_w']=0
            write_configs(rd/'measurements.csv',rows)
        self.assertEqual(advance(self.destination,bad)['completed_rounds'],0)
        (rd/'measurements.csv').unlink()
        self.assertEqual(advance(self.destination,full)['completed_rounds'],1)
        model=load_bundle(rd/'predictor_after.joblib')
        self.assertEqual(model['targets'][-1],'dynamic_energy_per_token_mj')
        report=update_accuracy(self.destination)
        self.assertEqual(len(report['history']),2)
        self.assertEqual(report['history'][-1]['validation'][-1]['target'],'dynamic_energy_per_token_mj')
        self.assertEqual(report['baseline'],baseline['baseline'])

    def test_bad_historical_baseline_refuses_migration_without_filtering(self):
        snapshot=read_json(self.snapshot)
        snapshot['records']['batch1'][0]['baseline_power_w']=0
        atomic_json(self.snapshot,snapshot)
        with self.assertRaisesRegex(ValueError,'baseline'):
            switch_to_dynamic(self.root,self.destination,self.snapshot,'Switch target')
        self.assertFalse(self.destination.exists())

    def test_dynamic_replay_retains_real_baseline_validation(self):
        from ..active_learning.replay import ReplayBackend
        switch_to_dynamic(self.root,self.destination,self.snapshot,'Switch target')
        # Exercise the backend directly: only offline workspaces may use it.
        c=candidate_pool()[0]
        raw=raw_row(c)
        oracle={c['config_id']:dict(decode_tok_s=20.,ttft_ms=200.,dynamic_energy_per_token_mj=25.)}
        backend=ReplayBackend(oracle,{c['config_id']:raw})
        rd=Path(self.temp.name)/'replay'
        rd.mkdir()
        protocol=verify_workspace(self.destination)[2].protocol
        backend(rd,dict(schedule=[c]),protocol,dict(mode='offline_replay'))
        import csv
        with (rd/'measurements.csv').open() as stream:
            measured=next(csv.DictReader(stream))
        self.assertEqual(result_metrics(measured,c,protocol)['dynamic_energy_per_token_mj'],25.)
        with self.assertRaisesRegex(ValueError,'offline'):
            backend(rd,dict(schedule=[c]),protocol,dict(mode='live'))

    def test_explicit_quarantine_preserves_bad_training_row_but_never_filters_holdouts(self):
        snapshot=read_json(self.snapshot)
        first=snapshot['records']['batch1'][0]
        first['baseline_power_w']=first['active_power_w']+0.1
        first['dynamic_energy_per_token_mj']=0
        atomic_json(self.snapshot,snapshot)
        switch_to_dynamic(self.root,self.destination,self.snapshot,'Quarantine unusable energy',
                          quarantine_invalid_training_energy=True)
        dataset=verify_workspace(self.destination)[2]
        self.assertEqual(sum(r['split']=='train' for r in dataset.observations),7)
        record=read_json(self.destination/'provenance/energy_transition.json')
        self.assertEqual(record['quarantined_training'][0]['raw_row'],first)
        heldout=next(r for r in self.dataset.observations if r['split']=='validation')
        raw=next(r for r in snapshot['records']['batch1'] if r['config_id']==heldout['config_id'])
        raw['baseline_power_w']=0
        atomic_json(self.snapshot,snapshot)
        with self.assertRaisesRegex(ValueError,'validation'):
            switch_to_dynamic(self.root,Path(self.temp.name)/'bad_holdout',self.snapshot,'Never filter holdouts',
                              quarantine_invalid_training_energy=True)

    def test_monitor_refuses_gross_dynamic_baseline_mix(self):
        switch_to_dynamic(self.root,self.destination,self.snapshot,'Switch target')
        update_accuracy(self.destination)
        path=self.destination/'monitoring/baseline.json'
        baseline=read_json(path)
        baseline['energy_target']='gross'
        atomic_json(path,baseline)
        with self.assertRaisesRegex(ValueError,'target/split'):
            update_accuracy(self.destination)

    def test_run_requires_explicit_hardware_authority(self):
        from ..active_learning.cli import main
        with patch.object(sys,'argv',['active','run','--workspace',str(self.root),'--serial','fake']),\
             patch('scripts.prediction.active_learning.monitoring.run_monitored',side_effect=AssertionError('No hardware')):
            with self.assertRaises(SystemExit) as result:
                main()
        self.assertEqual(result.exception.code,2)


if __name__=='__main__':
    unittest.main()
