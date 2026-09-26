"""Offline integration tests; no ADB, benchmark, or production-data writes."""
import copy
import csv
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

from .test_rounds import example_document
from ..data.dataset import MeasurementDataset, architecture_key
from ..active_learning.committee import fit_committee, committee_predictions
from ..active_learning.engine import initialize, advance, verify_workspace
from ..active_learning.hardware import AndroidBackend, merge_attempts, preflight
from ..active_learning.quality import anchor_report, result_metrics
from ..active_learning.sampling import select_batch, unseen_candidates
from ..active_learning.storage import read_json, write_configs, locked, atomic_json, digest, atomic_bundle


def candidate_pool():
    prototype = example_document()['observations'][0]['architecture']
    return [dict(prototype,config_id=f'candidate_{i}',n_layer=20+i,d_model=512,d_mlp=1536) for i in range(8)]


def mock_results(proposal, drift=False):
    rows = []
    for c in proposal['schedule']:
        scale = 2 if drift and c['config_id'].endswith('_POST') else 1
        rows.append(dict(c,decode_tok_s=(50-int(c['n_layer']))*scale,
                         ttft_ms=(100+10*int(c['n_layer']))*scale,
                         dynamic_energy_per_token_mj=(10+int(c['n_layer']))*scale,notes=''))
    return rows


class ActiveLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'workspace'
        self.dataset = MeasurementDataset(example_document())
        self.pool = candidate_pool()
        self.settings = dict(batch_size=2,members=2,anchors=2,max_rounds=2,min_params_m=.001,max_params_m=1000.)

    def init(self):
        return initialize(self.root,self.dataset,self.pool,self.settings)

    def test_dry_step_does_not_access_hardware_and_freezes_proposal(self):
        self.init()
        with patch('scripts.prediction.active_learning.hardware.preflight',side_effect=AssertionError('No ADB allowed')):
            state = advance(self.root)
            self.assertEqual(state['status'],'awaiting_measurements')
            proposal = (self.root/'rounds/0001/proposal.json').read_bytes()
            advance(self.root)
            self.assertEqual((self.root/'rounds/0001/proposal.json').read_bytes(),proposal)

    def test_initial_checkpoint_is_loaded_without_refitting_predictor(self):
        from ..training.pipeline import fit_dataset
        model = fit_dataset(self.dataset)
        checkpoint = Path(self.temp.name)/'existing.joblib'
        atomic_bundle(checkpoint,model)
        initialize(self.root,self.dataset,self.pool,self.settings,checkpoint)
        with patch('scripts.prediction.active_learning.engine.fit_dataset',side_effect=AssertionError('Do not refit initial checkpoint')):
            advance(self.root)
        proposal = read_json(self.root/'rounds/0001/proposal.json')
        self.assertTrue(proposal['predictor_origin']['loaded'])
        self.assertEqual(read_json(self.root/'initial_model_source.json')['sha256'],digest(checkpoint))

    def test_initial_checkpoint_rejects_mismatched_dataset(self):
        from ..training.pipeline import fit_dataset
        model = fit_dataset(self.dataset)
        model['dataset_sha256']='different-dataset'
        checkpoint = Path(self.temp.name)/'wrong.joblib'
        atomic_bundle(checkpoint,model)
        with self.assertRaisesRegex(ValueError,'exact starting dataset'):
            initialize(self.root,self.dataset,self.pool,self.settings,checkpoint)

    def test_complete_two_rounds_and_budget_stop(self):
        self.init()
        called = []
        def backend(rd,proposal,protocol,settings):
            called.append(proposal['round'])
            write_configs(rd/'measurements.csv',mock_results(proposal))
        state = advance(self.root,backend,rounds=2)
        self.assertEqual(state['status'],'budget_complete')
        self.assertEqual(called,[1,2])
        _,_,dataset = verify_workspace(self.root)
        self.assertEqual(len(dataset.observations),len(self.dataset.observations)+4)
        self.assertEqual([r for r in dataset.observations if r['split']!='train'],
                         [r for r in self.dataset.observations if r['split']!='train'])
        advance(self.root,lambda *a: self.fail('Budget exhausted; no backend calls'))

    def test_missing_results_pause_without_partial_ingestion(self):
        self.init()
        def partial(rd,proposal,*args):
            write_configs(rd/'measurements.csv',mock_results(proposal)[:-1])
        state = advance(self.root,partial)
        self.assertEqual(state['status'],'awaiting_measurements')
        self.assertFalse((self.root/'rounds/0001/dataset_after.json').exists())
        self.assertEqual(verify_workspace(self.root)[2].fingerprint,self.dataset.fingerprint)

    def test_anchor_drift_pauses(self):
        self.init()
        def drift(rd,proposal,*args):
            write_configs(rd/'measurements.csv',mock_results(proposal,True))
        state = advance(self.root,drift)
        self.assertEqual(state['status'],'paused_anchor_drift')
        self.assertFalse((self.root/'rounds/0001/dataset_after.json').exists())

    def test_resume_after_ingestion_does_not_remeasure_or_duplicate(self):
        self.init()
        from ..active_learning import engine
        real_fit = engine.fit_dataset
        calls = []
        def fail_after_ingestion(*a,**kw):
            calls.append(1)
            if len(calls)==2:
                raise RuntimeError('simulated training interruption')
            return real_fit(*a,**kw)
        def backend(rd,proposal,*args):
            write_configs(rd/'measurements.csv',mock_results(proposal))
        with patch.object(engine,'fit_dataset',side_effect=fail_after_ingestion):
            with self.assertRaisesRegex(RuntimeError,'interruption'):
                advance(self.root,backend)
        self.assertTrue((self.root/'rounds/0001/ingested.json').exists())
        advance(self.root,lambda *a:self.fail('Must not remeasure a committed batch'))
        self.assertEqual(len(verify_workspace(self.root)[2].observations),len(self.dataset.observations)+2)

    def test_frozen_inputs_reject_edits(self):
        self.init()
        settings = read_json(self.root/'settings.json')
        settings['batch_size']=8
        atomic_json(self.root/'settings.json',settings)
        with self.assertRaisesRegex(ValueError,'input changed'):
            advance(self.root)

    def test_checkpoint_proposal_tamper_rejected(self):
        self.init()
        advance(self.root)
        path = self.root/'rounds/0001/schedule.csv'
        with path.open('a') as stream:
            stream.write('\n')
        with self.assertRaisesRegex(ValueError,'artifact changed'):
            advance(self.root)

    def test_lock_releases_on_exception(self):
        path = Path(self.temp.name)/'lock'
        with self.assertRaisesRegex(RuntimeError,'crash'):
            with locked(path):
                raise RuntimeError('crash')
        with locked(path):
            pass

    def test_hardware_requires_acknowledgment_and_consistent_policy(self):
        with self.assertRaises(ValueError):
            AndroidBackend('device')
        backend = AndroidBackend('device',True)
        with self.assertRaisesRegex(ValueError,'Anchor schedule'):
            backend(Path(self.temp.name),dict(anchors=[],schedule=[]),{},dict(anchors=3))
        with self.assertRaisesRegex(ValueError,'adapter'):
            backend(Path(self.temp.name),dict(anchors=[],schedule=[]),{},dict(anchors=0))
        with self.assertRaisesRegex(ValueError,'Offline'):
            backend(Path(self.temp.name),{}, {},dict(mode='offline_replay'))

    def test_bootstrap_and_acquisition_do_not_use_test_labels(self):
        original = self.dataset
        document = original.to_dict()
        for row in document['observations']:
            if row['split']=='test':
                row['metrics']={k:v*1000 for k,v in row['metrics'].items()}
        changed = MeasurementDataset(document)
        left,right = fit_committee(original,2,42),fit_committee(changed,2,42)
        a,b = committee_predictions(left,self.pool),committee_predictions(right,self.pool)
        np.testing.assert_allclose(a[0],b[0])
        np.testing.assert_allclose(a[1],b[1])
        self.assertEqual(select_batch(original,self.pool,left,3),select_batch(changed,self.pool,right,3))
        train_ids = {r['measurement_id'] for r in original.observations if r['split']=='train'}
        for member in left['members']:
            self.assertTrue(set(member['bootstrap_ids']) <= train_ids)

    def test_pool_excludes_holdout_architecture_aliases(self):
        heldout = self.dataset.observations[-1]
        alias = dict(heldout['architecture'],config_id='alias_of_test')
        self.assertEqual(len(unseen_candidates(self.dataset,self.pool+[alias])),len(self.pool))

    def test_raw_invalid_rows_quarantined_not_deleted(self):
        rd = Path(self.temp.name)/'round'
        attempt = rd/'attempts/001'
        attempt.mkdir(parents=True)
        proposal = dict(schedule=self.pool[:2],anchors=[])
        rows = mock_results(proposal)
        rows[0]['dynamic_energy_per_token_mj']=0
        raw = attempt/'raw_results.csv'
        write_configs(raw,rows)
        before=raw.read_bytes()
        valid = merge_attempts(rd,proposal,self.dataset.protocol)
        self.assertEqual(set(valid),{self.pool[1]['config_id']})
        self.assertEqual(raw.read_bytes(),before)
        self.assertEqual(len(read_json(rd/'attempt_audit.json')['rejected']),1)

    def test_drift_review_requires_exact_result_hash_and_records_override(self):
        self.init()
        def drift(rd,proposal,*args):
            write_configs(rd/'measurements.csv',mock_results(proposal,True))
        advance(self.root,drift)
        rd = self.root/'rounds/0001'
        atomic_json(rd/'drift_review.json',dict(reason='Wrong file',results_sha256='not-the-hash'))
        self.assertEqual(advance(self.root)['status'],'paused_anchor_drift')
        from ..active_learning.cli import main
        with patch.object(sys,'argv',['active','review-drift','--workspace',str(self.root),
                                    '--accept','--reason','Synthetic test: known injected factor-two drift']):
            main()
        approval = read_json(rd/'drift_review.json')
        self.assertEqual(approval['results_sha256'],digest(rd/'measurements.csv'))
        self.assertEqual(advance(self.root)['completed_rounds'],1)
        self.assertEqual(read_json(rd/'completion.json')['drift_override'],approval)

    def test_sampling_does_not_reward_predicted_performance(self):
        committee = dict(targets=self.dataset.targets)
        dispersion = np.arange(len(self.pool)*3).reshape(-1,3)*.001
        with patch('scripts.prediction.active_learning.sampling.committee_predictions',
                   return_value=(np.ones((len(self.pool),3)),dispersion)):
            left = select_batch(self.dataset,self.pool,committee,4)
        means = np.arange(1,len(self.pool)*3+1).reshape(-1,3)*100
        with patch('scripts.prediction.active_learning.sampling.committee_predictions',
                   return_value=(means,dispersion)):
            right = select_batch(self.dataset,self.pool,committee,4)
        self.assertEqual([(r['config']['config_id'],r['reason']) for r in left],
                         [(r['config']['config_id'],r['reason']) for r in right])

    def test_gross_energy_does_not_require_positive_dynamic_energy(self):
        c = self.pool[0]
        row = dict(c,decode_tok_s=10,ttft_ms=100,total_energy_j=3.2,
                   dynamic_energy_per_token_mj=-1,notes='')
        protocol = dict(self.dataset.protocol,energy_target='gross')
        values = result_metrics(row,c,protocol)
        self.assertAlmostEqual(values['gross_energy_per_token_mj'],100.)
        with self.assertRaises(ValueError):
            result_metrics(row,c,self.dataset.protocol)

    def test_preflight_refuses_charger_low_battery_and_busy_device(self):
        for charging,level,busy,expected in [('true',80,'','Unplug'),('false',30,'','charge'),
                                           ('false',80,'123','already active')]:
            def run(command,**kwargs):
                script = command[-1]
                if script=='dumpsys battery':
                    output=f'AC powered: {charging}\nUSB powered: false\nWireless powered: false\nlevel: {level}\nscale: 100\n'
                elif script.startswith('pidof '):
                    output=busy
                else:
                    output='device'
                return subprocess.CompletedProcess(command,0,stdout=output,stderr='')
            with self.subTest(expected=expected),patch('scripts.prediction.active_learning.hardware.subprocess.run',side_effect=run):
                with self.assertRaisesRegex(RuntimeError,expected):
                    preflight('fake-device',adb='fake-adb')

    def test_lock_refuses_concurrent_owner(self):
        path = Path(self.temp.name)/'lock'
        with locked(path):
            with self.assertRaises(RuntimeError):
                with locked(path):
                    self.fail('Concurrent owner must not acquire the lock')

    def test_android_adapter_mocked_worker_resumes_missing_rows_only(self):
        self._adapter_resume(anchors=True)

    def test_android_adapter_without_anchors_resumes_only_missing_candidates(self):
        self._adapter_resume(anchors=False)

    def test_android_adapter_passes_strict45_policy_and_records_provenance(self):
        self._adapter_resume(anchors=False,strict=True)

    def _adapter_resume(self, anchors, strict=False):
        from ..active_learning import hardware
        fake_repo = Path(self.temp.name)/'repo'
        (fake_repo/'src').mkdir(parents=True)
        (fake_repo/'src/runq_llmforge.c').write_text('Synthetic kernel identity; never compiled')
        rd = self.root/'rounds/0001'
        rd.mkdir(parents=True)
        c = self.pool[0]
        proposal = dict(schedule=[dict(c,config_id='anchor_PRE'),self.pool[1],self.pool[2],
                                  dict(c,config_id='anchor_POST')],
                        anchors=[dict(config_id=c['config_id'],before='anchor_PRE',after='anchor_POST')])
        if not anchors:
            proposal = dict(schedule=self.pool[1:3],anchors=[])
        kernel = digest(fake_repo/'src/runq_llmforge.c')
        protocol = dict(self.dataset.protocol,adapter='legacy_random_sweep_v1',kernel_id=kernel,
                        temperature_ceiling=40.,min_battery_percent=30)
        settings = dict(kernel_id=kernel,temperature_ceiling=40.,min_battery_percent=30,max_measurement_attempts=3)
        if strict:
            settings.update(temperature_ceiling=45.,thermal_policy=dict(ceiling_c=45.,comparison='lt',previous_ceiling_c=40.,migration='reviewed-test'))
        snapshot = dict(hardware_serial='fake-physical-serial',abi='armeabi-v7a',build='fake-build')
        commands = []
        class FakeWorker:
            def __init__(self,command,**kwargs):
                commands.append(command)
                output = Path(command[command.index('--out')+1])
                configs = Path(command[command.index('--config')+1])
                with configs.open(newline='') as stream:
                    schedule = list(csv.DictReader(stream))
                # First attempt stops after one candidate; next measures missing
                # candidate and fresh POST, never the accepted candidate/PRE.
                rows = mock_results(dict(schedule=schedule))
                write_configs(output,rows[:(2 if anchors else 1)] if len(commands)==1 else rows)
                atomic_json(output.with_name('original_device_state.json'),dict(settings={}))
                atomic_json(output.with_name('restoration.json'),dict(settings_restored=True))
            def wait(self):
                return 75 if len(commands)==1 else 0
        with patch.object(hardware,'ROOT',fake_repo),patch.object(hardware,'preflight',return_value=snapshot),\
             patch.object(hardware,'toolchain_info',return_value={'mock':True}),\
             patch.object(hardware,'adb_path',return_value='fake-adb'),\
             patch.object(hardware.subprocess,'Popen',FakeWorker),\
             patch.object(hardware.tempfile,'gettempdir',return_value=self.temp.name):
            backend = AndroidBackend('fake-device',True)
            backend(rd,proposal,protocol,settings)
            preserved = (rd/'attempts/001/raw_results.csv').read_bytes()
            backend(rd,proposal,protocol,settings)
            backend(rd,proposal,protocol,settings)  # Complete: no third execution.
        self.assertEqual(len(commands),2)
        for command in commands:
            self.assertEqual('--strict-temperature' in command,strict)
        if strict:
            self.assertEqual(read_json(rd/'attempts/002/thermal_policy.json')['ceiling_c'],45.)
            self.assertTrue(all(v['thermal_policy']['comparison']=='lt' for v in read_json(rd/'attempt_audit.json')['accepted_sources'].values()))
        self.assertEqual((rd/'attempts/001/raw_results.csv').read_bytes(),preserved)
        with (rd/'attempts/002/configs.csv').open(newline='') as stream:
            pending = list(csv.DictReader(stream))
        self.assertEqual([r['config_id'] for r in pending],
                         [self.pool[2]['config_id']]+(['anchor_POST'] if anchors else []))
        with (rd/'measurements.csv').open(newline='') as stream:
            self.assertEqual(len(list(csv.DictReader(stream))),4 if anchors else 2)

    def test_default_has_no_anchors_and_ingests_only_valid_complete_batches(self):
        settings = dict(self.settings)
        settings.pop('anchors')
        initialize(self.root,self.dataset,self.pool,settings)
        advance(self.root)
        rd = self.root/'rounds/0001'
        proposal = read_json(rd/'proposal.json')
        self.assertEqual(proposal['anchors'],[])
        self.assertEqual(len(proposal['schedule']),2)
        rows = mock_results(proposal)
        rows[0]['decode_tok_s'] = 0
        write_configs(rd/'measurements.csv',rows)
        self.assertEqual(advance(self.root)['completed_rounds'],0)
        # Simulate a repaired aggregate, keeping raw-data preservation to adapter tests.
        (rd/'measurements.csv').unlink()
        write_configs(rd/'measurements.csv',mock_results(proposal))
        self.assertEqual(advance(self.root)['completed_rounds'],1)
        quality = read_json(rd/'anchor_report.json')
        self.assertFalse(quality['enabled'])
        self.assertIn('not assessed',quality['warning'])
        self.assertFalse(read_json(rd/'completion.json')['quality_policy']['anchor_gate'])

    def test_migrate_pending_batch_preserves_source_and_resumes_without_anchors(self):
        from ..active_learning.migration import disable_anchors
        self.init()
        def complete(rd,proposal,*args):
            write_configs(rd/'measurements.csv',mock_results(proposal))
        advance(self.root,complete)
        advance(self.root)
        rd = self.root/'rounds/0002'
        proposal = read_json(rd/'proposal.json')
        rows = [r for r in mock_results(proposal) if not r['config_id'].endswith('_POST')]
        attempt = rd/'attempts/001'
        attempt.mkdir(parents=True)
        write_configs(attempt/'raw_results.csv',rows)
        write_configs(rd/'measurements.csv',rows)
        advance(self.root)
        before = {str(p.relative_to(self.root)):digest(p) for p in self.root.rglob('*') if p.is_file() and p.name!='RUNNING.lock'}
        destination = Path(self.temp.name)/'without_anchors'
        disable_anchors(self.root,destination,'User requested removal of repeated controls')
        self.assertEqual(before,{name:digest(self.root/name) for name in before})
        self.assertEqual(verify_workspace(destination)[0]['anchors'],0)
        new_rd = destination/'rounds/0002'
        revised = read_json(new_rd/'proposal.json')
        self.assertEqual(revised['selected'],proposal['selected'])
        self.assertEqual(len(revised['schedule']),2)
        self.assertEqual(digest(new_rd/'attempts/001/raw_results.csv'),digest(attempt/'raw_results.csv'))
        merged = merge_attempts(new_rd,revised,self.dataset.protocol)
        self.assertEqual(len(merged),2)
        self.assertEqual(len(read_json(new_rd/'attempt_audit.json')['excluded']),2)
        with patch('scripts.prediction.active_learning.hardware.preflight',side_effect=AssertionError('No ADB')):
            state = advance(destination)
        self.assertEqual(state['completed_rounds'],2)
        child = verify_workspace(destination)[2]
        self.assertEqual(len(child.observations),len(self.dataset.observations)+4)
        self.assertEqual([r for r in child.observations if r['split']!='train'],
                         [r for r in self.dataset.observations if r['split']!='train'])
        advance(destination,lambda *a:self.fail('No duplicate execution'))

    def test_migration_refuses_unacknowledged_or_unrelated_source_changes(self):
        from ..active_learning import migration
        self.init()
        baseline = migration.source_hashes()
        destination = Path(self.temp.name)/'without_anchors'
        changed = dict(baseline)
        changed['scripts/prediction/active_learning/quality.py']='changed'
        with patch.object(migration,'source_hashes',return_value=changed):
            with self.assertRaisesRegex(ValueError,'accept-source-changes'):
                migration.disable_anchors(self.root,destination,'Remove controls')
        changed['src/runq_llmforge.c']='changed-kernel'
        with patch.object(migration,'source_hashes',return_value=changed):
            with self.assertRaisesRegex(ValueError,'Non-anchor'):
                migration.disable_anchors(self.root,destination,'Remove controls',True)
        self.assertFalse(destination.exists())

    def test_migration_rejects_tampered_inputs_and_incomplete_restore(self):
        from ..active_learning.migration import disable_anchors
        self.init()
        destination = Path(self.temp.name)/'without_anchors'
        snapshot = self.root/'rounds/0001/attempts/001/original_device_state.json'
        atomic_json(snapshot,dict(settings={}))
        with self.assertRaisesRegex(ValueError,'Restore device'):
            disable_anchors(self.root,destination,'Remove controls')
        atomic_json(snapshot.with_name('restoration.json'),dict(settings_restored=True))
        settings = read_json(self.root/'settings.json')
        settings['anchors']=0
        atomic_json(self.root/'settings.json',settings)
        with self.assertRaisesRegex(ValueError,'input changed'):
            disable_anchors(self.root,destination,'Remove controls')
        self.assertFalse(destination.exists())


if __name__=='__main__':
    unittest.main()
