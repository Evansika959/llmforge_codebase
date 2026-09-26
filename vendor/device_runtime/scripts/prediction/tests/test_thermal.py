"""Offline thermal boundary and explicit operating-policy migration tests."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from .test_active_learning import candidate_pool, mock_results
from .test_rounds import example_document
from ..data.dataset import MeasurementDataset
from ..active_learning import thermal
from ..active_learning.engine import initialize, advance, verify_workspace
from ..active_learning.migration import change_temperature
from ..active_learning.storage import read_json, write_configs, digest


def telemetry(temp=44.9,**kwargs):
    return dict(cpu_temp_c=temp,cooling_state=0,max_freq_mhz=1900.,nominal_max_freq_mhz=1900.,voltage_v=4.,**kwargs)


class ThermalTests(unittest.TestCase):
    def legacy(self,samples):
        return SimpleNamespace(check_battery_or_stop=Mock(return_value=70.),
                               get_device_telemetry=Mock(side_effect=samples))

    @patch.object(thermal.time,'sleep')
    def test_exact_45_blocks_and_449_passes_without_stale_reread(self,sleep):
        expected = telemetry()
        legacy = self.legacy([telemetry(45.),expected])
        self.assertIs(thermal.wait_ready(legacy,[]),expected)
        self.assertEqual(legacy.get_device_telemetry.call_count,2)
        self.assertEqual(legacy.check_battery_or_stop.call_count,2)

    @patch.object(thermal.time,'sleep')
    @patch.object(thermal.time,'monotonic',side_effect=[0.,200.,401.])
    def test_valid_cooldown_continues_beyond_old_timeout(self,clock,sleep):
        legacy = self.legacy([telemetry(46.),telemetry()])
        self.assertEqual(thermal.wait_ready(legacy,[])['cpu_temp_c'],44.9)

    @patch.object(thermal.time,'sleep')
    @patch.object(thermal.time,'monotonic',side_effect=[0.,1.,182.])
    def test_bad_telemetry_still_times_out(self,clock,sleep):
        legacy = self.legacy([None,telemetry(float('nan'))])
        with self.assertRaisesRegex(SystemExit,'telemetry unavailable'):
            thermal.wait_ready(legacy,[])

    @patch.object(thermal.time,'sleep')
    def test_battery_guard_remains_active_while_hot(self,sleep):
        legacy = self.legacy([telemetry(46.)])
        legacy.check_battery_or_stop.side_effect=[31.,SystemExit('battery 30%; charge')]
        with self.assertRaisesRegex(SystemExit,'battery 30'):
            thermal.wait_ready(legacy,[])

    @patch.object(thermal.time,'sleep')
    def test_frequency_cooling_and_voltage_guards_remain(self,sleep):
        clamped,cooling,low_voltage=telemetry(40.),telemetry(40.),telemetry(40.)
        clamped['max_freq_mhz']=1200.
        cooling['cooling_state']=1
        low_voltage['voltage_v']=3.4
        legacy = self.legacy([clamped,cooling,low_voltage,telemetry()])
        thermal.wait_ready(legacy,[])
        self.assertEqual(legacy.get_device_telemetry.call_count,4)

    def test_migration_keeps_partial_candidates_and_checkpoint_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/'old'
            destination = Path(temporary)/'under45'
            doc = example_document()
            doc['protocol']['temperature_ceiling']=40.
            doc['protocol']['min_battery_percent']=30
            dataset = MeasurementDataset(doc)
            initialize(root,dataset,candidate_pool(),dict(batch_size=2,members=2,anchors=0,min_params_m=.001,max_params_m=1000.))
            advance(root)
            rd = root/'rounds/0001'
            proposal = read_json(rd/'proposal.json')
            write_configs(rd/'measurements.csv',mock_results(proposal)[:1])
            original = {str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file() and p.name!='RUNNING.lock'}
            change_temperature(root,destination,'User requested strict <45 C')
            settings,state,migrated = verify_workspace(destination)
            self.assertEqual(migrated.fingerprint,dataset.fingerprint)
            self.assertEqual(settings['temperature_ceiling'],45.)
            self.assertEqual(settings['thermal_policy']['comparison'],'lt')
            self.assertEqual(digest(destination/'rounds/0001/measurements.csv'),digest(rd/'measurements.csv'))
            self.assertEqual(read_json(destination/'rounds/0001/proposal.json')['selected'],proposal['selected'])
            self.assertEqual(advance(destination)['completed_rounds'],0)
            self.assertEqual(original,{name:digest(root/name) for name in original})
            with self.assertRaises(ValueError):
                change_temperature(root,destination,'Do not overwrite')


if __name__=='__main__':
    unittest.main()
