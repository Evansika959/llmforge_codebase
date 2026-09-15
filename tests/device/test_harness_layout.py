"""The measurement harness imports without side effects and never picks a device implicitly."""
import importlib
import os
import sys
import unittest
from unittest.mock import patch


class HarnessLayoutTests(unittest.TestCase):
    def test_modules_import_without_changing_directory(self):
        cwd = os.getcwd()
        for name in ['llmforge.hw.device.measurement.sweep.run_sweep_configs',
                     'llmforge.hw.device.prediction.active_learning.worker',
                     'llmforge.hw.device.package_deliverable']:
            importlib.import_module(name)
        self.assertEqual(os.getcwd(), cwd)

    def test_sweep_requires_an_explicit_serial(self):
        sweep = importlib.import_module('llmforge.hw.device.measurement.sweep.run_sweep_configs')
        environment = {k: v for k, v in os.environ.items() if k != 'ANDROID_SERIAL'}
        with patch.dict(os.environ, environment, clear=True), patch.object(sys, 'argv', ['run_sweep_configs']):
            with self.assertRaises(SystemExit) as stop:
                sweep.main()
        self.assertEqual(stop.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
