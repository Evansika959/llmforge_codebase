import unittest
from unittest.mock import patch

from .watch_monitor import assess, sound


class MonitorTests(unittest.TestCase):
    def test_running(self):
        self.assertEqual(assess({'pid': 42, 'status': 'measuring'}, 42, True, True)[0], 'ok')

    def test_disconnect(self):
        self.assertEqual(assess({'pid': 42, 'status': 'uploading'}, 42, True, False)[0], 'alert')

    def test_dead_runner(self):
        self.assertEqual(assess({'pid': 42, 'status': 'uploading'}, 42, False, True)[0], 'alert')

    def test_pause_preserves_reason(self):
        self.assertEqual(assess({'pid': 42, 'status': 'paused', 'message': 'Battery 30%'},
                                42, False, False), ('alert', 'Battery 30%'))

    def test_completion(self):
        self.assertEqual(assess({'pid': 42, 'status': 'complete'}, 42, False, False)[0], 'complete')

    def test_replacement(self):
        self.assertEqual(assess({'pid': 43, 'status': 'measuring'}, 42, True, True)[0], 'alert')

    @patch('scripts.sweep.layerwise.reporting.watch_monitor.subprocess.run')
    def test_three_chimes_and_speech(self, run):
        self.assertEqual(sound('Test alarm'), [])
        self.assertEqual(run.call_count, 4)
        self.assertEqual(run.call_args.args[0], ['/usr/bin/say', 'Test alarm'])


if __name__ == '__main__':
    unittest.main()
