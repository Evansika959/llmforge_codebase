import unittest
from .active_watch_monitor import assess


class MonitorTests(unittest.TestCase):
    def test_running(self):
        self.assertIsNone(assess({'pid': 123}, 123, True, True))

    def test_terminal_and_disconnect(self):
        for record, runner, supervisor, connected in [
            ({'pid': 123, 'ended_utc': 'now', 'returncode': 0}, True, True, True),
            ({'pid': 123}, False, True, True),
            ({'pid': 123}, True, False, True),
            ({'pid': 123}, True, True, False),
            ({'pid': 456}, True, True, True),
        ]:
            with self.subTest(record=record, runner=runner, supervisor=supervisor, connected=connected):
                self.assertIsNotNone(assess(record, 123, runner, supervisor, connected))


if __name__ == '__main__':
    unittest.main()
