"""Offline integration checks against the prepared replay, never the live DB."""
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
import unittest

from scripts.sweep.layerwise.database import digest
from scripts.sweep.layerwise.gs64_resweep import DEFAULT_OUTPUT, validate, report


@unittest.skipUnless((DEFAULT_OUTPUT / 'replay_manifest.json').exists(), 'Prepared GS64 integration fixture unavailable')
class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name) / 'replay'
        self.folder.mkdir()
        for name in ['candidates.sqlite','replay_manifest.json','search_space.json','architectures.jsonl']:
            shutil.copy2(DEFAULT_OUTPUT / name, self.folder / name)
        shutil.copytree(DEFAULT_OUTPUT / 'baseline', self.folder / 'baseline')

    def tearDown(self):
        self.temp.cleanup()

    def test_exact_subset_and_versioned_reports(self):
        before = digest(self.folder / 'baseline/dataset_2000.json')
        result = validate(self.folder / 'candidates.sqlite')
        self.assertEqual(result['candidates'], 664)
        self.assertEqual(result['splits'], dict(train=536, validation=64, test=64))
        a, b = report(self.folder), report(self.folder)
        self.assertNotEqual(a['report'], b['report'])
        self.assertTrue(Path(a['report']).exists())
        self.assertEqual(before, digest(self.folder / 'baseline/dataset_2000.json'))

    def test_header_tamper_rejected(self):
        with sqlite3.connect(self.folder / 'candidates.sqlite') as con:
            con.execute('UPDATE candidates SET q8_group_size=32')
        with self.assertRaisesRegex(ValueError, 'header'):
            validate(self.folder / 'candidates.sqlite')

    def test_db_spec_tamper_rejected(self):
        with sqlite3.connect(self.folder / 'candidates.sqlite') as con:
            con.execute("UPDATE metadata SET value_json='{}' WHERE key='search_space'")
        with self.assertRaisesRegex(ValueError, 'specification'):
            validate(self.folder / 'candidates.sqlite')

    def test_weight_hash_tamper_rejected(self):
        path = self.folder / 'replay_manifest.json'
        value = json.loads(path.read_text())
        value['expected_model_hashes'][next(iter(value['expected_model_hashes']))] = '0'*64
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'weight hash'):
            validate(self.folder / 'candidates.sqlite')


if __name__ == '__main__':
    unittest.main()
