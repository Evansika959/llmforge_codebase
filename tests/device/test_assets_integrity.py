"""Integrity of the shipped Pixel Watch 5 assets after the release rewrite."""
import copy
import hashlib
import json
import re
import unittest
from pathlib import Path

from llmforge import paths
from llmforge.hw.device.prediction.data.dataset import MeasurementDataset
from llmforge.hw.device.prediction.models.serialization import load_bundle

ASSETS = paths.DEVICE_ASSETS / 'pixel_watch5'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load(relative):
    return json.loads((ASSETS / relative).read_text())


class AssetIntegrityTests(unittest.TestCase):
    def test_manifest_digests_match_files(self):
        manifest = load('manifest.json')
        for relative, sha in manifest['artifact_sha256'].items():
            self.assertEqual(digest(ASSETS / relative), sha, relative)
        for relative, entry in manifest['copied_files'].items():
            self.assertEqual(digest(ASSETS / relative), entry['sha256'], relative)
        self.assertNotIn('provenance/device_identity.json', manifest['artifact_sha256'])
        self.assertFalse((ASSETS / 'provenance/device_identity.json').exists())

    def test_fingerprints_agree_across_records_and_bundles(self):
        best = MeasurementDataset.load(ASSETS / 'data/dataset_best_model.json')
        latest = MeasurementDataset.load(ASSETS / 'data/dataset_latest.json')
        manifest, state = load('manifest.json'), load('provenance/state.json')
        self.assertEqual(manifest['best_model_dataset_sha256'], best.fingerprint)
        self.assertEqual(manifest['latest_dataset_sha256'], latest.fingerprint)
        self.assertEqual(state['dataset_sha256'], latest.fingerprint)
        self.assertEqual(manifest['source_state'], state)
        self.assertEqual(load_bundle(ASSETS / 'models/predictor_best.joblib')['dataset_sha256'], best.fingerprint)
        self.assertEqual(load_bundle(ASSETS / 'models/predictor_latest.joblib')['dataset_sha256'], latest.fingerprint)
        contract = load('provenance/contract.json')
        self.assertEqual(contract['inputs']['provenance/energy_transition.json'],
                         digest(ASSETS / 'provenance/energy_transition.json'))

    def test_round_chain_rebuilds_the_recorded_fingerprints(self):
        best, latest = load('data/dataset_best_model.json'), load('data/dataset_latest.json')
        recorded = [c['dataset_sha256'] for c in load('evaluation/model_selection.json')['candidates']]
        rows, first_round = best['observations'], set(best['ingestions'][0]['round_ids'])
        start = len(rows)
        while start and rows[start - 1]['round_id'] in first_round:
            start -= 1
        initial = copy.deepcopy(best)
        del initial['parent_sha256'], initial['ingestions']
        initial['observations'] = initial['observations'][:start]
        documents, position = [initial], start
        for ingestion in latest['ingestions']:
            end = position
            while end < len(latest['observations']) and latest['observations'][end]['round_id'] in ingestion['round_ids']:
                end += 1
            document = copy.deepcopy(documents[-1])
            document['parent_sha256'] = fingerprint(documents[-1])
            document['observations'].extend(latest['observations'][position:end])
            document.setdefault('ingestions', []).append(ingestion)
            documents.append(document)
            position = end
        self.assertEqual([fingerprint(d) for d in documents], recorded)
        self.assertEqual(documents[1], best)
        self.assertEqual(documents[-1], latest)

    def test_no_machine_paths_or_device_identity(self):
        pattern = re.compile(r'/Users/|/home/|hardware_serial|ro\.serialno|ro\.build\.fingerprint')
        for path in ASSETS.rglob('*'):
            if path.is_file() and path.suffix in ('.json', '.csv', '.md'):
                self.assertIsNone(pattern.search(path.read_text()), path.name)


if __name__ == '__main__':
    unittest.main()
