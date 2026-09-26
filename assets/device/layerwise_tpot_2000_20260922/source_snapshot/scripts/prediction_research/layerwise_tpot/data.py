"""Audited TPOT labels from frozen hardware timings, never predictor inputs."""
import hashlib
import json
from pathlib import Path

import numpy as np

TARGETS = ['tpot_ms', 'ttft_ms', 'dynamic_energy_per_token_mj']
UNITS = ['ms/decode_token', 'ms', 'mJ/output_token']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def label_row(row, raw):
    timing = raw['timing']
    count = timing['decode_tokens']
    duration = timing['end'] - timing['prefill_end']
    if count != 31 or timing['output_tokens'] != 32 or duration <= 0:
        raise ValueError('Expected 32 outputs and 31 positive-duration decode forwards')
    tpot = 1000 * duration / count
    np.testing.assert_allclose(tpot, row['derived_metrics']['tpot_ms'], rtol=1e-10)
    np.testing.assert_allclose(tpot, 1000 / raw['decode_tok_s'], rtol=1e-10)
    for target in ('decode_tok_s', *TARGETS[1:]):
        if raw[target] != row['metrics'][target]:
            raise ValueError(f'Raw/snapshot mismatch: {target}')
    labels = [tpot, raw['ttft_ms'], raw['dynamic_energy_per_token_mj']]
    if not np.isfinite(labels).all() or not (np.asarray(labels) > 0).all():
        raise ValueError('Missing, nonfinite or nonpositive target')
    return labels


def load_labels(snapshot):
    snapshot = Path(snapshot).resolve()
    data = json.loads(snapshot.read_text())
    rows = data['rows']
    if len({r['candidate_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate architecture IDs')
    labels = []
    for row in rows:
        artifact = snapshot.parent / row['artifact_relative_path']
        if sha(artifact) != row['artifact_sha256']:
            raise ValueError(f'Artifact hash mismatch: {artifact}')
        labels.append(label_row(row, json.loads(artifact.read_text())))
    splits = np.array([r['split'] for r in rows])
    if set(splits) != {'train', 'validation', 'test'}:
        raise ValueError('Expected train/validation/test splits')
    groups = [{r['permutation_group'] for r in rows if r['split'] == s}
              for s in ('train', 'validation', 'test')]
    if any(groups[a] & groups[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError('Permutation-group leakage across splits')
    return data, np.asarray(labels, dtype=float), splits
