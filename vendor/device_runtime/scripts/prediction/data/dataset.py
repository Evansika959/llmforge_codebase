"""Versioned measurement datasets with architecture-grouped, frozen holdouts.

Only the architecture dictionary becomes a predictor input. Acquisition metadata
and measured metrics never become features. Appending creates a new value/file.
"""
import copy
import hashlib
import json
import math
from pathlib import Path

from ..config import FEATURES
from ..features.physics import architecture_stats

SCHEMA_VERSION = 1
ENERGY_TARGETS = {'dynamic': 'dynamic_energy_per_token_mj', 'gross': 'gross_energy_per_token_mj'}


def architecture_key(config):
    values = []
    for key in FEATURES[:7] + ['vocab_size']:
        value = float(config.get(key, 50257) if key == 'vocab_size' else config[key])
        if not math.isfinite(value) or value <= 0 or not value.is_integer():
            raise ValueError(f'Invalid architecture dimension: {key}')
        values.append(int(value))
    if values[2] % values[3]:
        raise ValueError('n_h must be divisible by n_kv')
    group = architecture_stats(config)['G']
    if group not in (16, 32, 64) or float(config.get('q8_group_size', group)) != group:
        raise ValueError('Unsupported or mismatched quantization group')
    return tuple(values) + (group,)


class MeasurementDataset:
    def __init__(self, document):
        self._document = copy.deepcopy(document)
        self.validate()

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def to_dict(self):
        return copy.deepcopy(self._document)

    @property
    def fingerprint(self):
        content = json.dumps(self._document, sort_keys=True, allow_nan=False).encode()
        return hashlib.sha256(content).hexdigest()

    @property
    def protocol(self):
        return copy.deepcopy(self._document['protocol'])

    @property
    def observations(self):
        return copy.deepcopy(self._document['observations'])

    @property
    def targets(self):
        return ['decode_tok_s', 'ttft_ms', ENERGY_TARGETS[self.protocol['energy_target']]]

    def validate(self):
        doc = self._document
        if doc.get('schema_version') != SCHEMA_VERSION:
            raise ValueError('Unsupported dataset schema version')
        protocol = doc['protocol']
        for field in ['protocol_id', 'device_id', 'kernel_id']:
            if not isinstance(protocol.get(field), str) or not protocol[field].strip():
                raise ValueError(f'Explicit {field} is required')
        # Existing physics features include fixed-context decode proxies.
        if protocol.get('prompt_tokens') != 49 or protocol.get('output_tokens') != 32:
            raise ValueError('This feature version supports 49 actual prompt / 32 output tokens only')
        if protocol.get('energy_target') not in ENERGY_TARGETS:
            raise ValueError('energy_target must be dynamic or gross')
        observations = doc['observations']
        if not observations:
            raise ValueError('Dataset is empty')
        seen, assignments, config_ids = set(), {}, {}
        for row in observations:
            for key in ['measurement_id', 'config_id', 'round_id']:
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f'Missing {key}')
            if row['measurement_id'] in seen:
                raise ValueError('Duplicate measurement_id: '+row['measurement_id'])
            seen.add(row['measurement_id'])
            split = row['split']
            if split not in ['train', 'validation', 'test']:
                raise ValueError('Unknown split: '+str(split))
            arch = architecture_key(row['architecture'])
            if assignments.setdefault(arch, split) != split:
                raise ValueError('Architecture leakage across dataset splits')
            if config_ids.setdefault(row['config_id'], arch) != arch:
                raise ValueError('config_id maps to conflicting architectures')
            for target in self.targets:
                value = float(row['metrics'][target])
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f'Nonpositive/nonfinite log target: {target}')
        # Fail before training if the immutable split is incomplete.
        if set(assignments.values()) != {'train', 'validation', 'test'}:
            raise ValueError('Dataset needs nonempty train, validation and test partitions')

    def append_training(self, batch):
        """Append a protocol-matched batch without admitting held-out architectures."""
        if batch['protocol'] != self.protocol:
            raise ValueError('Measurement protocol/workload/energy target changed; keep a separate dataset')
        new_rows = copy.deepcopy(batch['observations'])
        if not new_rows:
            raise ValueError('New measurement batch is empty')
        heldout = {architecture_key(r['architecture']) for r in self.observations if r['split'] != 'train'}
        for row in new_rows:
            if row.get('split', 'train') != 'train':
                raise ValueError('An appended active-learning batch must be training-only')
            if architecture_key(row['architecture']) in heldout:
                raise ValueError('Cannot append a validation/test architecture to training')
            row['split'] = 'train'
        document = self.to_dict()
        document['parent_sha256'] = self.fingerprint
        document['observations'].extend(new_rows)
        document.setdefault('ingestions', []).append(dict(
            round_ids=sorted({r['round_id'] for r in new_rows}),
            source_hashes=copy.deepcopy(batch.get('source_hashes', {}))))
        return MeasurementDataset(document)

    def save(self, path):
        path = Path(path)
        content = json.dumps(self._document, indent=2, allow_nan=False) + '\n'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            stream.write(content)
