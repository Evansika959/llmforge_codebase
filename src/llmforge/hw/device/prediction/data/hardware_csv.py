"""Normalize one completed legacy-format hardware CSV batch, without modifying it.

Device/protocol identity must be supplied by the caller. No silent filtering,
baseline correction, deduplication or guessing of missing architecture fields.
"""
import csv
import hashlib
from pathlib import Path

from ..config import FEATURES
from .dataset import architecture_key


def normalize_batch(configs_path, measurements_path, protocol, round_id):
    if not round_id.strip():
        raise ValueError('round_id is required')
    with Path(configs_path).open(newline='') as stream:
        config_rows = list(csv.DictReader(stream))
    configs = {r['config_id']:r for r in config_rows}
    if len(configs) != len(config_rows):
        raise ValueError('Duplicate architecture config_id in configuration CSV')
    with Path(measurements_path).open(newline='') as stream:
        measurements = list(csv.DictReader(stream))
    if not measurements:
        raise ValueError('Measurement CSV is empty')
    rows = []
    for row in measurements:
        if None in row or any(value is None for value in row.values()):
            raise ValueError('Malformed/incomplete CSV row')
        config = configs[row['config_id']]
        architecture_key(config)
        if any(float(row[k]) != float(config[k]) for k in FEATURES[:7]):
            raise ValueError('Measured architecture differs from the configuration CSV')
        if row.get('notes', '').strip() or row.get('quality_flags', '').strip() not in ('', 'none'):
            raise ValueError('Flagged measurement requires explicit review before import')
        if row.get('output_tokens') and int(row['output_tokens']) != protocol['output_tokens']:
            raise ValueError('Measured output length differs from the declared protocol')
        values = {k:float(row[k]) for k in ['decode_tok_s','ttft_ms']}
        if row.get('dynamic_energy_per_token_mj'):
            values['dynamic_energy_per_token_mj'] = float(row['dynamic_energy_per_token_mj'])
        if row.get('total_energy_j'):
            values['gross_energy_per_token_mj'] = float(row['total_energy_j'])*1000/protocol['output_tokens']
        # Repeats require a run/measurement ID or explicit repeat column.
        identifier = row.get('measurement_id') or row.get('run_id') or row['config_id']+':'+row.get('repeat','1')
        rows.append(dict(measurement_id=round_id+':'+identifier,config_id=row['config_id'],round_id=round_id,
                         split='train',architecture=config,metrics=values))
    return dict(protocol=protocol,observations=rows,
                source_hashes={str(Path(p)):hashlib.sha256(Path(p).read_bytes()).hexdigest()
                               for p in [configs_path,measurements_path]})
