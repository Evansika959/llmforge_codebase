"""Import an initial legacy snapshot or append a normalized hardware batch."""
import argparse
import json
from pathlib import Path

from .dataset import MeasurementDataset
from .legacy import load_cohort
from .hardware_csv import normalize_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    initial = sub.add_parser('bootstrap', help='Preserve the historical 1100/150/314 split')
    initial.add_argument('--previous', required=True, type=Path)
    initial.add_argument('--protocol', required=True, type=Path, help='Explicit protocol identity JSON')
    initial.add_argument('--output', required=True, type=Path)
    append = sub.add_parser('append', help='New measurements enter training only')
    append.add_argument('--dataset', required=True, type=Path)
    append.add_argument('--batch', required=True, type=Path, help='Normalized batch JSON with protocol and observations')
    append.add_argument('--output', required=True, type=Path)
    hardware = sub.add_parser('import-csv', help='Append a reviewed, completed hardware CSV batch')
    hardware.add_argument('--dataset', required=True, type=Path)
    hardware.add_argument('--configs', required=True, type=Path)
    hardware.add_argument('--measurements', required=True, type=Path)
    hardware.add_argument('--protocol', required=True, type=Path)
    hardware.add_argument('--round-id', required=True)
    hardware.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.operation == 'import-csv':
        dataset = MeasurementDataset.load(args.dataset)
        batch = normalize_batch(args.configs, args.measurements, json.loads(args.protocol.read_text()), args.round_id)
        result = dataset.append_training(batch)
    elif args.operation == 'append':
        dataset = MeasurementDataset.load(args.dataset)
        result = dataset.append_training(json.loads(args.batch.read_text()))
    else:
        snapshot, _, rows, configs, ix, _, gross, _ = load_cohort(args.previous)
        protocol = json.loads(args.protocol.read_text())
        validation, test = set(ix['fixed_validation']), set(ix['pooled_test'])
        observations = []
        for i,(row,config) in enumerate(zip(rows,configs)):
            observations.append(dict(measurement_id='legacy:'+row['config_id'],config_id=row['config_id'],
                round_id='legacy_snapshot',split='test' if i in test else 'validation' if i in validation else 'train',
                architecture=config,metrics=dict(decode_tok_s=float(row['decode_tok_s']),ttft_ms=float(row['ttft_ms']),
                    dynamic_energy_per_token_mj=float(row['dynamic_energy_per_token_mj']),gross_energy_per_token_mj=float(gross[i]))))
        result = MeasurementDataset(dict(schema_version=1,protocol=protocol,observations=observations,
                                         source_hashes=snapshot['hashes']))
    result.save(args.output)
    print(f'Saved {len(result.observations)} observations: {args.output} ({result.fingerprint})')
