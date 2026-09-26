"""Predict from architecture CSV using a trusted, locally trained model bundle.

The workload is fixed to the original sweep; prompt/decode lengths are not inputs.
Joblib can execute code: load only bundles you trust.
"""
import argparse
import csv
from pathlib import Path

from ..models.serialization import load_bundle
import torch

from ..data.dataset import architecture_key
from .predictor import bundle_targets, predict_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--configs', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new filename.')
    with args.configs.open() as f:
        configs = list(csv.DictReader(f))
    if not configs:
        parser.error('Architecture CSV is empty.')
    for c in configs:
        try:
            architecture_key(c)
        except (KeyError, TypeError, ValueError) as error:
            parser.error(str(error))
    torch.set_num_threads(4)
    pack = load_bundle(args.model)
    targets = bundle_targets(pack)
    predictions = predict_bundle(pack, configs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['config_id'] + targets)
        writer.writeheader()
        for i, (c, pred) in enumerate(zip(configs, predictions)):
            writer.writerow(dict(config_id=c.get('config_id', str(i)), **dict(zip(targets, pred))))
    print(f'Predicted {len(configs)} architectures: {args.output}')
    print('Fixed sweep workload only; predictions outside the training architecture range are unvalidated.')


if __name__ == '__main__':
    main()
