"""Resume two sequential 500-point registries; stop on any hardware pause."""
import argparse
from pathlib import Path

from scripts.prediction.active_learning.storage import locked
from scripts.sweep.layerwise.database import ROOT, validate_database
from scripts.sweep.layerwise.runner import run, status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serial', required=True)
    parser.add_argument('--acknowledge-protocol', action='store_true', required=True)
    args = parser.parse_args()
    folder = ROOT / 'scripts/sweep/outputs/watch5_layerwise_1000_variable_kv_v3'
    import torch
    torch.set_num_threads(4)
    with locked(folder / 'QUEUE.lock'):
        for part in ('part1', 'part2'):
            target = folder / part
            validate_database(target / 'candidates.sqlite')
            completed = status(target)['jobs'].get('complete', 0)
            print(f'{part}: {completed}/500 complete', flush=True)
            if completed == 500:
                continue
            if (folder / 'STOP').exists():
                print('Queue STOP requested; progress preserved.', flush=True)
                return 2
            result = run(target, args.serial, 500, args.acknowledge_protocol)
            if result or status(target)['jobs'].get('complete', 0) != 500:
                print('Queue paused; resume this same command after resolving the cause.', flush=True)
                return result or 2
        print('All 1000 measurements completed.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
