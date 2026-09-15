"""Export an audited, hardware-free snapshot without modifying an active-learning workspace.

    python -m llmforge.hw.device.package_deliverable --source WORKSPACE --destination NEW_DIRECTORY

Joblib inputs must be trusted local checkpoints. Test labels are never evaluated. Recorded
provenance paths are relative to the directory that holds the workspace. The device identity record
is left out unless --include-device-identity is given.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import shutil

from llmforge.hw.device.prediction.active_learning.engine import _evaluate
from llmforge.hw.device.prediction.active_learning.migration import _source_state
from llmforge.hw.device.prediction.active_learning.storage import (atomic_json, digest, locked,
                                                                    portable_path)
from llmforge.hw.device.prediction.config import OUTPUTS
from llmforge.hw.device.prediction.data.dataset import MeasurementDataset
from llmforge.hw.device.prediction.models.serialization import load_bundle


def export(source, destination, include_device_identity=False):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists() or source in destination.parents:
        raise ValueError('Use a new destination outside the source workspace')
    base = source.parent
    with locked(source/'RUNNING.lock'):
        contract, state, latest, _ = _source_state(source, False)
        if latest.protocol['energy_target'] != 'dynamic':
            raise ValueError('This delivery is for the baseline-subtracted dynamic stage')
        candidates, files = [], {}
        paths = [(0, source/'initial_model.joblib', source/'dataset_000.json')]
        paths += [(n, source/f'rounds/{n:04d}/predictor_after.joblib',
                   source/f'rounds/{n:04d}/dataset_after.json')
                  for n in range(1, state['completed_rounds']+1)]
        holdouts = [r for r in latest.observations if r['split'] != 'train']
        for number, model_path, data_path in paths:
            dataset = MeasurementDataset.load(data_path)
            model = load_bundle(model_path)
            if (model['dataset_sha256'] != dataset.fingerprint or
                    model['targets'] != latest.targets or model['protocol'] != latest.protocol or
                    [r for r in dataset.observations if r['split'] != 'train'] != holdouts):
                raise ValueError('Checkpoint/dataset/holdout mismatch')
            scores = _evaluate(model, dataset, 'validation')
            files[number] = (model_path, data_path)
            candidates.append(dict(round=number, model=portable_path(model_path, base),
                dataset=portable_path(data_path, base), dataset_sha256=dataset.fingerprint,
                validation=scores, mean_mape=sum(m['mape'] for m in scores)/len(scores)))
        best = min(candidates, key=lambda c: (c['mean_mape'], c['round']))
        best_model, best_dataset = files[best['round']]
        destination.mkdir(parents=True)
        copied = {}

        def include(path, relative):
            target = destination/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            before = digest(path)
            shutil.copy2(path, target)
            if digest(target) != before or digest(path) != before:
                raise ValueError('Source or copy changed: '+str(path))
            copied[relative] = dict(source=portable_path(path, base), sha256=before)

        include(source/state['current_dataset'], 'data/dataset_latest.json')
        include(best_dataset, 'data/dataset_best_model.json')
        include(best_model, 'models/predictor_best.joblib')
        include(files[candidates[-1]['round']][0], 'models/predictor_latest.joblib')
        names = ['state.json', 'settings.json', 'contract.json']
        if include_device_identity:
            names.append('device_identity.json')
        for name in names:
            include(source/name, 'provenance/'+name)
        include(source/'provenance/energy_transition.json', 'provenance/energy_transition.json')
        for number in range(1, state['completed_rounds']+1):
            include(source/f'rounds/{number:04d}/measurements.csv',
                    f'provenance/measurements/round_{number:04d}.csv')
        report = dict(scope='Current real-watch dynamic stage, initial checkpoint and all completed rounds',
            criterion='Lowest arithmetic mean of the three fixed-validation MAPEs; tie: earliest round',
            best_round=best['round'], candidates=candidates,
            test_evaluated=False, warning='Validation-selected best within this stage, not a global or unbiased test-set winner.')
        atomic_json(destination/'evaluation/model_selection.json', report)
        notes = ['Original workspace retained; this package is not an AL resume workspace.',
                 'Best checkpoint was not trained on the latest dataset: use its matching dataset for checkpoint initialization.',
                 'Source protocol retains historical 40 C; provenance/settings.json records the strict <45 C amendment.',
                 'Zero dynamic-energy training label retained in energy_transition.json quarantine, not log-trained.',
                 'Partial uncommitted round is not included in the validated dataset. Raw attempts remain in source workspace.']
        if not include_device_identity:
            notes.append('Device identity record excluded from this export.')
        manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), source_workspace=source.name,
            source_state=state, protocol=latest.protocol, targets=latest.targets,
            latest_dataset_sha256=latest.fingerprint, latest_counts=dict(Counter(r['split'] for r in latest.observations)),
            best_model_round=best['round'], best_model_dataset_sha256=best['dataset_sha256'],
            best_model_counts=dict(Counter(r['split'] for r in MeasurementDataset.load(best_dataset).observations)),
            model_selection=report['criterion'], copied_files=copied,
            source_hashes=contract['source_hashes'],
            package_versions={name:version(name) for name in ['numpy','scipy','scikit-learn','xgboost','torch','joblib']},
            notes=notes)
        manifest['artifact_sha256'] = {str(p.relative_to(destination)):digest(p)
                                      for p in destination.rglob('*') if p.is_file()}
        atomic_json(destination/'manifest.json', manifest)
        print(f'Exported {len(latest.observations)} observations; best round {best["round"]}; latest round {state["completed_rounds"]}')
        print(f'Best mean validation MAPE: {best["mean_mape"]:.4f}%')
        print(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', type=Path, default=OUTPUTS/'active_learning_watch5_dynamic_under45')
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--include-device-identity', action='store_true',
                        help='Also copy device_identity.json, which holds device serial and build identifiers')
    args = parser.parse_args()
    export(args.source, args.destination, args.include_device_identity)


if __name__ == '__main__':
    main()
