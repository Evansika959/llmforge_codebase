"""Version a layerwise dataset, then package validation-selected fitted models."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import zipfile

import joblib
import numpy as np

from scripts.prediction_research.layerwise_refit.data import snapshot, TARGETS
from scripts.prediction_research.layerwise_refit.features import matrix
from scripts.prediction_research.ranking_metrics import score
from .training import predict


ROOT = Path(__file__).resolve().parents[3]
CAMPAIGNS = ['watch5_layerwise_500_v1', 'watch5_layerwise_500_variable_kv_v2',
             'watch5_layerwise_1000_variable_kv_v3/part1',
             'watch5_layerwise_1000_variable_kv_v3/part2']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def freeze(destination, expected_rows):
    folders = [ROOT / 'scripts/sweep/outputs' / name for name in CAMPAIGNS]
    data = snapshot(folders)
    if len(data['rows']) != expected_rows:
        raise ValueError(f"Expected {expected_rows} accepted rows, got {len(data['rows'])}")
    for row in data['rows']:
        raw = json.loads(Path(row['artifact_path']).read_text())
        if any(raw[t] != row['metrics'][t] for t in TARGETS):
            raise ValueError('Raw/database metric mismatch')
    destination.mkdir(parents=True, exist_ok=False)
    write(destination / 'dataset_snapshot.json', data)
    archives = []
    # Back up each database consistently, including superseded attempts, and
    # ensure it agrees with the earlier accepted-row snapshot.
    with zipfile.ZipFile(destination / 'measurement_evidence.zip', 'x',
                         compression=zipfile.ZIP_DEFLATED) as archive:
        for index, folder in enumerate(folders, 1):
            bundle = destination / 'databases' / f'campaign{index}'
            bundle.mkdir(parents=True)
            database = bundle / 'candidates.sqlite'
            with sqlite3.connect((folder / 'candidates.sqlite').as_uri() + '?mode=ro', uri=True) as src:
                with sqlite3.connect(database) as dst:
                    src.backup(dst)
                    assert dst.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
                    records = dst.execute("""SELECT m.candidate_id, m.artifact_path,
                        m.decode_tok_s, m.ttft_ms, m.dynamic_energy_per_token_mj
                        FROM measurements m JOIN jobs j USING(candidate_id)
                        WHERE m.accepted=1 AND j.status='complete'""").fetchall()
                    backed = {r[0]: (str(folder / r[1] / 'result.json'), *r[2:]) for r in records}
                    frozen = {r['candidate_id']: (r['artifact_path'], *(r['metrics'][t] for t in TARGETS))
                              for r in data['rows'] if r['source'] == str(folder)}
                    if backed != frozen:
                        raise ValueError('Registry changed during export; discard incomplete release')
            shutil.copy2(folder / 'hardware_contract.json', bundle / 'hardware_contract.json')
            count = 0
            # All attempt traces/results/logs, including excluded/retested attempts.
            # Synthetic weight files are reproducible and deliberately excluded.
            for tree in ('attempts', 'retests'):
                for path in sorted((folder / tree).rglob('*')):
                    if path.is_file() and path.suffix in ('.json', '.csv', '.log', '.txt'):
                        archive.write(path, f'campaign{index}/{path.relative_to(folder).as_posix()}')
                        count += 1
            archives.append(dict(campaign=index, source=str(folder), evidence_files=count,
                                 database_sha256=sha(database)))
    metadata = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    rows=len(data['rows']), splits=dict(Counter(r['split'] for r in data['rows'])),
                    valid_targets={t: sum(r['metrics'][t] is not None and
                                         np.isfinite(r['metrics'][t]) and r['metrics'][t] > 0
                                         for r in data['rows']) for t in TARGETS},
                    dataset_sha256=sha(destination / 'dataset_snapshot.json'),
                    evidence_sha256=sha(destination / 'measurement_evidence.zip'),
                    campaigns=archives,
                    policy='Accepted complete rows only; original registries unchanged; old attempts retained in backups/evidence.')
    # Convert NumPy scalar counts for strict JSON serialization.
    metadata['valid_targets'] = {k: int(v) for k, v in metadata['valid_targets'].items()}
    write(destination / 'data_manifest.json', metadata)
    print(json.dumps(metadata, indent=2), flush=True)


def package(destination, run):
    if sha(destination / 'dataset_snapshot.json') != sha(run / 'dataset_snapshot.json'):
        raise ValueError('Training/release dataset mismatch')
    result = json.loads((run / 'metrics.json').read_text())
    manifest = json.loads((run / 'manifest.json').read_text())
    validation = result['validation']
    family = min(validation, key=lambda name: float(np.mean(validation[name])))
    seed_index = int(np.argmin(np.mean(validation[family], axis=1)))
    seed = manifest['seeds'][seed_index]
    checkpoint = run / f'{family}_seed{seed}.joblib'
    data = json.loads((destination / 'dataset_snapshot.json').read_text())
    test = [r for r in data['rows'] if r['split'] == 'test']
    pack = joblib.load(checkpoint)
    x, _ = matrix([r['architecture'] for r in test], pack['features'])
    predicted = predict(pack, x)
    stored = np.load(run / 'test_predictions.npz', allow_pickle=False)
    ids = np.array([r['candidate_id'] for r in test])
    np.testing.assert_array_equal(ids, stored['ids'])
    np.testing.assert_array_equal(predicted, stored[family][seed_index])
    actual = stored['actual']
    selected_scores = {}
    for j, target in enumerate(TARGETS):
        valid = np.isfinite(actual[:, j])
        selected_scores[target] = score(actual[valid, j], predicted[valid, j], ids[valid],
                                        j == 0, min(32, int(valid.sum())))
    selection = dict(model=family, seed=seed, original_checkpoint=checkpoint.name,
                     rule='Family: minimum mean validation MAPE across three targets and three seeds; seed: minimum mean validation MAPE within selected family. No test-based selection.',
                     family_validation_mape={n: float(np.mean(v)) for n, v in validation.items()},
                     selected_validation_mape=validation[family][seed_index],
                     test_metrics=selected_scores, dataset_sha256=manifest['dataset_sha256'],
                     model_sha256=sha(checkpoint), units=['tokens/s', 'ms', 'mJ/output token'],
                     deployment='Offline release only; no active-learning workspace/model state changed.')
    evaluation = destination / 'evaluation'
    models = destination / 'models'
    evaluation.mkdir(); models.mkdir()
    for path in run.glob('*.joblib'):
        shutil.copy2(path, models / path.name)
    shutil.copy2(checkpoint, models / 'predictor_final.joblib')
    for name in ('metrics.json', 'manifest.json', 'test_predictions.npz',
                 'checkpoint_hashes.json', 'training_progress.json', 'README.md'):
        shutil.copy2(run / name, evaluation / name)
    for path in run.glob('*_history.json'):
        shutil.copy2(path, evaluation / path.name)
    shutil.copytree(run / 'source_snapshot', destination / 'source_snapshot')
    write(evaluation / 'model_selection.json', selection)
    hashes = {str(p.relative_to(destination)): sha(p)
              for p in sorted(destination.rglob('*')) if p.is_file() and p.name != 'release_hashes.json'}
    write(destination / 'release_hashes.json', hashes)
    print(json.dumps(selection, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['freeze', 'package'])
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--expected-rows', type=int, default=2000)
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze(args.destination.resolve(), args.expected_rows)
    else:
        if args.run is None:
            parser.error('package requires --run')
        package(args.destination.resolve(), args.run.resolve())


if __name__ == '__main__':
    main()
