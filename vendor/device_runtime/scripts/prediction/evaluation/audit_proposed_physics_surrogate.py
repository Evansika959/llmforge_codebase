"""Audit split isolation and compute paired test-row bootstrap comparisons."""
import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import joblib
from ..models.serialization import load_bundle
import numpy as np

from ..config import ROOT

from ..data.io import write_csv
from ..features.physics import FEATURES32
from ..features.physics import TOKEN_GROUPS
from ..features.physics import architecture_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=ROOT / 'scripts/prediction/outputs/proposed_physics_surrogate_936')
    args = parser.parse_args()
    out = args.results
    meta = json.loads((out / 'metadata.json').read_text())
    original = json.loads((ROOT / 'scripts/prediction/outputs/surrogate_comparison_936/metadata.json').read_text())
    assert hashlib.sha256(Path(original['source']).read_bytes()).hexdigest() == meta['source_sha256']
    assert meta['splits'] == original['splits']
    train, val, test = (set(meta['splits'][k]) for k in ['train', 'validation', 'test'])
    assert len(train | val | test) == 936 and not train & val and not train & test and not val & test
    folds = json.loads((out / 'oof_folds.json').read_text())
    for seed in meta['seeds']:
        held = []
        for fold in [f for f in folds if f['seed'] == seed]:
            fit, stop, hold = (set(fold[k]) for k in ['fitting', 'stopping', 'held_out'])
            assert not fit & stop and not fit & hold and not stop & hold
            assert fit | stop | hold == train
            held.extend(hold)
        assert len(held) == len(train) and set(held) == train
        oof = np.load(out / f'oof_seed{seed}.npz')
        assert oof['predictions'].shape == (598, 3, 4)
        assert np.isfinite(oof['predictions']).all()
        assert list(oof['config_ids']) == meta['splits']['train']
        pack = load_bundle(out / f'stacked_ensemble_seed{seed}.joblib')
        assert all((m.coef_ >= 0).all() for m in pack['combiners'])
    assert len(FEATURES32) == 32 and sorted(sum(TOKEN_GROUPS, [])) == list(range(32))
    with (ROOT / 'scripts/sweep/configs/watch5_random_1000_50M_150M_sweep.csv').open() as f:
        configs = {c['config_id']: c for c in csv.DictReader(f)}
    configs = [configs[k] for k in meta['splits']['train']]
    stats = [architecture_stats(c) for c in configs]
    assert all(s['G'] == int(c['q8_group_size']) for c, s in zip(configs, stats))
    assert all(abs(s['params'] / 1e6 - float(c['total_params_M'])) <= .0005001 for c, s in zip(configs, stats))
    x = pack['base']['profile'].transform(configs)
    audit = dict(passed=True, checked='Source hash, original disjoint splits, 15 isolated OOF folds, nonnegative weights, 32 features, architecture parameter counts and quantization group sizes',
                 constant_features_on_training=[n for i, n in enumerate(FEATURES32) if np.ptp(x[:, i]) == 0],
                 train_layer_weight_mib_range=[min(s['layer'] for s in stats) / 2**20, max(s['layer'] for s in stats) / 2**20],
                 full_training_profile=asdict(pack['base']['profile']))
    (out / 'audit.json').write_text(json.dumps(audit, indent=2))
    with (out / 'test_predictions.csv').open() as f:
        rows = list(csv.DictReader(f))
    errors = {}
    for row in rows:
        key = row['model'], row['target'], row['config_id']
        error = abs(float(row['prediction']) - float(row['actual'])) / float(row['actual']) * 100
        errors.setdefault(key, []).append(error)
    rng = np.random.default_rng(20260912)
    draws = rng.integers(0, len(test), size=(10000, len(test)))
    comparisons = []
    for model in ['base_xgboost', 'grouped_transformer', 'stacked_ensemble']:
        for target in meta['targets']:
            delta = np.array([np.mean(errors['original_xgboost', target, k]) - np.mean(errors[model, target, k]) for k in sorted(test)])
            lo, hi = np.quantile(delta[draws].mean(axis=1), [.025, .975])
            comparisons.append(dict(model=model, target=target, mape_improvement_pp=float(delta.mean()), ci95_lower=float(lo), ci95_upper=float(hi)))
    write_csv(out / 'paired_bootstrap.csv', comparisons)
    lines = ['# Integrity checks and uncertainty', '',
             'All integrity assertions passed; details are in audit.json.', '',
             'Paired bootstrap resamples the 188 test architectures 10,000 times. Each architecture contributes its mean absolute percentage error across three seeds. Positive differences favor the proposed model. These are exploratory, unadjusted intervals on a previously inspected split, not fresh confirmatory evidence or learning-curve estimates.', '',
             '| Model | Target | MAPE improvement (percentage points) | 95% interval |',
             '|---|---|---:|---:|']
    for c in comparisons:
        lines.append(f"| {c['model']} | {c['target']} | {c['mape_improvement_pp']:.2f} | [{c['ci95_lower']:.2f}, {c['ci95_upper']:.2f}] |")
    lines += ['', '## Interpretation', '',
              '- No improvement interval is strictly positive. The stack does not demonstrate a reliable gain over the original XGBoost on this split.',
              '- Training-layer weights span 1.28–7.77 MiB: all exceed the 512 KiB shared L2. Cached-layer bytes are constant, and spilled bytes are an affine transformation of layer bytes. This dataset does not identify a whole-layer cache-residency transition; this does not imply individual weight tiles cannot benefit from cache.',
              '- The Transformer has 807,939 parameters and selects epochs 9, 9, and 16. This run alone cannot establish whether more samples will improve it or prove measurement noise is the limiting factor.',
              '- Effective fitted bandwidth is about 1.92 GB/s, compute is 3.01 GMAC/s, per-layer decode overhead is 0.53 ms, and prefill overhead is 3.22 ms. These are regression coefficients, not independent hardware measurements.',
              '- Keep the next acquisition batch untouched until the model and selection rules are fixed, then use it to test cross-batch generalization. Repeated anchor measurements would help distinguish environmental variation from architectural effects.']
    (out / 'AUDIT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(audit, indent=2))
    print(json.dumps(comparisons, indent=2))


if __name__ == '__main__':
    main()
