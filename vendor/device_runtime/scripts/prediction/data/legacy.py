"""Read the historical fixed 1564-architecture cohort for reproducibility only."""
import json
import numpy as np
from ..config import FEATURES

def load_cohort(previous):
    snapshot = json.loads((previous / 'input_snapshot.json').read_text())
    meta = json.loads((previous / 'metadata.json').read_text())
    raw = snapshot['records']['batch1'] + snapshot['records']['batch2']
    lookup = {r['config_id']: r for r in raw}
    assert len(lookup) == len(raw), 'Duplicate measurement IDs'
    split = meta['splits']
    ids = split['old_train'] + split['batch2_train'] + split['fixed_validation'] + split['pooled_test']
    assert len(ids) == len(set(ids)) == 1564
    rows = [lookup[k] for k in ids]
    cfg = {c['config_id']: c for k in ['config1', 'config2'] for c in snapshot['records'][k]}
    configs = [cfg[k] for k in ids]
    assert len({tuple(int(c[k]) for k in FEATURES[:7]) for c in configs}) == len(rows)
    for r, c in zip(rows, configs):
        assert all(int(r[k]) == int(c[k]) for k in FEATURES[:7])
        assert not r['notes']
    index = {k: i for i, k in enumerate(ids)}
    ix = {k: np.array([index[a] for a in v]) for k, v in split.items()
          if k != 'new_test_plus_flagged_stalls'}
    y = np.array([[float(r[k]) for k in ['decode_tok_s', 'ttft_ms', 'dynamic_energy_per_token_mj']]
                  for r in rows], dtype=np.float32)
    gross = np.array([float(r['total_energy_j']) * 1000 / 32 for r in rows], dtype=np.float32)
    assert np.isfinite(y).all() and (y > 0).all()
    assert np.isfinite(gross).all() and (gross > 0).all()
    # CSV power/duration/energy fields were independently rounded to 4 decimals.
    duration = np.array([float(r['duration_s']) for r in rows])
    active = np.array([float(r['active_power_w']) for r in rows])
    baseline = np.array([float(r['baseline_power_w']) for r in rows])
    total = np.array([float(r['total_energy_j']) for r in rows])
    gross_tolerance_j = .00005 * (duration + active) + .000051
    assert np.all(abs(total - active * duration) <= gross_tolerance_j)
    reconstructed = np.maximum(active - baseline, 0) * duration * 1000 / 32
    dynamic_tolerance_mj = (.0001 * duration + .00005 * abs(active - baseline)) * 1000 / 32 + .0001
    assert np.all(abs(reconstructed - y[:, 2]) <= dynamic_tolerance_mj)
    audit = dict(rows=len(rows), unique_architectures=len(rows),
                 max_gross_reconciliation_error_j=float(max(abs(total-active*duration))),
                 max_dynamic_reconciliation_error_mj=float(max(abs(reconstructed-y[:, 2]))),
                 excluded_from_frozen_snapshot=[r['config_id'] for r in raw if r['config_id'] not in index])
    return snapshot, meta, rows, configs, ix, y, gross, audit
