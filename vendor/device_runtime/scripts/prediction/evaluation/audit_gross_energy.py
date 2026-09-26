"""Independent checkpoint/metric checks and post-hoc state diagnostics (no fitting)."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
from ..models.serialization import load_bundle
import numpy as np

from ..config import ROOT

from ..data.legacy import load_cohort

from ..data.io import write_json

from ..config import FEATURES

from ..features.analytic import physical_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'scripts/prediction/outputs/gross_energy_comparison_1564')
    args = parser.parse_args()
    out = args.output
    snapshot, _, rows, configs, ix, _, gross, _ = load_cohort(ROOT/'scripts/prediction/outputs/batch2_progress_632')
    predictions = json.loads((out/'predictions.json').read_text())
    summary = json.loads((out/'summary.json').read_text())
    index = {r['config_id']: i for i,r in enumerate(rows)}
    x14 = np.array([[physical_features(c)[0][f] for f in FEATURES] for c in configs],dtype=np.float32)
    checks = 0
    for seed in [42,123,2026]:
        for name in ['xgboost14','xgboost32']:
            filename = f'xgboost14_plus_all_seed{seed}.joblib' if name=='xgboost14' else f'xgboost32_seed{seed}.joblib'
            pack = load_bundle(out/filename)
            x = x14 if name=='xgboost14' else pack['profile'].transform(configs)
            pred = np.exp(pack['models'][2].predict(x))
            records = [r for r in predictions if (r['model'],r['energy_target'],r['stage'],r['test'],r['metric'],r['seed']) ==
                       (name,'gross','plus_all','pooled_test','energy',seed)]
            assert len(records)==314
            for r in records:
                i = index[r['config_id']]
                np.testing.assert_allclose(r['actual'],gross[i],rtol=1e-7)
                np.testing.assert_allclose(r['prediction'],pred[i],rtol=1e-7)
                checks += 1
    error_correlations = {}
    residual = {}
    for kind in ['gross','dynamic']:
        rs = [r for r in predictions if (r['model'],r['energy_target'],r['stage'],r['test'],r['metric']) ==
              ('xgboost14',kind,'plus_all','pooled_test','energy')]
        grouped = {}
        for r in rs:
            grouped.setdefault(r['config_id'],[]).append(np.log(r['prediction']/r['actual']))
        ids = list(grouped)
        residual[kind] = {k:float(np.mean(grouped[k])) for k in ids}
        error_correlations[kind] = {k:float(np.corrcoef([residual[kind][i] for i in ids],
                                    [float(rows[index[i]][k]) for i in ids])[0,1])
                                    for k in ['baseline_power_w','active_power_w','temp_cpu_start_c','duration_s']}
    bands = []
    for label,lo,hi in [('below_150mW',0,.15),('150_to_200mW',.15,.2),('at_least_200mW',.2,float('inf'))]:
        ids = [rows[i]['config_id'] for i in ix['pooled_test'] if lo <= float(rows[i]['baseline_power_w']) < hi]
        bands.append(dict(baseline_band=label,n=len(ids),
                          median_prediction_to_actual_ratio={k:float(np.median([np.exp(residual[k][i]) for i in ids])) for k in residual}))
    # Recompute headline metrics from the saved individual predictions.
    for r in summary:
        records = [p for p in predictions if all(p[k]==r[k] for k in ['model','energy_target','stage','test','metric'])]
        mape = np.mean([abs(p['prediction']/p['actual']-1)*100 for p in records])
        assert abs(mape-r['mape']) < 1e-5
    actual_hashes = {k:hashlib.sha256(Path(p).read_bytes()).hexdigest() for k,p in snapshot['paths'].items()}
    assert actual_hashes==snapshot['hashes'], 'Current measurement/config sources differ from frozen snapshot'
    diagnostic = dict(saved_tree_prediction_checks=checks,summary_checks=len(summary),
        current_sources_match_frozen_hashes=True,
        baseline_power_w_quantiles=dict(zip(['p10','median','p90'],np.quantile([float(r['baseline_power_w']) for r in rows],[.1,.5,.9]).tolist())),
        residual_state_correlations=error_correlations,baseline_bands=bands,
        notes='Post-hoc diagnostic only: observed power, temperature and batch/order are never predictor inputs. Correlation is not causation. Baseline bands are descriptive, not filtering/tuning rules. Signed residual is mean across seeds of log(prediction/actual).')
    write_json(out/'diagnostics.json',diagnostic)
    print(json.dumps(diagnostic,indent=2))


if __name__=='__main__':
    main()
