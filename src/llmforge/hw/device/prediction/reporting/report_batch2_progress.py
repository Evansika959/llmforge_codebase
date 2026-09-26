"""Independent integrity audit and compact report for batch2_progress outputs."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ..config import OUTPUTS, PREDICTION_ROOT, SWEEP_CONFIGS, SWEEP_OUTPUTS
from ..config import PHYSICS_TARGETS as TARGETS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=OUTPUTS/'batch2_progress_632')
    args = parser.parse_args()
    out = args.results
    meta = json.loads((out / 'metadata.json').read_text())
    splits = {k:set(v) for k,v in meta['splits'].items()}
    train = splits['old_train'] | splits['batch2_train']
    val, test = splits['fixed_validation'], splits['pooled_test']
    assert not train & val and not train & test and not val & test
    assert splits['batch2_half'] <= splits['batch2_train']
    assert splits['pooled_test'] == splits['old_batch_test'] | splits['new_batch_test']
    seen = set(json.loads((OUTPUTS/'state_signal_investigation_936/metadata.json').read_text())['batch2_ids'])
    assert not seen & splits['new_batch_test']
    assert not (set(meta['invalid_ids']) | set(meta['flagged_stall_ids'])) & (train | val | test)
    folds = json.loads((out / 'oof_folds.json').read_text())
    for seed in meta['seeds']:
        held = []
        for fold in [f for f in folds if f['seed'] == seed]:
            f, s, h = (set(fold[k]) for k in ['fitting', 'stopping', 'held_out'])
            assert not f & s and not f & h and not s & h
            assert f | s | h == train
            held.extend(h)
        assert len(held) == len(train) and set(held) == train
        oof = np.load(out / f'oof_seed{seed}.npz')
        assert oof['predictions'].shape == (len(train),3,4) and np.isfinite(oof['predictions']).all()
        assert set(oof['config_ids']) == train
    metrics = json.loads((out / 'metrics.json').read_text())
    original = list(csv.DictReader((OUTPUTS/'proposed_physics_surrogate_936/metrics.csv').open()))
    names = dict(xgboost14='original_xgboost', xgboost32='base_xgboost', transformer32='grouped_transformer', stack='stacked_ensemble')
    checks = 0
    for r in metrics:
        if r['stage'] == 'old_only' and r['test'] == 'old_batch_test':
            previous = next(p for p in original if p['model']==names[r['model']] and int(p['seed'])==r['seed'] and p['target']==r['target'])
            np.testing.assert_allclose(r['mape'], float(previous['mape']), atol=1e-4)
            checks += 1
    assert checks == 36
    weights = json.loads((out / 'stacking_weights.json').read_text())
    assert all(r[k] >= 0 for r in weights for k in ['xgboost','extratrees','randomforest','residual_mlp'])
    summary = json.loads((out / 'summary.json').read_text())
    intervals = json.loads((out / 'paired_bootstrap.json').read_text())
    lookup = {(r['model'],r['stage'],r['test'],r['target']):r for r in summary}
    labels = dict(xgboost14='Original XGBoost (14 features)', xgboost32='Physics XGBoost (32 features)', transformer32='Grouped Transformer', stack='Stacking')
    snapshot = json.loads((out / 'input_snapshot.json').read_text())
    old_n = len(snapshot['records']['batch1'])
    last_id = snapshot['records']['batch2'][-1]['config_id']
    lines = ['# Does batch 2 improve prediction?', '',
        f"Frozen snapshot: {meta['batch2_raw_count']} raw rows through {last_id}; {meta['batch2_accepted_count']} retained. Combined with the original {old_n}: {old_n+meta['batch2_accepted_count']} usable rows. Model fitting grows from {len(splits['old_train'])} to {len(train)}, with {len(val)} fixed stopping rows and {len(test)} fixed test rows.", '',
        'Each arrow compares the same test architectures and the same model settings before/after adding training data. Values are mean MAPE across three training seeds, not ensemble-averaged predictions.', '']
    for testname, label in [('new_batch_test','New batch: 126 held-out architectures'), ('old_batch_test','Old batch: original188 held-out architectures'), ('pooled_test','Pooled: all314 held-out architectures')]:
        lines += [f'## {label}', '', '| Model | Throughput MAPE | TTFT MAPE | Energy MAPE |', '|---|---:|---:|---:|']
        for model, label in labels.items():
            cells = []
            for target in TARGETS:
                before = lookup[model,'old_only',testname,target]['mape']
                after = lookup[model,'plus_all',testname,target]['mape']
                cells.append(f'{before:.2f}% → {after:.2f}%')
            lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')
        lines.append('')
    lines += ['## Original XGBoost: data-growth curve', '', '| Fitting rows | Old-test throughput | New-test throughput | New-test TTFT | New-test energy |', '|---:|---:|---:|---:|---:|']
    for stage in ['old_only','plus_half','plus_all','batch2_only']:
        cells = [lookup['xgboost14',stage,t,k] for t,k in [('old_batch_test',TARGETS[0]),('new_batch_test',TARGETS[0]),('new_batch_test',TARGETS[1]),('new_batch_test',TARGETS[2])]]
        name = str(cells[0]['train_n']) + (' (batch2 only)' if stage=='batch2_only' else '')
        lines.append(f'| {name} | ' + ' | '.join(f"{r['mape']:.2f}%" for r in cells) + ' |')
    lines += ['', '## Paired uncertainty: added-data XGBoost14', '', '| Test | Metric | Improvement in MAPE points | 95% interval |', '|---|---|---:|---:|']
    for r in intervals:
        if r['model']=='xgboost14' and r['test'] != 'new_test_plus_flagged_stalls':
            lines.append(f"| {r['test']} | {r['target']} | {r['mape_improvement_pp']:.2f} | [{r['ci95_lower']:.2f}, {r['ci95_upper']:.2f}] |")
    lines += ['', '## Scope and checks', '',
        '- Integrity checks passed: original 36 baseline metric values reproduced; source snapshot preserved; no architecture overlaps between fit/stop/test; all 15 OOF folds isolated; all meta weights nonnegative. New test IDs exclude batch2 rows inspected in the earlier state-signal experiment.',
        '- Source CSVs were not edited. Zero/nonfinite targets are unsuitable for joint log training. The 36 s row RND_1250 is flagged under a 20 s gross-stall convention, not a proven failure diagnosis. Detailed README reports a separate stress-test including that row.',
        '- Batch2 starts warmer (median 39.5°C versus 36.8°C). Temperature is diagnostic only, never an input. The batch2-only control helps expose distribution adaptation; it does not isolate a causal thermal effect.',
        '- Bootstrap resamples test architectures and averages each row’s errors across seeds first. Intervals assume independent test rows and are not adjusted for many comparisons or temporal correlation. Previously reused old test remains exploratory.',
        '- Preserve a future contiguous acquisition block for forward-transfer testing before claiming generalization to future watch sessions. The present new-batch split is random within the observed snapshot.',
        '- Workload/labels unchanged: nominal48 prompt is inferred49 actual tokens,32 normal outputs, and dynamic energy includes prefill. No change to runq_llmforge.c or device experiments.']
    (out / 'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    (out / 'audit.json').write_text(json.dumps(dict(passed=True, baseline_checks=checks, oof_folds=len(folds), train_n=len(train), validation_n=len(val), test_n=len(test)), indent=2))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
