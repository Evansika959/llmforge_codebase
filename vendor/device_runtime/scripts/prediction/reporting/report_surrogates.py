"""Summarize held-out errors; paired bootstrap conditional on the fixed test split."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from ..config import TARGETS
from ..data.io import write_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    with (args.output/'test_predictions.csv').open() as f:
        predictions=list(csv.DictReader(f))
    meta=json.loads((args.output/'metadata.json').read_text())
    ids=meta['splits']['test']
    rng=np.random.default_rng(20260909)
    boot=rng.integers(0,len(ids),size=(5000,len(ids)))
    comparisons=[]
    for target in TARGETS:
        errors={}
        for family in ['xgboost','transformer']:
            records=[r for r in predictions if r['target']==target and r['model']==family]
            errors[family]=np.array([np.mean([abs(float(r['prediction'])-float(r['actual']))/float(r['actual'])*100 for r in records if r['config_id']==i]) for i in ids])
        delta=errors['transformer']-errors['xgboost']
        ci=np.quantile(delta[boot].mean(1),[.025,.975])
        comparisons.append(dict(target=target,transformer_minus_xgboost_mape_pp=float(delta.mean()),ci95_lower=float(ci[0]),ci95_upper=float(ci[1])))
    write_csv(args.output/'paired_bootstrap.csv',comparisons)
    with (args.output/'summary.csv').open() as f: summary=list(csv.DictReader(f))
    lines=['# XGBoost vs small Transformer on 936 architectures','',
           'Architecture-only inputs. Fixed split: 598 training, 150 validation, 188 test architectures, stratified by Q8 group size. All 936 architectures are unique.',
           'Both models have four validation-selected hyperparameter candidates, followed by three training seeds (42, 123, 2026) on the same split. Test data are used only for final evaluation.',
           'XGBoost fits one model per target. The Transformer uses feature tokens, two attention layers, four heads, and a joint three-target head. Both learn log targets; Transformer feature/target normalization uses training rows only.',
           '', '| Model | Target | MAE | MAPE (%) | R² | Spearman |', '|---|---|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['model']} | {r['target']} | {float(r['mae_mean']):.3f} | {float(r['mape_percent_mean']):.2f} ± {float(r['mape_percent_std']):.2f} | {float(r['r2_mean']):.3f} | {float(r['spearman_mean']):.3f} |")
    lines += ['', 'MAPE ± sample SD across training seeds. These are not confidence intervals across independent train/test splits.', '',
              'Paired test-row bootstrap (5,000 resamples) of Transformer minus XGBoost MAPE, averaging seed-specific errors per row. Positive values favor XGBoost:']
    for r in comparisons:
        lines.append(f"- {r['target']}: {r['transformer_minus_xgboost_mape_pp']:.2f} percentage points; conditional 95% interval [{r['ci95_lower']:.2f}, {r['ci95_upper']:.2f}].")
    lines += ['', '## Interpretation limits', '',
              '- One random held-out split supports within-distribution comparison, not generalization to unseen devices, kernels, workload lengths, or future measurement sessions.',
              '- Random weights and uncontrolled measurement noise remain in the source data. No repeated measurements or session identifiers are available.',
              '- Dynamic energy includes prefill and is divided by requested output tokens. It is not decode-only energy.',
              '- No thermal, voltage, measured performance, config ID, or acquisition-order variables are model inputs.',
              '- Seed dispersion is not calibrated epistemic uncertainty for active learning.',
              '- Checkpoints are trained on the training partition only; validation controls early stopping. Do not mix these held-out evaluation results with a later all-data refit.',
              '', '## Reproduce', '', '```bash', 'conda activate nanollmforge',
              'python -m scripts.prediction compare-surrogates --output scripts/prediction/outputs/surrogate_comparison_repeat',
              'python -m scripts.prediction report-surrogates scripts/prediction/outputs/surrogate_comparison_repeat', '```', '',
              'Metadata stores input SHA256, exact split IDs, feature ordering, scalers, and library versions. Separate metrics, tuning scores, held-out predictions, selected hyperparameters, and checkpoints are included.']
    (args.output/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(comparisons,indent=2))


if __name__=='__main__':
    main()
