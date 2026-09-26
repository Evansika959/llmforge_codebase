"""Explicit cost-direction evaluation and human-readable training report."""
import numpy as np
from scripts.prediction_research.ranking_metrics import score
from .data import TARGETS


def evaluate(actual, predictions, ids):
    result = {}
    for j, target in enumerate(TARGETS):
        values = [score(actual[:, j], pred[:, j], ids, higher_is_better=False,
                        k=min(32, len(ids))) for pred in predictions]
        keys = ('mape', 'spearman', 'kendall_tau_b', 'pairwise_accuracy_pct',
                'pairwise_gap10_pct', 'recall_at_k_pct', 'ndcg_at_k',
                'selection_regret_pct', 'topk_regret_pct')
        result[target] = dict(per_seed=values, mean={
            key: float(np.mean([v[key] for v in values]))
            if all(v[key] is not None for v in values) else None for key in keys})
        result[target]['mape_std_across_seeds'] = float(np.std([v['mape'] for v in values]))
    return result


def render(result, manifest):
    lines = ['# XGBoost: TPOT / TTFT / dynamic energy', '',
             f"Dataset: {sum(manifest['counts'].values())} architectures; split {manifest['counts']}.",
             '110 architecture-only features; no temperature, battery, voltage, or measured timing inputs.',
             'Three independent log-space XGBoost regressors per seed; maximum depth 3, up to 1500 trees,',
             'learning rate 0.03, validation log-RMSE early stopping after 50 rounds.',
             'Seeds: 42, 123, 2026. All three targets are lower-is-better.', '',
             '## Test results: mean across three seeds', '',
             '| Target | MAPE ↓ | Spearman ↑ | Kendall τ-b ↑ | Pairwise accuracy ↑ | Recall@32 ↑ |',
             '|---|---:|---:|---:|---:|---:|']
    for target in TARGETS:
        m = result['test'][target]['mean']
        lines.append(f"| {target} | {m['mape']:.2f}% | {m['spearman']:.4f} | "
                     f"{m['kendall_tau_b']:.4f} | {m['pairwise_accuracy_pct']:.2f}% | "
                     f"{m['recall_at_k_pct']:.2f}% |")
    lines += ['', '## Saved checkpoint', '',
              f"`models/predictor_final.joblib`: seed {result['selected_seed']}, selected ONLY by the",
              'mean validation MAPE across the three targets, before test evaluation.',
              'This is one seed, not an ensemble or a model refitted on validation/test rows.', '',
              '| Target | Selected-checkpoint test MAPE |', '|---|---:|']
    for target in TARGETS:
        lines.append(f"| {target} | {result['selected_test'][target]['mean']['mape']:.2f}% |")
    lines += ['', '## Definitions and limitations', '',
              '- TPOT = 1000 × (end − prefill_end) / 31 ms/decode token, verified against raw timing and reciprocal throughput.',
              '- TTFT uses the actual 49-token prompt (nominally 48); output count is 32.',
              '- Dynamic energy includes prefill AND decode, baseline-subtracted and divided by 32 output tokens; it is not decode-only energy.',
              '- All 417 baseline-drift warnings remain. All 2000 labels are positive and finite.',
              '- GS64 uses the optimized kernel, GS16/32 retain earlier measurements; this is a mixed-kernel, non-contemporaneous dataset.',
              '- Reused fixed test rows make this exploratory evaluation, not a fresh confirmatory holdout.',
              '- Changing from throughput to reciprocal TPOT is not itself evidence of improved prediction. Their MAPE denominators differ.',
              '- No new Transformer was trained in this release. Old throughput-based checkpoints and datasets are untouched.',
              '- Target order and units are stored in every checkpoint; do not feed this checkpoint to old throughput-only reports/AL consumers.', '',
              '## Reproduce', '', 'From the repository root, in the nanollmforge environment:', '',
              '```bash', 'python -m scripts.prediction_research.layerwise_tpot.run \\',
              '  --snapshot diliverable/layerwise_2000_gs64_refresh_20260922/dataset_snapshot.json \\',
              '  --output scripts/prediction/outputs/layerwise_tpot_reproduction', '```', '',
              'The output directory must be new. `dataset_snapshot.json` is an exact copy of the source;',
              '`labels.json` records the actual three training targets and preserved splits.',
              '`test_predictions.npz`, `metrics.json`, `manifest.json`, and `source_snapshot/` record evaluation and provenance.', '',
              'Inference (trusted local joblib files only):', '', '```python', 'import joblib',
              'from scripts.prediction_research.layerwise_refit.features import predict',
              'bundle = joblib.load("diliverable/layerwise_tpot_2000_20260922/models/predictor_final.joblib")',
              'values = predict(bundle, architectures)  # list of ordered-layer architecture dictionaries',
              'print(bundle["targets"], bundle["units"], values)', '```', '']
    return '\n'.join(lines)
