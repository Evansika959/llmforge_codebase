"""Offline ranking and selection metrics from already saved predictions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr


TARGETS = ['decode_tok_s', 'ttft_ms', 'dynamic_energy_per_token_mj']
NAMES = ['吞吐', 'TTFT', '扣基线能耗']
LABELS = {
    'xgboost14': 'XGBoost-14', 'physics32': 'XGBoost-32',
    'transformer128x4': 'Transformer-4×128', 'transformer': 'Transformer-2×64',
    'active_seed42': '当前 AL XGBoost-32（单种子）',
    'params_kv_group': 'XGBoost-3',
}


def score(y, pred, ids, higher_is_better, k):
    y, pred, ids = np.asarray(y, float), np.asarray(pred, float), np.asarray(ids, str)
    assert y.ndim==1 and pred.shape==y.shape==ids.shape and len(set(ids))==len(y)
    assert np.isfinite(y).all() and np.isfinite(pred).all() and (y>0).all() and (pred>0).all()
    assert 1<=k<=len(y)
    sign = 1 if higher_is_better else -1
    truth = np.lexsort((ids, -sign*y))
    order = np.lexsort((ids, -sign*pred))
    a,b = np.triu_indices(len(y), 1)
    delta = np.sign(y[a]-y[b]); guess = np.sign(pred[a]-pred[b])
    eligible = delta!=0
    correct = (delta==guess).astype(float) + .5*(guess==0)
    separated = eligible & (np.maximum(y[a],y[b])/np.minimum(y[a],y[b])>=1.10)
    best = y[truth[0]]
    picked = y[order[0]]
    pool_best = np.max(y[order[:k]]) if higher_is_better else np.min(y[order[:k]])
    # Linear discounted gains: normalized throughput, or inverse normalized cost.
    # No exponential gain transform; gain is monotone in actual desirability.
    gain = y/best if higher_is_better else best/y
    discount = 1/np.log2(np.arange(2,k+2))
    rho = spearmanr(y,pred).statistic
    tau = kendalltau(y,pred,variant='b').statistic
    return dict(mape=float(np.mean(abs(pred-y)/y)*100),
                spearman=float(rho) if np.isfinite(rho) else None,
                kendall_tau_b=float(tau) if np.isfinite(tau) else None,
                pairwise_accuracy_pct=float(correct[eligible].mean()*100) if eligible.any() else None,
                pairwise_gap10_pct=float(correct[separated].mean()*100) if separated.any() else None,
                eligible_pairs=int(eligible.sum()), gap10_pairs=int(separated.sum()),
                recall_at_k_pct=100*len(set(truth[:k]) & set(order[:k]))/k,
                ndcg_at_k=float(np.sum(gain[order[:k]]*discount)/np.sum(gain[truth[:k]]*discount)),
                selection_regret_pct=float(sign*(best-picked)/best*100),
                topk_regret_pct=float(sign*(best-pool_best)/best*100),
                selected_id=str(ids[order[0]]), actual_best_id=str(ids[truth[0]]))


def table(rows):
    lines=['| 模型 | 指标 | Kendall τ-b ↑ | 两两准确率 ↑ | Recall@K ↑ | NDCG@K ↑ | 选择 regret ↓ | Top-K regret ↓ |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        m=r['mean']; lines.append(f'| {LABELS[r["model"]]} | {NAMES[TARGETS.index(r["target"])]} | '
            f'{m["kendall_tau_b"]:.4f} | {m["pairwise_accuracy_pct"]:.2f}% | {m["recall_at_k_pct"]:.2f}% | '
            f'{m["ndcg_at_k"]:.4f} | {m["selection_regret_pct"]:.2f}% | {m["topk_regret_pct"]:.2f}% |')
    return lines


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]/'prediction/outputs'
    paths={
        'historical':root/'compact_surrogates_1564/test_predictions.npz',
        'physics':root/'state_inputs_1564/test_predictions.npz',
        'minimal':root/'minimal_features_al50_with_controls/validation_predictions.npz',
        'minimal_manifest':root/'minimal_features_al50_with_controls/manifest.json',
    }
    a=np.load(paths['historical'],allow_pickle=False)
    b=np.load(paths['physics'],allow_pickle=False)
    c=np.load(paths['minimal'],allow_pickle=False)
    np.testing.assert_array_equal(a['config_ids'], b['config_ids'])
    np.testing.assert_array_equal(a['actual'], b['actual'])
    manifest=json.loads(paths['minimal_manifest'].read_text())
    assert manifest['completed_round']==50 and manifest['seeds']==[42,123,2026]
    checkpoint=Path(manifest['workspace'])/manifest['source_model']
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==manifest['source_model_sha256']
    sources=[('historical_test_314',a['actual'],a['config_ids'],
              {'xgboost14':a['xgboost14'],'physics32':b['physics32'],
               'transformer128x4':a['transformer128x4'],'transformer':a['transformer']}),
             ('active_validation_150',c['actual'],c['measurement_ids'],
              {'active_seed42':c['physics32'][:1], 'physics32':c['physics32'],
               'params_kv_group':c['params_kv_group']})]
    reports=[]
    fields=['mape','spearman','kendall_tau_b','pairwise_accuracy_pct','pairwise_gap10_pct',
            'recall_at_k_pct','ndcg_at_k','selection_regret_pct','topk_regret_pct']
    for split,y,ids,models in sources:
        k=int(np.ceil(len(y)*.1))
        for name,predictions in models.items():
            assert predictions.shape[1:]==y.shape
            for j,target in enumerate(TARGETS):
                individual=[score(y[:,j],p[:,j],ids,j==0,k) for p in predictions]
                assert all(r[f] is not None for r in individual for f in fields)
                reports.append(dict(split=split,model=name,target=target,n=len(y),k=k,seeds=len(individual),
                    mean={f:float(np.mean([r[f] for r in individual])) for f in fields},
                    seed_sd={f:float(np.std([r[f] for r in individual],ddof=1)) if len(individual)>1 else None for f in fields},
                    per_seed=individual))
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'summary.json').write_text(json.dumps(dict(source_sha256={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in paths.items()},
                                                   reports=reports),indent=2,allow_nan=False)+'\n')
    lines=['# Ranking and selection evaluation','',
           'No retraining, new hardware measurements, model promotion or changes to AL state.', '',
           '## Historical test: 1,100 train / 150 validation / 314 test; K=32','',
           'Three seeds, each scored separately before averaging. Existing reused test set; exploratory results.', '']
    lines+=table([r for r in reports if r['split']=='historical_test_314'])
    lines+=['','## AL round 50: 1,962 train / 150 validation; K=15','',
            'Current AL checkpoint is seed 42; offline models use three seeds. Validation also controls early stopping.','']
    lines+=table([r for r in reports if r['split']=='active_validation_150'])
    lines+=['','## Definitions and limitations','',
            '- K = ceil(10% of evaluation rows), fixed before scoring.',
            '- Throughput: larger is better; TTFT and dynamic energy: smaller is better.',
            '- Kendall uses tau-b tie adjustment. Pairwise accuracy excludes exact truth ties and gives prediction ties half credit. Noisy near-ties remain in the main metric.',
            '- Additional pairwise_gap10_pct in JSON considers only pairs whose actual max/min >=1.10; this is a descriptive sensitivity analysis, not a measured noise threshold.',
            '- Recall@K is the overlap of predicted and actual Top-K, divided by K. Top-K ties are broken by lexicographic measurement/config ID, never by true performance.',
            '- NDCG uses linear gains: throughput / maximum throughput, or minimum cost / cost for TTFT and energy; logarithmic position discount. No exponential relevance transform.',
            '- Selection regret is the predicted winner\'s actual relative performance loss against the actual best in this evaluation set. Throughput: (best-selected)/best; cost: (selected-best)/best.',
            '- Top-K regret uses the actual best among predicted Top-K instead of the predicted winner. It assumes K follow-up measurements, not a guarantee about a future search space.',
            '- Ground-truth optima/ranks are used for evaluation only. Both regret metrics can be sensitive to noisy extreme observations.',
            '- Summary JSON includes MAPE, Spearman, seed SD, and all per-seed metrics. Seed SD is not an evaluation-split confidence interval.',
            '- Historical test and AL validation have different training/evaluation populations and must not be compared as a learning curve.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':
    main()
