"""Common error/ranking evaluation and paired group bootstrap."""
import numpy as np
from scripts.prediction_research.ranking_metrics import score

FIELDS=['mape','spearman','kendall_tau_b','pairwise_accuracy_pct',
        'recall_at_k_pct','ndcg_at_k','selection_regret_pct','topk_regret_pct']


def evaluate(y, predictions, ids, groups, targets, cohorts):
    rows=[]
    for cohort, mask in cohorts.items():
        for model, array in predictions.items():
            for j,t in enumerate(targets):
                take=mask & np.isfinite(y[:,j])
                n=int(take.sum())
                if n<3: continue
                k=min(32,n)
                scores=[score(y[take,j],p[take,j],ids[take],j==0,k) for p in array]
                rows.append(dict(cohort=cohort,model=model,target=t,n=n,k=k,
                    mean={f:float(np.mean([s[f] for s in scores])) for f in FIELDS},
                    seed_sd={f:float(np.std([s[f] for s in scores],ddof=1)) for f in FIELDS},per_seed=scores))
    bootstrap=[]
    for model in predictions:
        if model=='XGBoost': continue
        for j,t in enumerate(targets):
            valid=np.isfinite(y[:,j]); yy=y[valid,j]
            # First average paired absolute percentage errors across training seeds.
            delta=(np.abs(predictions[model][:,valid,j]-yy)/yy
                   -np.abs(predictions['XGBoost'][:,valid,j]-yy)/yy).mean(0)*100
            unique=np.unique(groups[valid]); index=[np.flatnonzero(groups[valid]==g) for g in unique]
            sums=np.array([delta[ix].sum() for ix in index]); counts=np.array([len(ix) for ix in index])
            rng=np.random.default_rng(20260921)
            draws=rng.integers(0,len(index),size=(3000,len(index)))
            values=sums[draws].sum(1)/counts[draws].sum(1)
            bootstrap.append(dict(model=model,target=t,mean_mape_delta_pp=float(delta.mean()),
                 ci95=np.quantile(values,[.025,.975]).tolist(),groups=len(index),
                 interpretation='Positive means Transformer MAPE is worse. Conditional on fitted models; not a confidence interval over training datasets.'))
    return dict(rows=rows,paired_group_bootstrap=bootstrap)


def render(result):
    lines=['# XGBoost versus Transformer: layerwise measurements','',
           'Three-seed means on the same frozen test architectures. ± is training-seed SD. K=32.', '',
           '| Model | Metric | Test N | MAPE (%) | Spearman | Kendall tau-b | Pairwise (%) | Recall@32 (%) |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in result['rows']:
        if r['cohort']!='all': continue
        m=r['mean'];sd=r['seed_sd']
        lines.append(f"| {r['model']} | {r['target']} | {r['n']} | {m['mape']:.2f} ± {sd['mape']:.2f} | "
                     f"{m['spearman']:.4f} | {m['kendall_tau_b']:.4f} | {m['pairwise_accuracy_pct']:.2f} | {m['recall_at_k_pct']:.2f} |")
    lines+=['','## Paired uncertainty for MAPE differences','',
            'Delta = Transformer − XGBoost, in percentage points. 3,000 paired bootstrap draws of whole layer-multiset groups.', '',
            '| Transformer | Metric | Delta | 95% interval |','|---|---|---:|---:|']
    for r in result['paired_group_bootstrap']:
        lo,hi=r['ci95']
        lines.append(f"| {r['model']} | {r['target']} | {r['mean_mape_delta_pp']:.2f} | [{lo:.2f}, {hi:.2f}] |")
    return '\n'.join(lines)+'\n'
