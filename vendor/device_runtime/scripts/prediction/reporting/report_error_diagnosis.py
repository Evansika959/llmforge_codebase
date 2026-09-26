"""Starting-temperature-only follow-up and learning-curve report."""
import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
import numpy as np
import xgboost as xgb
from ..config import ROOT
from ..config import FEATURES
from ..config import TARGETS
from ..data.io import write_csv
from ..features.analytic import physical_features
from ..evaluation.metrics import metrics


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('output',type=Path); args=parser.parse_args(); out=args.output
    meta=json.loads((out/'metadata.json').read_text())
    with (ROOT/'scripts/sweep/outputs/watch5_random_50M_150M_40C_decode32_results_clean.csv').open() as f: rows=list(csv.DictReader(f))
    with (ROOT/'scripts/sweep/configs/watch5_random_1000_50M_150M_sweep.csv').open() as f: cfg={r['config_id']:r for r in csv.DictReader(f)}
    index={r['config_id']:i for i,r in enumerate(rows)}
    tr,va,te=(np.asarray([index[i] for i in meta['splits'][k]]) for k in ['train','validation','test'])
    X=np.asarray([[physical_features(cfg[r['config_id']])[0][k] for k in FEATURES]+[float(r['temp_cpu_start_c'])] for r in rows],dtype=np.float32)
    Y=np.asarray([[float(r[k]) for k in TARGETS] for r in rows],dtype=np.float32)
    result=[]; predictions=[]
    for seed in [42,123,2026]:
        for j,t in enumerate(TARGETS):
            m=xgb.XGBRegressor(n_estimators=1500,learning_rate=.03,tree_method='hist',n_jobs=4,early_stopping_rounds=50,
                max_depth=2,min_child_weight=3,reg_lambda=5,subsample=.85,colsample_bytree=.9,
                objective='reg:squarederror',random_state=seed)
            m.fit(X[tr],np.log(Y[tr,j]),eval_set=[(X[va],np.log(Y[va,j]))],verbose=False)
            pred=np.exp(m.predict(X[te])); m.save_model(out/f'temperature_conditioned_seed{seed}_{t}.json')
            loaded=xgb.XGBRegressor(); loaded.load_model(out/f'temperature_conditioned_seed{seed}_{t}.json')
            np.testing.assert_allclose(pred,np.exp(loaded.predict(X[te])),rtol=1e-6)
            result.append(dict(variant='start_temperature_only',seed=seed,target=t,**metrics(Y[te,j],pred)))
            for i,a,p in zip(te,Y[te,j],pred):
                predictions.append(dict(seed=seed,target=t,config_id=rows[i]['config_id'],actual=float(a),prediction=float(p)))
    write_csv(out/'temperature_conditioned_metrics.csv',result)
    write_csv(out/'temperature_conditioned_predictions.csv',predictions)
    (out/'temperature_conditioned_metadata.json').write_text(json.dumps(dict(features=FEATURES+['temp_cpu_start_c'],targets=TARGETS,
        splits=meta['splits'],target_transform='log, inverse exp',requires='Valid CPU temperature measured before inference; device/workload as original data',
        limitation='Exploratory random split, not causal temperature effect or prospective/session-held-out validation'),indent=2))
    with (out/'metrics.csv').open() as f: all_metrics=list(csv.DictReader(f))
    all_metrics+=result
    summary=[]
    for variant in ['baseline','optimized','start_temperature_only']:
        for t in TARGETS:
            group=[r for r in all_metrics if r['variant']==variant and r['target']==t]
            summary.append(dict(variant=variant,target=t,**{k:float(np.mean([float(r[k]) for r in group])) for k in ['mae','mape','r2','spearman']}))
    write_csv(out/'summary.csv',summary)
    with (out/'learning_curve.csv').open() as f: curve=list(csv.DictReader(f))
    os.environ.setdefault('MPLCONFIGDIR',tempfile.mkdtemp(prefix='surrogate-mpl-'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for ax,t,title in zip(axes,TARGETS,['TPOT','TTFT','Dynamic energy / output token']):
        for split,color in [('train','#64748b'),('validation','#2563eb')]:
            sizes=[120,240,360,480,598]
            vals=[[float(r['mape']) for r in curve if int(r['n_train'])==n and r['target']==t and r['split']==split] for n in sizes]
            means=np.mean(vals,axis=1); sd=np.std(vals,axis=1,ddof=1)
            ax.plot(sizes,means,'o-',color=color,label=split)
            ax.fill_between(sizes,means-sd,means+sd,color=color,alpha=.15)
        ax.set(title=title,xlabel='Training architectures',ylabel='MAPE (%)'); ax.grid(alpha=.2); ax.legend()
    fig.suptitle('Fixed validation set; mean ± SD over three seeds',fontsize=12)
    fig.savefig(out/'learning_curve.png',dpi=160); plt.close(fig)
    lines=['# Why are surrogate errors high?','',
        '936 observations, fixed 598 training / 150 validation / 188 test split. Three training seeds. Learning curves use the fixed validation set and stratified training subsets. Validation still controls early stopping, so it is not a new independent test.',
        '', '| Variant | Target | MAPE (%) | MAE | R² |','|---|---|---:|---:|---:|']
    for r in summary: lines.append(f"| {r['variant']} | {r['target']} | {r['mape']:.2f} | {r['mae']:.3f} | {r['r2']:.3f} |")
    lines+=['','![Learning curve](learning_curve.png)','',
        '## Methods','',
        '- Baseline reproduces the original XGBoost feature set and settings.',
        '- Optimized: target-specific search over 25 settings (original/physics features, depth, child weight, squared/absolute log loss). Three-fold CV runs entirely inside the 598 training rows. Every fold has a separate early-stopping subset. Hyperparameters are frozen before test evaluation.',
        '- Temperature-conditioned: original baseline settings plus only starting CPU temperature. No ending voltage or measured performance is an input. Checkpoints and feature metadata are saved separately.',
        '- An additional state diagnostic includes ending voltage; it is explanatory only and is not a deployable architecture-only surrogate.',
        '', '## Interpretation limits','',
        'Starting temperature can encode operating conditions or measurement-session differences. Improved prediction does not prove temperature causes the performance change. It changes the prediction task from architecture-only to architecture conditional on known starting state. Random split results do not establish generalization across charging cycles, sessions, devices or temperatures outside the data.',
        'The old test set has already been inspected repeatedly, so these are exploratory results. Prospective measurements and repeated architectures under matched conditions are needed before a strong accuracy claim. Neither a flat learning curve nor temperature improvement measures irreducible measurement noise.',
        '', '## Reproduce','', '```bash','conda activate nanollmforge',
        'python -m scripts.prediction diagnose-surrogate-error --output scripts/prediction/outputs/xgboost_error_diagnosis_repeat',
        'python -m scripts.prediction report-error-diagnosis scripts/prediction/outputs/xgboost_error_diagnosis_repeat','```']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
