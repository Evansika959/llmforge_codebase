"""Evaluate the requested grouped Transformer and five-fold stacked ensemble."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from dataclasses import asdict
import joblib
from ..models.serialization import load_bundle
import numpy as np
import torch
import xgboost as xgb
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from ..config import OUTPUTS, PREDICTION_ROOT, SWEEP_CONFIGS, SWEEP_OUTPUTS
from ..config import FEATURES
from ..data.io import write_csv
from ..features.physics import HardwareProfile
from ..features.physics import FEATURES32
from ..features.physics import TOKEN_GROUPS
from ..training.neural import fit_neural
from ..inference.neural import predict_neural

from ..config import PHYSICS_TARGETS as TARGETS
from ..training.ensemble import BASE_NAMES, fit_bases
from ..inference.ensemble import predict_bases


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default=str(OUTPUTS/'proposed_physics_surrogate_936'))
    parser.add_argument('--cache-bytes',type=float,default=524288.)
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    old=OUTPUTS/'surrogate_comparison_936';meta=json.loads((old/'metadata.json').read_text())
    source=Path(meta['source']);assert hashlib.sha256(source.read_bytes()).hexdigest()==meta['source_sha256']
    with source.open() as f:rows=list(csv.DictReader(f))
    with (SWEEP_CONFIGS/'watch5_random_1000_50M_150M_sweep.csv').open() as f:cfg={r['config_id']:r for r in csv.DictReader(f)}
    configs=[cfg[r['config_id']] for r in rows];index={r['config_id']:i for i,r in enumerate(rows)}
    tr,va,te=(np.array([index[i] for i in meta['splits'][k]]) for k in ['train','validation','test'])
    y=np.asarray([[float(r[k]) for k in TARGETS] for r in rows],dtype=np.float32);z=np.log(y)
    gs=np.array([int(c['q8_group_size']) for c in configs]);fold_records=[];results=[];predictions=[];weights=[];runtimes=[]
    manifest=dict(source_sha256=meta['source_sha256'],features=FEATURES32,token_groups=TOKEN_GROUPS,targets=TARGETS,splits=meta['splits'],
        seeds=[42,123,2026],hardware=dict(device='Pixel Watch 5',soc_sysfs='monaco',chip_public='Snapdragon W5 Gen 2 Accelerated',
            cores_public='4x Cortex-A53',shared_l2_bytes=args.cache_bytes,l1_data_per_core_bytes=32768,
            cache_source='adb sysfs cpu0/cache/index2: level 2, Unified, 512K, shared_cpu_list 0-3; read 2026-09-12',
            bandwidth_compute_overheads='Nonnegative effective coefficients calibrated only from respective fitting rows'),
        workload=dict(nominal_prompt_tokens=48,actual_prompt_tokens_inferred=49,decode_forward_calls=31,normal_output_tokens=32,energy_includes_prefill=True),
        sources=['https://store.google.com/au/product/pixel_watch_5_specs?hl=en-GB','https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/snapdragon-w5-plus-gen-2-product-brief.pdf'],
        notes='Same previously inspected test split, exploratory. Neural models use standardized log targets, SmoothL1, AdamW, cosine schedule. Constants and scalers never use test labels. No temperature, voltage or acquisition order inputs.')
    (out/'metadata.json').write_text(json.dumps(manifest,indent=2))
    def evaluate(name,seed,pred):
        assert np.isfinite(pred).all() and (pred>0).all()
        for j,t in enumerate(TARGETS):
            actual=y[te,j];p=pred[:,j]
            results.append(dict(model=name,seed=seed,target=t,mae=float(np.mean(abs(p-actual))),
                mape=float(np.mean(abs(p-actual)/actual)*100),r2=float(r2_score(actual,p)),spearman=float(spearmanr(actual,p).statistic)))
            for i,a,b in zip(te,actual,p):predictions.append(dict(model=name,seed=seed,target=t,config_id=rows[i]['config_id'],actual=float(a),prediction=float(b)))
        write_csv(out/'metrics.csv',results);write_csv(out/'test_predictions.csv',predictions)
    # Original architecture-only baseline, transformed from its TPOT model to throughput.
    from ..features.analytic import physical_features
    oldx=np.array([[physical_features(c)[0][k] for k in FEATURES] for c in configs],dtype=np.float32)
    for seed in manifest['seeds']:
        old_predictions=[]
        for j,t in enumerate(['tpot_ms','ttft_ms','dynamic_energy_per_token_mj']):
            m=xgb.XGBRegressor();m.load_model(old/f'xgboost_seed{seed}_{t}.json');p=np.exp(m.predict(oldx[te]))
            old_predictions.append(1000/p if j==0 else p)
        evaluate('original_xgboost',seed,np.column_stack(old_predictions))
        start=time.monotonic()
        profile=HardwareProfile(cache_bytes=args.cache_bytes).fit([configs[i] for i in tr],y[tr]);x=profile.transform(configs)
        pack=fit_neural('transformer',x[tr],z[tr],x[va],z[va],seed)
        pack['profile']=profile
        joblib.dump(pack,out/f'grouped_transformer_seed{seed}.joblib')
        predicted=np.exp(predict_neural(pack,x[te]));evaluate('grouped_transformer',seed,predicted)
        loaded=load_bundle(out/f'grouped_transformer_seed{seed}.joblib')
        np.testing.assert_allclose(predicted,np.exp(predict_neural(loaded,loaded['profile'].transform([configs[i] for i in te]))),rtol=1e-5)
        runtimes.append(dict(seed=seed,stage='transformer',seconds=time.monotonic()-start,best_epoch=pack['best_epoch'],total_epochs=pack['total_epochs']))
        print('TRANSFORMER',seed,'parameters',pack['parameter_count'],'epochs',pack['best_epoch'],pack['total_epochs'],flush=True)
        # A row's OOF prediction must not depend on its label for calibration,
        # scaling, tree/neural fitting, or early stopping.
        oof=np.full((len(tr),3,4),np.nan);assigned=np.zeros(len(tr),dtype=int)
        for fold,(fitloc,holdloc) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(tr,gs[tr])):
            fitids,stopids=train_test_split(tr[fitloc],test_size=.15,random_state=seed,stratify=gs[tr[fitloc]])
            held=tr[holdloc]
            assert not set(held)&(set(fitids)|set(stopids)) and not set(held)&set(te)
            started=time.monotonic();base=fit_bases(configs,y,fitids,stopids,seed,args.cache_bytes)
            oof[holdloc]=predict_bases(base,[configs[i] for i in held]);assigned[holdloc]+=1
            fold_records.append(dict(seed=seed,fold=fold,fitting=[rows[i]['config_id'] for i in fitids],
                stopping=[rows[i]['config_id'] for i in stopids],held_out=[rows[i]['config_id'] for i in held],profile=asdict(base['profile'])))
            print('OOF',seed,fold,'seconds',round(time.monotonic()-started,1),flush=True)
        assert np.isfinite(oof).all() and (assigned==1).all()
        np.savez(out/f'oof_seed{seed}.npz',predictions=oof,targets=z[tr],config_ids=np.array([rows[i]['config_id'] for i in tr]))
        combiners=[]
        for j,t in enumerate(TARGETS):
            ridge=Ridge(alpha=.1,positive=True,solver='lbfgs',tol=1e-7,max_iter=10000).fit(oof[:,j,:],z[tr,j]);combiners.append(ridge)
            assert (ridge.coef_>=0).all()
            weights.append(dict(seed=seed,target=t,intercept=float(ridge.intercept_),**{name:float(w) for name,w in zip(BASE_NAMES,ridge.coef_)}))
        base=fit_bases(configs,y,tr,va,seed,args.cache_bytes)
        final=dict(base=base,combiners=combiners,features=FEATURES32,targets=TARGETS)
        joblib.dump(final,out/f'stacked_ensemble_seed{seed}.joblib')
        basepred=predict_bases(base,[configs[i] for i in te])
        for k,name in enumerate(BASE_NAMES):evaluate('base_'+name,seed,np.exp(basepred[:,:,k]))
        stacked=np.exp(np.column_stack([m.predict(basepred[:,j,:]) for j,m in enumerate(combiners)]))
        evaluate('stacked_ensemble',seed,stacked)
        reloaded=load_bundle(out/f'stacked_ensemble_seed{seed}.joblib');bp=predict_bases(reloaded['base'],[configs[i] for i in te])
        np.testing.assert_allclose(stacked,np.exp(np.column_stack([m.predict(bp[:,j,:]) for j,m in enumerate(reloaded['combiners'])])),rtol=1e-5)
        write_csv(out/'stacking_weights.csv',weights);write_csv(out/'training.csv',runtimes)
        (out/'oof_folds.json').write_text(json.dumps(fold_records,indent=2))
        print('SEED COMPLETE',seed,flush=True)
    summary=[]
    for name in dict.fromkeys(r['model'] for r in results):
        for t in TARGETS:
            rs=[r for r in results if r['model']==name and r['target']==t]
            summary.append(dict(model=name,target=t,**{k:float(np.mean([r[k] for r in rs])) for k in ['mae','mape','r2','spearman']},
                mape_seed_sd=float(np.std([r['mape'] for r in rs],ddof=1))))
    write_csv(out/'summary.csv',summary)
    lines=['# Proposed physics-informed surrogate experiment','',
        '936 architectures; original 598 fitting / 150 stopping / 188 test split; seeds 42, 123, 2026. All inputs are architecture-only. Results reuse a previously explored test set.',
        '', '| Predictor | Target | MAPE % | MAE | R² |','|---|---|---:|---:|---:|']
    for r in summary:lines.append(f"| {r['model']} | {r['target']} | {r['mape']:.2f} ± {r['mape_seed_sd']:.2f} | {r['mae']:.3f} | {r['r2']:.3f} |")
    lines+=['','## Implementation','',
        '- Exactly 32 documented features, grouped into seven disjoint semantic tokens. Transformer: CLS + learned positional embeddings, four encoder layers, eight heads, width 128, feedforward width 512, regression head.',
        '- Four stacking bases: XGBoost, ExtraTrees, Random Forest, residual MLP. Five-fold OOF within the 598 fitting rows; each fold has its own calibration, normalization, and separate internal stopping rows. Nonnegative Ridge per target fits only the OOF predictions.',
        '- Neural loss: Smooth L1 on training-standardized natural-log targets; AdamW with cosine schedule. Final outputs are exponentiated. Features are log1p transformed and train-standardized for neural models.',
        '- Base-model OOF fits use fewer fitting rows than full bases because early-stopping rows are held separately. This can create stacking distribution shift; base predictions and weights are reported to expose it.',
        '', '## Hardware and label provenance','',
        '- Google public specification: Snapdragon W5 Gen 2 Accelerated; Qualcomm public W5 Gen 2 brief: four A53 cores and LPDDR4 2133MHz. See sources in metadata.json.',
        '- Connected Pixel Watch 5 reports shared L2 512KiB (CPUs 0–3) and L1 data cache 32KiB per core. Used as static cache-capacity prior.',
        '- Effective bandwidth, compute MAC/s, synchronization and dispatch terms use nonnegative fits on training labels only. They are not published peak specifications or independently measured hardware constants.',
        '- Cache residency/spill estimates are capacity proxies, not measured hit rates or DRAM traffic. Weight transfer and prefill compute formulas are roofline-inspired, not calibrated cycle-accurate simulation.',
        '- Existing nominal 48-token measurements imply 49 actual prompt tokens and 31 decode forward calls, normally 32 output tokens. This experiment uses 49 in physical work estimates and preserves existing TTFT labels; it does not claim exact-48-token accuracy.',
        '- Dynamic energy labels include prefill. They are not decode-only energy. The old baseline throughput is obtained by inverting its TPOT predictions, so its throughput MAPE differs from TPOT MAPE.',
        '- No runq_llmforge.c changes, no benchmark launches, no environmental inputs. All saved model predictions were reproduced after reload.',
        '', '## Reproduce','', '```bash',
        'python -m llmforge.hw.device.prediction train-proposed-physics-surrogate --output runs/device/proposed_physics_surrogate_repeat','```']
    lines += ['', '## Audit and architecture-only inference', '',
        'Run the audit script for split checks and paired bootstrap intervals (AUDIT.md). Model bundles are trusted-local joblib files; do not load untrusted bundles.', '',
        '```bash',
        'python -m llmforge.hw.device.prediction audit-proposed-physics-surrogate --results runs/device/proposed_physics_surrogate_repeat',
        'python -m llmforge.hw.device.prediction predict-proposed-physics-surrogate \\',
        '  --model runs/device/proposed_physics_surrogate_repeat/stacked_ensemble_seed42.joblib \\',
        '  --configs src/llmforge/hw/device/measurement/sweep/configs/watch5_random_1000_50M_150M_sweep.csv \\',
        '  --output runs/device/proposed_predictions.csv', '```', '',
        'The Transformer bundle can be used with the same inference command. Predictions apply only to the fixed original sweep workload; extrapolation is unvalidated.']
    (out/'README.md').write_text('\n'.join(lines)+'\n');print('COMPLETE',out,flush=True)


if __name__=='__main__':main()
