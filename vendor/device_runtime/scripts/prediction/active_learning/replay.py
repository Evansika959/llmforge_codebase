"""Retrospective active-vs-random replay. Oracle labels are invisible to acquisition."""
import copy
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr

from ..data.dataset import MeasurementDataset
from ..training.pipeline import fit_dataset
from .datasets import legacy_dataset
from .engine import initialize, advance, _evaluate
from .storage import atomic_json, read_json, write_configs


class ReplayBackend:
    def __init__(self, oracle, raw_oracle=None):
        self.oracle = oracle
        self.raw_oracle = raw_oracle or {}

    def __call__(self, rd, proposal, protocol, config):
        if config.get('mode')!='offline_replay':
            raise ValueError('Replay backend requires an explicitly offline workspace')
        rows = []
        for c in proposal['schedule']:
            observed = self.oracle[c['config_id']]
            row = dict(c,decode_tok_s=observed['decode_tok_s'],ttft_ms=observed['ttft_ms'],notes='')
            if protocol['energy_target']=='gross':
                row['total_energy_j'] = observed['gross_energy_per_token_mj']*protocol['output_tokens']/1000
            else:
                row['dynamic_energy_per_token_mj'] = observed['dynamic_energy_per_token_mj']
                if protocol.get('energy_label_validation')=='baseline_reconciled_v1':
                    raw = self.raw_oracle[c['config_id']]
                    for name in ['active_power_w','baseline_power_w','duration_s']:
                        row[name] = raw[name]
            rows.append(row)
        if not (rd/'measurements.csv').exists():
            write_configs(rd/'measurements.csv',rows)


def run_replay(output, previous, initial=300, batch_size=20, rounds=3, members=3, seeds=(42,123,2026), energy='dynamic'):
    if rounds < 1 or batch_size < 1 or len(set(seeds))!=len(seeds):
        raise ValueError('Invalid replay budgets/seeds')
    dataset = legacy_dataset(previous,energy=energy)
    snapshot = read_json(Path(previous)/'input_snapshot.json')
    raw_oracle = {r['config_id']:r for name in ['batch1','batch2'] for r in snapshot['records'][name]}
    training = [r for r in dataset.observations if r['split']=='train']
    holdout = [r for r in dataset.observations if r['split']!='train']
    if not 10<=initial<len(training) or initial+batch_size*rounds>len(training):
        raise ValueError('Replay budget exceeds the hidden training pool')
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'replay_protocol.json',dict(mode='offline_replay',source_dataset_sha256=dataset.fingerprint,
        initial=initial,batch_size=batch_size,rounds=rounds,members=members,seeds=list(seeds),energy_target=energy,
        warning='Historical test already inspected; retrospective single-measurement oracle, not prospective hardware evidence. No anchor noise can be evaluated in this replay.'))
    results = []
    for seed in seeds:
        ids = np.arange(len(training))
        strata = [int(r['architecture']['q8_group_size']) for r in training]
        first,hidden = train_test_split(ids,train_size=initial,random_state=seed,stratify=strata)
        doc = dataset.to_dict()
        doc['observations'] = [training[i] for i in first]+holdout
        initial_dataset = MeasurementDataset(doc)
        candidates = [dict(training[i]['architecture'],config_id=training[i]['config_id']) for i in hidden]
        oracle = {training[i]['config_id']:training[i]['metrics'] for i in hidden}
        baseline = _evaluate(fit_dataset(initial_dataset,seed=seed),initial_dataset,'test')
        for strategy in ['hybrid','random']:
            workspace = output/f'{strategy}_seed{seed}'
            initialize(workspace,initial_dataset,candidates,dict(mode='offline_replay',seed=seed,strategy=strategy,
                members=members,anchors=0,batch_size=batch_size,max_rounds=rounds))
            results.extend(dict(seed=seed,strategy=strategy,round=0,train_n=initial,**r) for r in baseline)
            advance(workspace,ReplayBackend(oracle,raw_oracle),rounds=rounds,evaluate_test=True)
            for number in range(1,rounds+1):
                rd = workspace/'rounds'/f'{number:04d}'
                report = read_json(rd/'completion.json')
                results.extend(dict(seed=seed,strategy=strategy,round=number,train_n=initial+batch_size*number,**r)
                               for r in report['exploratory_test_after'])
            atomic_json(output/'metrics.json',results)
            print(f'Replay complete: {strategy}, seed {seed}',flush=True)
    summary = []
    for strategy in ['hybrid','random']:
        for number in range(rounds+1):
            for target in dataset.targets:
                values = [r for r in results if (r['strategy'],r['round'],r['target'])==(strategy,number,target)]
                summary.append(dict(strategy=strategy,round=number,target=target,train_n=initial+batch_size*number,
                    mean_mape=float(np.mean([r['mape'] for r in values])),seed_sd_mape=float(np.std([r['mape'] for r in values]))))
    atomic_json(output/'summary.json',summary)
    lines = ['# Offline active-learning replay','',
        f'Energy target: {energy} (gross = no baseline subtraction).', '',
        f'{initial} initial training architectures; {rounds} rounds × {batch_size} new labels; {len(seeds)} seeds. Identical fixed validation/test and hidden candidate pool within each paired seed.', '',
        '| Strategy | Fit architectures | Throughput MAPE | TTFT MAPE | Energy MAPE |','|---|---:|---:|---:|---:|']
    for strategy in ['hybrid','random']:
        for number in range(rounds+1):
            values = [next(r for r in summary if (r['strategy'],r['round'],r['target'])==(strategy,number,t)) for t in dataset.targets]
            lines.append(f'| {strategy} | {values[0]["train_n"]} | '+' | '.join(f'{r["mean_mape"]:.2f}%' for r in values)+' |')
    lines += ['', 'Limits: This is retrospective engineering/learning-curve evidence, not proof of prospective sample efficiency. Acquisition sees architecture features and training/validation labels only. Oracle labels are revealed only for selected rows. Fixed test labels are used for reporting, never selection. Each stored architecture has one historical measurement, so this test cannot validate live noise, charging, thermal behavior or anchor gates. No hardware was used.']
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    print('Replay report:',output/'README.md',flush=True)
