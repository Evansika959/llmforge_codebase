"""Transactional active-learning rounds. Only an explicit backend can touch hardware."""
import copy
import csv
import hashlib
from pathlib import Path

import numpy as np

from ..config import ROOT, PREDICTION_ROOT, FEATURES
from ..data.dataset import MeasurementDataset, architecture_key
from ..features.physics import architecture_stats
from ..inference.predictor import predict_bundle, bundle_targets
from ..models.serialization import load_bundle
from ..training.pipeline import fit_dataset
from ..evaluation.metrics import metrics
from .committee import fit_committee
from .sampling import generate_pool, select_batch, unseen_candidates
from .quality import result_metrics, anchor_report
from .storage import atomic_json, atomic_bundle, read_json, write_configs, digest, locked

DEFAULTS = dict(batch_size=20,members=5,seed=42,strategy='hybrid',target_weights=[1.,1.,1.],
                anchors=0,anchor_tolerance=.25,max_rounds=5,pool_size=5000,
                min_params_m=50.,max_params_m=150.,temperature_ceiling=40.,min_battery_percent=30,
                max_measurement_attempts=3)


def source_hashes():
    paths = [p for p in PREDICTION_ROOT.rglob('*.py') if 'outputs' not in p.parts and 'tests' not in p.parts]
    paths += [ROOT/'scripts/sweep/run_sweep_configs.py',ROOT/'src/runq_llmforge.c',ROOT/'src/power_sampler.c',
              ROOT/'llmforge_bridge/export_llmforge_hetero.py',ROOT/'scripts/sweep/generate_watch5_random_50m_150m.py']
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted(paths)}


def normalized_config(c):
    stat = architecture_stats(c)
    return dict(config_id=c['config_id'],category=c.get('category','ActiveLearning'),
                **{k:int(c[k]) for k in FEATURES[:7]},vocab_size=stat['vocab'],q8_group_size=stat['G'],
                total_params_M=stat['params']/1e6,layer_size_kb=stat['layer']/1024,
                mlp_ratio=stat['M']/stat['D'],kv_ratio=stat['K']/stat['H'])


def _choose_anchors(dataset, count):
    unique = {architecture_key(r['architecture']):dict(r['architecture'],config_id=r['config_id'])
              for r in dataset.observations if r['split']=='train'}
    configs = sorted(unique.values(),key=lambda c:architecture_stats(c)['params'])
    if count > len(configs):
        raise ValueError('Not enough distinct training architectures for anchors')
    indices = np.linspace(0,len(configs)-1,count+2,dtype=int)[1:-1]
    if len(set(indices)) != count:
        raise ValueError('Anchor selection would duplicate architectures')
    return [normalized_config(configs[i]) for i in indices]


def initialize(workspace, dataset, candidates=None, settings=None, initial_model=None):
    workspace = Path(workspace)
    config = dict(DEFAULTS,**(settings or {}))
    if config['batch_size'] < 1 or config['members'] < 2 or config['max_rounds'] < 1 or config['anchors'] < 0:
        raise ValueError('Invalid active-learning budget')
    if config['anchor_tolerance'] <= 0 or config['temperature_ceiling'] <= 0 or config['min_battery_percent'] < 30:
        raise ValueError('Invalid drift/safety settings (battery cutoff must be >=30%)')
    if config['strategy'] not in ('hybrid','random') or config['max_measurement_attempts'] < 1:
        raise ValueError('Invalid strategy/attempt budget')
    weights = np.asarray(config['target_weights'])
    if weights.shape != (3,) or not np.isfinite(weights).all() or (weights<0).any() or weights.sum()<=0:
        raise ValueError('Invalid metric weights')
    if candidates is None:
        candidates = generate_pool(dataset,config['pool_size'],config['seed']+1000,config['min_params_m'],config['max_params_m'])
    candidates = [normalized_config(c) for c in unseen_candidates(dataset,candidates)]
    candidates = [c for c in candidates if config['min_params_m'] <= architecture_stats(c)['params']/1e6 <= config['max_params_m']]
    if len(candidates) < config['batch_size']:
        raise ValueError('Pool smaller than one requested batch')
    anchors = _choose_anchors(dataset,config['anchors'])
    initial_bundle = None
    if initial_model is not None:
        initial_bundle = load_bundle(initial_model)
        if (initial_bundle.get('dataset_sha256') != dataset.fingerprint or
                initial_bundle.get('protocol') != dataset.protocol or
                bundle_targets(initial_bundle) != dataset.targets):
            raise ValueError('Initial checkpoint must match the exact starting dataset and protocol')
        predict_bundle(initial_bundle,[dataset.observations[0]['architecture']])
    workspace.mkdir(parents=True,exist_ok=False)
    dataset.save(workspace/'dataset_000.json')
    write_configs(workspace/'pool.csv',candidates)
    atomic_json(workspace/'anchors.json',anchors)
    atomic_json(workspace/'settings.json',config)
    inputs = ['dataset_000.json','pool.csv','anchors.json','settings.json']
    if initial_bundle is not None:
        atomic_bundle(workspace/'initial_model.joblib',initial_bundle)
        inputs.append('initial_model.joblib')
        atomic_json(workspace/'initial_model_source.json',dict(path=str(Path(initial_model).resolve()),sha256=digest(initial_model)))
        inputs.append('initial_model_source.json')
    atomic_json(workspace/'contract.json',dict(version=1,protocol=dataset.protocol,source_hashes=source_hashes(),
        inputs={name:digest(workspace/name) for name in inputs}))
    atomic_json(workspace/'state.json',dict(completed_rounds=0,current_dataset='dataset_000.json',
        dataset_sha256=dataset.fingerprint,status='ready',message='No hardware work has started'))
    return workspace


def verify_workspace(workspace):
    workspace = Path(workspace)
    if (workspace/'MIGRATION_IN_PROGRESS.json').exists():
        raise ValueError('Migration incomplete; preserve this directory and migrate to a new destination')
    contract = read_json(workspace/'contract.json')
    for name,sha in contract['inputs'].items():
        if digest(workspace/name) != sha:
            raise ValueError('Workspace input changed: '+name)
    if contract['source_hashes'] != source_hashes():
        raise ValueError('Protocol/source code changed since initialization; use a new workspace')
    state = read_json(workspace/'state.json')
    dataset = MeasurementDataset.load(workspace/state['current_dataset'])
    if dataset.fingerprint != state['dataset_sha256'] or dataset.protocol != contract['protocol']:
        raise ValueError('Current dataset integrity/protocol mismatch')
    return read_json(workspace/'settings.json'),state,dataset


def _evaluate(model, dataset, split):
    rows = [r for r in dataset.observations if r['split']==split]
    pred = predict_bundle(model,[r['architecture'] for r in rows])
    result = []
    for j,t in enumerate(dataset.targets):
        values = metrics(np.array([r['metrics'][t] for r in rows]),pred[:,j])
        result.append(dict(target=t,n=len(rows),**{k:float(v) if np.isfinite(v) else None for k,v in values.items()}))
    return result


def _proposal(workspace, rd, dataset, config, number):
    proposal_path = rd/'proposal.json'
    if proposal_path.exists():
        proposal = read_json(proposal_path)
        if proposal['dataset_sha256'] != dataset.fingerprint:
            raise ValueError('Proposal belongs to a different dataset')
        for name,sha in proposal['artifacts'].items():
            if digest(rd/name)!=sha:
                raise ValueError('Frozen acquisition artifact changed: '+name)
        return proposal
    rd.mkdir(parents=True,exist_ok=True)
    checkpoint = workspace/'initial_model.joblib' if number==1 else workspace/'rounds'/f'{number-1:04d}'/'predictor_after.joblib'
    if checkpoint.exists():
        if number>1 and read_json(checkpoint.with_name('completion.json'))['model_sha256']!=digest(checkpoint):
            raise ValueError('Previous round checkpoint changed')
        model = load_bundle(checkpoint)
        if model.get('dataset_sha256')!=dataset.fingerprint or model.get('protocol')!=dataset.protocol:
            raise ValueError('Acquisition checkpoint does not match current dataset/protocol')
        model_origin = dict(path=str(checkpoint),sha256=digest(checkpoint),loaded=True)
        print(f'Round {number}: loaded existing predictor checkpoint {checkpoint}',flush=True)
    else:
        print(f'Round {number}: fitting initial predictor',flush=True)
        model = fit_dataset(dataset,'xgboost',config['seed'])
        model_origin = dict(loaded=False)
    print(f'Round {number}: fitting {config["members"]}-member acquisition committee',flush=True)
    committee = fit_committee(dataset,config['members'],config['seed']+number*100)
    with (workspace/'pool.csv').open(newline='') as stream:
        pool = list(csv.DictReader(stream))
    selected = select_batch(dataset,pool,committee,config['batch_size'],config['seed']+number,
                            config['strategy'],config['target_weights'])
    predictor_values = predict_bundle(model,[s['config'] for s in selected])
    for selection,values in zip(selected,predictor_values):
        selection['predictor_prediction'] = dict(zip(dataset.targets,values.astype(float).tolist()))
    anchors = read_json(workspace/'anchors.json')
    before,after,anchor_ids = [],[],[]
    for i,c in enumerate(anchors):
        a,b = f'ANCHOR_{number:04d}_{i}_PRE',f'ANCHOR_{number:04d}_{i}_POST'
        before.append(dict(c,config_id=a))
        after.append(dict(c,config_id=b))
        anchor_ids.append(dict(config_id=c['config_id'],before=a,after=b))
    atomic_bundle(rd/'predictor_before.joblib',model)
    atomic_bundle(rd/'committee.joblib',committee)
    schedule = before+[s['config'] for s in selected]+after
    # An uncommitted proposal has never been handed to a backend; retry safely.
    if (rd/'schedule.csv').exists():
        (rd/'schedule.csv').rename(rd/'schedule.uncommitted.csv')
    write_configs(rd/'schedule.csv',schedule)
    proposal = dict(round=number,dataset_sha256=dataset.fingerprint,targets=dataset.targets,
                    quality_policy=dict(anchor_gate=bool(anchors),measurement_validation=True),
                    thermal_policy=config.get('thermal_policy'),
                    predictor_origin=model_origin,
                    strategy=config['strategy'],selected=selected,anchors=anchor_ids,
                    schedule=schedule,validation_before=_evaluate(model,dataset,'validation'),
                    artifacts={name:digest(rd/name) for name in ['schedule.csv','predictor_before.joblib','committee.joblib']},
                    warning='Disagreement is an uncalibrated acquisition proxy. No objective rewards fast/low-energy architectures.')
    atomic_json(proposal_path,proposal)
    return proposal


def advance(workspace, backend=None, rounds=1, evaluate_test=False):
    """Advance bounded rounds. A missing backend pauses at the hardware boundary."""
    workspace = Path(workspace).resolve()
    if rounds < 1:
        raise ValueError('rounds must be positive')
    with locked(workspace/'RUNNING.lock'):
        config,state,dataset = verify_workspace(workspace)
        stop = min(state['completed_rounds']+rounds,config['max_rounds'])
        while state['completed_rounds'] < stop:
            number = state['completed_rounds']+1
            rd = workspace/'rounds'/f'{number:04d}'
            proposal = _proposal(workspace,rd,dataset,config,number)
            if state.get('pending_round')==number and state.get('proposal_sha256') not in (None,digest(rd/'proposal.json')):
                raise ValueError('Pending proposal was modified')
            result_path = rd/'measurements.csv'
            if not (rd/'ingested.json').exists():
                controls = f' + {len(proposal["anchors"])*2} anchor measurements' if proposal['anchors'] else ' (no anchors)'
                state.update(status='awaiting_measurements',message=f'Round {number}: {len(proposal["selected"])} candidates'+controls,pending_round=number,proposal_sha256=digest(rd/'proposal.json'))
                atomic_json(workspace/'state.json',state)
                if backend is not None:
                    backend(rd,proposal,dataset.protocol,config)
                if not result_path.exists():
                    print(state['message']+f'. Supply {result_path} or explicitly enable hardware execution.',flush=True)
                    return state
                with result_path.open(newline='') as stream:
                    results = list(csv.DictReader(stream))
                expected = {c['config_id']:c for c in proposal['schedule']}
                ids = [r.get('config_id') for r in results]
                if len(set(ids))!=len(ids) or set(ids)-set(expected):
                    raise ValueError('Duplicate or unexpected measurement IDs')
                checked,issues = {},{}
                for row in results:
                    try:
                        checked[row['config_id']] = result_metrics(row,expected[row['config_id']],dataset.protocol)
                    except (ValueError,KeyError,TypeError) as error:
                        issues[row['config_id']] = str(error)
                missing = sorted(set(expected)-set(checked))
                atomic_json(rd/'measurement_audit.json',dict(missing_or_invalid=missing,issues=issues,results_sha256=digest(result_path)))
                if missing:
                    state.update(status='awaiting_measurements',message=f'{len(missing)} missing/invalid measurements. Valid rows retained; no labels ingested.')
                    atomic_json(workspace/'state.json',state)
                    return state
                quality = anchor_report(proposal['anchors'],checked,state.get('anchor_reference'),config['anchor_tolerance'])
                atomic_json(rd/'anchor_report.json',quality)
                approval = read_json(rd/'drift_review.json') if (rd/'drift_review.json').exists() else None
                accepted = approval and approval.get('results_sha256')==digest(result_path) and approval.get('reason')
                if not quality['passed'] and not accepted:
                    state.update(status='paused_anchor_drift',message='Anchor drift gate failed. Inspect anchor_report.json; no labels ingested.')
                    atomic_json(workspace/'state.json',state)
                    return state
                rows = []
                attempt_audit = read_json(rd/'attempt_audit.json') if (rd/'attempt_audit.json').exists() else {}
                for selection in proposal['selected']:
                    c = selection['config']
                    rows.append(dict(measurement_id=f'active:{number}:{c["config_id"]}',config_id=c['config_id'],
                        round_id=f'active_{number:04d}',split='train',architecture=c,metrics=checked[c['config_id']]))
                    context = attempt_audit.get('accepted_sources',{}).get(c['config_id'],{}).get('thermal_policy')
                    if context:
                        rows[-1]['measurement_context'] = dict(thermal_policy=context)
                sources = {str(result_path):digest(result_path),str(rd/'schedule.csv'):digest(rd/'schedule.csv')}
                sources[str(rd/'proposal.json')] = digest(rd/'proposal.json')
                if (rd/'attempt_audit.json').exists():
                    sources[str(rd/'attempt_audit.json')] = digest(rd/'attempt_audit.json')
                child = dataset.append_training(dict(protocol=dataset.protocol,observations=rows,
                    source_hashes=sources))
                atomic_json(rd/'dataset_after.json',child.to_dict())
                prequential = [{**s,'actual':checked[s['config']['config_id']]} for s in proposal['selected']]
                atomic_json(rd/'prequential.json',prequential)
                atomic_json(rd/'ingested.json',dict(dataset_sha256=child.fingerprint,results_sha256=digest(result_path),
                    anchor_reference=quality.get('reference'),dataset_file_sha256=digest(rd/'dataset_after.json'),
                    drift_override=approval if not quality['passed'] else None))
            ingest = read_json(rd/'ingested.json')
            if digest(result_path)!=ingest['results_sha256'] or digest(rd/'dataset_after.json')!=ingest['dataset_file_sha256']:
                raise ValueError('Committed measurements/dataset were modified')
            child = MeasurementDataset.load(rd/'dataset_after.json')
            if not (rd/'completion.json').exists():
                print(f'Round {number}: retraining on {sum(r["split"]=="train" for r in child.observations)} observations',flush=True)
                model = fit_dataset(child,'xgboost',config['seed'])
                atomic_bundle(rd/'predictor_after.joblib',model)
                report = dict(round=number,new_architectures=len(proposal['selected']),hardware_measurements=len(proposal['schedule']),
                    quality_policy=proposal.get('quality_policy',dict(anchor_gate=bool(proposal['anchors']),measurement_validation=True)),
                    validation_before=proposal['validation_before'],validation_after=_evaluate(model,child,'validation'),
                    dataset_sha256=child.fingerprint,model_sha256=digest(rd/'predictor_after.joblib'),
                    drift_override=ingest.get('drift_override'),
                    warning='Validation is used for stopping and is not an unbiased final test. No automatic best-model promotion.')
                if evaluate_test:
                    report['exploratory_test_after'] = _evaluate(model,child,'test')
                atomic_json(rd/'completion.json',report)
            completion = read_json(rd/'completion.json')
            if completion['dataset_sha256']!=child.fingerprint or completion['model_sha256']!=digest(rd/'predictor_after.joblib'):
                raise ValueError('Committed round model mismatch')
            state.update(completed_rounds=number,current_dataset=str((rd/'dataset_after.json').relative_to(workspace)),
                         dataset_sha256=child.fingerprint,status='ready',message=f'Round {number} complete',
                         anchor_reference=ingest['anchor_reference'],latest_model=str((rd/'predictor_after.joblib').relative_to(workspace)))
            state.pop('pending_round',None)
            state.pop('proposal_sha256',None)
            atomic_json(workspace/'state.json',state)
            dataset = child
        if state['completed_rounds']>=config['max_rounds']:
            state.update(status='budget_complete',message='Configured round budget exhausted; no further measurements scheduled')
            atomic_json(workspace/'state.json',state)
        return state
