"""Audited gross-to-dynamic learning-stage transition; no device access."""
import copy
import csv
from datetime import datetime, timezone
from pathlib import Path
import shutil

from ..data.dataset import MeasurementDataset
from ..training.pipeline import fit_dataset
from .engine import initialize, source_hashes, verify_workspace
from .migration import _source_state
from .quality import result_metrics
from .sampling import unseen_candidates
from .storage import atomic_json, atomic_bundle, digest, locked, read_json

ALLOWED = {f'scripts/prediction/active_learning/{name}.py' for name in
           ['cli','datasets','quality','energy','monitoring','replay']}


def switch_to_dynamic(source, destination, legacy_snapshot, reason, accept_source_changes=False,
                      quarantine_invalid_training_energy=False):
    source,destination,legacy_snapshot = map(lambda p:Path(p).resolve(),[source,destination,legacy_snapshot])
    if not reason.strip() or destination.exists() or source in destination.parents:
        raise ValueError('Supply a reason and a new destination outside the source workspace')
    with locked(source/'RUNNING.lock'):
        contract,state,old,changes = _source_state(source,accept_source_changes,ALLOWED)
        if old.protocol['energy_target']!='gross':
            raise ValueError('This transition expects a gross source workspace')
        config = read_json(source/'settings.json')
        if config['anchors']:
            raise ValueError('Use a reviewed no-anchor workspace for this energy transition')
        protocol = old.protocol
        protocol.update(energy_target='dynamic',energy_label_validation='baseline_reconciled_v1',
            protocol_id=protocol['protocol_id']+'_dynamic_v1',
            energy_definition='Recorded max(active-baseline,0)*inference_duration*1000/output_tokens; includes prefill; validated within rounding tolerance')
        sources = {str(source/'contract.json'):digest(source/'contract.json'),
                   str(source/state['current_dataset']):digest(source/state['current_dataset'])}
        legacy = {}
        if any(not r['round_id'].startswith('active_') for r in old.observations):
            snapshot = read_json(legacy_snapshot)
            sources[str(legacy_snapshot)] = digest(legacy_snapshot)
            for name in ['batch1','batch2']:
                for row in snapshot['records'][name]:
                    if row['config_id'] in legacy:
                        raise ValueError('Duplicate legacy measurement ID')
                    legacy[row['config_id']] = row
        cache = {}
        def measured(number):
            if number not in cache:
                path = source/f'rounds/{number:04d}/measurements.csv'
                sources[str(path)] = digest(path)
                with path.open(newline='') as stream:
                    rows = list(csv.DictReader(stream))
                mapping = {r['config_id']:r for r in rows}
                if len(mapping)!=len(rows):
                    raise ValueError('Duplicate measured architecture ID')
                cache[number] = mapping
            return cache[number]
        converted, audits, quarantined = [],[],[]
        def convert(obs, raw):
            config_row = dict(obs['architecture'],config_id=obs['config_id'])
            original_metrics = result_metrics(raw,config_row,old.protocol)
            for name in ['decode_tok_s','ttft_ms']:
                if abs(original_metrics[name]-obs['metrics'][name])>1e-6:
                    raise ValueError('Historical performance label changed: '+obs['config_id'])
            gross = float(raw['total_energy_j'])*1000/protocol['output_tokens']
            if abs(gross-obs['metrics']['gross_energy_per_token_mj'])>1e-5:
                raise ValueError('Historical gross label changed: '+obs['config_id'])
            try:
                metrics = result_metrics(raw,config_row,protocol)
            except (ValueError,KeyError,TypeError) as error:
                if not quarantine_invalid_training_energy or obs['split']!='train':
                    raise ValueError(f'{obs["config_id"]} ({obs["split"]}): {error}') from error
                quarantined.append(dict(observation=copy.deepcopy(obs),raw_row=copy.deepcopy(raw),reason=str(error)))
                return None
            item = copy.deepcopy(obs)
            item['metrics'].update(metrics)
            audits.append(dict(config_id=item['config_id'],measurement_id=item['measurement_id'],split=item['split'],
                baseline_power_w=float(raw['baseline_power_w']),active_power_w=float(raw['active_power_w']),
                duration_s=float(raw['duration_s']),dynamic_energy_per_token_mj=metrics['dynamic_energy_per_token_mj']))
            return item
        for obs in old.observations:
            raw = (measured(int(obs['round_id'].split('_')[-1]))[obs['config_id']]
                   if obs['round_id'].startswith('active_') else legacy[obs['config_id']])
            item = convert(obs,raw)
            if item is not None:
                converted.append(item)
        # Complete labels from a partial gross batch become seed observations of
        # the new stage, NOT a fabricated completion of that old round. Unmeasured
        # candidates return to the pool for dynamic-driven acquisition.
        pending = state.get('pending_round')
        recovered,returned = [],[]
        if pending is not None:
            rd = source/f'rounds/{pending:04d}'
            if (rd/'ingested.json').exists():
                raise ValueError('Finish already-committed ingestion before switching targets')
            proposal_path = rd/'proposal.json'
            proposal = read_json(proposal_path)
            if digest(proposal_path)!=state['proposal_sha256'] or proposal['dataset_sha256']!=old.fingerprint:
                raise ValueError('Pending proposal integrity mismatch')
            for name,sha in proposal['artifacts'].items():
                if digest(rd/name)!=sha:
                    raise ValueError('Pending acquisition artifact changed')
            sources[str(proposal_path)] = digest(proposal_path)
            selected = {s['config']['config_id']:s['config'] for s in proposal['selected']}
            raw_rows = measured(pending) if (rd/'measurements.csv').exists() else {}
            attempt_audit = {}
            if (rd/'attempt_audit.json').exists():
                attempt_audit = read_json(rd/'attempt_audit.json')
                sources[str(rd/'attempt_audit.json')] = digest(rd/'attempt_audit.json')
            if set(raw_rows)-set(selected):
                raise ValueError('Unexpected pending result IDs')
            for ident,c in selected.items():
                if ident not in raw_rows:
                    returned.append(ident)
                    continue
                raw = raw_rows[ident]
                old_metrics = result_metrics(raw,c,old.protocol)
                obs = dict(measurement_id=f'transition:source_round_{pending}:{ident}',config_id=ident,
                    round_id=f'source_pending_{pending:04d}',split='train',architecture=c,metrics=old_metrics)
                context = attempt_audit.get('accepted_sources',{}).get(ident,{}).get('thermal_policy')
                if context:
                    obs['measurement_context'] = dict(thermal_policy=context)
                item = convert(obs,raw)
                if item is not None:
                    converted.append(item)
                    recovered.append(ident)
                else:
                    returned.append(ident)
        # Preserve old measurement IDs and split assignments; replace only the
        # target contract and derive dynamic labels from actual recorded power.
        document = dict(schema_version=1,protocol=protocol,observations=converted,
            source_hashes=sources,energy_transition=dict(source_workspace=str(source),
                source_dataset_sha256=old.fingerprint,source_completed_rounds=state['completed_rounds'],
                recovered_pending_measurements=recovered,returned_unmeasured_candidates=returned,
                quarantined_training_ids=[r['observation']['config_id'] for r in quarantined],
                reason=reason.strip(),prior_ingestions=old.to_dict().get('ingestions',[])))
        dataset = MeasurementDataset(document)
        with (source/'pool.csv').open(newline='') as stream:
            pool = unseen_candidates(dataset,list(csv.DictReader(stream)))
        sources[str(source/'pool.csv')] = digest(source/'pool.csv')
        remaining = config['max_rounds']-state['completed_rounds']
        config['max_rounds'] = min(remaining,len(pool)//config['batch_size'])
        if config['max_rounds']<1:
            raise ValueError('No remaining acquisition budget')
        if quarantined:
            print(f'Quarantined {len(quarantined)} invalid training energy labels; original rows retained in provenance. Holdouts unchanged.',flush=True)
        print(f'Training dynamic initial checkpoint on {sum(r["split"]=="train" for r in converted)} observations; no hardware access.',flush=True)
        model = fit_dataset(dataset,'xgboost',config['seed'])
        for path,sha in sources.items():
            if digest(path)!=sha:
                raise ValueError('Frozen migration input changed: '+path)
        initialize(destination,dataset,pool,config)
        marker = destination/'MIGRATION_IN_PROGRESS.json'
        atomic_json(marker,dict(source=str(source),kind='gross_to_dynamic'))
        provenance = destination/'provenance'
        provenance.mkdir()
        # Keep local copies of every input used for label conversion. Old models
        # and full trace history remain in the unmodified source workspace.
        copies = {}
        for i,(path,sha) in enumerate(sources.items()):
            target = provenance/f'{i:03d}_{Path(path).name}'
            shutil.copy2(path,target)
            if digest(target)!=sha:
                raise ValueError('Provenance copy mismatch')
            copies[str(target.relative_to(destination))] = sha
        if (source/'migrations').exists():
            shutil.copytree(source/'migrations',destination/'migrations')
        if (source/'device_identity.json').exists():
            shutil.copy2(source/'device_identity.json',destination/'device_identity.json')
        atomic_bundle(destination/'initial_model.joblib',model)
        record = dict(created_utc=datetime.now(timezone.utc).isoformat(),reason=reason.strip(),
            source_workspace=str(source),source_changes=changes,source_file_hashes=sources,
            source_completed_rounds=state['completed_rounds'],old_energy_target='gross',new_energy_target='dynamic',
            inherited_observations=len(old.observations),recovered_pending_measurements=recovered,
            quarantine_invalid_training_energy=quarantine_invalid_training_energy,quarantined_training=quarantined,
            returned_unmeasured_candidates=returned,training_rows=sum(r['split']=='train' for r in converted),
            validation_rows=sum(r['split']=='validation' for r in converted),test_rows=sum(r['split']=='test' for r in converted),
            label_audit=audits,initial_model_sha256=digest(destination/'initial_model.joblib'),
            note='New dynamic stage starts at round 1. No old round is relabeled as completed, no gross uncertainty/predictions are reused, and no labels are silently filtered. Explicitly quarantined invalid training labels remain in this audit and their architectures can be selected again from the pool. Invalid holdout labels always block migration. Fixed holdout identities remain unchanged.')
        atomic_json(provenance/'energy_transition.json',record)
        new_contract = read_json(destination/'contract.json')
        new_contract['inputs'].update(copies)
        for name in ['initial_model.joblib','provenance/energy_transition.json']:
            new_contract['inputs'][name]=digest(destination/name)
        new_contract['source_hashes']=source_hashes()
        atomic_json(destination/'contract.json',new_contract)
        marker.unlink()
        verify_workspace(destination)
    return destination
