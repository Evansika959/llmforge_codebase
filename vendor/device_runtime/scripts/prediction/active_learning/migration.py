"""Explicit, copy-on-migration removal of anchors; never accesses hardware.

Only acquisition/quality-policy code changes can be acknowledged here. Changes
to inference, measurement workers, features or training require another review.
The measurement/target protocol stays unchanged; the quality policy is versioned
separately so existing checkpoints and complete candidate labels remain usable.
"""
import copy
import csv
from datetime import datetime, timezone
from pathlib import Path
import shutil

from ..data.dataset import MeasurementDataset
from .engine import source_hashes, verify_workspace
from .storage import atomic_json, digest, locked, read_json, write_configs

POLICY_FILES = {f'scripts/prediction/active_learning/{name}.py'
                for name in ['engine','hardware','quality','cli','migration']}


def _source_state(source, accept_source_changes, allowed_files=POLICY_FILES):
    if (source/'MIGRATION_IN_PROGRESS.json').exists():
        raise ValueError('Cannot migrate an incomplete migration')
    contract = read_json(source/'contract.json')
    for name, sha in contract['inputs'].items():
        if digest(source/name) != sha:
            raise ValueError('Workspace input changed: '+name)
    current = source_hashes()
    previous = contract['source_hashes']
    changes = {p:dict(before=previous.get(p),after=current.get(p))
               for p in previous.keys() | current.keys() if previous.get(p)!=current.get(p)}
    if set(changes)-allowed_files:
        raise ValueError('Non-anchor/unreviewed source changes require separate protocol review: '+str(sorted(set(changes)-allowed_files)))
    if changes and not accept_source_changes:
        raise ValueError('Review anchor-policy source changes and explicitly pass --accept-source-changes')
    state = read_json(source/'state.json')
    parent = MeasurementDataset.load(source/'dataset_000.json')
    for number in range(1,state['completed_rounds']+1):
        rd = source/'rounds'/f'{number:04d}'
        proposal, ingest, completion = [read_json(rd/name) for name in ['proposal.json','ingested.json','completion.json']]
        for name, sha in proposal['artifacts'].items():
            if digest(rd/name)!=sha:
                raise ValueError('Committed proposal artifact changed: '+str(rd/name))
        child = MeasurementDataset.load(rd/'dataset_after.json')
        if (child.to_dict().get('parent_sha256')!=parent.fingerprint or
                proposal['dataset_sha256']!=parent.fingerprint or child.protocol!=contract['protocol'] or
                ingest['dataset_sha256']!=child.fingerprint or completion['dataset_sha256']!=child.fingerprint or
                digest(rd/'dataset_after.json')!=ingest['dataset_file_sha256'] or
                digest(rd/'measurements.csv')!=ingest['results_sha256'] or
                digest(rd/'predictor_after.joblib')!=completion['model_sha256']):
            raise ValueError('Committed round integrity mismatch: '+str(number))
        parent = child
    dataset = MeasurementDataset.load(source/state['current_dataset'])
    if dataset.fingerprint!=state['dataset_sha256'] or dataset.fingerprint!=parent.fingerprint or dataset.protocol!=contract['protocol']:
        raise ValueError('Current dataset integrity/protocol mismatch')
    # Do not carry unresolved device mutations into the new workspace.
    for snapshot in source.glob('rounds/*/attempts/*/original_device_state.json'):
        restoration = snapshot.with_name('restoration.json')
        if not restoration.exists() or not read_json(restoration).get('settings_restored'):
            raise ValueError('Restore device settings before migration: '+str(snapshot))
    return contract, state, dataset, changes


def disable_anchors(source, destination, reason, accept_source_changes=False):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not reason.strip():
        raise ValueError('A nonempty policy-change reason is required')
    if destination.exists() or source==destination or source in destination.parents:
        raise ValueError('Migration requires a new destination outside the source workspace')
    with locked(source/'RUNNING.lock'):
        contract, state, dataset, changes = _source_state(source,accept_source_changes)
        settings = read_json(source/'settings.json')
        if not settings['anchors']:
            raise ValueError('Anchors are already disabled')
        pending = state.get('pending_round')
        proposal = None
        if pending is not None:
            rd = source/'rounds'/f'{pending:04d}'
            if pending!=state['completed_rounds']+1 or (rd/'ingested.json').exists():
                raise ValueError('Finish committed ingestion before migrating a pending round')
            proposal = read_json(rd/'proposal.json')
            if proposal['dataset_sha256']!=dataset.fingerprint or digest(rd/'proposal.json')!=state['proposal_sha256']:
                raise ValueError('Pending proposal integrity mismatch')
            for name, sha in proposal['artifacts'].items():
                if digest(rd/name)!=sha:
                    raise ValueError('Frozen acquisition artifact changed: '+name)
        # Verify byte-for-byte copy and retain a complete provenance manifest.
        manifest = {str(p.relative_to(source)):digest(p) for p in source.rglob('*')
                    if p.is_file() and p.name!='RUNNING.lock'}
        destination.mkdir(parents=True,exist_ok=False)
        marker = destination/'MIGRATION_IN_PROGRESS.json'
        atomic_json(marker,dict(source=str(source),reason=reason.strip()))
        shutil.copytree(source,destination,dirs_exist_ok=True,ignore=shutil.ignore_patterns('RUNNING.lock'))
        for name, sha in manifest.items():
            if digest(destination/name)!=sha:
                raise ValueError('Migration copy integrity mismatch: '+name)
        audit = destination/'migrations/0001_disable_anchors'
        backup = audit/'before'
        def preserve(relative):
            path = destination/relative
            if path.exists():
                target = backup/relative
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(path,target)
        for name in ['contract.json','settings.json','anchors.json','state.json']:
            preserve(name)
        policy = dict(version=2,anchor_gate=False,measurement_validation=True,
                      reason=reason.strip(),migration='migrations/0001_disable_anchors/record.json')
        settings['anchors'] = 0
        atomic_json(destination/'settings.json',settings)
        atomic_json(destination/'anchors.json',[])
        if proposal is not None:
            relative = Path('rounds')/f'{pending:04d}'
            rd = destination/relative
            for name in ['proposal.json','schedule.csv','measurements.csv','measurement_audit.json','anchor_report.json','drift_review.json']:
                preserve(relative/name)
            revised = copy.deepcopy(proposal)
            selected_ids = {s['config']['config_id'] for s in revised['selected']}
            retired_ids = {a[k] for a in revised['anchors'] for k in ['before','after']}
            if selected_ids & retired_ids or {c['config_id'] for c in revised['schedule']}!=selected_ids|retired_ids:
                raise ValueError('Pending schedule does not match selected candidates and anchors')
            revised['retired_anchor_configs'] = [c for c in revised['schedule'] if c['config_id'] in retired_ids]
            revised['schedule'] = [c for c in revised['schedule'] if c['config_id'] in selected_ids]
            revised['anchors'] = []
            revised['quality_policy'] = policy
            # The before/ snapshots retain the original frozen acquisition plan.
            (rd/'schedule.csv').unlink()
            write_configs(rd/'schedule.csv',revised['schedule'])
            revised['artifacts']['schedule.csv'] = digest(rd/'schedule.csv')
            atomic_json(rd/'proposal.json',revised)
            if (rd/'measurements.csv').exists():
                with (rd/'measurements.csv').open(newline='') as stream:
                    rows = list(csv.DictReader(stream))
                ids = [r.get('config_id') for r in rows]
                if len(ids)!=len(set(ids)) or set(ids)-(selected_ids|retired_ids):
                    raise ValueError('Duplicate or unexpected pending measurement IDs')
                (rd/'measurements.csv').unlink()
                retained = [r for r in rows if r['config_id'] in selected_ids]
                if retained:
                    write_configs(rd/'measurements.csv',retained)
            for name in ['measurement_audit.json','anchor_report.json','drift_review.json']:
                if (rd/name).exists():
                    (rd/name).unlink()  # Old policy reports retained in before/.
            state.update(status='awaiting_measurements',proposal_sha256=digest(rd/'proposal.json'),
                         message='Anchor requirement explicitly removed; candidate results await normal validation/ingestion')
        state.pop('anchor_reference',None)
        state['quality_policy'] = policy
        atomic_json(destination/'state.json',state)
        record = dict(created_utc=datetime.now(timezone.utc).isoformat(),source=str(source),
                      reason=reason.strip(),quality_policy=policy,source_changes=changes,
                      source_file_hashes=manifest,completed_rounds_preserved=state['completed_rounds'],
                      pending_round=pending,measurement_protocol_unchanged=True,
                      warning='Anchor drift was not assessed for the migrated pending batch; no new labels ingested by migration')
        atomic_json(audit/'record.json',record)
        updated = copy.deepcopy(contract)
        updated.update(source_hashes=source_hashes(),quality_policy=policy)
        updated['inputs'] = {name:digest(destination/name) for name in updated['inputs']}
        updated['inputs'][str((audit/'record.json').relative_to(destination))] = digest(audit/'record.json')
        atomic_json(destination/'contract.json',updated)
        marker.unlink()
        verify_workspace(destination)
    return destination


def change_temperature(source, destination, reason, accept_source_changes=False):
    """Reviewed 45 C admission amendment, retaining historical label provenance.

The dataset protocol remains the historical workload/checkpoint identity. Its
temperature field is NOT a claim that all new labels were collected below 40 C:
the run contract and per-attempt/per-observation thermal context are authoritative.
"""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not reason.strip() or destination.exists() or source in destination.parents:
        raise ValueError('A reason and a new destination outside the source are required')
    allowed = POLICY_FILES | {'scripts/prediction/active_learning/worker.py','scripts/prediction/active_learning/thermal.py'}
    with locked(source/'RUNNING.lock'):
        contract, state, dataset, changes = _source_state(source,accept_source_changes,allowed)
        config = read_json(source/'settings.json')
        if config.get('thermal_policy') or config['temperature_ceiling']!=40:
            raise ValueError('This migration only supports the original 40 C admission policy')
        number = state.get('pending_round')
        if number is not None:
            rd = source/'rounds'/f'{number:04d}'
            proposal = read_json(rd/'proposal.json')
            if number!=state['completed_rounds']+1 or (rd/'ingested.json').exists():
                raise ValueError('Finish committed ingestion before migrating')
            if digest(rd/'proposal.json')!=state['proposal_sha256'] or proposal['dataset_sha256']!=dataset.fingerprint:
                raise ValueError('Pending proposal integrity mismatch')
            for name,sha in proposal['artifacts'].items():
                if digest(rd/name)!=sha:
                    raise ValueError('Pending artifact changed: '+name)
        manifest = {str(p.relative_to(source)):digest(p) for p in source.rglob('*') if p.is_file() and p.name!='RUNNING.lock'}
        destination.mkdir(parents=True,exist_ok=False)
        marker = destination/'MIGRATION_IN_PROGRESS.json'
        atomic_json(marker,dict(source=str(source),reason=reason))
        shutil.copytree(source,destination,dirs_exist_ok=True,ignore=shutil.ignore_patterns('RUNNING.lock'))
        for name,sha in manifest.items():
            if digest(destination/name)!=sha:
                raise ValueError('Migration copy mismatch: '+name)
        audit = destination/'migrations/0002_strict_under45'
        originals = ['contract.json','settings.json','state.json']
        if number is not None:
            originals.append(f'rounds/{number:04d}/proposal.json')
        for name in originals:
            path = audit/'before'/name
            path.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(destination/name,path)
        policy = dict(version=1,ceiling_c=45.,comparison='lt',previous_ceiling_c=dataset.protocol['temperature_ceiling'],
                      valid_cooldown_timeout=False,invalid_telemetry_timeout_seconds=180,
                      inference_temperature_polling=False,reason=reason.strip(),
                      migration='migrations/0002_strict_under45/record.json')
        config.update(temperature_ceiling=45.,thermal_policy=policy)
        atomic_json(destination/'settings.json',config)
        if number is not None:
            rd = destination/'rounds'/f'{number:04d}'
            proposal['thermal_policy'] = policy
            proposal['thermal_transition'] = 'Existing complete rows retain the old admission conditions; only remaining/new attempts use strict <45 C.'
            atomic_json(rd/'proposal.json',proposal)
            state['proposal_sha256'] = digest(rd/'proposal.json')
        state['thermal_policy'] = policy
        atomic_json(destination/'state.json',state)
        atomic_json(audit/'record.json',dict(source=str(source),created_utc=datetime.now(timezone.utc).isoformat(),
                    source_file_hashes=manifest,source_changes=changes,reason=reason.strip(),thermal_policy=policy,
                    warning='Admission conditions changed, not workload/energy definitions. Historical 40 C and new 45 C labels may have different state distributions; thermal contexts are not predictor inputs.'))
        contract.update(source_hashes=source_hashes(),thermal_policy=policy)
        contract['inputs'] = {name:digest(destination/name) for name in contract['inputs']}
        contract['inputs'][str((audit/'record.json').relative_to(destination))] = digest(audit/'record.json')
        atomic_json(destination/'contract.json',contract)
        marker.unlink()
        verify_workspace(destination)
    return destination
