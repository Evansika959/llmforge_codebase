"""Publish a new 2,000-row snapshot with coherent GS64 remeasurements."""
import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import zipfile

import numpy as np

from .data import PROTOCOL_KEYS, TARGETS
from scripts.sweep.layerwise.candidates import fingerprint
from scripts.sweep.layerwise.measurement import parse

ROOT = Path(__file__).resolve().parents[3]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def replace_rows(old_rows, replacements):
    """Pure identity-preserving join; no partial replacement or metric splicing."""
    ids = [r['candidate_id'] for r in old_rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate baseline architecture IDs')
    wanted = {r['candidate_id'] for r in old_rows if r['architecture']['q8_group_size'] == 64}
    if set(replacements) != wanted:
        raise ValueError('Replacement IDs must equal the complete GS64 subset')
    result = []
    for old in old_rows:
        ident = old['candidate_id']
        if ident not in replacements:
            result.append(copy.deepcopy(old))
            continue
        new = replacements[ident]
        for key in ('candidate_id', 'architecture', 'split', 'permutation_group'):
            if new[key] != old[key]:
                raise ValueError('Replacement identity/split changed: '+key)
        row = copy.deepcopy(old)
        row.update(copy.deepcopy(new))
        result.append(row)
    return result


def statistics(rows):
    result = []
    for family in ['all'] + sorted({r['architecture']['family'] for r in rows}):
        for group in [None, 16, 32, 64]:
            take = [r for r in rows if (family == 'all' or r['architecture']['family'] == family)
                    and (group is None or r['architecture']['q8_group_size'] == group)]
            if not take:
                continue
            metrics = {}
            for t in TARGETS:
                values = np.array([r['metrics'][t] for r in take if r['metrics'][t] is not None], float)
                metrics[t] = dict(n=len(values), median=float(np.median(values)),
                                  p10=float(np.quantile(values, .1)), p90=float(np.quantile(values, .9)))
            result.append(dict(family=family, group=group, n=len(take), metrics=metrics))
    return result


def publish(baseline, campaign, output):
    baseline, campaign, output = map(lambda p: Path(p).resolve(), (baseline, campaign, output))
    if output.exists():
        raise FileExistsError('Refusing to overwrite an existing release')
    old_path = baseline/'dataset_snapshot.json'
    old = json.loads(old_path.read_text())
    old_hashes = json.loads((baseline/'release_hashes.json').read_text())
    for name, expected in old_hashes.items():
        if sha(baseline/name) != expected:
            raise ValueError('Original release integrity failure: '+name)
    replay = json.loads((campaign/'replay_manifest.json').read_text())
    if sha(old_path) != replay['source_dataset_sha256']:
        raise ValueError('Replay used a different baseline dataset')
    contract = json.loads((campaign/'hardware_contract.json').read_text())
    if replay['new_kernel_sha256'] != contract['kernel_sha256']:
        raise ValueError('Optimized kernel identity mismatch')
    common = {k: old['protocol'][k] for k in PROTOCOL_KEYS if k != 'kernel_sha256'}
    if any(contract[k] != v for k, v in common.items()):
        raise ValueError('Workload/device protocol changed beyond the kernel')
    if len(old['rows']) != 2000:
        raise ValueError('Expected 2000 baseline architectures')
    old_by_id = {r['candidate_id']: r for r in old['rows']}
    replacements = {}; full_results = {}; source_files = {}; changes = []
    db = campaign/'candidates.sqlite'
    # Keep one SQLite read transaction while freezing accepted replay rows.
    with sqlite3.connect(db.as_uri()+'?mode=ro', uri=True) as con:
        con.row_factory = sqlite3.Row
        con.execute('BEGIN')
        if con.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise ValueError('Replay registry integrity failure')
        if con.execute("SELECT count(*) FROM jobs WHERE status='complete'").fetchone()[0] != 664:
            raise ValueError('The GS64 replay must be complete')
        measured = con.execute("""SELECT c.candidate_id,c.architecture_json,c.split,c.permutation_group,
            m.artifact_path,m.kernel_sha256,m.protocol_sha256,m.decode_tok_s,m.ttft_ms,
            m.dynamic_energy_per_token_mj FROM candidates c JOIN jobs j USING(candidate_id)
            JOIN measurements m USING(candidate_id) WHERE m.accepted=1 AND j.status='complete'""").fetchall()
        for m in measured:
            ident = m['candidate_id']
            if ident in replacements:
                raise ValueError('Duplicate accepted replay labels')
            if m['kernel_sha256'] != contract['kernel_sha256'] or m['protocol_sha256'] != fingerprint(contract):
                raise ValueError('Replay row protocol mismatch')
            path = campaign/m['artifact_path']/'result.json'
            raw = json.loads(path.read_text()); parsed = parse(path.parent)
            if raw != parsed:
                raise ValueError('Replay result does not reproduce from raw trace/timing')
            provenance = json.loads(path.with_name('provenance.json').read_text())
            if provenance['model']['model_sha256'] != replay['expected_model_hashes'][ident]:
                raise ValueError('Replay weights differ from baseline')
            if any(raw[t] != m[t] for t in TARGETS):
                raise ValueError('Replay database/raw labels differ')
            previous = old_by_id[ident]
            old_raw = json.loads(Path(previous['artifact_path']).read_text())
            if old_raw['timing']['last_token'] != raw['timing']['last_token']:
                raise ValueError('Final generated token mismatch')
            replacements[ident] = dict(candidate_id=ident, architecture=json.loads(m['architecture_json']),
                split=m['split'], permutation_group=m['permutation_group'], metrics={t:raw[t] for t in TARGETS},
                source=str(campaign), protocol_sha256=m['protocol_sha256'], artifact_path=str(path),
                artifact_sha256=sha(path), energy_warning=raw.get('energy_warning'))
            changes.append(dict(candidate_id=ident, old_metrics=previous['metrics'], new_metrics=replacements[ident]['metrics'],
                                old_artifact_sha256=previous['artifact_sha256'], new_artifact_sha256=sha(path),
                                model_sha256=provenance['model']['model_sha256'], last_token_matches=True))
    rows = replace_rows(old['rows'], replacements)
    if len(replacements) != 664:
        raise ValueError('Expected 664 replacements')
    groups = {}
    for row in rows:
        ident = row['candidate_id']; before = old_by_id[ident]
        for key in ('architecture', 'split', 'permutation_group', 'cohort'):
            if row[key] != before[key]:
                raise ValueError('Original architecture order/split/cohort changed')
        if groups.setdefault(row['permutation_group'], row['split']) != row['split']:
            raise ValueError('Permutation-group split leakage')
        path = Path(row['artifact_path'])
        if sha(path) != row['artifact_sha256']:
            raise ValueError('Measurement artifact hash mismatch')
        raw = json.loads(path.read_text())
        if any(raw[t] != row['metrics'][t] for t in TARGETS):
            raise ValueError('Snapshot/raw label mismatch')
        if any(not isinstance(raw[t], (int,float)) or not np.isfinite(raw[t]) or raw[t] <= 0 for t in TARGETS):
            raise ValueError('Updated dataset must have all positive valid target labels')
        row['measurement_revision'] = 'gs64_decode_v1' if ident in replacements else 'baseline_retained'
        row['kernel_sha256'] = contract['kernel_sha256'] if ident in replacements else old['protocol']['kernel_sha256']
        row['original_artifact_path'] = str(path)
        row['artifact_relative_path'] = f'evidence/results/{ident}.json'
        row['artifact_path'] = str(output/row['artifact_relative_path'])
        dt = raw['timing']['end']-raw['timing']['prefill_end']
        row['derived_metrics'] = dict(decode_duration_s=dt,tpot_ms=1000*dt/31,
                                     gross_energy_per_token_mj=raw.get('gross_energy_per_token_mj'),
                                     decode_gross_energy_per_forward_mj=raw.get('decode_gross_energy_per_forward_mj'))
        full_results[ident] = raw; source_files[ident] = path
    output.mkdir(parents=True, exist_ok=False)
    (output/'evidence/results').mkdir(parents=True)
    (output/'provenance').mkdir()
    with zipfile.ZipFile(output/'measurement_evidence.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        for ident, path in source_files.items():
            shutil.copy2(path,output/f'evidence/results/{ident}.json')
            for name in ('trace.csv','timing.json','infer.log','provenance.json','result.json','exit.json','command.json'):
                source = path.with_name(name)
                if source.exists():
                    archive.write(source,f'{ident}/{name}')
    shutil.copy2(old_path,output/'provenance/baseline_dataset_snapshot.json')
    shutil.copy2(campaign/'hardware_contract.json',output/'provenance/optimized_hardware_contract.json')
    shutil.copy2(campaign/'replay_manifest.json',output/'provenance/replay_manifest.json')
    shutil.copy2(campaign/'kernel_validation.json',output/'provenance/kernel_validation.json')
    # Keep both actual C implementations as evidence, not only their hashes.
    shutil.copy2(campaign/'baseline/runq_llmforge.c',output/'provenance/runq_llmforge_baseline.c')
    shutil.copy2(campaign/'source_snapshot/src/runq_llmforge.c',output/'provenance/runq_llmforge_optimized.c')
    with sqlite3.connect(db.as_uri()+'?mode=ro',uri=True) as src:
        with sqlite3.connect(output/'provenance/gs64_registry.sqlite') as dst:
            src.backup(dst)
    data = dict(rows=rows, sources=[dict(source=s,count=n) for s,n in sorted(Counter(r['source'] for r in rows).items())],
                dataset_version='layerwise_2000_gs64_refresh_20260922',
                protocol=dict(version='mixed_kernel_layerwise_gs64_refresh_v1',common_measurement_protocol=common,
                    kernel_by_group={str(g):contract['kernel_sha256'] if g==64 else old['protocol']['kernel_sha256'] for g in (16,32,64)},
                    note='GS16/32 retain historical measurements. Only GS64 was remeasured with the optimized source; not a single-kernel contemporaneous campaign.'))
    write(output/'dataset_snapshot.json',data)
    write(output/'replacement_audit.json',dict(baseline_dataset_sha256=sha(old_path),replacements=changes,
        unchanged_ids=[r['candidate_id'] for r in rows if r['candidate_id'] not in replacements]))
    write(output/'plotted_records.json',[dict(candidate_id=r['candidate_id'],family=r['architecture']['family'],
        group=r['architecture']['q8_group_size'],split=r['split'],metrics=r['metrics'],energy_warning=r['energy_warning'],
        measurement_revision=r['measurement_revision']) for r in rows])
    with sqlite3.connect(output/'dataset.sqlite') as con:
        con.execute('CREATE TABLE observations (candidate_id TEXT PRIMARY KEY, split TEXT, family TEXT, group_size INTEGER, decode_tok_s REAL, ttft_ms REAL, dynamic_energy_per_token_mj REAL, kernel_sha256 TEXT, row_json TEXT)')
        con.executemany('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?)',[(r['candidate_id'],r['split'],r['architecture']['family'],r['architecture']['q8_group_size'],*(r['metrics'][t] for t in TARGETS),r['kernel_sha256'],json.dumps(r)) for r in rows])
    summary = dict(created_utc=datetime.now(timezone.utc).isoformat(),rows=len(rows),replaced=664,retained=1336,
        splits=dict(Counter(r['split'] for r in rows)),groups=dict(Counter(str(r['architecture']['q8_group_size']) for r in rows)),
        valid_targets={t:len(rows) for t in TARGETS},warnings=dict(Counter(str(r['energy_warning']) for r in rows)),
        replaced_warnings=dict(Counter(str(r['energy_warning']) for r in rows if r['candidate_id'] in replacements)),
        dataset_sha256=sha(output/'dataset_snapshot.json'),baseline_dataset_sha256=sha(old_path),statistics=statistics(rows),
        no_retraining=True,protocol=data['protocol'])
    write(output/'manifest.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('statistics','protocol')},indent=2),flush=True)
    return data, old


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,default=ROOT/'diliverable/layerwise_2000_20260921')
    p.add_argument('--campaign',type=Path,default=ROOT/'scripts/sweep/outputs/watch5_gs64_decode_v1_664')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    publish(args.baseline,args.campaign,args.output)


if __name__=='__main__':main()
