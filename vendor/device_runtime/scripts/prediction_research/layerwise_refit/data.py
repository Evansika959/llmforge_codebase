"""Transactionally snapshot completed rows, preserving registry splits."""
import hashlib
import json
import sqlite3
from pathlib import Path

TARGETS = ['decode_tok_s', 'ttft_ms', 'dynamic_energy_per_token_mj']
PROTOCOL_KEYS = ['version', 'hardware_serial', 'abi', 'os_build', 'kernel_sha256',
                 'wrapper_sha256', 'prompt', 'prompt_tokens', 'output_tokens',
                 'decode_forwards', 'cpu_threads', 'cpuset', 'energy',
                 'power_interval_ms', 'pre_idle_s', 'post_idle_s',
                 'inference_temperature_polling', 'temperature_admission_lt_c',
                 'battery_stop_le_percent', 'synthetic_weights']


def snapshot(folders):
    rows, sources, groups, ids = [], [], {}, set()
    reference = None
    for index, folder in enumerate(map(Path, folders)):
        contract = folder / 'hardware_contract.json'
        with sqlite3.connect((folder/'candidates.sqlite').resolve().as_uri()+'?mode=ro', uri=True) as con:
            con.row_factory = sqlite3.Row
            con.execute('BEGIN')
            records = con.execute('''SELECT c.*, m.* FROM candidates c
                JOIN jobs j USING(candidate_id) JOIN measurements m USING(candidate_id)
                WHERE j.status='complete' AND m.accepted=1 ORDER BY j.ordinal''').fetchall()
        if not records:
            continue
        protocol = json.loads(contract.read_text())
        semantic = {k: protocol[k] for k in PROTOCOL_KEYS}
        if reference is None:
            reference = semantic
        if semantic != reference:
            raise ValueError(f'Incompatible measurement protocol: {folder}')
        for r in records:
            if r['candidate_id'] in ids:
                raise ValueError('Duplicate architecture measurement')
            ids.add(r['candidate_id'])
            if groups.setdefault(r['permutation_group'], r['split']) != r['split']:
                raise ValueError('Permutation-group split leakage')
            artifact = folder / r['artifact_path'] / 'result.json'
            raw = json.loads(artifact.read_text())
            rows.append(dict(candidate_id=r['candidate_id'], split=r['split'],
                permutation_group=r['permutation_group'], cohort='initial1000' if index<2 else 'new1000',
                source=str(folder), architecture=json.loads(r['architecture_json']),
                metrics={t:r[t] for t in TARGETS}, energy_warning=raw.get('energy_warning'),
                protocol_sha256=r['protocol_sha256'], artifact_path=str(artifact),
                artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest()))
        sources.append(dict(folder=str(folder), completed=len(records),
                            contract_sha256=hashlib.sha256(contract.read_bytes()).hexdigest()))
    return dict(rows=rows, sources=sources, protocol=reference)
