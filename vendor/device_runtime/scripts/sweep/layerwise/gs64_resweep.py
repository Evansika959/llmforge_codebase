"""Replay frozen GS64 architectures into a NEW registry; never alter baseline labels."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import shutil
import sqlite3
import time

from .candidates import canonical, fingerprint, validate_architecture, architecture_summary
from .database import ROOT, digest
from . import runner

DEFAULT_RELEASE = ROOT / 'diliverable/layerwise_2000_20260921'
DEFAULT_OUTPUT = ROOT / 'scripts/sweep/outputs/watch5_gs64_decode_v1_664'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def prepare(release, output, baseline_kernel):
    release, output = Path(release).resolve(), Path(output).resolve()
    data = json.loads((release / 'dataset_snapshot.json').read_text())
    selected = [r for r in data['rows'] if r['architecture']['q8_group_size'] == 64]
    if len(data['rows']) != 2000 or len(selected) != 664:
        raise ValueError('Expected frozen 2000-row baseline with exactly 664 GS64 architectures')
    if digest(baseline_kernel) != data['protocol']['kernel_sha256']:
        raise ValueError('Baseline kernel does not match the frozen measurement protocol')
    if digest(ROOT / 'src/runq_llmforge.c') == digest(baseline_kernel):
        raise ValueError('New kernel is identical to baseline')
    # Validate release files before trusting architecture identities or provenance.
    hashes = json.loads((release / 'release_hashes.json').read_text())
    for name in ['dataset_snapshot.json', 'measurement_evidence.zip']:
        if digest(release / name) != hashes[name]:
            raise ValueError(f'Release hash mismatch: {name}')
    output.mkdir(parents=True, exist_ok=False)
    baseline = output / 'baseline'; baseline.mkdir()
    shutil.copy2(release / 'dataset_snapshot.json', baseline / 'dataset_2000.json')
    shutil.copy2(baseline_kernel, baseline / 'runq_llmforge.c')
    raw_records, schema, expected_models = {}, None, {}
    for index in range(1, 5):
        db = release / f'databases/campaign{index}/candidates.sqlite'
        if digest(db) != hashes[str(db.relative_to(release))]:
            raise ValueError('Release database hash mismatch')
        with sqlite3.connect(db.as_uri() + '?mode=ro', uri=True) as con:
            con.row_factory = sqlite3.Row
            if schema is None:
                schema = [r[0] for r in con.execute("SELECT sql FROM sqlite_master WHERE type='table' ORDER BY rowid")]
            for r in con.execute('SELECT * FROM candidates WHERE q8_group_size=64'):
                raw_records[r['candidate_id']] = tuple(r)
        if index == 2:
            with sqlite3.connect(db.as_uri() + '?mode=ro', uri=True) as con:
                spec = json.loads(con.execute("SELECT value_json FROM metadata WHERE key='search_space'").fetchone()[0])
    spec.pop('exclusions', None)
    spec['require_previous_campaign_exclusion'] = False
    spec['require_variable_kv_in_heterogeneous_candidates'] = False
    spec['replay'] = dict(source_dataset_sha256=digest(baseline / 'dataset_2000.json'),
                          selection='q8_group_size == 64; no performance-label filtering', count=664)
    write(output / 'search_space.json', spec)
    random.Random(20260921).shuffle(selected)
    scheduled = []
    for ordinal, row in enumerate(selected, 1):
        artifact = Path(row['artifact_path'])
        if digest(artifact) != row['artifact_sha256']:
            raise ValueError('Baseline raw artifact hash mismatch')
        raw = json.loads(artifact.read_text())
        provenance = json.loads(artifact.with_name('provenance.json').read_text())
        expected_models[row['candidate_id']] = provenance['model']['model_sha256']
        write(baseline / (row['candidate_id'] + '.json'), dict(row=row, result=raw, provenance=provenance))
        scheduled.append(dict(candidate_id=row['candidate_id'], architecture=row['architecture'],
                              split=row['split'], permutation_group=row['permutation_group'],
                              ordinal=ordinal, batch=(ordinal-1)//10+1))
    (output / 'architectures.jsonl').write_text(''.join(canonical(r)+'\n' for r in scheduled))
    with sqlite3.connect(output / 'candidates.sqlite') as con:
        con.executescript(';\n'.join(schema)+';')
        con.execute('PRAGMA foreign_keys=ON')
        for key, value in [('schema_version', 1), ('search_space', spec), ('hardware_enabled', False)]:
            con.execute('INSERT INTO metadata VALUES (?,?)', (key, canonical(value)))
        for row in scheduled:
            ident = row['candidate_id']; a = row['architecture']
            con.execute('INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?,?,?,?)', raw_records[ident])
            con.executemany('INSERT INTO layers VALUES (?,?,?,?,?,?,?)',
                            [(ident, i, r['n_h'], r['n_kv'], r['d_qk'], r['d_v'], r['d_mlp']) for i, r in enumerate(a['layers'])])
            con.execute('INSERT INTO jobs (ordinal,batch,candidate_id) VALUES (?,?,?)', (row['ordinal'], row['batch'], ident))
    manifest = dict(version='gs64_decode_replay_v1', count=664, shuffle_seed=20260921,
                    release=str(release), source_dataset_sha256=digest(baseline / 'dataset_2000.json'),
                    old_kernel_sha256=digest(baseline / 'runq_llmforge.c'),
                    new_kernel_sha256=digest(ROOT / 'src/runq_llmforge.c'),
                    expected_model_hashes=expected_models,
                    splits=dict(Counter(r['split'] for r in selected)),
                    baseline_files={p.name: digest(p) for p in baseline.iterdir()},
                    specification_sha256=digest(output / 'search_space.json'),
                    schedule_sha256=digest(output / 'architectures.jsonl'),
                    policy='New full measurements, same exact weights/architecture IDs/splits. Baseline untouched. No old/new energy/timing splicing.')
    write(output / 'replay_manifest.json', manifest)
    return validate(output / 'candidates.sqlite')


def validate(database):
    folder = Path(database).resolve().parent
    manifest = json.loads((folder / 'replay_manifest.json').read_text())
    for name, sha in manifest['baseline_files'].items():
        if digest(folder / 'baseline' / name) != sha:
            raise ValueError('Frozen baseline changed')
    if digest(folder / 'search_space.json') != manifest['specification_sha256'] or digest(folder / 'architectures.jsonl') != manifest['schedule_sha256']:
        raise ValueError('Frozen replay schedule/specification changed')
    spec = json.loads((folder / 'search_space.json').read_text())
    data = json.loads((folder / 'baseline/dataset_2000.json').read_text())
    if digest(folder / 'baseline/dataset_2000.json') != manifest['source_dataset_sha256']:
        raise ValueError('Baseline dataset fingerprint changed')
    expected = {r['candidate_id']: r for r in data['rows'] if r['architecture']['q8_group_size'] == 64}
    for ident in expected:
        old = json.loads((folder / 'baseline' / (ident+'.json')).read_text())
        if manifest['expected_model_hashes'].get(ident) != old['provenance']['model']['model_sha256']:
            raise ValueError('Expected weight hash differs from baseline provenance')
    schedule = [json.loads(line) for line in (folder / 'architectures.jsonl').read_text().splitlines()]
    with sqlite3.connect(Path(database).resolve().as_uri()+'?mode=ro', uri=True) as con:
        con.row_factory = sqlite3.Row
        if con.execute('PRAGMA integrity_check').fetchone()[0] != 'ok' or con.execute('PRAGMA foreign_key_check').fetchall():
            raise ValueError('Registry integrity failure')
        if json.loads(con.execute("SELECT value_json FROM metadata WHERE key='search_space'").fetchone()[0]) != spec:
            raise ValueError('Database specification differs from frozen file')
        rows = con.execute('SELECT * FROM candidates').fetchall()
        if {r['candidate_id'] for r in rows} != set(expected) or len(rows) != manifest['count']:
            raise ValueError('Replay is not the exact GS64 subset')
        jobs = con.execute('SELECT * FROM jobs ORDER BY ordinal').fetchall()
        if [(r['ordinal'], r['batch'], r['candidate_id']) for r in jobs] != [(r['ordinal'], r['batch'], r['candidate_id']) for r in schedule]:
            raise ValueError('Replay schedule mismatch')
        for row in rows:
            ident = row['candidate_id']; a = json.loads(row['architecture_json']); e = expected[ident]
            validate_architecture(a, spec)
            if a != e['architecture'] or fingerprint(a) != row['architecture_sha256'] or ident != 'LW_'+fingerprint(a)[:20]:
                raise ValueError('Architecture changed')
            if any(row[k] != a[k] for k in ('family','n_layer','d_model','q8_group_size')):
                raise ValueError('Candidate header changed')
            if row['split'] != e['split'] or row['permutation_group'] != e['permutation_group']:
                raise ValueError('Baseline split/group changed')
            if json.loads(row['summary_json']) != architecture_summary(a):
                raise ValueError('Architecture summary changed')
            layers = con.execute('SELECT layer_index,n_h,n_kv,d_qk,d_v,d_mlp FROM layers WHERE candidate_id=? ORDER BY layer_index', (ident,)).fetchall()
            if [tuple(r) for r in layers] != [(i, *(r[k] for k in ('n_h','n_kv','d_qk','d_v','d_mlp'))) for i, r in enumerate(a['layers'])]:
                raise ValueError('Layer table changed')
        completed = con.execute("SELECT count(*) FROM jobs WHERE status='complete'").fetchone()[0]
        if con.execute('SELECT candidate_id FROM measurements WHERE accepted=1 GROUP BY candidate_id HAVING count(*)>1').fetchall():
            raise ValueError('Duplicate accepted remeasurements')
    return dict(candidates=len(rows), completed=completed, splits=manifest['splits'], baseline_preserved=True)


def report(folder):
    folder = Path(folder).resolve(); state = validate(folder / 'candidates.sqlite')
    manifest = json.loads((folder / 'replay_manifest.json').read_text())
    pairs = []
    with sqlite3.connect((folder / 'candidates.sqlite').as_uri()+'?mode=ro', uri=True) as con:
        for ident, artifact, kernel in con.execute("SELECT m.candidate_id,m.artifact_path,m.kernel_sha256 FROM measurements m JOIN jobs j USING(candidate_id) WHERE m.accepted=1 AND j.status='complete' ORDER BY j.ordinal"):
            if kernel != manifest['new_kernel_sha256']:
                raise ValueError('Unexpected measurement kernel')
            old = json.loads((folder / 'baseline' / (ident+'.json')).read_text())['result']
            path = folder / artifact / 'result.json'; new = json.loads(path.read_text())
            old_dt = old['timing']['end'] - old['timing']['prefill_end']
            new_dt = new['timing']['end'] - new['timing']['prefill_end']
            pairs.append(dict(candidate_id=ident, baseline=old, optimized=new,
                optimized_artifact=str(path), optimized_artifact_sha256=digest(path),
                decode_update=dict(decode_tok_s=new['decode_tok_s'], decode_duration_s=new_dt,
                                   tpot_ms=1000*new_dt/31,
                                   decode_gross_energy_per_forward_mj=new.get('decode_gross_energy_per_forward_mj')),
                comparison=dict(throughput_ratio=new['decode_tok_s']/old['decode_tok_s'],
                                old_tpot_ms=1000*old_dt/31, new_tpot_ms=1000*new_dt/31,
                                last_token_matches=old['timing']['last_token']==new['timing']['last_token'])))
    output = folder / 'reports' / str(time.time_ns()); output.mkdir(parents=True)
    write(output / 'decode_updates.json', dict(status=state, baseline_dataset_sha256=manifest['source_dataset_sha256'],
        baseline_kernel_sha256=manifest['old_kernel_sha256'], optimized_kernel_sha256=manifest['new_kernel_sha256'],
        policy='Versioned overlay keyed by architecture ID; original dataset unchanged. Full coherent measurements retained alongside decode updates. Noncontemporaneous runs are not a controlled causal speedup estimate.', pairs=pairs))
    return dict(**state, report=str(output / 'decode_updates.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare','validate','run','status','report','restore'])
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--release', type=Path, default=DEFAULT_RELEASE)
    parser.add_argument('--baseline-kernel', type=Path)
    parser.add_argument('--serial')
    parser.add_argument('--max-jobs', type=int, default=664)
    parser.add_argument('--acknowledge-protocol', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        if not args.baseline_kernel: parser.error('--baseline-kernel is required')
        result = prepare(args.release, args.output, args.baseline_kernel)
    elif args.command == 'validate': result = validate(args.output / 'candidates.sqlite')
    elif args.command == 'report': result = report(args.output)
    elif args.command == 'status': result = runner.status(args.output)
    elif args.command == 'restore':
        if not args.serial: parser.error('--serial is required')
        result = runner.restore(args.output, args.serial)
    else:
        if not args.serial: parser.error('--serial is required')
        manifest = json.loads((args.output / 'replay_manifest.json').read_text())
        if digest(ROOT / 'src/runq_llmforge.c') != manifest['new_kernel_sha256']:
            parser.error('Kernel differs from the prepared optimization; use a new experiment version')
        import torch
        torch.set_num_threads(4)
        expected_tokens = {ident: json.loads((args.output / 'baseline' / (ident+'.json')).read_text())['result']['timing']['last_token']
                           for ident in manifest['expected_model_hashes']}
        code = runner.run(args.output, args.serial, args.max_jobs, args.acknowledge_protocol,
                          validator=validate, expected_model_hashes=manifest['expected_model_hashes'],
                          expected_last_tokens=expected_tokens)
        print(json.dumps(report(args.output), indent=2))
        return code
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
