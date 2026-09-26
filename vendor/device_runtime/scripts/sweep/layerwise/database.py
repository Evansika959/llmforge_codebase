"""SQLite candidate registry, not a table of invented hardware measurements."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

from .candidates import canonical, fingerprint, generate, validate_architecture, architecture_summary, layer_shapes

ROOT = Path(__file__).resolve().parents[3]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(spec_path, output, exclude_databases=None):
    spec_path, output = Path(spec_path), Path(output)
    spec=json.loads(spec_path.read_text())
    if exclude_databases:
        excluded_ids=set(); excluded_groups=set(); sources=[]
        for source in exclude_databases:
            source=Path(source).resolve(strict=True)
            validate_database(source)
            with sqlite3.connect(source.as_uri()+'?mode=ro',uri=True) as prior:
                records=prior.execute('SELECT architecture_sha256,permutation_group FROM candidates').fetchall()
            excluded_ids.update(r[0] for r in records); excluded_groups.update(r[1] for r in records)
            sources.append(dict(path=str(source),sha256=digest(source),candidates=len(records)))
        spec['exclusions']=dict(sources=sources,architecture_sha256=sorted(excluded_ids),
                                permutation_groups=sorted(excluded_groups),
                                policy='Exclude all prior architectures and layer-multiset permutations; no label-based selection.')
    if spec.get('require_previous_campaign_exclusion') and not spec.get('exclusions',{}).get('sources'):
        raise ValueError('This next-batch specification requires --exclude-database')
    rows=generate(spec)
    output.mkdir(parents=True,exist_ok=False)
    (output/'search_space.json').write_text(json.dumps(spec,indent=2)+'\n')
    with (output/'architectures.jsonl').open('x') as stream:
        for row in rows: stream.write(canonical(row)+'\n')
    database=output/'candidates.sqlite'
    with sqlite3.connect(database) as con:
        con.executescript('''
        PRAGMA foreign_keys=ON;
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value_json TEXT NOT NULL);
        CREATE TABLE candidates (
          candidate_id TEXT PRIMARY KEY, architecture_sha256 TEXT UNIQUE NOT NULL,
          family TEXT NOT NULL, n_layer INTEGER NOT NULL, d_model INTEGER NOT NULL,
          q8_group_size INTEGER NOT NULL CHECK(q8_group_size IN (16,32,64)),
          pattern TEXT NOT NULL, permutation_group TEXT NOT NULL,
          split TEXT NOT NULL CHECK(split IN ('train','validation','test')),
          architecture_json TEXT NOT NULL, summary_json TEXT NOT NULL);
        CREATE TABLE layers (
          candidate_id TEXT REFERENCES candidates(candidate_id), layer_index INTEGER NOT NULL,
          n_h INTEGER NOT NULL, n_kv INTEGER NOT NULL, d_qk INTEGER NOT NULL,
          d_v INTEGER NOT NULL, d_mlp INTEGER NOT NULL,
          PRIMARY KEY(candidate_id,layer_index));
        CREATE TABLE jobs (
          ordinal INTEGER PRIMARY KEY, batch INTEGER NOT NULL,
          candidate_id TEXT UNIQUE NOT NULL REFERENCES candidates(candidate_id),
          status TEXT NOT NULL DEFAULT 'planned' CHECK(status IN ('planned','running','complete','failed')));
        CREATE TABLE measurements (
          candidate_id TEXT REFERENCES candidates(candidate_id), attempt INTEGER NOT NULL,
          artifact_path TEXT NOT NULL, kernel_sha256 TEXT NOT NULL, protocol_sha256 TEXT NOT NULL,
          accepted INTEGER NOT NULL CHECK(accepted IN (0,1)),
          decode_tok_s REAL, ttft_ms REAL, dynamic_energy_per_token_mj REAL,
          baseline_power_w REAL, active_power_w REAL, duration_s REAL,
          PRIMARY KEY(candidate_id,attempt));
        ''')
        for key,value in [('schema_version',1),('search_space',spec),('hardware_enabled',False)]:
            con.execute('INSERT INTO metadata VALUES (?,?)',(key,canonical(value)))
        for row in rows:
            a=row['architecture']
            con.execute('INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?,?,?,?)',(
                row['candidate_id'],row['architecture_sha256'],a['family'],a['n_layer'],a['d_model'],
                a['q8_group_size'],row['pattern'],row['permutation_group'],row['split'],canonical(a),canonical(row['summary'])))
            con.executemany('INSERT INTO layers VALUES (?,?,?,?,?,?,?)',[
                (row['candidate_id'],i,r['n_h'],r['n_kv'],r['d_qk'],r['d_v'],r['d_mlp']) for i,r in enumerate(a['layers'])])
            con.execute('INSERT INTO jobs (ordinal,batch,candidate_id) VALUES (?,?,?)',
                        (row['ordinal'],row['batch'],row['candidate_id']))
    report=validate_database(database)
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),candidate_count=len(rows),
        software_config_confirmed=spec['software_config_confirmed'],hardware_enabled=False,
        status='prepared_not_measured',
        pending_decisions=spec['pending_decisions'],validation=report,
        specification_source=str(spec_path.resolve()),specification_sha256=digest(spec_path),
        source_hashes={str(p.relative_to(ROOT)):digest(p) for p in
            [ROOT/'src/runq_llmforge.c',ROOT/'llmforge_bridge/export_llmforge_hetero.py']},
        artifacts={name:digest(output/name) for name in ['candidates.sqlite','architectures.jsonl','search_space.json']},
        note='Candidate identities contain ordered layer lists. No measurements, predictions, or performance labels were fabricated. Global group size is inferred across ALL layers. No old 50–150M filter was applied.')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def validate_database(path):
    con=sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True)
    con.row_factory=sqlite3.Row
    try:
        if con.execute('PRAGMA integrity_check').fetchone()[0]!='ok' or con.execute('PRAGMA foreign_key_check').fetchall():
            raise ValueError('SQLite integrity failure')
        spec=json.loads(con.execute("SELECT value_json FROM metadata WHERE key='search_space'").fetchone()[0])
        rows=con.execute('SELECT * FROM candidates').fetchall()
        if len(rows)!=500: raise ValueError('Expected 500 candidates')
        counts=Counter(); splits=Counter(); patterns=Counter(); assignments={}; params=defaultdict(list)
        coverage=defaultdict(set); kv_counts=defaultdict(Counter); kv_variable=Counter()
        exclusions=spec.get('exclusions',{})
        excluded_ids=set(exclusions.get('architecture_sha256',[]))
        excluded_groups=set(exclusions.get('permutation_groups',[]))
        for row in rows:
            a=json.loads(row['architecture_json']); validate_architecture(a,spec)
            if fingerprint(a)!=row['architecture_sha256'] or row['candidate_id']!='LW_'+fingerprint(a)[:20]:
                raise ValueError('Candidate identity changed')
            for key in ['family','n_layer','d_model','q8_group_size']:
                if row[key]!=a[key]: raise ValueError('Database header mismatch')
            indexed=con.execute('SELECT * FROM layers WHERE candidate_id=? ORDER BY layer_index',(row['candidate_id'],)).fetchall()
            if [r['layer_index'] for r in indexed]!=list(range(a['n_layer'])):
                raise ValueError('Layer indices are not contiguous')
            if [{k:r[k] for k in ['n_h','n_kv','d_qk','d_v','d_mlp']} for r in indexed]!=a['layers']:
                raise ValueError('Ordered layer records differ')
            summary=architecture_summary(a)
            if json.loads(row['summary_json'])!=summary: raise ValueError('Summary mismatch')
            expected_group=fingerprint(dict(a,layers=sorted(a['layers'],key=canonical)))
            if row['permutation_group']!=expected_group: raise ValueError('Permutation group mismatch')
            if assignments.setdefault(expected_group,row['split'])!=row['split']:
                raise ValueError('Layer-permutation leakage across splits')
            if row['architecture_sha256'] in excluded_ids or expected_group in excluded_groups:
                raise ValueError('Candidate overlaps a prior architecture or layer-permutation group')
            if (summary['unique_layer_shapes']==1)!=(row['pattern']=='uniform_control'):
                raise ValueError('Pattern does not match layerwise heterogeneity')
            varying=len({r['n_kv'] for r in a['layers']})>1
            if spec.get('require_variable_kv_in_heterogeneous_candidates') and row['pattern']!='uniform_control' and not varying:
                raise ValueError('Heterogeneous candidate must vary KV heads across layers')
            kv_variable[a['family']]+=int(varying)
            kv_counts[a['family']].update((r['n_h'],r['n_kv']) for r in a['layers'])
            counts[(a['family'],a['q8_group_size'])]+=1; splits[row['split']]+=1; patterns[row['pattern']]+=1
            params[a['family']].append(summary['total_params'])
            coverage[a['family']].update(canonical(r) for r in a['layers'])
        expected={(family,g):n for family in spec['families'] for g,n in [(16,84),(32,83),(64,83)]}
        if dict(counts)!=expected or dict(splits)!={'train':400,'validation':50,'test':50}:
            raise ValueError('Stratum/split allocation mismatch')
        jobs=con.execute('SELECT * FROM jobs ORDER BY ordinal').fetchall()
        if [j['ordinal'] for j in jobs]!=list(range(1,501)) or any(j['batch']!=(j['ordinal']-1)//10+1 for j in jobs):
            raise ValueError('Invalid sampling schedule')
        for batch in range(1,51):
            families=con.execute('SELECT c.family,count(*) FROM jobs j JOIN candidates c USING(candidate_id) WHERE batch=? GROUP BY c.family',(batch,)).fetchall()
            if dict(families)!={name:5 for name in spec['families']}: raise ValueError('Unbalanced batch')
        report=dict(candidates=500,layers=sum(r['n_layer'] for r in rows),patterns=dict(patterns),splits=dict(splits),
            strata=[dict(family=k[0],group_size=k[1],count=n) for k,n in sorted(counts.items())],
            parameter_ranges_m={family:[min(values)/1e6,max(values)/1e6] for family,values in params.items()},
            distinct_layer_shapes={family:len(shapes) for family,shapes in coverage.items()},
            measurement_count=con.execute('SELECT count(*) FROM measurements').fetchone()[0],
            jobs_by_status=dict(con.execute('SELECT status,count(*) FROM jobs GROUP BY status')))
        if spec['schema_version']==2:
            report.update(allowed_layer_shapes={family:len(layer_shapes(f)) for family,f in spec['families'].items()},
                          variable_kv_candidates=dict(kv_variable),
                          layer_head_pair_counts={family:[dict(n_h=h,n_kv=k,layers=n) for (h,k),n in sorted(counts.items())]
                                                  for family,counts in kv_counts.items()},
                          excluded_architectures=len(excluded_ids),excluded_permutation_groups=len(excluded_groups),
                          prior_architecture_overlap=0,prior_permutation_overlap=0)
        return report
    finally:
        con.close()


def read_candidate(path, candidate_id):
    with sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True) as con:
        row=con.execute('SELECT architecture_json FROM candidates WHERE candidate_id=?',(candidate_id,)).fetchone()
        if row is None: raise ValueError('Unknown candidate ID')
        spec=json.loads(con.execute("SELECT value_json FROM metadata WHERE key='search_space'").fetchone()[0])
    arch=json.loads(row[0]); validate_architecture(arch,spec)
    if candidate_id!='LW_'+fingerprint(arch)[:20]: raise ValueError('Candidate fingerprint mismatch')
    return arch
