"""Remeasure one completed architecture whose accepted energy label is missing.

Preserve the original attempt and replace acceptance atomically only after a
valid retest and successful device restoration. Never edit the frozen runner.
"""
import argparse
import json
import signal
import sqlite3
import tempfile
import time
from pathlib import Path

from scripts.sweep.layerwise import runner
from scripts.sweep.layerwise.database import digest, read_candidate, validate_database
from scripts.sweep.layerwise.candidates import fingerprint
from scripts.sweep.layerwise.export import export_mock
from scripts.prediction.active_learning.storage import atomic_json, locked


def promote(con, original, attempt, folder, metrics, protocol):
    con.execute('BEGIN IMMEDIATE')
    current=con.execute('SELECT attempt,dynamic_energy_per_token_mj FROM measurements WHERE candidate_id=? AND accepted=1',
                        (original['candidate_id'],)).fetchall()
    if [tuple(r) for r in current]!=[(original['attempt'],None)]:
        raise ValueError('Accepted measurement changed before retest commit')
    if not metrics['energy_valid'] or metrics['dynamic_energy_per_token_mj']<=0:
        raise ValueError('Refusing invalid energy replacement')
    con.execute('UPDATE measurements SET accepted=0 WHERE candidate_id=? AND attempt=?',
                (original['candidate_id'],original['attempt']))
    con.execute('INSERT INTO measurements VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',(
        original['candidate_id'],int(attempt.name),str(attempt.relative_to(folder)),
        protocol['kernel_sha256'],fingerprint(protocol),1,metrics['decode_tok_s'],metrics['ttft_ms'],
        metrics['dynamic_energy_per_token_mj'],metrics['baseline_power_w'],metrics['active_power_w'],metrics['duration_s']))
    con.commit()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--candidate',required=True)
    parser.add_argument('--serial',required=True)
    args=parser.parse_args()
    folder=args.campaign.resolve(strict=True)
    db=folder/'candidates.sqlite'
    import torch
    torch.set_num_threads(4)
    def stop(signum,frame): raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    with locked(Path(tempfile.gettempdir())/'nanollmforge_active_hardware.lock'),locked(folder/'RUNNING.lock'):
        validate_database(db)
        protocol=json.loads((folder/'hardware_contract.json').read_text())
        if runner.source_hashes()!=protocol['source_hashes']:
            raise ValueError('Frozen measurement sources changed')
        d=runner.Device(args.serial)
        identity=runner.preflight(args.serial)
        if any(identity[k]!=protocol[p] for k,p in [('hardware_serial','hardware_serial'),('abi','abi'),('build','os_build')]):
            raise ValueError('Device/build differs from original protocol')
        if d.shell('pidof lw_measure runq_llmforge power_sampler 2>/dev/null || true'):
            raise ValueError('Device measurement already active')
        if runner.build(folder,identity['abi'])!=protocol['build']:
            raise ValueError('Frozen toolchain/binary differs')
        for path,key in [('architectures.jsonl','candidates_jsonl_sha256'),('search_space.json','specification_sha256')]:
            if digest(folder/path)!=protocol[key]: raise ValueError('Registry inputs changed')
        with sqlite3.connect(db) as con:
            con.row_factory=sqlite3.Row
            records=con.execute('SELECT * FROM measurements WHERE candidate_id=? AND accepted=1',(args.candidate,)).fetchall()
            if len(records)!=1 or records[0]['dynamic_energy_per_token_mj'] is not None:
                raise ValueError('Expected exactly one accepted measurement with missing energy')
            original=dict(records[0])
            old_attempt=folder/original['artifact_path']
            old_provenance=json.loads((old_attempt/'provenance.json').read_text())
            arch=read_candidate(db,args.candidate)
            audit=folder/'retests'/f'{time.time_ns()}_{args.candidate}'
            audit.mkdir(parents=True)
            with sqlite3.connect(audit/'database_before.sqlite') as backup: con.backup(backup)
            atomic_json(audit/'original_measurement.json',original)
            atomic_json(audit/'plan.json',dict(candidate_id=args.candidate,protocol_sha256=fingerprint(protocol),
                source_sha256=digest(__file__),selection='One missing-energy point explicitly requested for retest',
                policy='One retest; up to 3 admission retries only. Accept full new row if positive valid energy; retain old raw attempt and mark old accepted=0.'))
            remote='/data/local/tmp/nlf_lw_'+fingerprint(protocol)[:12]
            d.shell('mkdir -p '+remote)
            d.push(folder/'build/lw_measure',remote+'/lw_measure')
            d.push(protocol['build']['tokenizer'],remote+'/tokenizer.bin')
            d.shell('chmod 755 '+remote+'/lw_measure')
            with runner.controls(d,folder,identity):
                admission=d.ready(); print('Admission:',admission,flush=True)
                d.shell('input keyevent KEYCODE_WAKEUP')
                model=audit/'model.q8.rlm'
                metadata=export_mock(arch,model); metadata['model_sha256']=digest(model)
                if metadata['model_sha256']!=old_provenance['model']['model_sha256']:
                    raise ValueError('Regenerated weights do not match original bytes')
                print('Exact model hash verified; uploading',flush=True)
                d.push(model,remote+'/model.q8.rlm')
                if d.shell('sha256sum '+remote+'/model.q8.rlm').split()[0]!=metadata['model_sha256']:
                    raise ValueError('Upload checksum mismatch')
                for retry in range(3):
                    admission=d.ready()
                    jobfolder=old_attempt.parent
                    number=max(int(p.name) for p in jobfolder.iterdir() if p.is_dir() and p.name.isdigit())+1
                    attempt=jobfolder/f'{number:03d}';attempt.mkdir()
                    atomic_json(attempt/'provenance.json',dict(protocol_sha256=fingerprint(protocol),
                        architecture_sha256=fingerprint(arch),model=metadata,admission=admission,
                        retest_of=original['artifact_path']))
                    print('Measuring:',attempt,admission,flush=True)
                    metrics=runner.measure(d,folder,remote,attempt,protocol['prompt'])
                    if metrics is not None: break
                if metrics is None: raise ValueError('Thermal admission rejected three times')
                atomic_json(audit/'new_measurement.json',dict(artifact_path=str(attempt.relative_to(folder)),metrics=metrics))
                print(json.dumps(metrics),flush=True)
            # Device settings are verified restored before changing the accepted row.
            if not metrics['energy_valid']:
                atomic_json(audit/'outcome.json',dict(promoted=False,reason=metrics.get('energy_warning')))
                return 2
            promote(con,original,attempt,folder,metrics,protocol)
            atomic_json(audit/'outcome.json',dict(promoted=True,original_attempt=original['attempt'],
                new_attempt=int(attempt.name),candidate_id=args.candidate,database_after_sha256=digest(db)))
            atomic_json(folder/'progress.json',runner.status(folder))
            print('Retest accepted; original attempt preserved:',audit,flush=True)
    return 0


if __name__=='__main__': raise SystemExit(main())
