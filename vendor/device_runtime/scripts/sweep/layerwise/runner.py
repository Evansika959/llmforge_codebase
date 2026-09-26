"""Explicit, resumable hardware collection for the ordered layerwise registry."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import shlex
import signal
import sqlite3
import subprocess
import tempfile
import time

from .candidates import canonical, fingerprint
from .database import ROOT, digest, read_candidate, validate_database
from .export import export_mock
from .measurement import parse
from scripts.prediction.active_learning.storage import atomic_json, locked
from scripts.prediction.active_learning.hardware import adb_path, preflight

POWER='/sys/class/power_supply/battery'
TEMP='/sys/class/thermal/thermal_zone17/temp'
SETTINGS=['system:screen_off_timeout','system:screen_brightness','system:screen_brightness_mode']


class Paused(RuntimeError): pass


class Device:
    def __init__(self,serial): self.base=[adb_path(),'-s',serial]
    def command(self,args,timeout=30):
        return subprocess.run(self.base+args,capture_output=True,text=True,errors='replace',check=True,timeout=timeout).stdout.strip()
    def shell(self,command,timeout=30): return self.command(['shell',command],timeout)
    def push(self,local,remote):
        self.command(['push',str(local),remote],max(180,60+Path(local).stat().st_size/(512*1024)))
    def pull(self,remote,local): self.command(['pull',remote,str(local)],60)
    def state(self):
        battery=self.shell('dumpsys battery')
        fields=dict(line.strip().split(':',1) for line in battery.splitlines() if ':' in line)
        powered=[fields[k].strip().lower() for k in ['AC powered','USB powered','Wireless powered'] if k in fields]
        if not powered or any(p not in ('false','true') for p in powered): raise Paused('Cannot establish charging state')
        if 'true' in powered: raise Paused('Charging detected; unplug before resuming')
        level=100*float(fields['level'])/float(fields['scale'])
        if not 0<=level<=100 or level<=30: raise Paused(f'Battery {level:g}%; please charge above 30% and unplug before resuming')
        temp=float(self.shell('cat '+TEMP))/1000
        if not 0<temp<100: raise Paused('Invalid temperature telemetry')
        return dict(battery_percent=level,temperature_c=temp)
    def ready(self):
        last=0
        while True:
            state=self.state()
            if state['temperature_c']<45: return state
            if time.monotonic()-last>30:
                print(f'Cooling: {state["temperature_c"]:.1f}C; battery {state["battery_percent"]:g}%; waiting for <45C',flush=True)
                last=time.monotonic()
            time.sleep(5)


def status(folder):
    folder=Path(folder)
    with sqlite3.connect((folder/'candidates.sqlite').resolve().as_uri()+'?mode=ro',uri=True) as con:
        jobs=dict(con.execute('SELECT status,count(*) FROM jobs GROUP BY status'))
        completed=con.execute('SELECT c.family,c.q8_group_size,count(*) FROM jobs j JOIN candidates c USING(candidate_id) WHERE j.status="complete" GROUP BY c.family,c.q8_group_size').fetchall()
        metrics=con.execute('SELECT decode_tok_s,ttft_ms,dynamic_energy_per_token_mj FROM measurements WHERE accepted=1').fetchall()
    import statistics
    report=dict(jobs=jobs,completed_strata=completed,metrics={})
    for i,key in enumerate(['decode_tok_s','ttft_ms','dynamic_energy_per_token_mj']):
        values=[r[i] for r in metrics if r[i] is not None]
        report['metrics'][key]=dict(n=len(values),median=statistics.median(values),min=min(values),max=max(values)) if values else dict(n=0)
    if (folder/'run_state.json').exists(): report['runtime']=json.loads((folder/'run_state.json').read_text())
    return report


def source_hashes():
    paths=[p for p in Path(__file__).parent.glob('*') if p.suffix in ('.py','.c')]
    paths += [ROOT/'src/runq_llmforge.c',ROOT/'src/bpe.h',ROOT/'llmforge_bridge/export_llmforge_hetero.py',
              ROOT/'scripts/sweep/run_sweep_configs.py',ROOT/'scripts/prediction/active_learning/hardware.py',
              ROOT/'scripts/prediction/active_learning/storage.py']
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted(paths)}


def build(folder,abi):
    from scripts.sweep.run_sweep_configs import find_ndk_compiler, find_tokenizer_gpt2
    if abi not in ('armeabi-v7a','arm64-v8a'): raise ValueError('Unsupported device ABI')
    compiler=find_ndk_compiler('armv8l' if abi=='armeabi-v7a' else 'aarch64')
    if not compiler: raise ValueError('NDK compiler not found')
    builddir=folder/'build'; builddir.mkdir(exist_ok=True)
    executable=builddir/'lw_measure'
    flags=['-march=armv7-a','-mfpu=neon'] if abi=='armeabi-v7a' else ['-march=armv8-a']
    command=[compiler,'-O3',*flags,'-ffast-math','-fopenmp','-static-openmp','-pthread',
             '-I'+str(ROOT/'src'),str(Path(__file__).with_name('measure.c')),'-lm','-o',str(executable)]
    if not executable.exists(): subprocess.run(command,check=True,capture_output=True,timeout=120)
    tokenizer=ROOT/find_tokenizer_gpt2()
    return dict(command=command,compiler_sha256=digest(compiler),executable_sha256=digest(executable),
                tokenizer=str(tokenizer),tokenizer_sha256=digest(tokenizer))


@contextmanager
def controls(d,folder,identity):
    sessions=folder/'sessions'; sessions.mkdir(exist_ok=True)
    for previous in sorted(sessions.glob('*/original.json')):
        if not previous.with_name('restoration.json').exists():
            raise Paused(f'Unresolved device restoration: use restore --output {folder} --serial SERIAL')
        if not json.loads(previous.with_name('restoration.json').read_text())['restored']:
            raise Paused('Previous restoration incomplete; use restore before running')
    session=sessions/f'{time.time_ns()}'; session.mkdir()
    values={key:d.shell('settings get '+key.replace(':',' ')) for key in SETTINGS}
    for value in values.values():
        if value!='null' and not re.fullmatch(r'\d+',value): raise ValueError('Unexpected display setting value')
    atomic_json(session/'original.json',dict(settings=values,hardware_serial=identity['hardware_serial']))
    try:
        for key,value in [('system:screen_off_timeout','86400000'),('system:screen_brightness_mode','0'),('system:screen_brightness','1')]:
            d.shell('settings put '+key.replace(':',' ')+' '+value)
            if d.shell('settings get '+key.replace(':',' '))!=value: raise Paused('Display control readback failed')
        d.shell('input keyevent KEYCODE_WAKEUP')
        yield
    finally:
        errors=[]
        for key,value in values.items():
            try:
                restore_value(d,key,value)
            except Exception as error: errors.append(str(error))
        atomic_json(session/'restoration.json',dict(restored=not errors,errors=errors,
            note='Brightness/mode/timeout restored. Wakefulness is not forced back to sleep.'))
        if errors: raise Paused('Device restoration incomplete; inspect sessions/')


def restore_value(d,key,value):
    if key not in SETTINGS or (value!='null' and not re.fullmatch(r'\d+',value)): raise ValueError('Invalid saved setting')
    d.shell(('settings delete ' if value=='null' else 'settings put ')+key.replace(':',' ')+('' if value=='null' else ' '+value))
    if d.shell('settings get '+key.replace(':',' '))!=value: raise Paused('Restoration readback mismatch')


def restore(folder,serial):
    folder=Path(folder).resolve(); d=Device(serial)
    with locked(Path(tempfile.gettempdir())/'nanollmforge_active_hardware.lock'),locked(folder/'RUNNING.lock'):
        if d.shell('pidof lw_measure runq_llmforge power_sampler 2>/dev/null || true'): raise Paused('Device measurement is still active')
        for path in sorted((folder/'sessions').glob('*/original.json')):
            result=path.with_name('restoration.json')
            if result.exists() and json.loads(result.read_text())['restored']: continue
            saved=json.loads(path.read_text())
            if d.shell('getprop ro.serialno')!=saved['hardware_serial']: raise Paused('Physical device mismatch')
            for key,value in saved['settings'].items(): restore_value(d,key,value)
            atomic_json(result,dict(restored=True,manual=True,errors=[]))
    return dict(restored=True)


def stop_remote(d,remote):
    # Kill only the PID whose cmdline contains this campaign's exact executable.
    pid=d.shell('cat '+shlex.quote(remote+'/pid')+' 2>/dev/null || true')
    if re.fullmatch(r'[1-9]\d*',pid):
        cmd=d.shell(f'cat /proc/{pid}/cmdline 2>/dev/null || true')
        if remote.split('/attempts/')[0]+'/lw_measure' in cmd:
            d.shell(f'kill -TERM {pid}')


def measure(d,folder,remote,attempt,prompt):
    destination=remote+'/attempts/'+attempt.parent.name+'_'+attempt.name
    d.shell('mkdir -p '+shlex.quote(destination))
    command=f'echo $$ > {destination}/pid; export OMP_NUM_THREADS=4; exec taskset f {remote}/lw_measure {remote}/model.q8.rlm {remote}/tokenizer.bin {shlex.quote(prompt)} {POWER} {TEMP} {destination}/trace.csv > {destination}/timing.json 2> {destination}/infer.log'
    atomic_json(attempt/'command.json',dict(remote=destination,shell=command))
    with (attempt/'adb.log').open('x') as log:
        process=subprocess.Popen(d.base+['shell',command],stdout=log,stderr=subprocess.STDOUT)
        try:
            code=process.wait(timeout=330)
        except BaseException:
            try: stop_remote(d,destination)
            finally:
                process.terminate()
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
            raise
    atomic_json(attempt/'exit.json',dict(returncode=code))
    for name in ['trace.csv','timing.json','infer.log']:
        try: d.pull(destination+'/'+name,attempt/name)
        except subprocess.SubprocessError:
            if code==0: raise
    if code==75: raise Paused('Device battery/charger guard stopped the job; charge and unplug before resume')
    if code==76: return None
    if code: raise Paused(f'Measurement exited {code}; inspect {attempt}/infer.log')
    metrics=parse(attempt); atomic_json(attempt/'result.json',metrics)
    return metrics


def run(folder,serial,max_jobs=500,acknowledge=False,*,validator=None,expected_model_hashes=None,expected_last_tokens=None):
    if not acknowledge: raise ValueError('Review README and pass --acknowledge-protocol')
    if max_jobs<1: raise ValueError('max-jobs must be positive')
    folder=Path(folder).resolve(); db=folder/'candidates.sqlite'
    (validator or validate_database)(db)
    with sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True) as count_con:
        total_jobs=count_con.execute('SELECT count(*) FROM jobs').fetchone()[0]
    d=Device(serial); start_sources=source_hashes()
    state=dict(status='preflight',pid=os.getpid(),serial=serial)
    def update(**values):
        state.update(values,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        atomic_json(folder/'run_state.json',state)
    def interrupted(signum,frame): raise KeyboardInterrupt
    previous=signal.signal(signal.SIGTERM,interrupted); owns_workspace=False
    try:
        with locked(Path(tempfile.gettempdir())/'nanollmforge_active_hardware.lock'),locked(folder/'RUNNING.lock'):
            owns_workspace=True
            identity=preflight(serial)
            if d.shell('pidof lw_measure 2>/dev/null || true'): raise Paused('Another layerwise measurement is active')
            contract_path=folder/'hardware_contract.json'
            prior=json.loads(contract_path.read_text()) if contract_path.exists() else None
            if prior and prior['source_hashes']!=start_sources: raise Paused('Source changed since hardware protocol was frozen')
            built=build(folder,identity['abi'])
            from scripts.sweep.run_sweep_configs import build_prompt_for_tokens
            prompt=build_prompt_for_tokens(48)
            protocol=dict(version='layerwise_aligned_v1',kernel_sha256=digest(ROOT/'src/runq_llmforge.c'),
                wrapper_sha256=digest(Path(__file__).with_name('measure.c')),source_hashes=start_sources,build=built,
                hardware_serial=identity['hardware_serial'],abi=identity['abi'],os_build=identity['build'],
                candidates_jsonl_sha256=digest(folder/'architectures.jsonl'),specification_sha256=digest(folder/'search_space.json'),
                prompt=prompt,prompt_tokens=49,output_tokens=32,decode_forwards=31,fixed_eos_workload=True,
                cpu_threads=4,cpuset='f',temperature_admission_lt_c=45,battery_stop_le_percent=30,
                inference_temperature_polling=False,power_interval_ms=100,pre_idle_s=4,post_idle_s=4,
                energy='trapezoidal integral over monotonic prefill+decode interval minus pre-idle median times duration, divided by 32',
                baseline_drift='record >25% warning; no automatic exclusion of valid throughput/latency',
                synthetic_weights=True,software_skeleton_confirmed=True,
                operator_profile='same provisional kernel-compatible profile as candidate registry, retained for hardware shape sampling')
            if prior and prior!=protocol: raise Paused('Hardware/build/input contract differs; do not mix this protocol')
            if not prior:
                atomic_json(contract_path,protocol)
                atomic_json(folder/'confirmation.json',dict(skeleton_confirmed=True,source='User: 是一致 准备开跑并进行监测',
                    note='Prepared candidate identities retained; exact kernel and source hashes frozen in hardware_contract.json. This measurement protocol is separate from legacy AL.'))
            protocol_sha=fingerprint(protocol)
            remote='/data/local/tmp/nlf_lw_'+protocol_sha[:12]
            d.shell('mkdir -p '+remote)
            d.push(folder/'build/lw_measure',remote+'/lw_measure')
            d.push(built['tokenizer'],remote+'/tokenizer.bin')
            d.shell('chmod 755 '+remote+'/lw_measure')
            update(status='ready',remote=remote,protocol_sha256=protocol_sha)
            with controls(d,folder,identity),sqlite3.connect(db) as con:
                con.execute('PRAGMA foreign_keys=ON')
                jobs=con.execute('SELECT ordinal,batch,candidate_id FROM jobs WHERE status!="complete" ORDER BY ordinal').fetchall()
                for ordinal,batch,ident in jobs[:max_jobs]:
                    if (folder/'STOP').exists(): raise Paused('STOP requested; completed measurements retained')
                    if source_hashes()!=start_sources: raise Paused('Source changed during run')
                    update(status='admission',ordinal=ordinal,batch=batch,candidate_id=ident)
                    admission=d.ready(); print(f'[{ordinal}/{total_jobs}] batch {batch} {ident}: battery {admission["battery_percent"]:g}%, {admission["temperature_c"]:.1f}C',flush=True)
                    d.shell('input keyevent KEYCODE_WAKEUP')
                    arch=read_candidate(db,ident)
                    jobfolder=folder/'attempts'/f'{ordinal:04d}_{ident}'; jobfolder.mkdir(parents=True,exist_ok=True)
                    # A result written before an interrupted DB commit is reusable.
                    recovered=None
                    for result in sorted(jobfolder.glob('*/result.json')):
                        meta=json.loads(result.with_name('provenance.json').read_text())
                        if meta['protocol_sha256']!=protocol_sha or meta['architecture_sha256']!=fingerprint(arch): raise Paused('Attempt provenance mismatch')
                        if expected_model_hashes and meta['model']['model_sha256']!=expected_model_hashes[ident]:
                            raise Paused('Recovered model differs from baseline weights')
                        recovered=(result.parent,parse(result.parent)); break
                    model=folder/'build/model.q8.rlm'
                    if not recovered:
                        update(status='exporting',battery_percent=admission['battery_percent'])
                        # Scratch is exclusively generated by this runner; metadata retains identity.
                        for path in [model,model.with_suffix('.rlm.json')]:
                            if path.exists(): path.unlink()
                        metadata=export_mock(arch,model)
                        metadata['model_sha256']=digest(model)
                        if expected_model_hashes and metadata['model_sha256']!=expected_model_hashes[ident]:
                            raise Paused('Regenerated model differs from baseline weights; do not mix this comparison')
                        update(status='uploading'); d.push(model,remote+'/model.q8.rlm')
                        actual=d.shell('sha256sum '+remote+'/model.q8.rlm').split()[0]
                        if actual!=metadata['model_sha256']: raise Paused('Pushed model checksum mismatch')
                        con.execute('UPDATE jobs SET status="running" WHERE candidate_id=?',(ident,)); con.commit()
                        for retry in range(3):
                            admission=d.ready()
                            number=max([int(p.name) for p in jobfolder.iterdir() if p.is_dir()]+[0])+1
                            attempt=jobfolder/f'{number:03d}'; attempt.mkdir()
                            atomic_json(attempt/'provenance.json',dict(protocol_sha256=protocol_sha,architecture_sha256=fingerprint(arch),model=metadata,admission=admission))
                            update(status='measuring',attempt=number,battery_percent=admission['battery_percent'])
                            metrics=measure(d,folder,remote,attempt,prompt)
                            if metrics is not None: recovered=(attempt,metrics); break
                            print(f'[{ordinal}/{total_jobs}] admission became >=45C before inference; cooling, retry {retry+1}/3',flush=True)
                        if not recovered: raise Paused('Three pre-inference thermal rejections; measurements excluded, job retryable')
                    attempt,metrics=recovered
                    if expected_last_tokens and metrics['timing']['last_token']!=expected_last_tokens[ident]:
                        raise Paused('Generated token differs from the paired baseline; retain raw attempt for numerical investigation before accepting labels')
                    con.execute('INSERT INTO measurements VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',(
                        ident,int(attempt.name),str(attempt.relative_to(folder)),protocol['kernel_sha256'],protocol_sha,1,
                        metrics['decode_tok_s'],metrics['ttft_ms'],metrics['dynamic_energy_per_token_mj'],metrics['baseline_power_w'],metrics['active_power_w'],metrics['duration_s']))
                    con.execute('UPDATE jobs SET status="complete" WHERE candidate_id=?',(ident,)); con.commit()
                    print(f'[{ordinal}/{total_jobs}] saved: {metrics["decode_tok_s"]:.3f} tok/s, TTFT {metrics["ttft_ms"]:.1f} ms, dynamic {metrics["dynamic_energy_per_token_mj"]} mJ/token; {metrics.get("energy_warning")}',flush=True)
                    # Remove only this job's generated model; raw traces/metadata remain.
                    if model.exists(): model.unlink()
                    if model.with_suffix('.rlm.json').exists(): model.with_suffix('.rlm.json').unlink()
                    d.shell('rm -f '+remote+'/model.q8.rlm')
                    update(status='saved',completed=con.execute('SELECT count(*) FROM jobs WHERE status="complete"').fetchone()[0])
                    report=status(folder); atomic_json(folder/'progress.json',report)
                    if ordinal%10==0:
                        atomic_json(folder/f'batch_{batch:03d}_report.json',report)
                        print('BATCH COMPLETE '+json.dumps(report),flush=True)
            update(status='complete' if status(folder)['jobs'].get('complete',0)==total_jobs else 'paused',message='Requested job budget reached')
    except (Exception,KeyboardInterrupt) as error:
        message=str(error) or 'Interrupted; resume same command'
        if owns_workspace: update(status='paused',message=message)
        print('PAUSED: '+message,flush=True)
        return 2
    finally: signal.signal(signal.SIGTERM,previous)
    return 0
