"""Explicit Android adapter for the existing random-sweep measurement protocol.

Each invocation preserves raw attempt output. Only validated rows are aggregated.
No shell=True, implicit device selection, or automatic battery/thermal retries.
"""
import csv
import os
import re
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from llmforge.hw.device.runtime import (KERNEL_SOURCE, RUNTIME, SCRATCH_FILES, TOKENIZER_CANDIDATES,
                                        WORK, child_env)

from .quality import result_metrics
from .storage import atomic_json, read_json, locked, portable_path, write_configs

HARDWARE_LOCK = 'llmforge_device_hardware.lock'


def adb_path():
    choices = [os.environ.get('ADB'),shutil.which('adb')]
    choices += [str(Path(os.environ[name])/'platform-tools/adb') for name in ['ANDROID_HOME','ANDROID_SDK_ROOT'] if os.environ.get(name)]
    choices.append(str(Path.home()/'Library/Android/sdk/platform-tools/adb'))
    for path in choices:
        if path and Path(path).is_file():
            return str(Path(path).resolve())
    raise RuntimeError('ADB is unavailable; set ADB or install platform-tools')


def preflight(serial, adb=None, min_battery=30):
    if not serial or serial.startswith('-'):
        raise ValueError('An explicit ADB serial is required')
    command = [adb or adb_path(),'-s',serial]
    def shell(script):
        return subprocess.run(command+['shell',script],capture_output=True,text=True,check=True,timeout=15).stdout.strip()
    subprocess.run(command+['get-state'],capture_output=True,check=True,timeout=15)
    battery = shell('dumpsys battery')
    fields = dict(line.strip().split(':',1) for line in battery.splitlines() if ':' in line)
    level = 100*float(fields['level'])/float(fields['scale'])
    powered = [fields[k].strip().lower() for k in ['AC powered','USB powered','Wireless powered'] if k in fields]
    if not powered or any(v not in ('true','false') for v in powered):
        raise RuntimeError('Cannot establish charging state')
    if any(v=='true' for v in powered):
        raise RuntimeError('Unplug the charger before benchmarking')
    if not 0<=level<=100 or level<=min_battery:
        raise RuntimeError(f'Battery {level:g}%; charge above {min_battery}% and unplug before resuming')
    busy = shell('pidof runq_reallm power_sampler 2>/dev/null || true')
    if busy:
        raise RuntimeError('A runner/sampler is already active on the device; inspect it before resuming')
    return dict(serial=serial,hardware_serial=shell('getprop ro.serialno'),model=shell('getprop ro.product.model'),
                abi=shell('getprop ro.product.cpu.abi'),build=shell('getprop ro.build.fingerprint'),battery_percent=level,unplugged=True,
                warning='Read-only connection/battery check; temperature, NDK build and measurement quality are checked at execution.')


def toolchain_info():
    from .storage import digest
    roots = [Path(p) for p in [os.environ.get('NDK'),os.environ.get('ANDROID_NDK_HOME')] if p]
    roots += sorted((Path.home()/'Library/Android/sdk/ndk').glob('*'),reverse=True)
    for root in roots:
        for platform in ['darwin-x86_64','linux-x86_64']:
            folder = root/'toolchains/llvm/prebuilt'/platform/'bin'
            if (folder/'armv7a-linux-androideabi24-clang').is_file() and (folder/'aarch64-linux-android24-clang').is_file():
                properties = root/'source.properties'
                # Record the NDK release, never its install location.
                ndk = dict(revision=_ndk_revision(properties),host=platform,
                           properties_sha256=digest(properties) if properties.is_file() else None)
                break
        else:
            continue
        break
    else:
        raise RuntimeError('Android NDK compilers not found; set NDK/ANDROID_NDK_HOME before starting')
    for relative in TOKENIZER_CANDIDATES:
        if (RUNTIME/relative).is_file():
            return dict(ndk=ndk,tokenizer_sha256=digest(RUNTIME/relative))
    raise RuntimeError('GPT-2 tokenizer not found in the device runtime; refusing an unverified on-device tokenizer')


def _ndk_revision(properties):
    if properties.is_file():
        for line in properties.read_text(errors='replace').splitlines():
            key, _, value = line.partition('=')
            if key.strip() == 'Pkg.Revision':
                return value.strip()
    return None


def merge_attempts(rd, proposal, protocol):
    base = Path(rd).resolve().parents[2]
    expected = {c['config_id']:c for c in proposal['schedule']}
    found, provenance, rejected, excluded = {},{},[],[]
    retired = {c['config_id']:c for c in proposal.get('retired_anchor_configs',[])}
    after_ids = {a['after'] for a in proposal['anchors']}
    for path in sorted((rd/'attempts').glob('*/raw_results.csv')):
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        for number,row in enumerate(rows,2):
            ident = row.get('config_id')
            if ident in retired and ident not in expected:
                excluded.append(dict(path=portable_path(path,base),line=number,config_id=ident,
                                     reason='Anchor retired by recorded policy migration; raw row retained'))
                continue
            if ident not in expected:
                raise ValueError('Unexpected ID in raw hardware result: '+str(ident))
            try:
                result_metrics(row,expected[ident],protocol)
            except (ValueError,KeyError,TypeError) as error:
                rejected.append(dict(path=portable_path(path,base),line=number,config_id=ident,reason=str(error)))
                continue
            # First valid candidate/pre-anchor, latest valid post-anchor after retries.
            if ident not in found or ident in after_ids:
                found[ident]=row
                policy_path = path.with_name('thermal_policy.json')
                policy = read_json(policy_path) if policy_path.exists() else dict(
                    ceiling_c=protocol.get('temperature_ceiling'),comparison='legacy_le',
                    source='Historical dataset admission setting; sampled, not an inference peak bound')
                provenance[ident]=dict(path=portable_path(path,base),line=number,thermal_policy=policy)
    atomic_json(rd/'attempt_audit.json',dict(accepted_sources=provenance,rejected=rejected,excluded=excluded))
    return found


class AndroidBackend:
    def __init__(self, serial, acknowledge_protocol=False):
        if not acknowledge_protocol:
            raise ValueError('Hardware execution requires --acknowledge-protocol after reviewing protocol/comparability')
        self.serial = serial

    def __call__(self, rd, proposal, protocol, config):
        rd = Path(rd).resolve()
        if config.get('mode')=='offline_replay':
            raise ValueError('Offline replay workspaces cannot execute hardware')
        if len(proposal['anchors']) != config.get('anchors',len(proposal['anchors'])):
            raise ValueError('Anchor schedule differs from workspace settings')
        if any(int(c.get('vocab_size',50257))!=50257 for c in proposal['schedule']):
            raise ValueError('Legacy hardware adapter requires vocabulary size 50257')
        if protocol.get('adapter') != 'legacy_random_sweep_v1':
            raise ValueError('Dataset does not explicitly declare the legacy random-sweep adapter')
        if protocol['kernel_id'] != config.get('kernel_id'):
            raise ValueError('Kernel contract mismatch')
        from .storage import digest
        kernel = RUNTIME/KERNEL_SOURCE
        if not kernel.is_file():
            raise FileNotFoundError(f'Inference kernel {KERNEL_SOURCE} not found under {RUNTIME}; set LLMFORGE_DEVICE_RUNTIME')
        if protocol['kernel_id'] != digest(kernel):
            raise ValueError('Inference kernel changed')
        thermal = config.get('thermal_policy')
        if thermal and (thermal.get('comparison')!='lt' or thermal.get('ceiling_c')!=config['temperature_ceiling']
                        or not 0<thermal['ceiling_c']<=45 or thermal.get('previous_ceiling_c')!=protocol.get('temperature_ceiling')
                        or not thermal.get('migration')):
            raise ValueError('Invalid reviewed thermal-policy amendment')
        if (not thermal and protocol.get('temperature_ceiling') != config['temperature_ceiling']) or protocol.get('min_battery_percent') != config['min_battery_percent']:
            raise ValueError('Thermal/battery settings differ from the declared dataset protocol')
        # Global adapter lock: legacy runner uses shared local/remote file names.
        lock = Path(tempfile.gettempdir())/HARDWARE_LOCK
        with locked(lock):
            for snapshot_path in (rd/'attempts').glob('*/original_device_state.json'):
                restoration = snapshot_path.with_name('restoration.json')
                if not restoration.exists() or not read_json(restoration)['settings_restored']:
                    raise RuntimeError(f'Restore device settings first: {snapshot_path}')
            found = merge_attempts(rd,proposal,protocol)
            expected = {c['config_id'] for c in proposal['schedule']}
            missing = expected-set(found)
            if missing:
                attempts = sorted((rd/'attempts').glob('*'))
                if len(attempts)>=config['max_measurement_attempts']:
                    raise RuntimeError('Measurement-attempt budget exhausted; inspect raw attempts, do not blindly retry')
                for name in SCRATCH_FILES:
                    if (WORK/name).exists():
                        raise RuntimeError(f'Existing shared scratch file {WORK/name}; inspect it before running this adapter')
                snapshot = preflight(self.serial,min_battery=config['min_battery_percent'])
                snapshot['toolchain'] = toolchain_info()
                identity_path = rd.parents[1]/'device_identity.json'
                if identity_path.exists():
                    prior = read_json(identity_path)
                    if not snapshot['hardware_serial'] or prior['hardware_serial']!=snapshot['hardware_serial']:
                        raise ValueError('Connected physical device differs from the initialized live session')
                    if any(prior[k]!=snapshot[k] for k in ['abi','build','toolchain']):
                        raise ValueError('Device OS/ABI or local toolchain/tokenizer changed; use a new reviewed protocol')
                else:
                    if not snapshot['hardware_serial']:
                        raise ValueError('Cannot establish physical device identity')
                    atomic_json(identity_path,snapshot)
                attempt = rd/'attempts'/f'{len(attempts)+1:03d}'
                attempt.mkdir(parents=True,exist_ok=False)
                # Optional anchors: refresh POST only when explicitly configured.
                post = {a['after'] for a in proposal['anchors']}
                pending = [c for c in proposal['schedule'] if c['config_id'] in missing|post]
                write_configs(attempt/'configs.csv',pending)
                atomic_json(attempt/'preflight.json',snapshot)
                if thermal:
                    atomic_json(attempt/'thermal_policy.json',thermal)
                command = [sys.executable,'-u','-m','llmforge.hw.device.prediction.active_learning.worker',
                    '--config',str(attempt/'configs.csv'),'--out',str(attempt/'raw_results.csv'),
                    '--serial',self.serial,'--adb',adb_path(),'--work-dir',str(WORK),'--prefill-tokens','48','--steps','32',
                    '--cool-down-temp',str(config['temperature_ceiling']),
                    '--min-battery-percent',str(config['min_battery_percent'])]
                if thermal:
                    command.append('--strict-temperature')
                atomic_json(attempt/'command.json',command)
                print(f'Hardware attempt {attempt.name}: {len(pending)} measurements. Log: {attempt/"hardware.log"}',flush=True)
                with (attempt/'hardware.log').open('x') as log:
                    process = subprocess.Popen(command,env=child_env(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    try:
                        code = process.wait()
                    except BaseException:
                        os.killpg(process.pid,signal.SIGINT)
                        try:
                            process.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            # Do not kill unrelated device processes or hide unresolved restoration.
                            print('Worker still running. Inspect hardware.log/device state before restarting.',flush=True)
                        raise
                atomic_json(attempt/'exit.json',dict(returncode=code))
                found = merge_attempts(rd,proposal,protocol)
                if code:
                    print('Hardware paused/stopped; inspect hardware.log, charge/cool if indicated, then resume.',flush=True)
                if (attempt/'original_device_state.json').exists() and (
                        not (attempt/'restoration.json').exists() or not read_json(attempt/'restoration.json')['settings_restored']):
                    raise RuntimeError('Device settings restoration incomplete; use the restore subcommand before resuming')
            if found:
                destination = rd/'measurements.csv'
                temporary = rd/'measurements.next.csv'
                # The aggregate is regenerable. Raw attempt CSVs are never edited.
                if temporary.exists():
                    temporary.unlink()
                write_configs(temporary,[found[c['config_id']] for c in proposal['schedule'] if c['config_id'] in found])
                os.replace(temporary,destination)


def restore_settings(snapshot_path, serial):
    snapshot_path = Path(snapshot_path)
    snapshot = read_json(snapshot_path)
    identity = read_json(snapshot_path.with_name('preflight.json'))
    command = [adb_path(),'-s',serial]
    def shell(script):
        return subprocess.run(command+['shell',script],capture_output=True,text=True,check=True,timeout=15).stdout.strip()
    with locked(Path(tempfile.gettempdir())/HARDWARE_LOCK):
        if shell('getprop ro.serialno')!=identity['hardware_serial']:
            raise ValueError('Refusing to restore settings on a different device')
        if shell('pidof runq_reallm power_sampler 2>/dev/null || true'):
            raise RuntimeError('Device benchmark still active; wait/inspect before restoring')
        for location,value in snapshot['settings'].items():
            if location not in ('system:screen_off_timeout','global:stay_on_while_plugged_in'):
                raise ValueError('Unknown saved setting')
            if value!='null' and not re.fullmatch(r'\d+',value):
                raise ValueError('Invalid saved setting value')
        for location,value in snapshot['settings'].items():
            table,key = location.split(':')
            shell(f'settings delete {table} {key}' if value=='null' else f'settings put {table} {key} {value}')
            if shell(f'settings get {table} {key}')!=value:
                raise RuntimeError('Restoration read-back failed')
        atomic_json(snapshot_path.with_name('restoration.json'),dict(settings_restored=True,errors=[],restored_manually=True))
