"""Opt-in subprocess adapter; original measurement script and C sources unchanged.

Deterministic architecture-specific model weights make anchor repeats comparable.
Display settings changed by the legacy runner are saved/restored outside inference.
"""
import hashlib
import builtins
import csv
import os
import shutil
from pathlib import Path
import re
import signal
import subprocess
import sys

from .storage import atomic_json


def main():
    # This module is only started by the explicit hardware adapter.
    args = sys.argv[1:]
    strict_temperature = '--strict-temperature' in args
    if strict_temperature:
        args.remove('--strict-temperature')
        sys.argv = [sys.argv[0]]+args
    if '--serial' not in args or '--out' not in args or '--adb' not in args:
        raise ValueError('Worker requires explicit --serial, --adb and --out')
    serial = args[args.index('--serial')+1]
    adb = args[args.index('--adb')+1]
    folder = Path(args[args.index('--out')+1]).parent
    command = [adb,'-s',serial]
    def shell(script):
        return subprocess.run(command+['shell',script],capture_output=True,text=True,check=True,timeout=15).stdout.strip()
    state = {}
    for table,key in [('system','screen_off_timeout'),('global','stay_on_while_plugged_in')]:
        value = shell(f'settings get {table} {key}')
        if value!='null' and not re.fullmatch(r'\d+',value):
            raise ValueError('Unexpected display setting value')
        state[f'{table}:{key}']=value
    atomic_json(folder/'original_device_state.json',dict(serial=serial,settings=state))
    from scripts.sweep import run_sweep_configs as legacy
    if strict_temperature:
        from .thermal import wait_ready
        legacy.wait_for_thermal_and_voltage_recovery = lambda *a,**kw: wait_ready(legacy,*a,**kw)
        def legacy_print(*values,**kwargs):
            values = tuple(v.replace('Thermal Guard: Target CPU <=','Thermal Guard: Target CPU <') if isinstance(v,str) else v for v in values)
            builtins.print(*values,**kwargs)
        legacy.print = legacy_print
        print('Strict sampled admission enabled: temperature must be below the ceiling; no temperature queries during inference. Valid cooldown waits resume automatically.',flush=True)
    import torch
    config_path = Path(args[args.index('--config')+1])
    with config_path.open(newline='') as stream:
        scheduled = list(csv.DictReader(stream))
    current = dict(index=0,folder=None)
    original = legacy.generate_mock_ckpt
    def seeded_weights(*dimensions):
        ident = scheduled[current['index']]['config_id']
        current['index'] += 1
        current['folder'] = folder/'traces'/ident
        current['folder'].mkdir(parents=True,exist_ok=False)
        seed = int.from_bytes(hashlib.sha256(repr(dimensions).encode()).digest()[:4],'little')
        atomic_json(current['folder']/'model_seed.json',dict(seed=seed,dimensions=dimensions))
        torch.manual_seed(seed)
        return original(*dimensions)
    legacy.generate_mock_ckpt = seeded_weights
    class PreservingOS:
        def __getattr__(self,name):
            return getattr(os,name)
        def remove(self,path):
            if str(path) in ('temp_trace_raw.csv','temp_infer_log.txt') and current['folder'] is not None:
                shutil.copy2(path,current['folder']/Path(path).name)
            return os.remove(path)
    # Replace only the legacy module's os binding, never the global os.remove.
    legacy.os = PreservingOS()
    original_benchmark = legacy.run_benchmark_with_power
    def benchmark(*positional,**kwargs):
        result = original_benchmark(*positional,**kwargs)
        log = (current['folder']/'temp_infer_log.txt').read_text(errors='replace')
        prompt = re.search(r'prefill_tokens:\s*(\d+)',log)
        decode = re.search(r'decode_tokens:\s*(\d+)',log)
        if not prompt or not decode or int(prompt[1])!=49 or int(decode[1])!=31:
            raise ValueError('Inference did not complete the required 49-prompt/32-output workload; trace retained')
        return result
    legacy.run_benchmark_with_power = benchmark
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    try:
        legacy.main()
    finally:
        errors = []
        for location,value in state.items():
            table,key = location.split(':')
            try:
                shell(f'settings delete {table} {key}' if value=='null' else f'settings put {table} {key} {value}')
                if shell(f'settings get {table} {key}')!=value:
                    raise RuntimeError('Read-back mismatch')
            except Exception as error:
                errors.append(f'{location}: {error}')
        atomic_json(folder/'restoration.json',dict(settings_restored=not errors,errors=errors,
            note='Display setting values only. Wakefulness is not automatically restored. After forced termination, inspect device processes and the saved state.'))
        if errors:
            print('RESTORATION INCOMPLETE:',errors,flush=True)
            raise RuntimeError('Device setting restoration incomplete; see original_device_state.json')


if __name__=='__main__':
    main()
