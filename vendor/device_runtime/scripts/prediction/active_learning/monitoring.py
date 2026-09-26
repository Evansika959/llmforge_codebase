"""Target-aware local accuracy reporting and an explicitly enabled live runner."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from ..config import ROOT
from ..models.serialization import load_bundle
from .engine import verify_workspace, _evaluate, advance
from .hardware import AndroidBackend
from .storage import atomic_json, digest, locked, read_json


def update_accuracy(workspace):
    workspace = Path(workspace).resolve()
    config,state,dataset = verify_workspace(workspace)
    holdouts = [r for r in dataset.observations if r['split']!='train']
    holdout_hash = hashlib.sha256(json.dumps(holdouts,sort_keys=True,allow_nan=False).encode()).hexdigest()
    folder = workspace/'monitoring'
    folder.mkdir(exist_ok=True)
    with locked(folder/'REPORT.lock'):
        baseline_path = folder/'baseline.json'
        if baseline_path.exists():
            baseline = read_json(baseline_path)
        else:
            initial = workspace/'initial_model.joblib'
            if initial.exists():
                from ..data.dataset import MeasurementDataset
                initial_dataset = MeasurementDataset.load(workspace/'dataset_000.json')
                model = load_bundle(initial)
                if (model['dataset_sha256']!=initial_dataset.fingerprint or model['targets']!=dataset.targets or
                        model['protocol']!=dataset.protocol):
                    raise ValueError('Monitoring initial checkpoint mismatch')
                metrics = _evaluate(model,initial_dataset,'validation')
                baseline_round = 0
                model_hash = digest(initial)
            else:
                proposal = workspace/'rounds/0001/proposal.json'
                if not proposal.exists():
                    raise ValueError('Prepare a hardware-off step before starting accuracy monitoring')
                document = read_json(proposal)
                metrics = document['validation_before']
                baseline_round,model_hash = 0,document['artifacts']['predictor_before.joblib']
            baseline = dict(round=baseline_round,targets=dataset.targets,energy_target=dataset.protocol['energy_target'],
                model_sha256=model_hash,metrics=metrics,
                holdout_sha256=holdout_hash,
                holdout_ids=[r['measurement_id'] for r in dataset.observations if r['split']!='train'])
            atomic_json(baseline_path,baseline)
        if (baseline['targets']!=dataset.targets or baseline['energy_target']!=dataset.protocol['energy_target'] or
                baseline.get('holdout_sha256')!=holdout_hash or
                baseline['holdout_ids']!=[r['measurement_id'] for r in dataset.observations if r['split']!='train']):
            raise ValueError('Monitoring target/split changed; never compare gross and dynamic histories')
        reference = {m['target']:m for m in baseline['metrics']}
        history = [dict(round=0,validation=baseline['metrics'])]
        for number in range(1,state['completed_rounds']+1):
            path = workspace/f'rounds/{number:04d}/completion.json'
            completion = read_json(path)
            if [m['target'] for m in completion['validation_after']]!=dataset.targets:
                raise ValueError('Completed round has a different prediction target')
            values = []
            for metric in completion['validation_after']:
                prior = reference[metric['target']]['mape']
                values.append(dict(metric,mape_reduction_pp=prior-metric['mape'],
                    relative_mape_reduction_pct=100*(prior-metric['mape'])/prior if prior else None))
            history.append(dict(round=completion['round'],validation=values,model_sha256=completion['model_sha256']))
        report = dict(updated_utc=datetime.now(timezone.utc).isoformat(),state=state,targets=dataset.targets,
            energy_target=dataset.protocol['energy_target'],baseline=baseline,history=history,
            training_rows=sum(r['split']=='train' for r in dataset.observations),
            interpretation='Positive MAPE reduction means improvement. Validation is reused for stopping, not an untouched final test. No ADB or test evaluation. Different energy targets must not share an accuracy baseline.')
        atomic_json(folder/'accuracy_history.json',report)
        lines = [f'# Accuracy: {dataset.protocol["energy_target"]} energy','',
                 'Fixed validation. Positive reduction means improvement. Baseline is this stage’s initial checkpoint, not the previous energy target.',
                 '', '| Round | Target | MAPE % | Reduction (pp) | MAE | R² |', '|---:|---|---:|---:|---:|---:|']
        for row in history:
            for metric in row['validation']:
                lines.append(f"| {row['round']} | {metric['target']} | {metric['mape']:.3f} | {metric.get('mape_reduction_pp',0):+.3f} | {metric['mae']:.3f} | {metric['r2']:.4f} |")
        lines += ['',f"State: {state['status']}; completed {state['completed_rounds']}; training rows {report['training_rows']}."]
        temporary = folder/'ACCURACY.next.md'
        temporary.write_text('\n'.join(lines)+'\n')
        os.replace(temporary,folder/'ACCURACY.md')
        return report


def run_monitored(workspace, serial, rounds=500):
    """Caller must explicitly enable hardware. Does not retry a stopped child."""
    workspace = Path(workspace).resolve()
    if rounds<1:
        raise ValueError('rounds must be positive')
    AndroidBackend(serial,True)  # Same explicit device validation remains in step.
    verify_workspace(workspace)
    folder = workspace/'monitoring'
    folder.mkdir(exist_ok=True)
    with locked(folder/'SUPERVISOR.lock'):
        if not (workspace/'initial_model.joblib').exists() and not (workspace/'rounds/0001/proposal.json').exists():
            advance(workspace)  # Initial proposal only, hardware remains off.
        update_accuracy(workspace)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        path = folder/f'run_{stamp}.json'
        command = [sys.executable,'-u','-m','scripts.prediction','active','step','--workspace',str(workspace),
            '--execute-hardware','--acknowledge-protocol','--serial',serial,'--rounds',str(rounds)]
        with (folder/f'live_{stamp}.log').open('x') as log:
            process = subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            record = dict(pid=process.pid,supervisor_pid=os.getpid(),started_utc=stamp,command=command,
                          energy_target=read_json(workspace/'contract.json')['protocol']['energy_target'])
            atomic_json(path,record)
            print(f'Live loop started; log: {log.name}; accuracy: {folder/"ACCURACY.md"}',flush=True)
            previous_handler = signal.getsignal(signal.SIGTERM)
            def interrupted(signum,frame):
                raise KeyboardInterrupt
            signal.signal(signal.SIGTERM,interrupted)
            try:
                while True:
                    try:
                        code = process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        code = None
                    report = update_accuracy(workspace)
                    if code is not None:
                        print(f"Loop exited ({code}): {report['state']['message']}",flush=True)
                        if code:
                            raise RuntimeError(f'Hardware loop stopped with exit {code}; inspect {log.name}')
                        return report['state']
            finally:
                signal.signal(signal.SIGTERM,previous_handler)
                if process.poll() is None:
                    os.killpg(process.pid,signal.SIGINT)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        print('Runner still shutting down; inspect state before resuming.',flush=True)
                record.update(ended_utc=datetime.now(timezone.utc).isoformat(),returncode=process.poll())
                atomic_json(path,record)
                update_accuracy(workspace)
