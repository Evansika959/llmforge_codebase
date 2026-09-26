"""Accuracy-oriented active learning. Hardware is OFF unless explicitly enabled."""
import argparse
import csv
import json
import subprocess
from pathlib import Path

from ..config import OUTPUTS
from ..data.dataset import MeasurementDataset
from .datasets import legacy_dataset
from .engine import initialize, advance, verify_workspace
from .hardware import AndroidBackend, preflight, restore_settings
from .storage import read_json, locked, atomic_json, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    init = sub.add_parser('init',help='Freeze initial data, candidate pool and budgets; no hardware access')
    init.add_argument('--workspace',required=True,type=Path)
    source = init.add_mutually_exclusive_group()
    source.add_argument('--dataset',type=Path)
    source.add_argument('--previous',type=Path,default=OUTPUTS/'batch2_progress_632')
    init.add_argument('--candidates',type=Path)
    init.add_argument('--initial-model',type=Path,help='Trusted versioned checkpoint matching the exact initial dataset')
    init.add_argument('--pool-size',type=int,default=5000)
    init.add_argument('--batch-size',type=int,default=20)
    init.add_argument('--members',type=int,default=5)
    init.add_argument('--anchors',type=int,default=0,help='Optional reference architectures (default: none)')
    init.add_argument('--max-rounds',type=int,default=5)
    init.add_argument('--seed',type=int,default=42)
    init.add_argument('--strategy',choices=['hybrid','random'],default='hybrid')
    init.add_argument('--energy',choices=['dynamic','gross'],default='dynamic')
    init.add_argument('--temperature',type=float,default=40.)
    init.add_argument('--min-params-m',type=float,default=50.)
    init.add_argument('--max-params-m',type=float,default=150.)
    step = sub.add_parser('step',help='Prepare/resume bounded rounds; pauses without measurements')
    step.add_argument('--workspace',required=True,type=Path)
    step.add_argument('--rounds',type=int,default=1)
    step.add_argument('--execute-hardware',action='store_true')
    step.add_argument('--acknowledge-protocol',action='store_true')
    step.add_argument('--serial')
    step.add_argument('--evaluate-test',action='store_true',help='Exploratory only; final test is hidden by default')
    run = sub.add_parser('run',help='Continuous hardware loop with target-aware accuracy reports')
    run.add_argument('--workspace',required=True,type=Path)
    run.add_argument('--rounds',type=int,default=500)
    run.add_argument('--serial',required=True)
    run.add_argument('--execute-hardware',action='store_true')
    run.add_argument('--acknowledge-protocol',action='store_true')
    monitor = sub.add_parser('monitor',help='Refresh target-aware accuracy reports; no hardware access')
    monitor.add_argument('--workspace',required=True,type=Path)
    status = sub.add_parser('status')
    status.add_argument('--workspace',required=True,type=Path)
    check = sub.add_parser('preflight',help='Read-only device/charging check')
    check.add_argument('--serial',required=True)
    restore = sub.add_parser('restore',help='Explicitly restore saved display setting values')
    restore.add_argument('--state',required=True,type=Path)
    restore.add_argument('--serial',required=True)
    review = sub.add_parser('review-drift',help='Explicitly accept a reviewed drift-flagged batch, with recorded reason')
    review.add_argument('--workspace',required=True,type=Path)
    review.add_argument('--reason',required=True)
    review.add_argument('--accept',action='store_true')
    migrate = sub.add_parser('disable-anchors',help='Fork an audited workspace without anchors; no hardware or training')
    migrate.add_argument('--source',required=True,type=Path)
    migrate.add_argument('--workspace',required=True,type=Path)
    migrate.add_argument('--reason',required=True)
    migrate.add_argument('--accept-source-changes',action='store_true',help='Acknowledge recorded, limited anchor-policy code changes')
    thermal = sub.add_parser('relax-temperature',help='Fork a 40 C run to strict <45 C sampled admission; no hardware')
    thermal.add_argument('--source',required=True,type=Path)
    thermal.add_argument('--workspace',required=True,type=Path)
    thermal.add_argument('--reason',required=True)
    thermal.add_argument('--accept-source-changes',action='store_true')
    energy = sub.add_parser('switch-energy',help='Create a dynamic-learning stage from verified gross measurements; no hardware')
    energy.add_argument('--source',required=True,type=Path)
    energy.add_argument('--workspace',required=True,type=Path)
    energy.add_argument('--legacy-snapshot',type=Path,default=OUTPUTS/'batch2_progress_632/input_snapshot.json')
    energy.add_argument('--reason',required=True)
    energy.add_argument('--accept-source-changes',action='store_true')
    energy.add_argument('--quarantine-invalid-training-energy',action='store_true',
                        help='Explicitly preserve/exclude unusable training labels in an audit; never filter holdouts')
    replay = sub.add_parser('replay',help='Offline retrospective hybrid-vs-random comparison; no ADB')
    replay.add_argument('--output',required=True,type=Path)
    replay.add_argument('--previous',type=Path,default=OUTPUTS/'batch2_progress_632')
    replay.add_argument('--initial',type=int,default=300)
    replay.add_argument('--batch-size',type=int,default=20)
    replay.add_argument('--rounds',type=int,default=3)
    replay.add_argument('--members',type=int,default=3)
    replay.add_argument('--energy',choices=['dynamic','gross'],default='dynamic')
    replay.add_argument('--seeds',type=int,nargs='+',default=[42,123,2026])
    args = parser.parse_args()
    try:
        if args.command=='init':
            dataset = MeasurementDataset.load(args.dataset) if args.dataset else legacy_dataset(args.previous,args.energy,temperature=args.temperature)
            candidates = None
            if args.candidates:
                with args.candidates.open(newline='') as stream:
                    candidates = list(csv.DictReader(stream))
            settings = {k:getattr(args,k) for k in ['pool_size','batch_size','members','anchors','max_rounds','seed','strategy','min_params_m','max_params_m']}
            settings.update(temperature_ceiling=dataset.protocol.get('temperature_ceiling',args.temperature),
                            min_battery_percent=dataset.protocol.get('min_battery_percent',30),kernel_id=dataset.protocol['kernel_id'])
            initialize(args.workspace,dataset,candidates,settings,args.initial_model)
            print(f'Initialized {args.workspace}. Hardware OFF. Review contract.json and the active-learning README before enabling it.')
        elif args.command=='step':
            backend = None
            if args.execute_hardware:
                if not args.serial:
                    parser.error('--execute-hardware requires an explicit --serial')
                backend = AndroidBackend(args.serial,args.acknowledge_protocol)
            print(json.dumps(advance(args.workspace,backend,args.rounds,args.evaluate_test),indent=2))
        elif args.command=='run':
            if not args.execute_hardware or not args.acknowledge_protocol:
                parser.error('run requires --execute-hardware and --acknowledge-protocol')
            from .monitoring import run_monitored
            run_monitored(args.workspace,args.serial,args.rounds)
        elif args.command=='monitor':
            from .monitoring import update_accuracy
            print(json.dumps(update_accuracy(args.workspace),indent=2))
        elif args.command=='switch-energy':
            from .energy import switch_to_dynamic
            print(switch_to_dynamic(args.source,args.workspace,args.legacy_snapshot,args.reason,args.accept_source_changes,
                                    args.quarantine_invalid_training_energy))
        elif args.command=='status':
            config,state,dataset = verify_workspace(args.workspace)
            print(json.dumps(dict(state=state,budget=config,observations=len(dataset.observations)),indent=2))
        elif args.command=='preflight':
            print(json.dumps(preflight(args.serial),indent=2))
        elif args.command=='restore':
            restore_settings(args.state,args.serial)
            print('Saved device setting values restored and verified')
        elif args.command=='review-drift':
            if not args.accept or not args.reason.strip():
                parser.error('Review requires --accept and a nonempty --reason; do not bypass uninvestigated drift')
            with locked(args.workspace/'RUNNING.lock'):
                _,state,_ = verify_workspace(args.workspace)
                if state['status']!='paused_anchor_drift':
                    raise ValueError('Workspace is not paused for anchor drift')
                rd = args.workspace/'rounds'/f'{state["pending_round"]:04d}'
                atomic_json(rd/'drift_review.json',dict(reason=args.reason.strip(),results_sha256=digest(rd/'measurements.csv')))
            print('Review recorded for this exact result file. Rerun step to continue; no hardware was started.')
        elif args.command=='relax-temperature':
            from .migration import change_temperature
            print(change_temperature(args.source,args.workspace,args.reason,args.accept_source_changes))
        elif args.command=='disable-anchors':
            from .migration import disable_anchors
            print(disable_anchors(args.source,args.workspace,args.reason,args.accept_source_changes))
        elif args.command=='replay':
            from .replay import run_replay
            run_replay(args.output,args.previous,args.initial,args.batch_size,args.rounds,args.members,args.seeds,args.energy)
    except (ValueError,RuntimeError,FileNotFoundError,subprocess.SubprocessError) as error:
        parser.exit(2,f'Active learning paused: {error}\nNo automatic hardware retry was started.\n')
    except KeyboardInterrupt:
        parser.exit(130,'Interrupted. Inspect attempt logs/device state before resuming the same workspace.\n')
