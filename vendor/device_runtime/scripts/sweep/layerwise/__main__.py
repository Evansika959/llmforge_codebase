"""Prepare, validate, export or explicitly run the layerwise hardware campaign."""
import argparse
import json
from pathlib import Path

from .database import ROOT, prepare, validate_database, read_candidate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    create=sub.add_parser('prepare')
    create.add_argument('--spec',type=Path,default=ROOT/'scripts/sweep/configs/watch5_layerwise_500_space.json')
    create.add_argument('--output',type=Path,required=True)
    create.add_argument('--exclude-database',type=Path,action='append',default=[],
                        help='Exclude prior architectures AND their layer-order permutations; repeatable')
    check=sub.add_parser('validate'); check.add_argument('--database',type=Path,required=True)
    export=sub.add_parser('export'); export.add_argument('--database',type=Path,required=True)
    export.add_argument('--candidate',required=True); export.add_argument('--output',type=Path,required=True)
    run=sub.add_parser('run'); run.add_argument('--output',type=Path,required=True)
    run.add_argument('--serial',required=True); run.add_argument('--max-jobs',type=int,default=500)
    run.add_argument('--acknowledge-protocol',action='store_true')
    status=sub.add_parser('status'); status.add_argument('--output',type=Path,required=True)
    restore=sub.add_parser('restore'); restore.add_argument('--output',type=Path,required=True); restore.add_argument('--serial',required=True)
    args=parser.parse_args()
    try:
        if args.command=='prepare': result=prepare(args.spec,args.output,args.exclude_database)
        elif args.command=='validate': result=validate_database(args.database)
        elif args.command in ('run','status','restore'):
            from . import runner
            if args.command=='run':
                import torch
                torch.set_num_threads(4)
                raise SystemExit(runner.run(args.output,args.serial,args.max_jobs,args.acknowledge_protocol))
            result=runner.status(args.output) if args.command=='status' else runner.restore(args.output,args.serial)
        else:
            import torch
            torch.set_num_threads(4)
            from .export import export_mock
            result=export_mock(read_candidate(args.database,args.candidate),args.output)
        print(json.dumps(result,indent=2))
    except (ValueError,FileExistsError) as error:
        parser.exit(2,f'Layerwise preparation stopped: {error}\n')


if __name__=='__main__': main()
