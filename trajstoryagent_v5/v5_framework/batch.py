"""Independent v5 generation batches. Evaluation remains a separate operation."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('gpt', 'qwen'), required=True)
    parser.add_argument('--manifest', type=Path, default=ROOT/'trajstory_examples/insight/manifest.json')
    parser.add_argument('--repo-root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True, help='New batch directory; never a historical results directory')
    parser.add_argument('--case-ids', nargs='+', help='Default: all cases in the manifest')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--experiment', choices=('without_conditional_refinement','without_compression','without_memory','without_data_profiling','hyperparameter'))
    parser.add_argument('--source-generation', type=Path, help='Frozen generation directory containing CASE_ID/result.json')
    parser.add_argument('--checkpoint-manifest', type=Path, help='Original experiment manifest containing replay checkpoints')
    parser.add_argument('--partition-penalty', type=float, default=1.)
    parser.add_argument('--min-support', type=int, default=1)
    parser.add_argument('--tolerance-sd', type=float, default=.25)
    parser.add_argument('--distortion-weight', type=float, default=1.)
    parser.add_argument('--dry-run', action='store_true', help='Validate inputs and print plan without writing or API calls')
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.source_generation and args.checkpoint_manifest:
        parser.error('Choose --source-generation or --checkpoint-manifest')
    if bool(args.experiment) != bool(args.source_generation or args.checkpoint_manifest):
        parser.error('--experiment requires a checkpoint manifest or source generation directory')
    from trajstoryagent_v5.v5_framework.case_adapter import load_trajstory_case
    from trajstoryagent_v5.v5_framework.identity import IMPLEMENTATION_ID
    root = args.repo_root.resolve()
    manifest = args.manifest.resolve()
    cases = json.loads(manifest.read_text(encoding='utf-8'))['cases']
    ids = [c['case_id'] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate case IDs in manifest')
    requested = set(args.case_ids or ids)
    if requested - set(ids):
        raise ValueError('Unknown case IDs: '+', '.join(sorted(requested-set(ids))))
    cases = sorted((c for c in cases if c['case_id'] in requested), key=lambda c:c['case_id'])
    if not cases:
        raise ValueError('No cases selected')
    inputs = {str(manifest): digest(manifest)}
    for case in cases:
        cid = case['case_id']
        if not cid or cid in ('.', '..') or any(x in cid for x in '/\\:'):
            raise ValueError('Unsafe case ID')
        path = (root/case['json_path']).resolve()
        bundle = load_trajstory_case(path, root, input_csvs=True)
        for p in [path, *(f.path for f in bundle.files)]:
            inputs[str(p)] = digest(p)
    package = Path(__file__).resolve().parent.parent
    sources = {str(p.relative_to(ROOT)): digest(p) for p in sorted(package.rglob('*'))
               if p.is_file() and p.suffix in {'.py', '.md', '.json', '.txt', '.toml'}
               and 'tests' not in p.parts and '__pycache__' not in p.parts}
    plan = dict(implementation=IMPLEMENTATION_ID, model=args.model, cases=cases,
                input_sha256=inputs, source_sha256=sources,
                policy='One fresh attempt per selected case; no score-based selection; evaluation separate')
    experiment_args=[]
    if args.experiment:
        from trajstoryagent_v5.ablation.experiment import main as validate_experiment
        from trajstoryagent_v5.v5_framework.export_support import read as read_checkpoint
        frozen_plan=read_checkpoint(args.checkpoint_manifest) if args.checkpoint_manifest else None
        for case in cases:
            saved=frozen_plan['replay'][case['case_id']] if frozen_plan else read_checkpoint(args.source_generation/case['case_id']/'result.json')
            case['checkpoint']=saved['source_run']
        experiment_args=['--variant',args.experiment,'--partition-penalty',str(args.partition_penalty),
                        '--min-support',str(args.min_support),'--tolerance-sd',str(args.tolerance_sd),
                        '--distortion-weight',str(args.distortion_weight)]
        for case in cases:
            case['checkpoint_arguments']=['--checkpoint-manifest',str(args.checkpoint_manifest.resolve()),'--case-id',case['case_id']] if args.checkpoint_manifest else []
            validate_experiment(['--source-run',case['checkpoint'],'--model',args.model,
                '--output',str(args.output.resolve()/'generation'/case['case_id']),*experiment_args,*case['checkpoint_arguments'],'--dry-run'])
        plan['experiment']=dict(variant=args.experiment,arguments=experiment_args)
    elif (args.partition_penalty,args.min_support,args.tolerance_sd,args.distortion_weight)!=(1.,1,.25,1.):
        parser.error('Overrides require --experiment hyperparameter')
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'Refusing to overwrite an existing batch: {output}')
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    from trajstoryagent_v5.v5_framework.providers import credentials
    credentials(args.model)
    output.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents two launchers from sharing a batch directory.
    with (output/'batch_manifest.json').open('x', encoding='utf-8') as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)
    for rel, expected in sources.items():
        data = (ROOT/rel).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError('Source changed during preparation: '+rel)
        target = output/'source_snapshot'/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    def run(case):
        cid = case['case_id']
        dest = output/'generation'/cid
        log = output/'logs'/f'{cid}.log'
        log.parent.mkdir(exist_ok=True)
        command = [sys.executable, '-X', 'utf8', '-m', 'trajstoryagent_v5',
                   '--model', args.model, '--case', str(root/case['json_path']),
                   '--repo-root', str(root), '--output', str(dest)]
        if args.experiment:
            command=[sys.executable,'-X','utf8','-m','trajstoryagent_v5.ablation.experiment',
                     '--model',args.model,'--source-run',case['checkpoint'],'--output',str(dest),*experiment_args,*case['checkpoint_arguments']]
        with log.open('w', encoding='utf-8') as stream:
            proc = subprocess.run(command, cwd=output/'source_snapshot', stdout=stream, stderr=subprocess.STDOUT, check=False)
        success = proc.returncode == 0 and (dest/'result.json').is_file()
        return dict(case_id=cid, returncode=proc.returncode, state='completed' if success else 'failed', output=str(dest))

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run, cases))
    write(output/'batch_results.json', results)
    completed = sum(r['state']=='completed' for r in results)
    print(json.dumps(dict(completed=completed, total=len(results), output=str(output))))
    return 0 if completed == len(results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
