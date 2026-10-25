"""One coordinator for explicit Pandas batches and retries. No model calls by default.

prepare freezes an explicitly supplied case list into a fresh isolated batch.
run requires --run; resume reuses every recorded case, including failed cases.
The existing generator and two-round GPT judge provide the analysis protocol.
Historical result replacement/consolidation is deliberately not automatic.
"""
from __future__ import annotations

import sys as _baseline_sys
from pathlib import Path as _BaselinePath
_baseline_repo = next(p for p in _BaselinePath(__file__).resolve().parents if (p / "baselines" / "__init__.py").is_file())
_baseline_sys.path.insert(0, str(_baseline_repo))
import baselines  # Register the shared baseline module location.

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = 'pandas.explicit_batch.v1'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent,
                                     prefix=path.name+'.', suffix='.tmp', delete=False) as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        temp = f.name
    os.replace(temp, path)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def now():
    return datetime.now(timezone.utc).isoformat()

def batch_path(value):
    base = (ROOT/'results/pandas/batches').resolve()
    path = Path(value).resolve()
    if path == base or not path.is_relative_to(base):
        raise ValueError('Batch must be a child of '+str(base))
    return path

def source_paths():
    return [Path(__file__), ROOT/'baselines/scripts/run_pandas.py',
            ROOT/'baselines/pandas_multi_role/native_runner.py', ROOT/'baselines/scripts/pandas_repl_compat.py',
            ROOT/'baselines/scripts/pandas_usage_logging.py', ROOT/'baselines/scripts/pandas_native_budget.py',
            ROOT/'baselines/scripts/evaluate_pandas_checkpoint_batch.py',
            ROOT/'baselines/baseline_support/case_adapter.py',ROOT/'baselines/baseline_support/input_paths.py',
            *sorted((ROOT/'baselines/pandas_multi_role').glob('*.py')),*sorted((ROOT/'baselines/baseline_support').glob('*.py')),ROOT/'baselines/baseline_support/judge_prompts.json']

def prepare(batch, model, case_ids, workers=3, max_model_calls=60):
    batch = batch_path(batch)
    if model not in ('gpt','qwen') or not 1 <= workers <= 32 or max_model_calls < 1:
        raise ValueError('Invalid model, worker count or request budget')
    if not case_ids or len(case_ids) != len(set(case_ids)):
        raise ValueError('Supply a nonempty list of distinct case IDs')
    if batch.exists():
        raise FileExistsError('Fresh batch directory required; nothing overwritten')
    catalog = {c['case_id']: c for c in read(ROOT/'trajstory_examples/insight/manifest.json')['cases']}
    if set(case_ids)-catalog.keys():
        raise ValueError('Unknown cases: '+str(sorted(set(case_ids)-catalog.keys())))
    sys.path.insert(0,str(ROOT))
    from baseline_support.case_adapter import load_trajstory_case
    frozen = {str(p.resolve()):sha(p) for p in source_paths()}
    cases=[]
    for cid in case_ids:
        c=catalog[cid]
        bundle=load_trajstory_case(ROOT/c['json_path'],ROOT,input_csvs=True)
        for f in bundle.files:
            if not f.path.name.startswith('input_'):
                raise ValueError('Only formal input_*.csv files are allowed')
            frozen[str(f.path.resolve())]=sha(f.path)
        frozen[str((ROOT/c['json_path']).resolve())]=sha(ROOT/c['json_path'])
        cases.append({'case_id':cid,'json_path':c['json_path']})
    manifest={'protocol':PROTOCOL,'created_at':now(),'model':model,'case_ids':case_ids,
              'cases':cases,'workers':workers,'max_model_calls':max_model_calls,
              'whole_case_attempts':1,'projected_input_csvs':True,
              'python':str(ROOT/'.runtime/pandas_agent/Scripts/python.exe'),
              'judge':'gpt-6-luna','judge_rounds':2,'temperature':0,'reasoning_effort':'none',
              'replacement_policy':'None: results remain separate; no score-based selection',
              'frozen_sha256':frozen}
    batch.mkdir(parents=True,exist_ok=False)
    write(batch/'manifest.json',manifest)
    write(batch/'status.json',{'state':'prepared','total':len(cases),'completed':0})
    return manifest

def check(manifest):
    if manifest.get('protocol') != PROTOCOL:
        raise ValueError('Only a manifest prepared by this coordinator is accepted')
    for path, expected in manifest['frozen_sha256'].items():
        if sha(path) != expected:
            raise ValueError('Frozen input/source changed: '+path)

def generation_command(manifest, case, out):
    return [manifest['python'],'-X','utf8',str(ROOT/'baselines/scripts/run_pandas.py'),'--workflow','roles',
            '--case-id',case['case_id'],'--model',manifest['model'],
            '--max-model-calls',str(manifest['max_model_calls']),
            '--input-csvs','--results-dir',str(out),'--run']

def recorded_case(batch,case):
    """Never silently regenerate an exited, failed or interrupted job."""
    out=batch/'generation'/case['case_id']
    job=batch/'jobs'/(case['case_id']+'.json')
    if not job.exists():
        if out.exists():raise RuntimeError('Untracked existing generation: '+str(out))
        return None
    previous=read(job)
    if previous.get('state') != 'finished':
        raise RuntimeError('Running/interrupted job requires inspection: '+str(job))
    if (out/'job_result.json').is_file():return read(out/'job_result.json')
    state=read(out/'status.json') if (out/'status.json').is_file() else {'state':'failed'}
    return {'case_id':case['case_id'],'json_path':case['json_path'],'directory':str(out),
            'returncode':previous.get('returncode'),'eligible_for_scoring':False,
            'status':state,'returned_insight_count':0}

def generate(batch,manifest,case):
    previous=recorded_case(batch,case)
    if previous is not None:return previous
    out=batch/'generation'/case['case_id'];log=batch/'logs'/(case['case_id']+'.log')
    job=batch/'jobs'/(case['case_id']+'.json');log.parent.mkdir(parents=True,exist_ok=True)
    write(job,{'state':'starting','started_at':now()})
    with log.open('w',encoding='utf-8') as f:
        proc=subprocess.Popen(generation_command(manifest,case,out),cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        write(job,{'state':'running','pid':proc.pid,'started_at':now()})
        code=proc.wait()
    write(job,{'state':'finished','returncode':code,'finished_at':now()})
    return recorded_case(batch,case)

def judge(batch,manifest):
    # Reuse the maintained scorer, instead of writing a modified scorer for each retry.
    sys.path.insert(0,str(ROOT))
    path=ROOT/'baselines/scripts/evaluate_pandas_checkpoint_batch.py'
    spec=importlib.util.spec_from_file_location('pandas_batch_judge',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    mod.GENERATOR=manifest['model']
    frozen=mod.freeze(batch)
    return mod.evaluate(batch,frozen)

def run(batch,resume=False):
    batch=batch_path(batch);manifest=read(batch/'manifest.json');check(manifest)
    state=read(batch/'status.json')['state']
    if state=='completed':raise RuntimeError('Batch already completed; no rerun')
    if not resume and state!='prepared':raise RuntimeError('Use resume for an existing batch')
    # Check all existing jobs before launching any new work.
    for case in manifest['cases']:recorded_case(batch,case)
    lock=batch/'coordinator.lock'
    with lock.open('x',encoding='utf-8') as f:f.write(str(os.getpid()))
    try:
        rows=[]
        write(batch/'status.json',{'state':'generating','pid':os.getpid(),'total':len(manifest['cases']),'completed':0})
        with ThreadPoolExecutor(max_workers=manifest['workers']) as pool:
            jobs=[pool.submit(generate,batch,manifest,c) for c in manifest['cases']]
            for f in as_completed(jobs):
                rows.append(f.result());rows.sort(key=lambda x:x['case_id'])
                write(batch/'generation.json',rows)
                write(batch/'status.json',{'state':'generating','pid':os.getpid(),'total':len(jobs),'completed':len(rows)})
        check(manifest)
        write(batch/'status.json',{'state':'evaluating','total':len(rows),'completed':len(rows)})
        report=judge(batch,manifest);check(manifest)
        write(batch/'status.json',{'state':'completed','total':len(rows),'completed':len(rows),'completed_at':now()})
        return report
    except Exception as exc:
        write(batch/'status.json',{'state':'pipeline_failed','error_type':type(exc).__name__,'updated_at':now()})
        raise
    finally:
        lock.unlink(missing_ok=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare',help='Freeze explicit cases; no model calls')
    p.add_argument('--batch',type=Path,required=True);p.add_argument('--model',choices=['gpt','qwen'],required=True)
    p.add_argument('--case-id',action='append',required=True);p.add_argument('--workers',type=int,default=3)
    p.add_argument('--max-model-calls',type=int,default=60)
    for name in ('run','resume'):
        p=sub.add_parser(name);p.add_argument('--batch',type=Path,required=True);p.add_argument('--run',action='store_true',required=True)
    p=sub.add_parser('status',help='Read an existing batch status without running anything')
    p.add_argument('--batch',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='prepare':result=prepare(args.batch,args.model,args.case_id,args.workers,args.max_model_calls)
    elif args.command=='status':result=read(args.batch/'status.json')
    else:result=run(args.batch,args.command=='resume')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
