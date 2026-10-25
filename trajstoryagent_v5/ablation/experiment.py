"""Independent historical-protocol ablation / hyperparameter continuation entry."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from trajstoryagent_v5.v5_framework.export_support import read, write
from trajstoryagent_v5.v5_framework.checkpoint_paths import relocate_path
from trajstoryagent_v5.v5_framework.providers import MODELS, credentials
from trajstoryagent_v5.ablation.ablation_replay import ReplayAblation
from trajstoryagent_v5.ablation.ablation_memory_profile import MemoryAblation, DataProfilingAblation, bundle_from
from trajstoryagent_v5.hyperparameter_study.hyperparameter_replay import HyperparameterReplay

VARIANTS=('without_conditional_refinement','without_compression','without_memory','without_data_profiling','hyperparameter')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prepare(source, generator, variant):
    source=Path(relocate_path(str(source))).resolve()
    manifest=read(source/'run_manifest.json')
    if manifest['model'] != MODELS[generator][0]:
        raise ValueError('Checkpoint backbone differs from requested generator')
    for name,digest in manifest['input_sha256'].items():
        if sha(name)!=digest: raise ValueError('Frozen input changed: '+name)
    bundle=bundle_from(read(source/'00_input.json'))
    config=read(source/'00_config.json')
    if variant!='without_data_profiling':
        if read(source/'03_conflict_admission.json')['admitted_topic_count']!=0:
            raise ValueError('Historical replay protocol requires zero admitted first-round conflicts')
        # Instantiate all checkpoint loaders offline, before allowing API calls.
        if variant=='without_memory': MemoryAblation(None,generator,source)
        else: ReplayAblation(None,variant,source,generator=generator)
    files=[p for p in source.rglob('*') if p.is_file() and 'source_snapshot' not in p.relative_to(source).parts
           and p.suffix in {'.json','.py','.csv'}]
    frozen={str(p):sha(p) for p in files}
    return source,bundle,config,frozen


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',type=Path,required=True,help='Frozen case run directory, not generation/result.json')
    p.add_argument('--checkpoint-manifest',type=Path,help='Historical experiment manifest with replay[case_id] hashes')
    p.add_argument('--case-id',help='Required with --checkpoint-manifest')
    p.add_argument('--model',choices=MODELS,required=True)
    p.add_argument('--variant',choices=VARIANTS,required=True)
    p.add_argument('--output',type=Path,required=True,help='Fresh output; original results are never replaced')
    p.add_argument('--partition-penalty',type=float,default=1.)
    p.add_argument('--min-support',type=int,default=1)
    p.add_argument('--tolerance-sd',type=float,default=.25)
    p.add_argument('--distortion-weight',type=float,default=1.)
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args(argv)
    args.source_run=Path(relocate_path(str(args.source_run)))
    if bool(args.checkpoint_manifest)!=bool(args.case_id):p.error('--checkpoint-manifest and --case-id must be supplied together')
    if args.checkpoint_manifest:
        frozen_plan=read(args.checkpoint_manifest)
        checkpoint=frozen_plan['replay'][args.case_id]
        if Path(checkpoint['source_run']).resolve()!=args.source_run.resolve():raise ValueError('Source differs from frozen experiment selection')
        for path,digest in checkpoint['files'].items():
            if sha(path)!=digest:raise ValueError('Historical checkpoint hash mismatch: '+path)
    params=dict(partition_penalty=args.partition_penalty,min_support=args.min_support,
                tolerance_sd=args.tolerance_sd,distortion_weight=args.distortion_weight)
    if args.min_support<1 or any(not math.isfinite(v) or v<=0 for k,v in params.items()):
        p.error('Parameters must be finite and positive')
    defaults=dict(partition_penalty=1.,min_support=1,tolerance_sd=.25,distortion_weight=1.)
    if args.variant!='hyperparameter' and params!=defaults:
        p.error('Mathematical overrides require --variant hyperparameter')
    output=args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('Use a new empty experiment output directory')
    source,bundle,config,frozen=prepare(args.source_run,args.model,args.variant)
    if output==source or source in output.parents or output in source.parents:
        raise ValueError('Output must be separate from checkpoint tree')
    package=Path(__file__).resolve().parent.parent
    sources={str(p.relative_to(package.parent)):sha(p) for p in package.rglob('*') if p.is_file()
             and p.suffix in {'.py','.md','.json','.txt','.toml'} and 'tests' not in p.parts and '__pycache__' not in p.parts}
    plan=dict(generator=args.model,variant=args.variant,parameters=params,checkpoint_sha256=frozen,
              source_sha256=sources,first_round_source=str(source),
              protocol='fresh two-round run with raw schema only' if args.variant=='without_data_profiling' else 'frozen first-round continuation',
              selection='one attempt, no score-conditioned adoption; historical outputs unchanged')
    if args.checkpoint_manifest:
        plan['historical_manifest']=dict(path=str(args.checkpoint_manifest.resolve()),sha256=sha(args.checkpoint_manifest),case_id=args.case_id)
    if args.dry_run:
        print(json.dumps(dict(ready=True,variant=args.variant,checkpoint_files=len(frozen),protocol=plan['protocol'])))
        return 0
    credentials(args.model)
    output.mkdir(parents=True,exist_ok=True)
    with (output/'experiment_manifest.json').open('x',encoding='utf-8') as f:
        json.dump(plan,f,ensure_ascii=False,indent=2)
    for rel,digest in sources.items():
        data=(package.parent/rel).read_bytes()
        if hashlib.sha256(data).hexdigest()!=digest: raise RuntimeError('Runtime source changed')
        dest=output/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    from trajstoryagent_v5.v5_framework.transport import ResilientClient
    from trajstoryagent_v5.v5_framework.terminology import InsightClient
    from trajstoryagent_v5.v5_framework import export_support
    model,url=MODELS[args.model]
    key='OPENAI_API_KEY' if args.model=='gpt' else 'DASHSCOPE_API_KEY'
    client=ResilientClient(model=model,base_url=url,api_key=os.environ[key],temperature=0,reasoning_effort='none',max_tokens=16000)
    if args.variant=='hyperparameter':agent=HyperparameterReplay(client,source,args.model,**params)
    elif args.variant=='without_memory':agent=MemoryAblation(client,args.model,source)
    elif args.variant=='without_data_profiling':agent=DataProfilingAblation(client,args.model,config)
    else:agent=ReplayAblation(client,args.variant,source,generator=args.model)
    original=export_support.OpenAICompatibleClient
    old_audit=os.environ.get('TRAJSTORY_LLM_AUDIT_PATH')
    os.environ['TRAJSTORY_LLM_AUDIT_PATH']=str(output/'llm_calls.jsonl')
    export_support.OpenAICompatibleClient=lambda **kw:InsightClient(ResilientClient(**kw))
    try:
        report=agent.run(bundle,output)
        for path,digest in frozen.items():
            if sha(path)!=digest:raise RuntimeError('Checkpoint changed during execution: '+path)
        insights=[i.insight for i in report.insights]
        predictions=read(output/'predictions.json')['insights']
        if isinstance(predictions,dict):predictions=predictions['conditional_compressed']
        if not insights or insights!=predictions:raise RuntimeError('Prediction/report mismatch')
        framework=read(output/'00_framework.json')
        framework.update(experiment_variant=args.variant,experiment_parameters=params,
                         experiment_protocol=plan['protocol'])
        write(output/'00_framework.json',framework)
        write(output/'result.json',dict(generator=args.model,variant=args.variant,parameters=params,
              source_run=str(output),first_round_source=str(source),insights=insights))
    finally:
        export_support.OpenAICompatibleClient=original
        if old_audit is None:os.environ.pop('TRAJSTORY_LLM_AUDIT_PATH',None)
        else:os.environ['TRAJSTORY_LLM_AUDIT_PATH']=old_audit
    print(json.dumps(dict(state='completed',output=str(output))))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
