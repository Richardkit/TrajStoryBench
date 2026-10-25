"""Prepare or run one fresh LLM-only case with the original deterministic profile."""
import argparse
import json
import os
from pathlib import Path
import time
from baseline_support.case_adapter import load_trajstory_case
from baseline_support.models import FrameworkConfig
from baseline_support.profiling import StructuredTrajectoryProfiler
from baseline_support.runtime import credentials,read,write,sha

ROOT=Path(__file__).resolve().parents[2]
MODELS={'gpt':('gpt-6-luna','https://api.openai.com/v1','OPENAI_API_KEY'),
        'qwen':('qwen3.8-flash','https://dashscope.aliyuncs.com/compatible-mode/v1','DASHSCOPE_API_KEY')}


class NoPreprocessingLLM:
    def complete(self,**kwargs):raise AssertionError('Preprocessing must not call an LLM')


def prepare_input(case,root):
    bundle=load_trajstory_case(case,root,input_csvs=True)
    config=FrameworkConfig()
    profiler=StructuredTrajectoryProfiler(NoPreprocessingLLM(),config)
    files=[]
    for data_file in bundle.files:
        if not data_file.path.name.startswith('input_'):raise ValueError('Projected inputs required')
        profile=profiler._profile_file(data_file);frame=profiler._frames[data_file.file_id]
        files.append(dict(file_id=data_file.file_id,file_name=data_file.path.name,role=data_file.role,
            row_count=profile.row_count,column_count=len(profile.columns),
            columns=[dict(name=c.name,dtype=c.raw_dtype,physical_type=c.physical_type,
                missing_count=int(frame[c.name].isna().sum()),unique_count=c.unique_count,statistics=c.statistics) for c in profile.columns],
            head_rows=profiler._rows(frame,3,random=False),random_rows_seed_0=profiler._rows(frame,3,random=True)))
    value=dict(goal=bundle.goal,dataset_description=bundle.context,analyst_role=bundle.role,data_profile=dict(files=files),
               profile_scope='Column statistics use all rows. Sample rows are head(3) and sample(3, random_state=0); the two samples can overlap.')
    if len(json.dumps(value,ensure_ascii=False))>config.max_profile_text_chars:raise ValueError('Profile exceeds original size limit')
    return value,bundle


def build_request(value,model):
    prompts=read(Path(__file__).with_name('prompts.json'))
    user=prompts['INPUT_POLICY']+'\n\nAnalyze the following model-facing inputs and return the required insights and summary JSON:\n'+json.dumps(value,ensure_ascii=False,allow_nan=False)
    request=dict(model=MODELS[model][0],messages=[dict(role='system',content=prompts['SYSTEM']),dict(role='user',content=user)],
                 temperature=0,reasoning_effort='none',max_completion_tokens=16000,n=1,response_format=dict(type='json_object'))
    if model=='qwen':request.update(seed=0,extra_body=dict(enable_thinking=False,top_k=1))
    return request


def validate_response(response):
    if len(response.get('choices',[]))!=1:raise ValueError('Expected one choice')
    choice=response['choices'][0];message=choice['message']
    if choice.get('finish_reason')!='stop' or message.get('reasoning_content') or message.get('tool_calls'):raise ValueError('Incomplete or unexpected response')
    if ((response.get('usage') or {}).get('completion_tokens_details') or {}).get('reasoning_tokens') not in (None,0):raise ValueError('Unexpected reasoning')
    data=json.loads(message.get('content') or '')
    if not isinstance(data,dict) or list(data)!=['insights','summary'] or not isinstance(data['summary'],str) or not isinstance(data['insights'],list):raise ValueError('Invalid output schema')
    texts=[]
    for item in data['insights']:
        if not isinstance(item,dict) or list(item)!=['insight'] or not isinstance(item['insight'],str) or not item['insight'].strip() or len(item['insight'].split())>180:raise ValueError('Invalid insight')
        texts.append(item['insight'])
    return texts,data['summary']


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--repo-root',type=Path,default=ROOT)
    parser.add_argument('--model',choices=MODELS,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--run',action='store_true',help='Send one semantic generation; default only prepares inputs')
    args=parser.parse_args(argv);out=args.output.resolve();case=args.case.resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):raise FileExistsError('Use a fresh output directory')
    value,bundle=prepare_input(case,args.repo_root.resolve());request=build_request(value,args.model)
    sources=[p for d in ('llm_only_baseline','baseline_support') for p in (ROOT/"baselines"/d).rglob('*') if p.is_file() and p.suffix in ('.py','.json') and '__pycache__' not in p.parts]
    out.mkdir(parents=True,exist_ok=True)
    with (out/'manifest.json').open('x',encoding='utf-8') as f:
        json.dump(dict(generator=args.model,case=str(case),input_sha256={str(p):sha(p) for p in [case,*(x.path for x in bundle.files)]},source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},semantic_attempts=1),f,indent=2)
    for p in sources:
        dst=out/'source_snapshot'/p.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes())
    write(out/'model_input.json',value);write(out/'request.json',request)
    if not args.run:print(json.dumps(dict(state='prepared',output=str(out))));return
    from openai import OpenAI,APIConnectionError,APITimeoutError,RateLimitError,InternalServerError
    model,url,key=MODELS[args.model];credentials([key]);client=OpenAI(api_key=os.environ[key],base_url=url,timeout=180,max_retries=0)
    for attempt in range(1,4):
        write(out/f'attempt_{attempt}.json',dict(state='started'))
        try:response=client.chat.completions.create(**request).model_dump();break
        except (APIConnectionError,APITimeoutError,RateLimitError,InternalServerError):
            if attempt==3:raise
            time.sleep(2*attempt)
    write(out/'response.json',response)
    insights,summary=validate_response(response)
    write(out/'result.json',dict(generator=args.model,case_json=str(case),source_run=str(out),insights=insights,summary=summary,status='valid' if insights else 'empty'))
    print(json.dumps(dict(state='completed',insight_count=len(insights))))


if __name__=='__main__':main()
