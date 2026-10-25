"""Pinned upstream single-agent DSAEval, with minimal TrajStory I/O and transport adaptation."""
from __future__ import annotations

import sys as _baseline_sys
from pathlib import Path as _BaselinePath
_baseline_repo = next(p for p in _BaselinePath(__file__).resolve().parents if (p / "baselines" / "__init__.py").is_file())
_baseline_sys.path.insert(0, str(_baseline_repo))
import baselines  # Register the shared baseline module location.

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import ssl
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from baseline_support.case_adapter import load_trajstory_case

UPSTREAM=Path(__file__).with_name('origin_upstream')
RESULTS_ROOT=ROOT/'results/dsaeval_origin'
CATALOG=ROOT/'trajstory_examples/insight/manifest.json'
PROTOCOL='dsaeval_origin_single_persistent_agent_v1'
MODELS={'gpt':('gpt-6-luna','https://api.openai.com/v1','OPENAI_API_KEY'),
        'qwen':('qwen3.8-flash','https://dashscope.aliyuncs.com/compatible-mode/v1','DASHSCOPE_API_KEY')}

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf8');tmp.replace(p)

def verify_upstream():
    provenance=read(UPSTREAM/'provenance.json')
    for name,h in provenance['sha256'].items():
        if sha(UPSTREAM/name)!=h:raise RuntimeError('Upstream snapshot changed: '+name)
    return provenance

def load_upstream():
    """Load pristine sources without replacing modules used by the existing adapter."""
    verify_upstream();names=('prompts','tools','utils')
    saved={name:sys.modules.get(name) for name in names};oldpath=list(sys.path)
    try:
        sys.path.insert(0,str(UPSTREAM))
        for name in names:sys.modules.pop(name,None)
        def module(name,file):
            spec=importlib.util.spec_from_file_location(name,UPSTREAM/file)
            value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
        agent=module('_dsaeval_origin_agent','data_agent.py')
        sandbox=module('_dsaeval_origin_sandbox','jupyter_sandbox.py')
        return agent.DataScienceAgent,sandbox.JupyterSandbox,sys.modules['prompts']
    finally:
        sys.path[:]=oldpath
        for name,value in saved.items():
            if value is None:sys.modules.pop(name,None)
            else:sys.modules[name]=value

def load_cases(manifest,ids):
    catalog={c['case_id']:c for c in read(manifest)['cases']};cases=[]
    for cid in ids:
        if cid not in catalog:raise ValueError('Unknown case: '+cid)
        path=Path(catalog[cid]['json_path']);path=path if path.is_absolute() else ROOT/path
        bundle=load_trajstory_case(path,ROOT,input_csvs=True)
        sources=[]
        for f in bundle.files:
            if f.path.suffix.lower()!='.csv' or not f.path.name.startswith('input_'):
                raise ValueError('Projected input CSV required: '+str(f.path))
            sources.append(dict(source=str(f.path),role=f.role))
        cases.append(dict(case_id=cid,goal=bundle.goal,sources=sources,json_path=str(path)))
    return cases

def system_prompt(template,files,artifacts,budget,timeout):
    lines=[]
    for line in template.splitlines():
        if line.startswith('- There are 4 * NVIDIA'):
            lines.append('- This Windows environment provides CPU-based scientific computing; no GPU is allocated.')
        elif line.startswith('Under no circumstances') or line.startswith('and you MUST not terminate'):continue
        else:lines.append(line)
    result='\n'.join(lines).replace('20 steps/iteration opportunities (from 0-19)',f'{budget} steps/iteration opportunities (from 0-{budget-1})')
    result=result.replace('1 hour',f'{timeout} seconds')
    return result.format(data_path='\n'+'\n'.join(f"- {f['path']} ({f['role']})" for f in files),working_path=str(artifacts))

def task_prompt(goal):
    return goal+'\n\nAim to report six factual insights supported by your executed analysis and a coherent summary. '+\
        'In section 4 of the original Markdown report, include one fenced JSON object with '+\
        '"insights" (a list of insight strings) and "summary" (a string). '+\
        'Related, coherent insights are allowed. Fewer than six insights or a missing summary is acceptable; do not invent findings to fill the count.'

def parse_report(report):
    candidates=[]
    for block in re.findall(r'```json\s*(.*?)```',report,flags=re.S|re.I):
        try:obj=json.loads(block)
        except ValueError:continue
        if isinstance(obj,dict) and 'insights' in obj:candidates.append(obj)
    if len(candidates)!=1:raise ValueError('Expected one result JSON object; no model-based extraction or recovery')
    obj=candidates[0];insights=obj.get('insights');summary=obj.get('summary') or ''
    if not isinstance(insights,list) or not all(isinstance(x,str) and x.strip() for x in insights) or not isinstance(summary,str):
        raise ValueError('Invalid insight/summary types')
    return insights,summary

class NetworkInterrupted(RuntimeError):pass

def network_error(exc):
    return isinstance(exc,(ConnectionError,TimeoutError,socket.gaierror,ssl.SSLError)) or any(
        c.__name__ in {'APIConnectionError','APITimeoutError','TransportError'} for c in type(exc).__mro__)

class Transport:
    """Request options, auditing and identical-request retries outside the unchanged agent loop."""
    def __init__(self,client,folder,budget,options):
        self.client=client;self.folder=Path(folder);self.budget=budget;self.options=options
        self.charged=0;self.records=[];self.streak=0;self.interrupted=False
        self.chat=SimpleNamespace(completions=SimpleNamespace(create=self.create))
    def usage(self):
        return dict(llm_calls=self.charged,model_calls_started=len(self.records),
            network_failures=sum(r.get('network_failure',False) for r in self.records),
            consecutive_network_failures=self.streak,
            **{k:sum((r.get('usage') or {}).get(k) or 0 for r in self.records) for k in ['prompt_tokens','completion_tokens','total_tokens']})
    def save(self):write(self.folder/'model_usage.json',dict(**self.usage(),calls=self.records))
    def create(self,**request):
        if self.interrupted:raise NetworkInterrupted('Five consecutive network failures')
        if self.charged>=self.budget:raise RuntimeError('WholeCaseModelBudgetExceeded')
        payload=dict(request,**self.options)
        while True:
            attempt=len(self.records)+1;dest=self.folder/'requests'/f'attempt_{attempt:03d}'
            record=dict(attempt=attempt,state='started',counts_toward_budget=False)
            self.records.append(record);write(dest/'request.json',payload);self.save()
            try:response=self.client.chat.completions.create(**payload)
            except Exception as exc:
                record.update(state='error',error_type=type(exc).__name__,status_code=getattr(exc,'status_code',None),error_code=getattr(exc,'code',None))
                if not network_error(exc):
                    self.charged+=1;self.streak=0;record['counts_toward_budget']=True;self.save();raise
                self.streak+=1;record['network_failure']=True
                self.interrupted=self.streak>=5;self.save()
                if self.interrupted:raise NetworkInterrupted('Five consecutive network failures; stop run') from exc
                time.sleep(min(2**(self.streak-1),8));continue
            self.charged+=1;self.streak=0
            usage=getattr(response,'usage',None)
            record.update(state='returned',counts_toward_budget=True,usage={k:getattr(usage,k,None) for k in ['prompt_tokens','completion_tokens','total_tokens']} if usage else None)
            self.save()
            if hasattr(response,'model_dump'):write(dest/'response.json',response.model_dump())
            return response

def run_case(case,folder,args,client,options,components=None):
    import shutil
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);artifacts=folder/'artifacts';artifacts.mkdir()
    Agent,Sandbox,prompts=components or load_upstream()
    staged=[]
    for i,item in enumerate(case['sources']):
        source=Path(item['source']);dest=folder/'input'/f'{i:02d}_{source.name}';dest.parent.mkdir(exist_ok=True)
        shutil.copy2(source,dest);staged.append(dict(path=str(dest),role=item['role']))
    template=prompts.DSAgent_PROMPT_VLM if args.is_multimodal else prompts.DSAgent_PROMPT_LLM
    system=system_prompt(template,staged,artifacts,args.max_model_calls,args.code_timeout);task=task_prompt(case['goal'])
    (folder/'system_prompt.txt').write_text(system,encoding='utf8');(folder/'task.txt').write_text(task,encoding='utf8')
    transport=Transport(client,folder,args.max_model_calls,options)
    agent=Agent(client=transport,model=args.model,system_prompt=system,max_iterations=args.max_model_calls,is_multimodal=args.is_multimodal)
    log={};report='';insights=[];summary='';error=None;started=time.monotonic()
    try:
        with Sandbox(timeout=args.code_timeout) as kernel:
            log,report=agent.run(task,str(artifacts),kernel,verbose=args.verbose)
        if report:insights,summary=parse_report(report)
    except Exception as exc:error=dict(type=type(exc).__name__,message=str(exc))
    finally:
        agent.save_notebook(str(folder/'analysis.ipynb'))
        write(folder/'agent_log.json',log);write(folder/'conversation.json',agent.messages)
        (folder/'final_report.md').write_text(report,encoding='utf8');transport.save()
    status='interrupted' if transport.interrupted else 'completed' if len(insights)==6 and summary and log.get('finished') else 'partial' if insights else 'failed'
    if not insights and error is None:error=dict(type='NoParsedInsight',native_finished=bool(log.get('finished')))
    row=dict(case_id=case['case_id'],model=args.model,status=status,pred_insights=insights,pred_summary=summary,
        eligible_for_scoring=bool(insights),exploration_protocol=PROTOCOL,agent_invocations=1,
        execution_mode='single_agent_single_conversation_single_persistent_kernel',
        native_finished=bool(log.get('finished')),max_model_calls_per_case=args.max_model_calls,
        token_usage=transport.usage(),error=error,elapsed_seconds=round(time.monotonic()-started,3))
    write(folder/'parsed_output.json',dict(answer=dict(insights=insights,summary=summary)))
    write(folder/'result.json',row);return row

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--generator',choices=MODELS,default='gpt');p.add_argument('--model');p.add_argument('--base-url');p.add_argument('--api-key-env')
    p.add_argument('--manifest',type=Path,default=CATALOG);p.add_argument('--case-ids',nargs='+');p.add_argument('--case-count',type=int,default=10)
    p.add_argument('--results-dir',type=Path);p.add_argument('--max-model-calls',type=int,default=83);p.add_argument('--code-timeout',type=int,default=180)
    p.add_argument('--is-multimodal',action='store_true');p.add_argument('--verbose',action='store_true');p.add_argument('--dry-run',action='store_true');p.add_argument('--run',action='store_true')
    args=p.parse_args();model,url,keyenv=MODELS[args.generator]
    args.model=args.model or model;url=args.base_url or url;keyenv=args.api_key_env or keyenv
    if args.max_model_calls<1 or args.code_timeout<1 or args.case_count<1:p.error('Positive budgets and case count required')
    ids=args.case_ids or [c['case_id'] for c in read(args.manifest)['cases'][:args.case_count]]
    if len(ids)!=len(set(ids)):p.error('Case IDs must be unique')
    cases=load_cases(args.manifest,ids);out=(args.results_dir or RESULTS_ROOT/args.generator).resolve()
    if not out.is_relative_to(RESULTS_ROOT.resolve()):p.error('Results must remain under results/dsaeval_origin')
    upstream=verify_upstream();options=dict(temperature=0,max_completion_tokens=16000,reasoning_effort='none')
    if args.generator=='qwen':options.update(extra_body=dict(enable_thinking=False,top_k=1),seed=0)
    sources=[Path(__file__),*list(UPSTREAM.glob('*.py')),UPSTREAM/'provenance.json',ROOT/'baselines/baseline_support/case_adapter.py',ROOT/'baselines/baseline_support/input_paths.py']
    config=dict(protocol=PROTOCOL,generator=args.generator,model=args.model,base_url=url,case_ids=ids,results_dir=str(out),
        max_model_calls=args.max_model_calls,upstream_default_iterations=20,code_timeout=args.code_timeout,
        upstream=upstream,agent_invocations_per_case=1,shared_context_within_case=True,private_persistent_kernel_per_case=True,
        fixed_branches=False,external_verifier=False,exact_citation_filter=False,forced_final_call=False,
        generation_options=options,network_failures_count_toward_budget=False,consecutive_network_failure_limit=5,sdk_retries=0,
        source_sha256={str(s):sha(s) for s in sources},input_sha256={c['case_id']:{s['source']:sha(s['source']) for s in c['sources']} for c in cases},
        primary_metrics=['g_eval','coverage_at_0_6'],judge='gpt-6-luna',judge_rounds=2)
    if args.dry_run or not args.run:
        print(json.dumps(dict(config=config,cases=cases),indent=2,ensure_ascii=False));return
    if out.exists() and any(out.iterdir()):p.error('Fresh model results directory required; do not overwrite old runs')
    components=load_upstream()
    secret=os.environ.get(keyenv)
    if not secret:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,'Environment') as reg:secret=winreg.QueryValueEx(reg,keyenv)[0]
    from openai import OpenAI
    client=OpenAI(api_key=secret,base_url=url,timeout=180,max_retries=0)
    os.environ['JUPYTER_PATH']=str(ROOT/'.runtime/dsaeval/share/jupyter')
    for name in list(os.environ):
        if name.endswith(('_API_KEY','_TOKEN')) or name in ('API_KEY','HF_TOKEN'):os.environ.pop(name,None)
    out.mkdir(parents=True,exist_ok=True);write(out/'manifest.json',config)
    for source in sources:
        dest=out/'source_snapshot'/source.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(source.read_bytes())
    rows=[];generation=[]
    write(out/'status.json',dict(state='generating',total=len(cases),finished=0,with_insights=0,max_model_calls=83 if args.max_model_calls==83 else args.max_model_calls))
    for case in cases:
        row=run_case(case,out/'generation'/case['case_id'],args,client,options,components);rows.append(row)
        write(out/'trajstory_case_results.json',rows)
        generation.append(dict(case_id=case['case_id'],json_path=case['json_path'],directory=str(out/'generation'/case['case_id']),eligible_for_scoring=bool(row['pred_insights'])))
        write(out/'evaluation/generation.json',generation)
        state='interrupted' if row['status']=='interrupted' else 'generated_pending_gpt_evaluation' if len(rows)==len(cases) else 'generating'
        write(out/'status.json',dict(state=state,total=len(cases),finished=len(rows),with_insights=sum(bool(r['pred_insights']) for r in rows)))
        print(json.dumps(dict(case_id=case['case_id'],status=row['status'],insights=len(row['pred_insights']),usage=row['token_usage'])),flush=True)
        if row['status']=='interrupted':raise SystemExit('Five consecutive network failures; run stopped; partial outputs saved')

if __name__=='__main__':main()
