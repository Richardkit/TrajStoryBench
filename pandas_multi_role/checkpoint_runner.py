"""Goal-driven official Pandas Agent with a six-round checkpoints and round-37 summary."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.pandas_checkpoint_budget import MAX_ROUNDS, ROUND_POLICY, apply_round_budget
os.environ['LANGCHAIN_TRACING_V2']='false'
os.environ['LANGSMITH_TRACING']='false'
MODELS={'gpt':('gpt-6-luna','https://api.openai.com/v1','OPENAI_API_KEY'),
        'qwen':('qwen3.8-flash','https://dashscope.aliyuncs.com/compatible-mode/v1','DASHSCOPE_API_KEY')}


def json_safe_keys(value):
    if isinstance(value,dict):
        return {(k if isinstance(k,(str,int,float,bool,type(None))) else str(k)):json_safe_keys(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [json_safe_keys(v) for v in value]
    return value


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(json_safe_keys(value),ensure_ascii=False,indent=2,default=str),encoding='utf8')


def task_input(bundle,tables,policy="minimal"):
    prompt = (
        ROUND_POLICY + 'Analyze the following goal using all available dataframes. Independently choose and '
        'investigate the analytical questions needed to discover SIX distinct, useful factual insights. '
        'Choose your own sequence of analyses and tool actions. Use computed evidence for numerical '
        'claims and retain important conditions. Finally return one JSON object containing '
        '"insights": [six insight strings] and "summary": "a coherent summary".\n\n'
        +json.dumps({'goal':bundle.goal,'context':bundle.context,'role':bundle.role,
                    'tables':tables},ensure_ascii=False)
    )

    if policy not in ("minimal", "bounded"):
        raise ValueError("Unknown checkpoint policy")
    if policy == "bounded":
        prompt = prompt.replace('Choose your own sequence of analyses and tool actions. Use computed evidence for numerical ', 'Keep analysis iterations to the minimum needed for reliable findings. Inspect each table once, reuse existing results, and combine related computations in a single tool call when practical. Do not repeat successful queries, repeatedly inspect the same schema, or keep refining already-supported findings. Once SIX distinct evidence-backed insights are available, submit the final JSON immediately. Final summary is on round 37 after six-round insight checkpoints; this is a ceiling, not a target. Use computed evidence for numerical ', 1)
    return prompt


def main(policy="minimal"):
    import pandas as pd
    from langchain_openai import ChatOpenAI
    from langchain_experimental.agents import create_pandas_dataframe_agent
    from baseline_support.case_adapter import load_trajstory_case
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case-id',required=True);p.add_argument('--model',choices=MODELS,default='gpt')
    p.add_argument('--results-dir',type=Path,required=True);p.add_argument('--run',action='store_true')
    p.add_argument('--smoke-offline',action='store_true')
    args=p.parse_args();out=args.results_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    catalog=json.loads((ROOT/'trajstory_examples/insight/manifest.json').read_text(encoding='utf8'))['cases']
    case=next(c for c in catalog if c['case_id']==args.case_id)
    bundle=load_trajstory_case(ROOT/case['json_path'],ROOT)
    frames=[pd.read_csv(f.path) if f.path.suffix.lower()=='.csv' else pd.read_parquet(f.path) for f in bundle.files]
    tables=[dict(dataframe='df' if len(frames)==1 else f'df{i}',file_id=f.file_id,role=f.role)
            for i,f in enumerate(bundle.files,1)]
    max_execution_time = 1800 if policy == 'bounded' else None
    prompt=task_input(bundle,tables,policy);(out/'task.txt').write_text(prompt,encoding='utf8')
    cfg=dict(case_id=args.case_id,model=MODELS[args.model][0],
        factory='langchain_experimental.agents.create_pandas_dataframe_agent',agent_type='tool-calling',
        invocations_per_case=1,requested_insights=6,summary=True,max_iterations=MAX_ROUNDS,max_execution_time=max_execution_time,
        prefix='official default',parser='official default',python_tool='official PythonAstREPLTool',
        round_limit=MAX_ROUNDS,round_counter='Runtime counter includes regular decisions, checkpoint submissions and final summary',
        checkpoint_policy='Rounds 6/12/18/24/30/36 require 1/2/3/4/5/6 cumulative insights; missed quota returns partial insights+summary; final summary round 37',
        forced_tool_choice=False,checkpoint_forced_text_submission=True,output_recovery=False,question_source='Agent autonomously explores from goal; no GT questions/answers supplied',
        versions={n:importlib.metadata.version(n) for n in ('langchain','langchain-core','langchain-experimental','langchain-openai','openai')},
        input_sha256={str(f.path.relative_to(ROOT)):hashlib.sha256(f.path.read_bytes()).hexdigest() for f in bundle.files},
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        round_budget_sha256=hashlib.sha256((ROOT/'scripts/pandas_checkpoint_budget.py').read_bytes()).hexdigest(),
        validation_mode='offline_fake' if args.smoke_offline else 'benchmark')
    if policy == 'bounded':
        cfg['task_policy'] = 'Minimize redundant iterations; combine computations; finish once six supported findings are available'
    if (out/'manifest.json').exists():
        if json.loads((out/'manifest.json').read_text(encoding='utf8'))!=cfg:raise ValueError('Frozen configuration changed; choose a new directory')
    else:write(out/'manifest.json',cfg)
    if not args.run and not args.smoke_offline:
        print(json.dumps({'prepared':True,'case_id':args.case_id,'invocations':1,'max_iterations':MAX_ROUNDS}));return
    if args.smoke_offline:
        from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
        from langchain_core.messages import AIMessage
        class FakeToolsModel(FakeMessagesListChatModel):
            def bind_tools(self,tools,**kwargs):return self
        llm=FakeToolsModel(responses=[
            AIMessage(content='',tool_calls=[{'name':'python_repl_ast','args':{'query':'len('+tables[0]['dataframe']+')'},'id':'offline_call'}]),
            AIMessage(content=json.dumps({'insights':[f'OFFLINE TEST ONLY {i}' for i in range(1,7)],'summary':'Offline protocol check.'}))])
    else:
        import winreg
        model,url,key_name=MODELS[args.model];key=os.environ.get(key_name)
        if not key:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER,'Environment') as k:key=winreg.QueryValueEx(k,key_name)[0]
        if args.model=='gpt':
            llm=ChatOpenAI(model=model,base_url=url,api_key=key,temperature=0,
                model_kwargs={'reasoning_effort':'none','max_completion_tokens':16000})
        else:
            llm=ChatOpenAI(model=model,base_url=url,api_key=key,temperature=0,max_tokens=16000,
                extra_body={'enable_thinking':False})
    # Official prompt, tools and parser; add only round counting and the requested hard limit.
    agent=create_pandas_dataframe_agent(llm,frames if len(frames)>1 else frames[0],
        agent_type='tool-calling',allow_dangerous_code=True,return_intermediate_steps=True,
        max_iterations=MAX_ROUNDS,max_execution_time=max_execution_time)
    agent=apply_round_budget(agent,llm,progress_callback=lambda data:write(out/'checkpoint_progress.json',data))
    # Remove provider credentials from the computation process environment.
    for name in list(os.environ):
        if name.endswith(('_API_KEY','_TOKEN')) or name in ('API_KEY','HF_TOKEN'):os.environ.pop(name,None)
    started=time.monotonic();write(out/'status.json',{'state':'running'})
    try:
        result=agent.invoke({'input':prompt})
        write(out/'raw_result.json',{'output':result['output'],'intermediate_steps':[
            {'tool':a.tool,'input':a.tool_input,'log':a.log,'observation':b} for a,b in result['intermediate_steps'] if a.tool!='checkpoint_progress'],
            'elapsed_seconds':time.monotonic()-started,'rounds_used':agent.rounds_used,
            'round_limit':MAX_ROUNDS,'completed_normally':agent.completed_normally,
            'termination_reason':agent.termination_reason,'checkpoint_reports':agent.checkpoints})
        if not agent.completed_normally:
            error_type='RoundLimitExceeded' if agent.rounds_used>=MAX_ROUNDS else 'AgentStoppedEarly'
            write(out/'status.json',{'state':'failed','error_type':error_type,
                'error':'Agent did not submit a final answer within the allowed budget',
                'structure_matches_request':False,'rounds_used':agent.rounds_used,
                'round_limit':MAX_ROUNDS,'tool_calls':sum(a.tool!='checkpoint_progress' for a,b in result['intermediate_steps'])})
            print(json.dumps({'state':'failed','error_type':error_type,'rounds_used':agent.rounds_used}))
            return
        # Record structure only; do not rewrite or send an invalid answer back to the model.
        from baseline_support.json_utils import extract_json_object
        try:
            obj=extract_json_object(result['output'])
            valid=isinstance(obj.get('insights'),list) and len(obj['insights'])==6 and all(isinstance(x,str) and x.strip() for x in obj['insights']) and isinstance(obj.get('summary'),str) and bool(obj['summary'].strip())
            write(out/'parsed_output.json',{'structure_matches_request':valid,'answer':obj})
        except ValueError:valid=False
        write(out/'status.json',{'state':'returned','structure_matches_request':valid,'insights_count':len(agent.insights),
            'output_kind':'complete' if valid else 'partial','termination_reason':agent.termination_reason,'rounds_used':agent.rounds_used,'round_limit':MAX_ROUNDS,'tool_calls':sum(a.tool!='checkpoint_progress' for a,b in result['intermediate_steps'])})
        print(json.dumps({'returned':True,'structure_matches_request':valid,'tool_calls':sum(a.tool!='checkpoint_progress' for a,b in result['intermediate_steps'])}))
    except Exception as exc:
        write(out/'status.json',{'state':'failed','error_type':type(exc).__name__,'error':str(exc),'rounds_used':agent.rounds_used,'round_limit':MAX_ROUNDS});raise


if __name__=='__main__':main()
