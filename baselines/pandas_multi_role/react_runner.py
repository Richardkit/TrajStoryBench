"""Official LangChain Pandas ReAct baseline with TrajStory input/result adapters."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import contextlib
import importlib.metadata
import io
import http.client
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time
import traceback

os.environ['LANGCHAIN_TRACING_V2']='false'
os.environ['LANGSMITH_TRACING']='false'

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'results/pandas/mimo15_20261012'
SOURCE=ROOT/'results/v3_result/mimo26_matched15_gpt_judge_20261012'
PYTHON=ROOT/'.runtime/pandas_agent/Scripts/python.exe'
REACT_SYSTEM = (
    'You are operating a live Pandas data analysis agent. The python_repl_ast tool is available. '
    'For computation output Thought:, Action: python_repl_ast, and Action Input: followed by actual nonempty Python code. '
    'Then stop and wait for the real tool observation; never invent an Observation. '
    'To finish output Final Answer: followed by the requested insights JSON object. '
    'Do not claim tools are unavailable without an actual tool error. An empty code block is not an action.'
)
TOOL_SYSTEM = (
    'You are operating a live Pandas data analysis agent with an available Python tool. '
    'Use native tool calls to compute evidence from the supplied dataframes and wait for the actual results. '
    'Every analysis round must contain a successful Python computation before its final answer. '
    'Return the requested insights JSON object as your final message. '
    'Do not simulate Action/Observation text, invent tool results, or say that a provided tool is unavailable.'
)


def resolved_agent_type(args):
    if args.agent_type!='auto':return args.agent_type
    return 'tool-calling' if args.model.startswith('gpt-') else 'zero-shot-react-description'


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str),encoding='utf8')
def sha(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def compatible_output_parser():
    from langchain.agents.output_parsers import ReActSingleInputOutputParser
    from langchain_core.agents import AgentFinish
    from langchain_core.exceptions import OutputParserException
    from langchain_experimental.tools.python.tool import sanitize_input
    class JSONOrReActParser(ReActSingleInputOutputParser):
        def parse(self,text):
            stripped=text.strip()
            # Do not interpret JSON embedded in a Python action as a final answer.
            action_present=re.search(r'(?m)^\s*Action\s*\d*\s*:',stripped)
            candidate=stripped
            if not action_present:
                final=re.search(r'(?im)^\s*Final\s+Answer\s*:\s*',stripped)
                if final:candidate=stripped[final.end():].strip()
                fence=re.fullmatch(r'```(?:json)?\s*\n?([\s\S]*?)\n?```',candidate,re.IGNORECASE)
                if fence:candidate=fence.group(1).strip()
            if not action_present and candidate.startswith('{'):
                try:
                    obj=json.loads(candidate)
                    if isinstance(obj,dict) and isinstance(obj.get('insights'),list):
                        return AgentFinish({'output':candidate},text)
                except ValueError:pass
            parsed=super().parse(text)
            if hasattr(parsed,'tool_input') and parsed.tool=='python_repl_ast' and not sanitize_input(parsed.tool_input).strip():
                raise OutputParserException('Empty Python action input',observation='Action Input must contain executable Python code, not an empty quoted or fenced block. Write the computation and wait for the actual tool result.',llm_output=text,send_to_llm=True)
            return parsed
    return JSONOrReActParser()


def validate_insights(payload):
    values=payload.get('insights')
    if not isinstance(values,list) or not 0<len(values)<=3 or any(not isinstance(x,str) or not x.strip() for x in values):
        raise ValueError('Expected 1-3 nonempty factual insight strings; perform goal-relevant computations before finalizing')
    failure=re.compile(r"^\s*I\s+(?:cannot|can['’]t|am unable to|could not|was unable to)\s+(?:provide|produce|generate|perform|compute)\b",re.IGNORECASE)
    if any(failure.search(x) for x in values):
        raise ValueError('A failure declaration is not an insight; report an analysis failure rather than submitting it as a finding')
    return values


def validate_code(code):
    allowed={'pandas','numpy','math','scipy','sklearn','datetime','collections','itertools','statistics','json','baseline_support'}
    forbidden={'open','eval','exec','compile','__import__','getattr','setattr','globals','locals','vars','input','breakpoint'}
    forbidden_attributes={'input_files','output_dir','read_csv','read_parquet','read_json','read_excel',
        'read_pickle','load','loads','save','savefig','to_csv','to_json','to_pickle','to_excel',
        'write_text','write_bytes','unlink','remove','rmdir','system','popen','environ'}
    for node in ast.walk(ast.parse(code)):
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            names=[x.name for x in node.names] if isinstance(node,ast.Import) else [node.module or '']
            if any(x.split('.')[0] not in allowed for x in names):raise ValueError('Import outside computation allowlist')
            if isinstance(node,ast.ImportFrom) and node.module and node.module.startswith('baseline_support') and node.module!='baseline_support.analysis_tools':raise ValueError('Only shared analysis tools may be imported')
            if isinstance(node,ast.Import) and any(x.name.startswith('baseline_support') and x.name!='baseline_support.analysis_tools' for x in node.names):raise ValueError('Only shared analysis tools may be imported')
        if isinstance(node,ast.Name) and node.id in forbidden:raise ValueError('Forbidden builtin: '+node.id)
        if isinstance(node,ast.Attribute) and (node.attr.startswith('_') or node.attr in forbidden_attributes):raise ValueError('Forbidden attribute: '+node.attr)


def worker(catalog,directory):
    # Computation process receives no provider credentials. This guard is not an OS sandbox.
    from langchain_experimental.tools.python.tool import PythonAstREPLTool,sanitize_input
    from baseline_support.analysis_tools import AnalysisSession
    import pandas as pd
    import numpy as np
    tables=read(catalog)
    session=AnalysisSession({x['file_id']:x['path'] for x in tables},directory)
    frames={x['file_id']:session.load_table(x['file_id']) for x in tables}
    local=dict(frames,session=session,pd=pd,np=np)
    if len(tables)==1:local['df']=next(iter(frames.values()))
    else:local.update({f'df{i}':frames[x['file_id']] for i,x in enumerate(tables,1)})
    repl=PythonAstREPLTool(locals=local,globals={})
    print(json.dumps(dict(ready=True)),flush=True)
    for line in sys.stdin:
        try:
            payload=json.loads(line);code=sanitize_input(payload['code'])
            if not code.strip():raise ValueError('Empty Python code; provide an actual computation')
            validate_code(code)
            capture=io.StringIO()
            with contextlib.redirect_stdout(capture),contextlib.redirect_stderr(capture):answer=repl.run(code)
            output=(capture.getvalue()+str(answer))[:16000]
            success=not re.match(r'^(?:[A-Za-z_]+Error|Exception):',str(answer))
            print(json.dumps(dict(output=output,success=success)),flush=True)
        except Exception as exc:print(json.dumps(dict(output=type(exc).__name__+': '+str(exc),success=False)),flush=True)


class Worker:
    def __init__(self,catalog,directory):
        self.directory=Path(directory);self.calls=0;self.successful_calls=0;self.queue=queue.Queue()
        env={k:v for k,v in os.environ.items() if k.upper() in {'SYSTEMROOT','WINDIR','PATH','TEMP','TMP','COMSPEC','PATHEXT'}}
        env.update(PYTHONUTF8='1',MPLCONFIGDIR=str(self.directory/'mpl'))
        self.errors=(self.directory/'worker.stderr.log').open('a',encoding='utf8')
        self.proc=subprocess.Popen([str(PYTHON),'-X','utf8','-u',str(Path(__file__).resolve()),'--worker',str(catalog),'--worker-dir',str(directory)],
            cwd=directory,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.errors,text=True,encoding='utf8',creationflags=subprocess.CREATE_NO_WINDOW)
        threading.Thread(target=self.consume,daemon=True).start()
        if self.queue.get(timeout=120).get('ready') is not True:raise RuntimeError('Worker initialization failed')
    def consume(self):
        for line in self.proc.stdout:
            try:self.queue.put(json.loads(line))
            except ValueError:self.queue.put({'output':'Worker protocol error'})
        self.queue.put({'output':'Worker exited'})
    def call(self,code):
        self.calls+=1;folder=self.directory/'calls'/f'{self.calls:03d}';folder.mkdir(parents=True,exist_ok=True)
        (folder/'code.py').write_text(code,encoding='utf8')
        if self.proc.poll() is not None:return 'Execution process is unavailable; stop and report failure.'
        self.proc.stdin.write(json.dumps({'code':code})+'\n');self.proc.stdin.flush()
        try:
            response=self.queue.get(timeout=180);result=response['output']
            if response.get('success'):self.successful_calls+=1
        except queue.Empty:
            self.proc.kill();result='Execution timed out after 180 seconds; stop and report failure.'
        write(folder/'observation.json',{'output':result,'success':response.get('success',False) if 'response' in locals() else False});return result
    def close(self):
        if self.proc.poll() is None:self.proc.kill()
        self.proc.wait(timeout=10);self.errors.close()


def prepare(args):
    from baseline_support.case_adapter import load_trajstory_case
    source=read(args.manifest);cases=source['cases']
    if args.case_ids:
        wanted=set(args.case_ids);cases=[c for c in cases if c['case_id'] in wanted]
        if wanted!={c['case_id'] for c in cases}:raise ValueError('Unknown case ID')
    versions={n:importlib.metadata.version(n) for n in ['langchain','langchain-core','langchain-experimental','pandas','numpy']}
    mode=resolved_agent_type(args)
    if mode=='tool-calling':versions.update({n:importlib.metadata.version(n) for n in ['langchain-openai','openai']})
    assert versions['langchain']=='0.2.17' and versions['langchain-experimental']=='0.0.65'
    inputs={};catalogs={}
    for i,c in enumerate(cases,1):
        b=load_trajstory_case(ROOT/c['json_path'],ROOT)
        catalogs[c['case_id']]=[{'file_id':f.file_id,'path':str(f.path),'role':f.role} for f in b.files]
        for p in [ROOT/c['json_path'],*[f.path for f in b.files]]:inputs[str(p.relative_to(ROOT))]=sha(p)
    import inspect
    from langchain_experimental.agents import create_pandas_dataframe_agent
    from langchain_experimental.tools.python.tool import PythonAstREPLTool
    official_source=Path(inspect.getfile(create_pandas_dataframe_agent))
    repl_source=Path(inspect.getfile(PythonAstREPLTool))
    cfg=dict(cases=cases,model=args.model,base_url=args.base_url,api_key_env=args.api_key_env,agent_type=mode,
        source_manifest=str(args.manifest.resolve()),source_manifest_sha256=sha(args.manifest),
        validation_mode='offline_fake' if args.smoke_offline else 'benchmark',official_agent_sha256=sha(official_source),official_repl_sha256=sha(repl_source),
        rounds=2,insights_per_round=3,max_iterations_per_round=args.max_iterations,max_execution_seconds_per_round=1800,temperature=0,reasoning_effort='none',
        workers=3,toolkit='baseline_support.analysis_tools; identical helper implementations',
        output_recovery='One continuation for invalid/empty final JSON per round; remaining iteration budget only, same worker and prior observations; no altered result or invented claims.',
        output_parser='Official ReAct parser plus standalone/fenced insights JSON and case-insensitive Final Answer prefix; rejects empty Python actions; preserves raw text.',
        output_admission='Reject empty outputs and explicit first-person failure declarations as insights; no GT-based filtering.',
        prompt_prefix='Official Pandas dataframe prefix retained, then schema/shared-tool documentation and the live agent contract.',
        system_prompt_sha256=hashlib.sha256((TOOL_SYSTEM if mode=='tool-calling' else REACT_SYSTEM).encode()).hexdigest(),
        execution_gate='Successful computation required in each round; native mode uses tool_choice=required until the first success, then auto.',
        transport_retries='Up to three attempts for a disconnected HTTP connection, retaining diagnostics; model parameters unchanged.',
        execution='Official PythonAstREPLTool in a credential-free persistent subprocess with computation AST guard and 180-second per-call timeout; not an OS sandbox.',
        output_limit=6,versions=versions,catalogs=catalogs,input_sha256=inputs,
        budget_note='Each round is one autonomous goal-driven ReAct run. Iterations are execution steps, not analytical question counts. No independent question planner or DPP.',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'baselines/scripts/evaluate_pandas_baseline.py',ROOT/'baselines/baseline_support/analysis_tools.py',ROOT/'baselines/baseline_support/case_adapter.py',ROOT/'baselines/baseline_support/llm.py',ROOT/'baselines/baseline_support/json_utils.py',ROOT/'baselines/baseline_support/llm_audit.py']})
    target=Path(args.results_dir).resolve();path=target/'manifest.json'
    if len(str(target/'c01/calls/999/observation.json'))>=240:raise ValueError('Use a shorter results directory')
    if path.exists():
        if read(path)!=cfg:raise RuntimeError('Frozen configuration changed; use a new results directory')
    else:
        write(path,cfg)
        snapshot=target/'source_snapshot';snapshot.mkdir(exist_ok=True)
        (snapshot/'pandas_agent_base.py').write_bytes(official_source.read_bytes())
        (snapshot/'python_repl_tool.py').write_bytes(repl_source.read_bytes())
        for rel in cfg['source_sha256']:
            dest=snapshot/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/rel).read_bytes())
    return cfg


def run_case(args,cfg,c,index,fake=False):
    from langchain_experimental.agents import create_pandas_dataframe_agent
    from langchain_experimental.agents.agent_toolkits.pandas.prompt import PREFIX,MULTI_DF_PREFIX,PREFIX_FUNCTIONS,MULTI_DF_PREFIX_FUNCTIONS
    from langchain_experimental.tools.python.tool import PythonAstREPLTool
    from langchain_core.language_models.llms import LLM
    from typing import Any
    from baseline_support.case_adapter import load_trajstory_case
    from baseline_support.analysis_tools import SHARED_TOOL_REFERENCE
    from baseline_support.json_utils import extract_json_object
    from baseline_support.llm import OpenAICompatibleClient
    import pandas as pd
    dest=Path(args.results_dir)/f'c{index:02d}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'predictions.json').exists():return read(dest/'predictions.json')
    bundle=load_trajstory_case(ROOT/c['json_path'],ROOT);tables=cfg['catalogs'][c['case_id']]
    frames=[pd.read_csv(f['path']) if Path(f['path']).suffix.lower()=='.csv' else pd.read_parquet(f['path']) for f in tables]
    schema=[dict(file_id=t['file_id'],repl_name='df' if len(tables)==1 else f'df{i}',role=t['role'],rows=len(df),columns={str(k):str(v) for k,v in df.dtypes.items()}) for i,(t,df) in enumerate(zip(tables,frames),1)]
    write(dest/'model_inputs.json',dict(goal=bundle.goal,context=bundle.context,role=bundle.role,tables=schema))
    write(dest/'catalog.json',tables);worker_process=Worker(dest/'catalog.json',dest)
    class ExternalREPL(PythonAstREPLTool):
        worker_handle: Any
        def _run(self,query,run_manager=None):return self.worker_handle.call(query)
    class Backbone(LLM):
        native: Any
        logs: Any
        calls: int=0
        @property
        def _llm_type(self):return 'trajstory_openai_compatible'
        def _call(self,prompt,stop=None,run_manager=None,**kwargs):
            self.calls+=1;write(self.logs/f'call_{self.calls:03d}_prompt.json',dict(prompt=prompt,stop=stop))
            for transport_attempt in range(3):
                try:
                    answer=self.native.complete(system=REACT_SYSTEM,user=prompt,json_mode=False)
                    break
                except http.client.RemoteDisconnected as exc:
                    write(self.logs/f'call_{self.calls:03d}_transport_{transport_attempt+1}.json',dict(error=type(exc).__name__,message=str(exc)))
                    if transport_attempt==2:raise
                    time.sleep(2*(transport_attempt+1))
            meta=self.native.last_response_metadata or {}
            if meta.get('reasoning_present') or (meta.get('usage',{}).get('completion_tokens_details') or {}).get('reasoning_tokens',0):raise RuntimeError('Unexpected reasoning in non-thinking benchmark')
            (self.logs/f'call_{self.calls:03d}_response.txt').write_text(answer,encoding='utf8')
            write(self.logs/f'call_{self.calls:03d}_metadata.json',self.native.last_response_metadata or {})
            if stop:
                for s in stop:
                    if s in answer:answer=answer.split(s,1)[0]
            return answer
    if cfg['agent_type']=='tool-calling' and not fake:
        from langchain_openai import ChatOpenAI
        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.outputs import ChatResult,ChatGeneration
        from langchain_core.utils.function_calling import convert_to_openai_tool
        class ToolBackbone(BaseChatModel):
            chat:Any
            logs:Any
            worker_handle:Any
            round_success_base:int=0
            calls:int=0
            @property
            def _llm_type(self):return 'native_openai_tool_calling'
            def bind_tools(self,tools,**kwargs):
                return self.bind(tools=[convert_to_openai_tool(t) for t in tools],**kwargs)
            def _generate(self,messages,stop=None,run_manager=None,**kwargs):
                self.calls+=1
                choice='required' if self.worker_handle.successful_calls<=self.round_success_base else 'auto'
                write(self.logs/f'call_{self.calls:03d}_prompt.json',dict(messages=[m.dict() for m in messages],tool_choice=choice))
                response=self.chat.invoke(messages,stop=stop,tool_choice=choice,**kwargs)
                write(self.logs/f'call_{self.calls:03d}_message.json',response.dict())
                (self.logs/f'call_{self.calls:03d}_response.txt').write_text(str(response.content),encoding='utf8')
                usage=response.response_metadata.get('token_usage') or {}
                if response.additional_kwargs.get('reasoning_content') or (usage.get('completion_tokens_details') or {}).get('reasoning_tokens',0):raise RuntimeError('Unexpected reasoning')
                write(self.logs/f'call_{self.calls:03d}_metadata.json',dict(usage=usage,response_metadata=response.response_metadata,tool_calls=response.tool_calls,tool_choice=choice))
                return ChatResult(generations=[ChatGeneration(message=response)])
        llm=ToolBackbone(chat=ChatOpenAI(model=args.model,base_url=args.base_url,api_key=os.environ[args.api_key_env],
            temperature=0,timeout=180,max_retries=3,model_kwargs={'reasoning_effort':'none','max_completion_tokens':16000}),
            logs=dest/'llm',worker_handle=worker_process)
    elif fake:
        from langchain_core.language_models.fake import FakeListLLM
        llm=FakeListLLM(responses=['Thought: I should compute the row count.\nAction: python_repl_ast\nAction Input: len(df1)',
            'Thought: I have computed evidence.\nFinal Answer: {"insights":["OFFLINE TEST ONLY: table contains rows."],"summary":"Offline smoke."}',
            'Thought: I should examine another aspect.\nAction: python_repl_ast\nAction Input: len(df1.columns)',
            'Thought: I have new computed evidence.\nFinal Answer: {"insights":["OFFLINE TEST ONLY: table contains columns."],"summary":"Offline second round."}'])
    else:llm=Backbone(native=OpenAICompatibleClient(model=args.model,base_url=args.base_url,api_key=os.environ[args.api_key_env],temperature=0,reasoning_effort='none',max_tokens=16000),logs=dest/'llm')
    extra='\nApproved tables and aliases: '+json.dumps(schema,ensure_ascii=False)+'\n'+SHARED_TOOL_REFERENCE
    extra+='\nUse existing dataframes or session.load_table(file_id); do not read paths, import OS/network modules, or write files directly. Use session.save_figure for any plots. Plain Python, pandas, numpy and shared analysis helpers are available. Each Python tool call preserves variables. Do not access hidden references or expected answers.\n'
    # LangChain parses the prefix as an f-string template; schemas/API examples are literals.
    if cfg['agent_type']=='tool-calling':
        official_prefix=MULTI_DF_PREFIX_FUNCTIONS.format(num_dfs=len(frames)) if len(frames)>1 else PREFIX_FUNCTIONS
        extra=official_prefix+extra+'\n'+TOOL_SYSTEM
        if len(frames)>1:extra=extra.replace('{','{{').replace('}','}}')
    else:
        official_prefix=MULTI_DF_PREFIX.format(num_dfs=len(frames)) if len(frames)>1 else PREFIX
        extra=official_prefix+extra.replace('{','{{').replace('}','}}')+'\n'+REACT_SYSTEM
    agent=create_pandas_dataframe_agent(llm,frames if len(frames)>1 else frames[0],prefix=extra,
        agent_type=cfg['agent_type'],allow_dangerous_code=True,include_df_in_prompt=True,
        number_of_head_rows=5,max_iterations=args.max_iterations,max_execution_time=1800,
        return_intermediate_steps=True,agent_executor_kwargs={'handle_parsing_errors':True})
    agent.tools=[ExternalREPL(worker_handle=worker_process)]
    from langchain_core.runnables import RunnableSequence
    if cfg['agent_type']=='zero-shot-react-description':agent.agent.runnable=RunnableSequence(*agent.agent.runnable.steps[:-1],compatible_output_parser())
    predictions=[]
    try:
        for rnd in range(1,3):
            round_success_base=worker_process.successful_calls
            if cfg['agent_type']=='tool-calling' and not fake:llm.round_success_base=round_success_base
            prompt=json.dumps(dict(goal=bundle.goal,context=bundle.context,role=bundle.role,round=rnd,previous_insights=predictions),ensure_ascii=False)
            prompt+='\nAutonomously investigate this goal using the available data and Python tool. Discover up to THREE distinct, goal-relevant factual insights this round, with explicit comparisons, numeric evidence and conditions where available. Compute evidence before making claims. In round 2 investigate new aspects rather than rephrasing previous insights. End with Final Answer containing JSON {"insights":["..."],"summary":"..."}. Do not include methodology, tool instructions, unsupported causes or uncomputed numbers in insights.'
            if cfg['agent_type']=='tool-calling':prompt+='\nUse actual native Python tool calls, not textual Action/Observation. Your final assistant message must contain the JSON object directly.'
            remaining=args.max_iterations;attempt_records=[]
            for recovery in range(2):
                agent.max_iterations=remaining
                result=agent.invoke({'input':prompt})
                steps=[dict(action=x.log,tool=x.tool,input=x.tool_input,observation=y) for x,y in result['intermediate_steps']]
                record=dict(output=result['output'],intermediate_steps=steps,remaining_iteration_budget=remaining)
                attempt_records.append(record);write(dest/f'round_{rnd}_attempt_{recovery+1}.json',record)
                remaining-=len(steps)+1
                try:
                    payload=extract_json_object(result['output']);values=validate_insights(payload)
                    if worker_process.successful_calls<=round_success_base:raise ValueError('No successful Python computation occurred in this round; execute the analysis before submitting findings')
                    if any(v.strip() in {x.strip() for x in predictions} for v in values):raise ValueError('Round 2 repeats an earlier insight verbatim; compute new goal-relevant findings')
                    break
                except ValueError as exc:
                    if recovery==1 or remaining<1:raise
                    prompt+='\nYour previous final answer did not complete the task: '+str(exc)
                    prompt+='\nThe Python session still retains previous variables. Continue the analysis using the tool; do not stop after just inspecting row counts. Previous output and executed steps:\n'+json.dumps(record,ensure_ascii=False)
            write(dest/f'round_{rnd}.json',dict(output=result['output'],intermediate_steps=[s for a in attempt_records for s in a['intermediate_steps']],attempts=len(attempt_records),successful_python_calls=worker_process.successful_calls-round_success_base))
            predictions.extend(values)
            write(dest/'partial_predictions.json',dict(case_id=c['case_id'],completed_rounds=rnd,predictions=predictions,status='partial' if rnd<2 else 'completed'))
        row=dict(case_id=c['case_id'],status='completed',predictions=predictions,pred_insights=predictions,pred_summary=payload.get('summary',''),model=args.model,temperature=0,python_tool_calls=worker_process.calls,agent_type=cfg['agent_type'])
        write(dest/'predictions.json',row);return row
    finally:worker_process.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--smoke-offline',action='store_true')
    p.add_argument('--worker',type=Path);p.add_argument('--worker-dir',type=Path)
    p.add_argument('--model',default='mimo-v2.6-flash');p.add_argument('--base-url',default='https://api.xiaomimimo.com/v1');p.add_argument('--api-key-env',default='XIAOMI_API_KEY')
    p.add_argument('--case-ids',nargs='+');p.add_argument('--results-dir',type=Path,default=OUT);p.add_argument('--max-iterations',type=int,default=15)
    p.add_argument('--manifest',type=Path,default=SOURCE/'manifest.json',help='Case catalog; defaults to the existing matched 15-case catalog')
    p.add_argument('--agent-type',choices=['auto','zero-shot-react-description','tool-calling'],default='auto')
    args=p.parse_args()
    if args.worker:worker(args.worker,args.worker_dir);return
    if args.max_iterations<1:raise ValueError('max_iterations must be positive')
    cfg=prepare(args)
    if not args.run and not args.smoke_offline:print(json.dumps(dict(prepared=True,cases=len(cfg['cases']),output=str(args.results_dir))));return
    if args.run:
        import winreg
        if not os.environ.get(args.api_key_env):
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER,'Environment') as k:os.environ[args.api_key_env]=winreg.QueryValueEx(k,args.api_key_env)[0]
        import msvcrt
        lock=(args.results_dir/'run.lock').open('a+b');lock.seek(0)
        if not lock.read(1):lock.write(b'0');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    errors=[];rows=[]
    selected=cfg['cases'][:1] if args.smoke_offline else cfg['cases']
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs={pool.submit(run_case,args,cfg,c,i,args.smoke_offline):c for i,c in enumerate(selected,1)}
        for f in as_completed(jobs):
            try:rows.append(f.result())
            except Exception:errors.append(dict(case_id=jobs[f]['case_id'],error=traceback.format_exc()))
            write(args.results_dir/'status.json',dict(state='generating',completed=len(rows),failed=len(errors),total=len(selected)))
    write(args.results_dir/'trajstory_case_results.json',rows);write(args.results_dir/'errors.json',errors)
    write(args.results_dir/'status.json',dict(state='failed' if errors else 'completed',completed=len(rows),failed=len(errors),total=len(selected)))
    print(json.dumps(dict(completed=len(rows),failed=len(errors),output=str(args.results_dir))))
    if errors:raise RuntimeError('Some cases failed; see errors.json')


if __name__=='__main__':main()
