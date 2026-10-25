"""Fresh Pandas role workflow; no scoring or generation is launched without --run."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baseline_support.json_utils import write_json
from pandas_multi_role.native_runner import make_llm, MODELS
from pandas_multi_role.workflow import Config, PandasRoleWorkflow, PROTOCOL
from pandas_multi_role.network_requests import NetworkRetryChatOpenAI


def offline_factory(rows, df_name):
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    class FakeToolsModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self
    def factory(role, recorder):
        if role.startswith('planner'):
            number = int(role[-1])
            payload = dict(questions=[dict(question=f'OFFLINE round {number} table check {i}',
                                           parent_evidence_ids=[]) for i in range(1,4)])
        elif role.startswith('analyst'):
            return FakeToolsModel(callbacks=[recorder],responses=[AIMessage(content='',tool_calls=[
                dict(name='python_repl_ast',args={'query':f'print(len({df_name}))'},id='offline_'+role)]),
                AIMessage(content='OFFLINE computation done')])
        elif role.startswith('interpreter') or role.startswith('verifier'):
            qid=role.rsplit('_',1)[-1]
            payload=dict(insight=f'OFFLINE {qid}: primary table has {rows} rows.',
                         citations=[dict(step=1,quote=str(rows))],limitations=['OFFLINE TEST ONLY'])
            if role.startswith('verifier'):payload.update(status='supported',reason='offline count')
        else:
            payload=dict(selected_evidence_ids=['E_Q1','E_Q2','E_Q3','E_F1','E_F2','E_F3'],
                         summary='OFFLINE TEST ONLY; no model generation or scoring.')
        return FakeToolsModel(callbacks=[recorder],responses=[AIMessage(content=json.dumps(payload))])
    return factory


def main():
    import pandas as pd
    from baseline_support.case_adapter import load_trajstory_case
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-id',required=True)
    parser.add_argument('--model',choices=MODELS,default='gpt')
    parser.add_argument('--max-model-calls',type=int,default=83,help='Whole workflow request budget, including every role')
    parser.add_argument('--results-dir',type=Path,required=True)
    parser.add_argument('--input-csvs',action='store_true')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--run',action='store_true')
    mode.add_argument('--smoke-offline',action='store_true')
    args=parser.parse_args();config=Config(args.max_model_calls)
    out=args.results_dir.resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error('use a fresh empty output directory')
    catalog=json.loads((ROOT/'trajstory_examples/insight/manifest.json').read_text(encoding='utf8'))['cases']
    case=next(c for c in catalog if c['case_id']==args.case_id)
    bundle=load_trajstory_case(ROOT/case['json_path'],ROOT,input_csvs=args.input_csvs)
    frames=[pd.read_csv(f.path) if f.path.suffix.lower()=='.csv' else pd.read_parquet(f.path) for f in bundle.files]
    tables=[dict(dataframe='df' if len(frames)==1 else f'df{i}',file_id=f.file_id,role=f.role)
            for i,f in enumerate(bundle.files,1)]
    sources=[Path(__file__),ROOT/'pandas_multi_role/native_runner.py',ROOT/'scripts/pandas_repl_compat.py',
             ROOT/'scripts/pandas_usage_logging.py',ROOT/'scripts/pandas_native_budget.py',
             ROOT/'baseline_support/case_adapter.py',ROOT/'baseline_support/input_paths.py',
             *sorted((ROOT/'pandas_multi_role').glob('*.py'))]
    from baseline_support.runtime import source_files as support_sources
    sources.extend(support_sources())
    sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    write_json(out/'manifest.json',dict(protocol=PROTOCOL,case_id=args.case_id,model=MODELS[args.model][0],
        rounds=2,questions_per_round=3,separate_role_contexts=True,shared_backbone=True,
        max_model_calls=config.max_model_calls,request_limit=config.max_model_calls,
        network_failure_limit=5,network_failures_consume_requests=False,
        analysis_calls_per_branch=config.analysis_calls_per_branch,
        fixed_role_call_ceiling=15,analysis_branch_count=6,whole_case_attempts=1,
        projected_input_csvs=args.input_csvs,validation_mode='offline_fake' if args.smoke_offline else 'benchmark',
        input_sha256={str(f.path.relative_to(ROOT)):sha(f.path) for f in bundle.files},
        case_json_sha256=sha(ROOT/case['json_path']),
        source_sha256={str(f.relative_to(ROOT)):sha(f) for f in sources},
        evidence_validation='model-based verifier plus exact output citation checks; not independent numeric reruns',
        mathematical_refinement=False,judge='separate two independent GPT-6-Luna rounds; not launched by this runner'))
    for f in sources:
        dest=out/'source_snapshot'/f.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(f.read_bytes())
    if not args.run and not args.smoke_offline:
        write_json(out/'status.json',dict(state='prepared',protocol=PROTOCOL))
        print(json.dumps(dict(state='prepared',analysis_calls_per_branch=config.analysis_calls_per_branch)))
        return
    os.environ['LANGCHAIN_TRACING_V2']='false';os.environ['LANGSMITH_TRACING']='false'
    if args.smoke_offline:
        factory=offline_factory(len(frames[0]),tables[0]['dataframe'])
    else:
        # Construct the provider client before clearing tool-visible credential env vars.
        # Each role gets a distinct wrapper/callback, not a shared message history.
        base=make_llm(args.model,False,llm_class=NetworkRetryChatOpenAI)
        def factory(role,recorder):
            # Pydantic copy omits excluded provider client fields in this environment.
            model=copy.copy(base)
            model.callbacks=[recorder]
            return model
    for name in list(os.environ):
        if name.endswith(('_API_KEY','_TOKEN')) or name in ('API_KEY','HF_TOKEN'):
            os.environ.pop(name,None)
    workflow=PandasRoleWorkflow(factory,config=config,offline=args.smoke_offline)
    try:
        state=workflow.run(bundle,frames,tables,out)
        if not args.smoke_offline:
            state.update(base.network_policy.summary())
            write_json(out/'status.json',state)
        write_json(out/'job_result.json',dict(case_id=args.case_id,json_path=case['json_path'],directory=str(out),
            returncode=0,status=state,eligible_for_scoring=state['eligible_for_scoring'],
            returned_insight_count=state['insights_count'],requests_used=state['requests_used'],
            tool_calls=state['tool_calls'],token_usage=state['token_usage']))
        print(json.dumps(state))
    except Exception as exc:
        import traceback
        write_json(out/'failure_diagnostic.json',dict(error_type=type(exc).__name__,
            attribute=getattr(exc,'name',None),frames=[dict(file=Path(f.filename).name,line=f.lineno,function=f.name) for f in traceback.extract_tb(exc.__traceback__)]))
        # Provider exception text may include sensitive data; record types only.
        summaries={r:c.summary() for r,c in workflow.recorders.items()}
        write_json(out/'model_usage.json',dict(roles=summaries))
        write_json(out/'status.json',dict(state='failed',protocol=PROTOCOL,error_type=type(exc).__name__,
            eligible_for_scoring=False,request_limit=config.max_model_calls,
            requests_used=sum(x['requests_used'] for x in summaries.values()),
            **(base.network_policy.summary() if not args.smoke_offline else {})))
        print(json.dumps(dict(state='failed',error_type=type(exc).__name__)))
        raise SystemExit(1)


if __name__=='__main__':main()
