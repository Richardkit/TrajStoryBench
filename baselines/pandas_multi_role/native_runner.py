"""TrajStory I/O adaptation of the official Pandas tool-calling agent."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from baselines.scripts.pandas_native_budget import (
    DEFAULT_MAX_CALLS, apply_total_budget, budget_policy, validate_answer,
)
from baseline_support.json_utils import extract_json_object, write_json
from baselines.scripts.pandas_repl_compat import apply_repl_compat
from baselines.scripts.pandas_usage_logging import UsageChatOpenAI, ProviderUsageRecorder

PROTOCOL = 'native_free_analysis_reserved_summary_v2'
MODELS = {
    'gpt': ('gpt-6-luna', 'https://api.openai.com/v1', 'OPENAI_API_KEY'),
    'qwen': ('qwen3.8-flash', 'https://dashscope.aliyuncs.com/compatible-mode/v1', 'DASHSCOPE_API_KEY'),
}


def task_input(bundle, tables, limit):
    return budget_policy(limit) + (
        'Analyze the following goal using the available dataframes. Aim to discover SIX '
        'distinct factual insights supported by your executed analysis. Choose your own '
        'analyses and tool actions. Use computed evidence for numerical claims and preserve '
        'the relevant conditions, units and scope. If fewer than six insights are supported, '
        'return only those supported; do not pad or invent findings. Return one JSON object '
        'with "insights": [up to six insight strings] and "summary": "a coherent summary".\n\n'
    ) + json.dumps(dict(goal=bundle.goal, context=bundle.context, role=bundle.role,
                        tables=tables), ensure_ascii=False)


def make_llm(model_key, offline, usage_recorder=None, llm_class=None):
    if offline:
        from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
        from langchain_core.messages import AIMessage

        class FakeToolsModel(FakeMessagesListChatModel):
            def bind_tools(self, tools, **kwargs):
                return self

        return FakeToolsModel(responses=[
            AIMessage(content='', tool_calls=[dict(name='python_repl_ast',
                args={'query': 'print("OFFLINE PROTOCOL CHECK")'}, id='offline_call')]),
            AIMessage(content=json.dumps(dict(insights=['OFFLINE TEST ONLY'],
                                              summary='Offline protocol check.'))),
        ])
    constructor = llm_class or UsageChatOpenAI
    retry_options = {'max_retries': 0} if llm_class else {}
    model, url, key_name = MODELS[model_key]
    key = os.environ.get(key_name)
    if not key and os.name == 'nt':
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as handle:
            key = winreg.QueryValueEx(handle, key_name)[0]
    if not key:
        raise ValueError(f'{key_name} is required for live generation')
    if model_key == 'gpt':
        return constructor(model=model, base_url=url, api_key=key, temperature=0, **retry_options,
                          callbacks=[usage_recorder] if usage_recorder else None,
                          model_kwargs={'reasoning_effort': 'none', 'max_completion_tokens': 16000})
    return constructor(model=model, base_url=url, api_key=key, temperature=0, **retry_options,
                      callbacks=[usage_recorder] if usage_recorder else None,
                      max_tokens=16000, extra_body={'enable_thinking': False})


def main():
    import pandas as pd
    from langchain_experimental.agents import create_pandas_dataframe_agent
    from baseline_support.case_adapter import load_trajstory_case

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-id', required=True)
    parser.add_argument('--model', choices=MODELS, default='gpt')
    parser.add_argument('--max-model-calls', type=int, default=DEFAULT_MAX_CALLS)
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--smoke-offline', action='store_true')
    parser.add_argument('--input-csvs',action='store_true',help='Use projected input_ CSVs without original fallback')
    args = parser.parse_args()
    if args.max_model_calls < 1:
        parser.error('--max-model-calls must be positive')
    if args.run and args.smoke_offline:
        parser.error('--run and --smoke-offline are mutually exclusive')
    out = args.results_dir.resolve()
    # Never relabel or overwrite a previous generation, including paused experiments.
    if out.exists() and any(out.iterdir()):
        parser.error('use a fresh empty results directory')
    catalog = json.loads((ROOT / 'trajstory_examples/insight/manifest.json').read_text(encoding='utf8'))['cases']
    case = next(c for c in catalog if c['case_id'] == args.case_id)
    bundle = load_trajstory_case(ROOT / case['json_path'], ROOT,input_csvs=args.input_csvs)
    frames = [pd.read_csv(f.path) if f.path.suffix.lower() == '.csv' else pd.read_parquet(f.path)
              for f in bundle.files]
    tables = [dict(dataframe='df' if len(frames) == 1 else f'df{i}', file_id=f.file_id, role=f.role)
              for i, f in enumerate(bundle.files, 1)]
    prompt = task_input(bundle, tables, args.max_model_calls)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'task.txt').write_text(prompt, encoding='utf8')
    sources = [Path(__file__), ROOT / 'baselines/scripts/pandas_native_budget.py',
               ROOT / 'baselines/scripts/pandas_repl_compat.py',
               ROOT / 'baselines/scripts/pandas_usage_logging.py',
               ROOT / 'baselines/baseline_support/case_adapter.py']
    from baseline_support.runtime import source_files as support_sources
    sources.extend(support_sources())
    sources.append(ROOT/'baselines/baseline_support/input_paths.py')
    write_json(out / 'manifest.json', dict(
        protocol=PROTOCOL, case_id=args.case_id, model=MODELS[args.model][0],
        factory='langchain_experimental.agents.create_pandas_dataframe_agent',
        agent_type='tool-calling', agent_invocations=1, generation_attempts=1,
        max_model_calls=args.max_model_calls, final_answer_counts=True,
        checkpoints=False, forced_final_call=True, output_recovery=False,
        analysis_call_limit=args.max_model_calls - 1, final_summary_inside_budget=True,
        prefix='official default', parser='official default',
        python_tool='PythonAstREPLTool with full stdout and shared namespace compatibility fix',
        input_mode='preloaded_dataframes_with_head',
        projected_input_csvs=args.input_csvs,
        requested_insights=6, partial_outputs_allowed=True,
        usage_logging='Provider usage per call; streamed calls request stream_options.include_usage; missing usage explicitly marked',
        evidence_validation='shape check only; factual support is not automatically verified',
        validation_mode='offline_fake' if args.smoke_offline else 'benchmark',
        versions={n: importlib.metadata.version(n) for n in
                  ('langchain', 'langchain-core', 'langchain-experimental', 'langchain-openai', 'openai')},
        input_sha256={str(f.path.relative_to(ROOT)): hashlib.sha256(f.path.read_bytes()).hexdigest()
                      for f in bundle.files},
        case_json_sha256=hashlib.sha256((ROOT / case['json_path']).read_bytes()).hexdigest(),
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    ))
    for source in sources:
        target = out / 'source_snapshot' / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    if not args.run and not args.smoke_offline:
        print(json.dumps(dict(prepared=True, case_id=args.case_id,
                              max_model_calls=args.max_model_calls)))
        return
    os.environ['LANGCHAIN_TRACING_V2'] = 'false'
    os.environ['LANGSMITH_TRACING'] = 'false'
    usage_recorder = ProviderUsageRecorder(out)
    llm = make_llm(args.model, args.smoke_offline, usage_recorder)
    official = create_pandas_dataframe_agent(
        llm, frames if len(frames) > 1 else frames[0], agent_type='tool-calling',
        allow_dangerous_code=True, return_intermediate_steps=True,
        max_iterations=args.max_model_calls, max_execution_time=None,
    )
    agent = apply_total_budget(apply_repl_compat(official), args.max_model_calls, llm=llm,
                              progress_callback=lambda data: write_json(out / 'progress.json', data))
    for name in list(os.environ):
        if name.endswith(('_API_KEY', '_TOKEN')) or name in ('API_KEY', 'HF_TOKEN'):
            os.environ.pop(name, None)
    started = time.monotonic()
    state = dict(state='running', round_limit=args.max_model_calls, protocol=PROTOCOL,
                 validation_mode='offline_fake' if args.smoke_offline else 'benchmark')
    write_json(out / 'status.json', state)
    try:
        result = agent.invoke({'input': prompt})
        steps = result['intermediate_steps']
        write_json(out / 'raw_result.json', dict(
            output=result['output'], intermediate_steps=[dict(tool=a.tool, input=a.tool_input,
                log=a.log, observation=b) for a, b in steps],
            elapsed_seconds=time.monotonic() - started, rounds_used=agent.rounds_used,
            tool_calls=len(steps), completed_normally=agent.completed_normally,
            completion_mode=agent.completion_mode,
            token_usage=usage_recorder.summary(),
        ))
        state.update(rounds_used=agent.rounds_used, tool_calls=len(steps),
                     completion_mode=agent.completion_mode)
        if not agent.completed_normally:
            state.update(state='failed', error_type='RoundLimitExceeded',
                         error='No final answer within the total model decision budget',
                         eligible_for_scoring=False, output_kind='failed')
        else:
            obj = extract_json_object(result['output'])
            kind = validate_answer(obj)
            write_json(out / 'parsed_output.json', dict(answer=obj, output_kind=kind,
                                                       structure_matches_request=True))
            state.update(state='returned', output_kind=kind, insights_count=len(obj['insights']),
                         structure_matches_request=True,
                         eligible_for_scoring=kind != 'empty' and not args.smoke_offline)
    except Exception as exc:
        # Do not persist exception text: provider errors may contain sensitive request data.
        state.update(state='failed', error_type=type(exc).__name__,
                     rounds_used=agent.rounds_used, eligible_for_scoring=False, output_kind='failed')
    state['token_usage'] = usage_recorder.summary()
    write_json(out / 'status.json', state)
    print(json.dumps(state))
    if state['state'] == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
