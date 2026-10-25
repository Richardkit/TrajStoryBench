"""Unified offline Pandas regressions: parser, budgets, REPL and usage logging."""

import sys as _baseline_sys
from pathlib import Path as _BaselinePath
_baseline_repo = next(p for p in _BaselinePath(__file__).resolve().parents if (p / "baselines" / "__init__.py").is_file())
_baseline_sys.path.insert(0, str(_baseline_repo))
import baselines  # Register the shared baseline module location.

from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
import json
from pathlib import Path
import sys
import unittest
from pandas_multi_role.react_runner import compatible_output_parser, validate_insights
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException

class OutputProtocolTests(unittest.TestCase):

    def setUp(self):
        self.parser = compatible_output_parser()
        self.answer = '{"insights":["A computed comparison."],"summary":"Evidence-backed output."}'

    def test_plain_json_without_prefix(self):
        result = self.parser.parse(self.answer)
        self.assertIsInstance(result, AgentFinish)
        self.assertEqual(result.return_values['output'], self.answer)

    def test_standard_and_lowercase_prefixes(self):
        for prefix in ('Final Answer: ', 'final answer: ', 'Thought: done\nFinal Answer: '):
            self.assertEqual(json.loads(self.parser.parse(prefix + self.answer).return_values['output']), json.loads(self.answer))

    def test_markdown_fences(self):
        for prefix in ('', 'Final Answer: '):
            self.assertEqual(self.parser.parse(prefix + '```json\n' + self.answer + '\n```').return_values['output'], self.answer)

    def test_python_action_containing_json_stays_action(self):
        response = 'Thought: compute\nAction: python_repl_ast\nAction Input: print(' + repr(self.answer) + ')'
        self.assertIsInstance(self.parser.parse(response), AgentAction)

    def test_empty_action_is_repairable_error(self):
        for empty in ('', '``', '```python\n```'):
            with self.assertRaises(OutputParserException) as caught:
                self.parser.parse('Thought: compute\nAction: python_repl_ast\nAction Input: ' + empty)
            self.assertTrue(caught.exception.send_to_llm)

    def test_invalid_json_not_accepted_as_final(self):
        with self.assertRaises(OutputParserException):
            self.parser.parse('{"insights":')

    def test_three_insights_missing_prefix_fixture(self):
        raw = json.dumps({'insights': ['Finding A', 'Finding B', 'Finding C'], 'summary': 'Evidence-backed output.'})
        result = self.parser.parse(raw)
        self.assertEqual(result.return_values['output'], raw.strip())
        self.assertEqual(len(json.loads(result.return_values['output'])['insights']), 3)

    def test_failure_statement_is_not_an_insight(self):
        with self.assertRaises(ValueError):
            validate_insights({'insights': ['I can’t provide new, evidence-based findings because analysis was not performed.']})
        self.assertEqual(validate_insights({'insights': ['Group means cannot be compared within destination because each destination occurs in one period.']}), ['Group means cannot be compared within destination because each destination occurs in one period.'])
import pandas as pd
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain_experimental.agents import create_pandas_dataframe_agent
from baselines.scripts.pandas_checkpoint_budget import apply_round_budget as checkpoint_budget_apply_round_budget, ROUND_POLICY as checkpoint_budget_ROUND_POLICY

class FakeToolsModel(FakeMessagesListChatModel):

    def bind_tools(self, tools, **kwargs):
        return self

class CountCalls(BaseCallbackHandler):

    def __init__(self):
        self.count = 0

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.count += 1

def checkpoint_budget_action():
    return AIMessage(content='', tool_calls=[{'name': 'python_repl_ast', 'args': {'query': 'len(df)'}, 'id': 'test_call'}])

def checkpoint_budget_submission(n):
    return AIMessage(content=json.dumps({'insights': [f'Finding {i}' for i in range(n)], 'summary': f'Summary of {n} findings'}))

def checkpoint_budget_run(responses):
    llm = FakeToolsModel(responses=responses)
    original = create_pandas_dataframe_agent(llm, pd.DataFrame({'x': [1, 2]}), agent_type='tool-calling', allow_dangerous_code=True, return_intermediate_steps=True)
    executor = checkpoint_budget_apply_round_budget(original, llm)
    count = CountCalls()
    result = executor.invoke({'input': checkpoint_budget_ROUND_POLICY + 'Find six useful insights.'}, config={'callbacks': [count]})
    return (executor, json.loads(result['output']), count.count)

class CheckpointTests(unittest.TestCase):

    def test_all_checkpoints_then_summary_exactly_round37(self):
        responses = []
        for n in range(1, 7):
            responses.extend([checkpoint_budget_action()] * 5 + [checkpoint_budget_submission(n)])
        responses.append(checkpoint_budget_submission(6))
        agent, answer, calls = checkpoint_budget_run(responses)
        self.assertEqual(calls, 37)
        self.assertEqual(agent.rounds_used, 37)
        self.assertEqual(len(answer['insights']), 6)
        self.assertEqual([x['round'] for x in agent.checkpoints], [6, 12, 18, 24, 30, 36, 37])
        self.assertEqual(agent.termination_reason, 'final_summary_round_37')

    def test_first_checkpoint_missing_insight_ends_with_summary(self):
        agent, answer, calls = checkpoint_budget_run([checkpoint_budget_action()] * 5 + [checkpoint_budget_submission(0), checkpoint_budget_submission(6)])
        self.assertEqual(calls, 6)
        self.assertEqual(answer, {'insights': [], 'summary': 'Summary of 0 findings'})
        self.assertEqual(agent.termination_reason, 'checkpoint_quota_missed_round_6')

    def test_second_checkpoint_outputs_existing_partial_insight(self):
        agent, answer, calls = checkpoint_budget_run([checkpoint_budget_action()] * 5 + [checkpoint_budget_submission(1)] + [checkpoint_budget_action()] * 5 + [checkpoint_budget_submission(1), checkpoint_budget_submission(6)])
        self.assertEqual(calls, 12)
        self.assertEqual(len(answer['insights']), 1)
        self.assertEqual(agent.termination_reason, 'checkpoint_quota_missed_round_12')

    def test_duplicate_insights_do_not_meet_quota(self):
        duplicate = AIMessage(content=json.dumps({'insights': ['Finding 0', 'Finding 0'], 'summary': 'One finding'}))
        agent, answer, calls = checkpoint_budget_run([checkpoint_budget_action()] * 5 + [checkpoint_budget_submission(1)] + [checkpoint_budget_action()] * 5 + [duplicate])
        self.assertEqual(calls, 12)
        self.assertEqual(len(answer['insights']), 1)
from baselines.scripts.pandas_native_budget import apply_total_budget, budget_policy, validate_answer
from baselines.scripts.pandas_repl_compat import CompatiblePythonAstREPLTool, apply_repl_compat

class Recorder(BaseCallbackHandler):

    def __init__(self):
        self.messages = []

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.messages.append(messages[0])

def native_action(queries=('len(df)',)):
    return AIMessage(content='', tool_calls=[dict(name='python_repl_ast', args={'query': query}, id=f'call{i}') for i, query in enumerate(queries)])

def native_execute(responses, limit=60):
    llm = FakeToolsModel(responses=responses)
    official = create_pandas_dataframe_agent(llm, pd.DataFrame({'x': [1, 2]}), agent_type='tool-calling', allow_dangerous_code=True, return_intermediate_steps=True, max_iterations=limit)
    agent = apply_total_budget(apply_repl_compat(official), limit, llm=llm)
    recorder = Recorder()
    result = agent.invoke({'input': budget_policy(limit) + 'Analyze the goal.'}, config={'callbacks': [recorder]})
    return (agent, result, recorder.messages)

class NativeTests(unittest.TestCase):

    def test_early_finish_is_preserved(self):
        agent, result, messages = native_execute([AIMessage(content='final')])
        self.assertTrue(agent.completed_normally)
        self.assertEqual(agent.completion_mode, 'autonomous')
        self.assertEqual(agent.rounds_used, 1)
        self.assertEqual(result['output'], 'final')
        self.assertIn('df.head()', messages[0][0].content)

    def test_sixth_call_stays_a_tool_turn_and_state_persists(self):
        responses = [native_action(('saved = 42',))] + [native_action()] * 4
        responses += [native_action(('saved + 1',)), AIMessage(content='done')]
        agent, result, messages = native_execute(responses)
        self.assertEqual(agent.rounds_used, 7)
        self.assertEqual(len(result['intermediate_steps']), 6)
        self.assertEqual(result['intermediate_steps'][-1][1], 43)
        self.assertTrue(any((m.type == 'tool' for m in messages[-1])))

    def test_final_call_inside_budget_succeeds(self):
        agent, result, messages = native_execute([native_action()] * 59 + [AIMessage(content='done')])
        self.assertTrue(agent.completed_normally)
        self.assertEqual(len(messages), 60)
        self.assertEqual(agent.completion_mode, 'budget_summary')
        self.assertEqual(len(result['intermediate_steps']), 59)
        self.assertTrue(any(('EXECUTED EVIDENCE' in str(m.content) for m in messages[-1])))
        self.assertFalse(any((m.type == 'tool' for m in messages[-1])))

    def test_no_extra_summary_or_recovery_call(self):
        agent, result, messages = native_execute([native_action()] * 2 + [AIMessage(content='done'), AIMessage(content='never')], 3)
        self.assertTrue(agent.completed_normally)
        self.assertEqual(len(messages), 3)
        self.assertEqual(len(result['intermediate_steps']), 2)
        self.assertNotEqual(result['output'], 'never')

    def test_final_call_with_no_tools_even_for_one_call_budget(self):
        agent, result, messages = native_execute([AIMessage(content='no evidence')], 1)
        self.assertEqual(agent.rounds_used, 1)
        self.assertEqual(len(result['intermediate_steps']), 0)
        self.assertEqual(agent.completion_mode, 'budget_summary')

    def test_unexpected_final_tool_call_or_empty_text_fails_without_recovery(self):
        for bad in (native_action(), AIMessage(content='')):
            with self.assertRaisesRegex(RuntimeError, 'FinalSummaryInvalid'):
                native_execute([bad], 1)

    def test_all_prints_and_final_expression_are_returned(self):
        tool = CompatiblePythonAstREPLTool()
        self.assertEqual(tool.invoke({'query': 'print("first")\nprint("second")\n3+4'}), 'first\nsecond\n7')

    def test_function_scope_and_later_call_share_variables(self):
        tool = CompatiblePythonAstREPLTool()
        self.assertEqual(tool.invoke({'query': 'import pandas as pd\ndef helper():\n    return pd.Series([1, 2]).sum()\nhelper()'}), 3)
        self.assertEqual(tool.invoke({'query': 'helper() + 1'}), 4)

    def test_error_does_not_execute_expression_twice(self):
        tool = CompatiblePythonAstREPLTool()
        tool.invoke({'query': 'calls = []\ndef fail():\n    calls.append(1)\n    raise ValueError("expected")'})
        self.assertIn('ValueError: expected', tool.invoke({'query': 'print("before")\nfail()'}))
        self.assertEqual(tool.invoke({'query': 'len(calls)'}), 1)

    def test_multiple_tools_share_one_decision(self):
        agent, result, messages = native_execute([native_action(('len(df)', 'df.x.sum()')), AIMessage(content='done')])
        self.assertEqual(agent.rounds_used, 2)
        self.assertEqual(len(result['intermediate_steps']), 2)

    def test_complete_partial_empty_and_invalid(self):
        for n, kind in ((6, 'complete'), (2, 'partial'), (0, 'empty')):
            self.assertEqual(validate_answer(dict(insights=['finding'] * n, summary='summary')), kind)
        for obj in (dict(insights=[''], summary='x'), dict(insights=['x'] * 7, summary='x'), dict(insights=['x'], summary=''), dict(insights='x', summary='x')):
            with self.assertRaises(ValueError):
                validate_answer(obj)

    def test_invalid_budget(self):
        for limit in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                apply_total_budget(None, limit)
from baselines.scripts.pandas_round_budget import apply_round_budget as round_budget_apply_round_budget, ROUND_POLICY as round_budget_ROUND_POLICY

class PromptRecorder(BaseCallbackHandler):

    def __init__(self):
        self.prompts = []

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.prompts.append(next((m.content for m in messages[0] if m.type == 'human')))

def round_budget_action(count=1):
    return AIMessage(content='', tool_calls=[{'name': 'python_repl_ast', 'args': {'query': 'len(df)'}, 'id': f'call{i}'} for i in range(count)])

def round_budget_run(responses):
    agent = round_budget_apply_round_budget(create_pandas_dataframe_agent(FakeToolsModel(responses=responses), pd.DataFrame({'x': [1, 2]}), agent_type='tool-calling', allow_dangerous_code=True, return_intermediate_steps=True, max_iterations=48))
    recorder = PromptRecorder()
    result = agent.invoke({'input': round_budget_ROUND_POLICY + 'Find six insights.'}, config={'callbacks': [recorder]})
    return (agent, result, recorder.prompts)

class BudgetTests(unittest.TestCase):

    def test_final_answer_on_round_48_succeeds(self):
        answer = json.dumps({'insights': ['test'] * 6, 'summary': 'test'})
        agent, result, prompts = round_budget_run([round_budget_action()] * 47 + [AIMessage(content=answer)])
        self.assertTrue(agent.completed_normally)
        self.assertEqual(agent.rounds_used, 48)
        self.assertEqual(len(prompts), 48)
        self.assertEqual(result['output'], answer)
        self.assertIn('CURRENT AGENT ROUND: 48/48', prompts[-1])
        self.assertIn('LAST ROUND', prompts[-1])

    def test_no_final_by_48_stops_without_49th_call(self):
        agent, result, prompts = round_budget_run([round_budget_action()] * 48 + [AIMessage(content='MUST NOT BE CALLED')])
        self.assertFalse(agent.completed_normally)
        self.assertEqual(agent.rounds_used, 48)
        self.assertEqual(len(prompts), 48)
        self.assertNotEqual(result['output'], 'MUST NOT BE CALLED')

    def test_two_tools_in_one_response_are_one_round(self):
        agent, result, prompts = round_budget_run([round_budget_action(2), AIMessage(content='finished')])
        self.assertTrue(agent.completed_normally)
        self.assertEqual(agent.rounds_used, 2)
        self.assertEqual(len(result['intermediate_steps']), 2)
        self.assertIn('CURRENT AGENT ROUND: 2/48', prompts[1])
import tempfile
from uuid import uuid4
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, LLMResult
from baselines.scripts.pandas_usage_logging import ProviderUsageRecorder, UsageChatOpenAI

class UsageTests(unittest.TestCase):

    def test_actual_langchain_stream_and_invoke_callbacks_without_network(self):

        class Stream:

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def __iter__(self):
                yield dict(id='offline', model='offline', choices=[dict(index=0, delta=dict(role='assistant', content='done'), finish_reason='stop')])
                yield dict(id='offline', model='offline', choices=[], usage=dict(prompt_tokens=10, completion_tokens=3, total_tokens=13))

        class Client:

            def __init__(self):
                self.payloads = []

            def create(self, **kwargs):
                self.payloads.append(kwargs)
                if kwargs.get('stream'):
                    return Stream()
                return dict(id='offline', model='offline', choices=[dict(message=dict(role='assistant', content='done'), finish_reason='stop')], usage=dict(prompt_tokens=20, completion_tokens=4, total_tokens=24))
        with tempfile.TemporaryDirectory() as directory:
            recorder = ProviderUsageRecorder(directory)
            client = Client()
            llm = UsageChatOpenAI(model='offline', api_key='offline_dummy', client=client, callbacks=[recorder])
            list(llm.stream('test'))
            llm.invoke('test')
            self.assertEqual(client.payloads[0]['stream_options'], {'include_usage': True})
            self.assertNotIn('stream_options', client.payloads[1])
            self.assertEqual(recorder.summary()['total_tokens'], 37)
            self.assertEqual(recorder.summary()['model_calls_started'], 2)
            self.assertTrue(recorder.summary()['complete'])

    def test_stream_and_nonstream_usage_count_once(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = ProviderUsageRecorder(directory)
            for message in (AIMessage(content='a', usage_metadata=dict(input_tokens=10, output_tokens=2, total_tokens=12)), AIMessageChunk(content='b', usage_metadata=dict(input_tokens=20, output_tokens=3, total_tokens=23))):
                run_id = uuid4()
                recorder.on_chat_model_start({}, [], run_id=run_id)
                recorder.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]), run_id=run_id)
            summary = recorder.summary()
            self.assertTrue(summary['complete'])
            self.assertEqual(summary['total_tokens'], 35)
            self.assertEqual(summary['calls_with_usage'], 2)
            self.assertEqual(json.loads((Path(directory) / 'model_usage.json').read_text())['summary'], summary)

    def test_missing_usage_and_errors_are_not_reported_as_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = ProviderUsageRecorder(directory)
            run_id = uuid4()
            recorder.on_chat_model_start({}, [], run_id=run_id)
            recorder.on_llm_end(LLMResult(generations=[[ChatGeneration(message=AIMessage(content='a'))]]), run_id=run_id)
            run_id = uuid4()
            recorder.on_chat_model_start({}, [], run_id=run_id)
            recorder.on_llm_error(RuntimeError('private provider details'), run_id=run_id)
            self.assertFalse(recorder.summary()['complete'])
            self.assertEqual(recorder.summary()['missing_usage_calls'], 2)
            self.assertNotIn('private provider details', (Path(directory) / 'model_usage.json').read_text())

if __name__ == '__main__':
    unittest.main()
