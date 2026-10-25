"""Check native dispatch is used without the former fixed-stage controller."""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import pytest

os.environ['TRAJSTORY_MINIMAL_IMPORTS'] = '1'
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('minimal_runner', ROOT / 'examples/run_trajstory.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

@pytest.mark.parametrize("count,summary,expected_status,audit_mode", [
    (6, 'Summary', 'complete', 'valid'),
    (2, None, 'partial', 'valid'),
    (6, '', 'partial', 'valid'),
    (7, 'Summary', 'count_mismatch', 'valid'),
    (6, 'Summary', 'complete', 'missing'),
    (6, 'Summary', 'complete', 'bypass'),
])
def test_native_config_and_single_dispatch_completion(tmp_path, count, summary, expected_status, audit_mode):
    from mmengine import Config
    from src.interaction import bus
    cfg = Config.fromfile(str(ROOT / 'configs/trajstory.py'))
    assert cfg.agent_names == ['planning', 'orchestra_analyst', 'orchestra_interpreter',
                               'orchestra_verifier', 'orchestra_synthesizer']
    assert cfg.planning_agent.prompt_name == 'planning'
    assert cfg.orchestra_analyst_agent.prompt_name == 'tool_calling'
    assert cfg.orchestra_analyst_agent.max_steps == 50
    assert cfg.orchestra_analyst_agent.max_tools == 10
    assert cfg.trajstory_analysis_tool.use_trajstory_helpers is False
    with patch('sys.argv', ['run_trajstory.py', '--case-ids', 'danish_ais_01_shared_corridor_day',
                            '--results-dir', str(tmp_path), '--max-rounds', '3']):
        args = runner.parse_args()
    payload = {'insights': [f'Finding {i}' for i in range(count)]}
    if summary is not None:
        payload['summary'] = summary
    answer = json.dumps(payload)
    submit = AsyncMock(return_value=SimpleNamespace(payload={'success': True, 'result': answer}))
    with patch.object(runner, '_configure_runtime'), \
         patch.object(runner, '_initialize_framework', new=AsyncMock()), \
         patch.object(bus, 'submit', new=submit), \
         patch.object(bus, 'shutdown', new=AsyncMock()), \
         patch('src.agent.orchestra_roles.role_audit', return_value={
             'missing_roles': ['orchestra_verifier'] if audit_mode == 'missing' else [],
             'separate_role_contexts': True,
             'synthesized_results': [] if audit_mode == 'bypass' else [answer]}):
        asyncio.run(runner.run(args))
    # A result can complete after one submit; no forced roster or result injection.
    assert submit.await_count == 1
    assert set(submit.call_args.kwargs) == {'session_ctx', 'max_rounds', 'history_result_max_chars',
                                           'planner_context_max_chars', 'max_model_requests'}
    assert submit.call_args.kwargs['max_model_requests'] == 83
    assert submit.call_args.kwargs['max_rounds'] == 3
    assert submit.call_args.kwargs['history_result_max_chars'] == 5000
    assert submit.call_args.kwargs['planner_context_max_chars'] == 6000
    task = submit.call_args.args[0].content
    assert 'six factual insights' in task
    assert 'distinct' not in task
    assert 'exactly these three' not in task
    rows = json.loads((tmp_path / 'trajstory_case_results.json').read_text(encoding='utf8'))
    assert rows[0]['status'] == ('completed' if audit_mode == 'valid' else 'failed')
    if audit_mode != 'valid':
        assert rows[0]['error_type'] == (
            'RoleWorkflowIncomplete' if audit_mode == 'missing' else 'PlannerBypassedSynthesizer')
    assert rows[0]['pred_insights'] == payload['insights']
    assert rows[0]['pred_summary'] == (summary or '')
    assert rows[0]['structure_matches_request'] == (expected_status == 'complete')
    assert rows[0]['protocol'] == 'native_planner_bounded_indexed_multi_role'
    assert rows[0]['output_status'] == expected_status
