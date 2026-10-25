"""Offline paper-protocol checks: no credentials or network calls required."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from baselines.scripts import run_insightbench_generation_no_rouge as paper

@pytest.mark.parametrize('model', ['gpt-6-luna', 'qwen3.8-flash'])
def test_paper_defaults_and_request(monkeypatch, model):
    captured = {}
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'offline-placeholder')
    monkeypatch.delenv('TRAJSTORY_LLM_AUDIT_PATH', raising=False)
    class Client:
        def __init__(self, **kwargs):
            captured['client'] = {k:v for k,v in kwargs.items() if k != 'api_key'}
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        def create(self, **kwargs):
            captured['request'] = kwargs
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='OFFLINE'))])
    monkeypatch.setattr(paper.au, 'OpenAI', Client)
    def run(args):
        assert (args.max_questions, args.branch_depth, args.n_retries) == (3, 2, 2)
        assert args.temperature == 0 and args.reasoning_effort == 'none'
        assert args.shared_analysis_tools and args.keep_artifacts
        assert not hasattr(args, 'max_model_requests')
        for name in ('TRAJSTORY_OPENAI_REASONING_EFFORT', 'TRAJSTORY_QWEN_REASONING_EFFORT'):
            monkeypatch.setenv(name, args.reasoning_effort)
        chat = paper.au.get_chat_model(args.model_name, args.temperature, args.seed)
        assert chat('test input') == 'OFFLINE'
        assert paper.benchmarks.evaluate_insights() is None
        assert paper.benchmarks.evaluate_summary() is None
    monkeypatch.setattr(paper.adapter, 'run', run)
    monkeypatch.setattr(sys, 'argv', ['paper', '--model-name', model])
    original = paper.au.get_chat_model
    paper.main()
    assert paper.au.get_chat_model is original
    req = captured['request']
    assert req['model'] == model
    assert req['temperature'] == 0 and req['reasoning_effort'] == 'none'
    assert req['messages'] == [{'role':'user', 'content':paper.INPUT_POLICY+'\n'+'test input'}]
    assert 'max_retries' not in captured['client']
    if model == 'qwen3.8-flash':
        assert captured['client']['base_url'] == 'https://dashscope.aliyuncs.com/compatible-mode/v1'
        assert req['seed'] == 0
        assert req['extra_body'] == {'enable_thinking':False, 'top_k':1}
    else:
        assert 'seed' not in req
