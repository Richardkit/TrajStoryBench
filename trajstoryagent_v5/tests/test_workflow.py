"""Formal v5 test workflow implementation."""
import ast
import inspect
import json
import textwrap
from types import SimpleNamespace

import pytest

from trajstoryagent_v5.json_utils import write_json
from trajstoryagent_v5.models import (
    AnalysisQuestion, DataBundle, DataFile, EvidencePackage,
    TrajectoryDataProfile,
)
from trajstoryagent_v5.reasoning_agents import CrossEvidenceExplorer, GoalDecomposer
from trajstoryagent_v5 import FrameworkConfig, TrajStoryAgent
from trajstoryagent_v5.models import BranchInsight, TrajStoryReport


class ScriptedLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, *, system, user, **kwargs):
        self.calls.append({'system': system, 'user': user})
        assert self.responses, 'Unexpected extra model call'
        return json.dumps(self.responses.pop(0))


def candidates(prefix):
    return {'candidates': [dict(
        question=f'{prefix} comparison {i}?', expected_new_information=f'Missing comparison {i}',
        analysis_granularity={'base': 'trajectory/trip', 'aggregation': None},
        related_evidence_ids=['EQ1'], relation_ids=[],
    ) for i in range(1, 4)]}


def test_formal_initialization_never_constructs_dpp_or_embedding(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Unexpected selector or encoder')
    agent = TrajStoryAgent(None, generator='gpt', refiner=None)
    assert isinstance(agent.explorer, CrossEvidenceExplorer)
    assert type(agent.decomposer) is GoalDecomposer
    assert agent.explorer.coverage_review_enabled
    assert not hasattr(agent, 'selector')
    assert not {'embedding', 'selection', 'initial_dpp_enabled'} & agent.config.to_dict().keys()
    assert agent.config.memory_manager_enabled
    assert agent.config.parallel_workers == 3
    assert agent.config.code_timeout_seconds == 180
    assert agent.config.min_support == 1



@pytest.mark.parametrize('override', [{'memory_manager_enabled': False}, {'coverage_review_enabled': False}])
def test_required_policy_cannot_be_silently_disabled(override):
    with pytest.raises(ValueError):
        FrameworkConfig(**override)


def test_planner_receives_first_round_refinement_and_regenerates(tmp_path, monkeypatch):
    llm = ScriptedLLM([
        candidates('First'),
        {'duplicates': [], 'coverage_issues': [{'follow_up_id': 'F2', 'reason': 'Missing goal aspect'}]},
        candidates('Revised'),
        {'duplicates': [], 'coverage_issues': []},
    ])
    calls, memory_claims = [], []

    def refine(slug, evidence, bundle, profile, directory):
        qid = evidence['question_id']
        calls.append(qid)
        assert bundle['context'] == 'Original context'
        status = {'Q2': 'declined', 'F3': 'failed'}.get(qid, 'applied')
        claim = 'refined ' + qid if status == 'applied' else evidence['insight']
        write_json(directory/'conditional_compressed/facts.json', [f'accepted fact {qid}'])
        return {'question_id': qid, 'insights': {a: claim for a in ('conditional_only', 'conditional_compressed')},
                'arm_status': {'conditional_compressed': status}}

    agent = TrajStoryAgent(llm, generator='gpt', refiner=refine,
                          config=FrameworkConfig(agreement_enabled=False))
    questions = [AnalysisQuestion(f'Q{i}', f'Initial {i}?') for i in range(1, 4)]
    agent.profiler = SimpleNamespace(build=lambda bundle: TrajectoryDataProfile([], []))
    agent.decomposer = SimpleNamespace(decompose=lambda bundle, profile: questions)

    def evidence(q):
        qid = getattr(q, 'question_id', None) or q.follow_up_id
        return EvidencePackage('E'+qid, qid, q.question, q.analysis_granularity, 'raw '+qid, [], [], {})

    def follow(**kw):
        assert calls == ['Q1', 'Q2', 'Q3']
        assert kw['follow_up'].question.startswith('Revised')
        assert kw['parent_evidence'][0].claim == 'raw Q1'
        assert 'FIRST-ROUND CONDITIONAL ANALYSIS SUPPLEMENT' in kw['bundle'].context
        return evidence(kw['follow_up'])

    agent.analysis.engine = SimpleNamespace(
        analyze_initial=lambda **kw: evidence(kw['question']), analyze_follow_up=follow)
    agent._verify_and_repair_initial = lambda **kw: kw['evidence']
    agent._verify_and_repair_follow_up = lambda **kw: kw['evidence']
    def summarize(**kw):
        assert calls == ['Q1', 'Q2', 'Q3', 'F1', 'F2', 'F3']
        assert [e.insight for e in kw['evidence']] == [
            'refined Q1', 'raw Q2', 'refined Q3', 'refined F1', 'refined F2', 'raw F3']
        assert (run/'predictions.json').exists()
        items = [BranchInsight(f'I{i}', e.question_id, e.insight, (e.evidence_id,))
                 for i,e in enumerate(kw['evidence'], 1)]
        return TrajStoryReport(items, 'A5 summary', kw['evidence'], kw['questions'], kw['follow_ups'])
    agent.synthesizer = SimpleNamespace(synthesize=summarize)

    def memory(_self, **kw):
        assert calls == ['Q1', 'Q2', 'Q3']
        memory_claims.extend(e.claim for e in kw['evidence'])
        return {'Question': ['Q1: initial objective'], 'Finding': list(memory_claims), 'Errors appear in coding': []}

    monkeypatch.setattr('trajstoryagent_v5.controller.MemoryManagerAgent.compress', memory)
    data = tmp_path/'data.csv'
    data.write_text('x\n1\n', encoding='utf-8')
    bundle = DataBundle('Compare trips', (DataFile('main', data),), context='Original context')
    agent.history.add('stale prior run')
    run = tmp_path/'run'
    report = agent.run(bundle, run)
    read = lambda name: json.loads((run/name).read_text(encoding='utf-8'))

    assert calls == ['Q1', 'Q2', 'Q3', 'F1', 'F2', 'F3']
    assert memory_claims == ['refined Q1', 'raw Q2', 'refined Q3']
    assert not llm.responses and len(llm.calls) == 4
    assert 'accepted fact Q1' in llm.calls[0]['user']
    assert 'accepted fact Q2' not in llm.calls[0]['user']
    assert all('accepted fact Q1' in call['user'] for call in llm.calls)
    assert 'refined Q1' in llm.calls[1]['user']
    assert len(read('04_duplicate_reviews.json')) == 2
    assert len(read('04_follow_up_questions.json')) == 3
    assert all(q['question'].startswith('Revised') for q in read('04_follow_up_questions.json'))
    assert [i.insight for i in report.insights] == [e.insight for e in report.evidence]
    assert all('claim' not in e.to_dict() for e in report.evidence)
    assert bundle.context == 'Original context'
    assert read('predictions.json')['insights']['conditional_compressed'] == [
        'refined Q1', 'raw Q2', 'refined Q3', 'refined F1', 'refined F2', 'raw F3']
    assert len(read('07_question_history.json')['questions']) == 6
    assert 'stale prior run' not in agent.history.snapshot()
    assert not list(run.rglob('*dpp*'))
    assert read('00_framework.json')['follow_up_dpp'] is False
    assert read('00_framework.json')['coverage_review_enabled'] is True
    assert read('predictions.json')['framework'] == 'v5_multi_role_insight_workflow'


def test_formal_refinement_precedes_summary_in_controller():
    """Guard the important stage ordering without any model calls."""
    source = inspect.getsource(TrajStoryAgent.run)
    assert source.index('final_evidence = self.refine_final(') < source.index('self.synthesizer.synthesize(')
    assert 'evidence=final_evidence' in source
    assert source.count('self.refine_final(') == 1



def test_runner_rejects_existing_outputs_before_loading_credentials(tmp_path):
    from trajstoryagent_v5.cli import main
    (tmp_path/'marker.txt').write_text('existing result', encoding='utf-8')
    with pytest.raises(FileExistsError, match='Refusing to overwrite'):
        main(['--case', 'missing.json', '--model', 'gpt', '--output', str(tmp_path)])
    assert (tmp_path/'marker.txt').read_text(encoding='utf-8') == 'existing result'
