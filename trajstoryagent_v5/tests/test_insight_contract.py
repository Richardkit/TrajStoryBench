"""Formal v5 test insight contract implementation."""
import json
import pytest
from trajstoryagent_v5.v5_framework.conditional_math import conditional_partition
from trajstoryagent_v5 import FrameworkConfig
from trajstoryagent_v5.v5_framework.models import EvidencePackage
from trajstoryagent_v5.v5_framework.result_interpretation import AnalysisResultInterpreter
from trajstoryagent_v5.v5_framework.summary import SummarySynthesizer
from trajstoryagent_v5.v5_framework.models import AnalysisGranularity


def test_singleton_condition_is_eligible_with_new_default():
    measure = dict(description='duration', unit='minutes', observation_unit='trip',
                   scope='all', conditions=['group'], cells=[
                       dict(condition={'group':'usual'}, n=100, mean=0., sse=0.),
                       dict(condition={'group':'exception'}, n=1, mean=100., sse=0.)])
    new = conditional_partition(measure, min_support=FrameworkConfig().min_support)
    old = conditional_partition(measure, min_support=2)
    assert new['split']=='group' and sorted(l['n'] for l in new['leaves'])==[1,100]
    assert old['split'] is None


def test_interpreter_requires_canonical_insight_field():
    payload=dict(insight='A computed exception.', justification='Measured values.',
                 limitations=[], evidence_strength='moderate')
    result=AnalysisResultInterpreter._parse(payload)
    assert result.insight==payload['insight']
    with pytest.raises(ValueError, match='insight'):
        AnalysisResultInterpreter._parse({**payload, 'insight':None, 'claim':'old field'})


def test_summary_does_not_regenerate_branch_insights():
    class LLM:
        def __init__(self):self.calls=[]
        def complete(self, **request):
            self.calls.append(request)
            return json.dumps({'summary':'An exception was observed.'})
    llm=LLM()
    evidence=EvidencePackage('E1','Q1','Which exception?',AnalysisGranularity("trajectory/trip"),
                             'The exception occurs once.',[],[],{})
    report=SummarySynthesizer(llm).synthesize(goal='Find exceptions',questions=[],follow_ups=[],evidence=[evidence])
    assert [i.insight for i in report.insights]==['The exception occurs once.']
    assert report.evidence[0].to_dict()['insight']==report.insights[0].insight
    assert 'claim' not in report.to_dict()['evidence'][0]
    assert 'The exception occurs once.' in llm.calls[0]['user']
    assert 'Return exactly {"summary":"..."}' in llm.calls[0]['user']


def test_summary_rejects_extra_generated_insights():
    class LLM:
        def complete(self, **request):
            return json.dumps({'summary':'Summary', 'insights':['new unrequested insight']})
    with pytest.raises(RuntimeError):
        SummarySynthesizer(LLM()).synthesize(goal='Goal',questions=[],follow_ups=[],evidence=[])
