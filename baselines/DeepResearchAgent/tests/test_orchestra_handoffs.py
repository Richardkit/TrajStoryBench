import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import asyncio
import pytest

from src.agent.orchestra_handoffs import (execution_artifact, role_input, store_for,
    write, validate_result)
from src.agent.orchestra_roles import OrchestraInterpreterAgent
from src.session import SessionContext

E='E_'+'a'*32
R='R_'+'b'*32


def task(refs):return json.dumps(dict(instruction='Interpret the selected evidence',evidence_ids=refs))


def test_large_log_has_small_index_and_lossless_chunks(tmp_path):
    output='N=42\n'+'raw numeric detail\n'*4000
    packet=execution_artifact(tmp_path,E,[dict(step=1,args={'code':'long code\n'*2000},observation=output)])
    assert len(json.dumps(packet))<1000
    saved=json.loads((tmp_path/(E+'.json')).read_text(encoding='utf8'))
    assert ''.join(saved['steps'][0]['stdout'])==output
    text,sources=role_input(tmp_path,task([E+'/S1/C1']), 'orchestra_interpreter')
    assert len(text)<2500
    assert sources[E+'/S1/C1']==output[:1600]
    assert 'long code' not in text
    with pytest.raises(ValueError,match='exceeds 12000'):
        role_input(tmp_path,task([E]), 'orchestra_interpreter')


def test_verifier_sources_deduplicated_and_synthesis_has_no_logs(tmp_path):
    execution_artifact(tmp_path,E,[dict(step=1,args={'code':'print(42)'},observation='N=42')])
    payload=dict(candidates=[dict(insight='N is 42',citations=[dict(evidence_id=E+'/S1/C1',quote='N=42')])])
    write(tmp_path/(R+'.json'),dict(role='orchestra_interpreter',payload=payload,source_ids=[E+'/S1/C1']))
    text,sources=role_input(tmp_path,task([R,R,E+'/S1/C1']), 'orchestra_verifier')
    assert len(sources)==1
    assert len(json.loads(text)['upstream_results'])==1
    write(tmp_path/(R+'.json'),dict(role='orchestra_verifier',payload=dict(reviews=[
        dict(status='supported',insight='N is 42',citations=[]),
        dict(status='rejected',insight='',citations=[])]),source_ids=[E+'/S1/C1']))
    text,sources=role_input(tmp_path,task([R]), 'orchestra_synthesizer')
    assert sources=={}
    assert len(json.loads(text)['upstream_results'][R]['reviews'])==1


def test_citations_are_exact_and_failed_steps_stay_out_of_inputs(tmp_path):
    execution_artifact(tmp_path,E,[dict(step=1,args={'code':'broken()'},observation='TrajStory analysis error: BAD'),
        dict(step=2,args={'code':'print(42)'},observation='N=42')])
    text,sources=role_input(tmp_path,task([E]), 'orchestra_interpreter')
    assert 'BAD' not in text and 'broken()' not in text
    payload=dict(candidates=[dict(insight='N is 43',citations=[dict(evidence_id=E+'/S2/C1',quote='N=43')])])
    with pytest.raises(ValueError,match='exactly match'):
        validate_result('orchestra_interpreter',payload,sources)


def test_other_cases_and_inline_logs_rejected_before_model_call(tmp_path):
    agent=OrchestraInterpreterAgent(workdir=str(tmp_path/'agent/orchestra_interpreter'),model_name='test')
    first,second=SessionContext(),SessionContext()
    execution_artifact(store_for(agent,first),E,[dict(step=1,args={'code':'print(42)'},observation='N=42')])
    model=AsyncMock(return_value=SimpleNamespace(success=True,message='{"candidates":[]}',extra=None))
    async def run():
        with patch('src.agent.orchestra_roles.model_manager',new=model):
            result=await agent(task=task([E]),ctx=second)
            assert not result.success
            result=await agent(task='full code and stdout '*4000,ctx=first)
            assert not result.success
    asyncio.run(run())
    assert model.await_count==0


def test_identical_content_under_different_ids_sent_only_once(tmp_path):
    execution_artifact(tmp_path,E,[dict(step=1,args={'code':'print(42)'},observation='N=42'),
        dict(step=2,args={'code':'print(42)'},observation='N=42')])
    text,sources=role_input(tmp_path,task([E]),'orchestra_interpreter')
    value=json.loads(text)
    assert len(value['evidence'])==1 and len(value['evidence_aliases'])==1
    assert len(sources)==2  # Both provenances remain valid for citations.
    assert text.count('N=42')==1


def test_repeated_results_and_citations_deduplicated_without_merging_distinct_scopes():
    cite=dict(evidence_id=E+'/S1/C1',quote='N=42')
    first=dict(insight='Group A has 42',citations=[cite,cite])
    second=dict(insight='Group B has 42',citations=[cite])
    payload=dict(candidates=[first,dict(first),second])
    validate_result('orchestra_interpreter',payload,{E+'/S1/C1':'N=42'})
    assert len(payload['candidates'])==2
    assert len(payload['candidates'][0]['citations'])==1
    output=dict(insights=['Same finding','Same finding','Different finding'],summary='Summary')
    validate_result('orchestra_synthesizer',output,{})
    assert output['insights']==['Same finding','Different finding']


def test_model_request_limits_stop_oversize_before_call():
    from src.agent.orchestra_handoffs import model_limits
    agent=SimpleNamespace(model_input_char_limit=100,model_output_token_limit=50)
    assert model_limits(agent,[SimpleNamespace(content='small')])=={'max_completion_tokens':50}
    with pytest.raises(ValueError,match='exceeds 100'):
        model_limits(agent,[SimpleNamespace(content='x'*101)])
