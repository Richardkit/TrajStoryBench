"""Regression coverage for repeated references and bounded native planning."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.agent.orchestra_handoffs import (PlannerEvidenceState, role_input, execution_artifact,
                                         write, dispatch_key, task_hint)
from src.agent.orchestra_roles import OrchestraInterpreterAgent, OrchestraVerifierAgent
from src.interaction.bus import AgentBus
from src.session import SessionContext
from src.task import Task

E = 'E_' + 'a' * 32
R = 'R_' + 'b' * 32


def test_repeated_handoffs_do_not_repeat_ids_quotes_or_failure_text():
    state = PlannerEvidenceState()
    task = json.dumps(dict(instruction='Compare groups', evidence_ids=[E]))
    packet = dict(status='ready', evidence_id=E,
                  index=[dict(step=2, stdout_chunks=2, code_chunks=1,
                              preview='PRIVATE_STDOUT' * 1000)])
    error = json.dumps(dict(detail='Citation must exactly match supplied execution evidence'))
    for _ in range(100):
        state.observe('orchestra_analyst', 'Compare groups', True, json.dumps(packet))
        state.observe('orchestra_interpreter', task, False, error)
    text = state.render(6000)
    value = json.loads(text)
    assert text.count(E) == 1
    assert 'PRIVATE_STDOUT' not in text
    assert value['recent_errors'][0]['attempts'] == 100
    assert len(value['recent_errors']) == 1
    assert value['artifacts'][0]['steps'] == [[2, 2, 1]]
    assert len(text) < 1000


def test_context_budget_drops_hints_but_preserves_all_reference_ids():
    state = PlannerEvidenceState()
    identifiers = ['R_' + f'{i:032x}' for i in range(35)]
    for identifier in identifiers:
        state.observe('orchestra_interpreter', 'Long descriptive topic ' * 20, True,
                      json.dumps(dict(result_id=identifier, count=1)))
    text = state.render(4000)
    assert len(text) <= 4000
    assert all(text.count(identifier) == 1 for identifier in identifiers)
    assert 'topic' not in text
    with pytest.raises(ValueError, match='index exceeds'):
        state.render(100)


def test_latest_synthesis_is_available_once_without_previous_reports():
    state = PlannerEvidenceState()
    for i in range(50):
        state.observe('orchestra_interpreter', 'A previous question', True,
                      json.dumps(dict(result_id='R_'+f'{i:032x}', count=1)))
    state.observe('orchestra_synthesizer', '', True,
                  json.dumps(dict(insights=['OLD_FINDING'], summary='old')))
    final = dict(insights=['FINAL_FINDING'], summary='final')
    for _ in range(20):
        state.observe('orchestra_synthesizer', '', True, json.dumps(final))
    text = state.render(6000)
    assert text.count('FINAL_FINDING') == 1 and 'OLD_FINDING' not in text
    assert json.loads(text)['latest_synthesis'] == final
    assert json.loads(text)['ready_artifact_count'] == 50
    assert 'artifacts' not in json.loads(text)


def test_new_evidence_restores_index_and_invalidates_old_synthesis():
    state = PlannerEvidenceState()
    packet = json.dumps(dict(evidence_id=E, successful_steps=[1, 4],
                             index=[dict(step=4, stdout_chunks=1, code_chunks=1)]))
    state.observe('orchestra_analyst', 'Compute speed by corridor', True, packet)
    state.observe('orchestra_synthesizer', '', True,
                  json.dumps(dict(insights=['OLD_REPORT'], summary='old')))
    # Reusing the very same result does not invalidate a report.
    state.observe('orchestra_analyst', 'Compute speed by corridor', True, packet)
    assert not state.synthesis_stale
    state.observe('orchestra_interpreter', 'Interpret new findings', True,
                  json.dumps(dict(result_id=R, count=2)))
    value = json.loads(state.render(6000))
    assert state.synthesis_stale
    assert {entry['id'] for entry in value['artifacts']} == {E, R}
    assert value['artifacts'][0]['preferred_steps'] == [1, 4]
    assert 'latest_synthesis' not in value and 'OLD_REPORT' not in state.render(6000)
    state.observe('orchestra_synthesizer', '', True,
                  json.dumps(dict(insights=['NEW_REPORT'], summary='new')))
    assert not state.synthesis_stale
    assert json.loads(state.render(6000))['latest_synthesis']['insights'] == ['NEW_REPORT']


def test_oversize_synthesis_raises_budget_error_not_missing_index_key():
    state = PlannerEvidenceState()
    state.observe('orchestra_synthesizer', '', True,
                  json.dumps(dict(insights=['x' * 3000], summary='summary')))
    with pytest.raises(ValueError, match='context budget'):
        state.render(1024)


def test_same_round_synthesis_cannot_hide_parallel_new_evidence():
    for synthesis_first in (False, True):
        state = PlannerEvidenceState()
        events = [
            ('orchestra_analyst', dict(evidence_id=E)),
            ('orchestra_synthesizer', dict(insights=['report'], summary='summary'))]
        if synthesis_first:
            events.reverse()
        for role, payload in events:
            state.observe(role, '', True, json.dumps(payload), synthesis_basis=())
        assert state.synthesis_stale
        assert json.loads(state.render(6000))['artifacts'][0]['id'] == E


def test_dispatch_keys_only_merge_equivalent_inputs_and_hints_keep_question():
    first = json.dumps(dict(instruction='Interpret speed', evidence_ids=[E, E, R]))
    second = json.dumps(dict(evidence_ids=[R, E], instruction='Interpret speed'))
    assert dispatch_key('interpreter', first) == dispatch_key('interpreter', second)
    assert dispatch_key('interpreter', first) != dispatch_key('verifier', first)
    assert dispatch_key('analyst', 'speed A') != dispatch_key('analyst', 'speed B')
    assert dispatch_key('analyst', 'speed', ['a.csv']) != dispatch_key('analyst', 'speed', ['b.csv'])
    task = r'Using trajstory_analysis with spec_path B:\results\long_case\case_input.json, compute speed by corridor'
    assert task_hint(task) == 'compute speed by corridor'


def test_verifier_receives_quoted_evidence_once_as_a_source_and_span(tmp_path):
    body = 'Header\nOne exact contiguous quote.\nFooter'
    quote = 'One exact contiguous quote.'
    execution_artifact(tmp_path, E, [dict(step=1, args={'code': 'print(data)'}, observation=body)])
    payload = dict(candidates=[dict(insight='A finding', citations=[
        dict(evidence_id=E+'/S1/C1', quote=quote)])])
    write(tmp_path/(R+'.json'), dict(role='orchestra_interpreter', payload=payload,
                                   source_ids=[E+'/S1/C1']))
    text, _ = role_input(tmp_path, json.dumps(dict(instruction='Verify', evidence_ids=[R])),
                         'orchestra_verifier')
    citation = json.loads(text)['upstream_results'][R]['candidates'][0]['citations'][0]
    assert text.count(quote) == 1 and 'quote' not in citation
    start, end = citation['quote_span']
    assert body[start:end] == quote
    assert json.loads((tmp_path/(R+'.json')).read_text(encoding='utf8'))['payload'] == payload


@pytest.mark.parametrize('role_class,key', [
    (OrchestraInterpreterAgent, 'candidates'), (OrchestraVerifierAgent, 'reviews')])
def test_bare_arrays_are_wrapped_but_invalid_quotes_still_fail(tmp_path, role_class, key):
    from src.agent.orchestra_handoffs import store_for
    agent = role_class(workdir=str(tmp_path/'agent'/role_class.__name__), model_name='test')
    ctx = SessionContext()
    store = store_for(agent, ctx)
    execution_artifact(store, E, [dict(step=1, args={'code': 'print(42)'}, observation='N=42')])
    write(store/(R+'.json'), dict(role='orchestra_interpreter', source_ids=[E+'/S1/C1'],
        payload=dict(candidates=[dict(insight='N is 42', citations=[dict(evidence_id=E+'/S1/C1', quote='N=42')])])))
    item = dict(insight='N is 42', citations=[dict(evidence_id=E+'/S1/C1', quote='N=42')])
    if key == 'reviews': item['status'] = 'supported'
    answers = [SimpleNamespace(success=True, message=json.dumps([item]), extra=None)]
    bad = dict(item, citations=[dict(evidence_id=E+'/S1/C1', quote='N=43')])
    answers.append(SimpleNamespace(success=True, message=json.dumps([bad]), extra=None))
    model = AsyncMock(side_effect=answers)
    request = json.dumps(dict(instruction='Interpret or review', evidence_ids=[E if key=='candidates' else R]))
    async def run():
        with patch('src.agent.orchestra_roles.model_manager', new=model):
            first = await agent(task=request, ctx=ctx)
            second = await agent(task=request, ctx=ctx)
        assert first.success and not second.success
        assert 'exactly match' in second.message
    asyncio.run(run())


@pytest.mark.parametrize('indexed', [False, True])
def test_native_multi_round_dispatch_receives_bounded_state_only_when_enabled(indexed):
    bus = AgentBus()
    bus.register_agent('orchestra_interpreter', 'Interpret evidence')
    histories = []
    calls = []
    ids = ['R_'+f'{i:032x}' for i in range(3)]
    async def planner(**kwargs):
        histories.append(kwargs['execution_history'])
        if kwargs['round_number'] == (3 if indexed else 10):
            return dict(is_done=True, final_result='finished', dispatches=[])
        return dict(is_done=False, plan_update='OLD_VERBOSE_PLAN ' * 80, dispatches=[
            dict(agent_name='orchestra_interpreter', task=f'Question {i}', files=[])
            for i in range(3)])
    async def dispatch(name, message, ctx=None):
        question = message.payload['content'];calls.append(question)
        identifier = ids[int(question[-1])]
        return SimpleNamespace(payload=dict(success=True, result=json.dumps(dict(
            result_id=identifier, count=1, previews=['REPEATED_BODY ' * 1000]))))
    async def run():
        with patch.object(bus, '_call_planner_raw', new=planner), \
             patch.object(bus, '_call_agent', new=dispatch):
            response = await bus.submit(Task(content='Focused analysis'), max_rounds=10,
                history_result_max_chars=5000,
                planner_context_max_chars=6000 if indexed else None)
        await bus.shutdown()
        assert response.payload['success']
    asyncio.run(run())
    assert len(calls) == (3 if indexed else 27)
    if indexed:
        assert max(map(len, histories)) <= 6000
        assert all(histories[-1].count(identifier) == 1 for identifier in ids)
        assert 'REPEATED_BODY' not in histories[-1] and 'OLD_VERBOSE_PLAN' not in histories[-1]
        assert json.loads(histories[-1])['reused_dispatches'] == 3
    else:
        assert len(histories[-1]) > 24000  # Legacy callers remain unchanged.


def test_indexed_bus_coalesces_parallel_duplicates_and_stops_unchanged_loop():
    bus = AgentBus()
    bus.register_agent('orchestra_interpreter', 'Interpret evidence')
    calls = []
    async def planner(**kwargs):
        return dict(is_done=False, dispatches=[dict(agent_name='orchestra_interpreter',
                    task='Same question', files=[]) for _ in range(3)])
    async def dispatch(name, message, ctx=None):
        calls.append(message.payload['content'])
        await asyncio.sleep(0)
        return SimpleNamespace(payload=dict(success=True, result=json.dumps(dict(result_id=R, count=1))))
    async def run():
        with patch.object(bus, '_call_planner_raw', new=planner), patch.object(bus, '_call_agent', new=dispatch):
            response = await bus.submit(Task(content='Analysis'), max_rounds=10, planner_context_max_chars=6000)
        await bus.shutdown()
        assert not response.payload['success']
        assert 'repeated unchanged dispatches' in response.payload['error']
    asyncio.run(run())
    assert calls == ['Same question']


@pytest.mark.parametrize('deterministic', [False, True])
def test_only_deterministic_handoff_failures_are_cached(deterministic):
    bus = AgentBus()
    bus.register_agent('orchestra_interpreter', 'Interpret evidence')
    calls = []
    async def planner(**kwargs):
        return dict(is_done=kwargs['round_number'] == 3, final_result='finished',
                    dispatches=[] if kwargs['round_number'] == 3 else [dict(
                        agent_name='orchestra_interpreter', task='Interpret', files=[])])
    async def dispatch(name, message, ctx=None):
        calls.append(name)
        detail = 'Invalid evidence ID: bad' if deterministic else 'Model request failed; see raw_response.json'
        return SimpleNamespace(payload=dict(success=False, result=json.dumps(dict(detail=detail))))
    async def run():
        with patch.object(bus, '_call_planner_raw', new=planner), patch.object(bus, '_call_agent', new=dispatch):
            await bus.submit(Task(content='Analysis'), max_rounds=3, planner_context_max_chars=6000)
        await bus.shutdown()
    asyncio.run(run())
    assert len(calls) == (1 if deterministic else 2)


def test_dispatch_reuse_is_case_scoped():
    bus = AgentBus()
    bus.register_agent('orchestra_interpreter', 'Interpret evidence')
    calls = []
    async def planner(**kwargs):
        return dict(is_done=kwargs['round_number'] == 2, final_result='finished',
                    dispatches=[] if kwargs['round_number'] == 2 else [dict(
                        agent_name='orchestra_interpreter', task='Interpret', files=[])])
    async def dispatch(name, message, ctx=None):
        calls.append(ctx.id)
        return SimpleNamespace(payload=dict(success=True, result=json.dumps(dict(result_id=R, count=1))))
    async def run():
        with patch.object(bus, '_call_planner_raw', new=planner), patch.object(bus, '_call_agent', new=dispatch):
            for _ in range(2):
                await bus.submit(Task(content='Analysis'), max_rounds=2, planner_context_max_chars=6000)
        await bus.shutdown()
    asyncio.run(run())
    assert len(calls) == len(set(calls)) == 2


def test_planner_cannot_finish_with_stale_synthesis_after_new_computation():
    bus = AgentBus()
    for role in ('orchestra_analyst', 'orchestra_synthesizer'):
        bus.register_agent(role, role)
    histories = []
    calls = []
    async def planner(**kwargs):
        round_number = kwargs['round_number']
        histories.append(json.loads(kwargs['execution_history']))
        if round_number in (3, 5):
            return dict(is_done=True, final_result='old' if round_number == 3 else 'new', dispatches=[])
        role = 'orchestra_analyst' if round_number == 2 else 'orchestra_synthesizer'
        return dict(is_done=False, dispatches=[dict(agent_name=role, task=f'Round {round_number}', files=[])])
    async def dispatch(name, message, ctx=None):
        calls.append(name)
        payload = dict(evidence_id=E) if name == 'orchestra_analyst' else dict(
            insights=['old' if len(calls) == 1 else 'new'], summary='report')
        return SimpleNamespace(payload=dict(success=True, result=json.dumps(payload)))
    async def run():
        with patch.object(bus, '_call_planner_raw', new=planner), patch.object(bus, '_call_agent', new=dispatch):
            response = await bus.submit(Task(content='Analysis'), max_rounds=5, planner_context_max_chars=6000)
        await bus.shutdown()
        assert response.payload['success'] and response.payload['result'] == 'new'
    asyncio.run(run())
    assert len(histories) == 5
    assert histories[2]['artifacts'][0]['id'] == E
    assert 'latest_synthesis' not in histories[2]
    assert histories[4]['latest_synthesis']['insights'] == ['new']


def test_reusing_cached_synthesis_cannot_approve_unprocessed_new_evidence():
    bus = AgentBus()
    for role in ('orchestra_analyst', 'orchestra_synthesizer'):
        bus.register_agent(role, role)
    calls = []
    histories = []
    async def planner(**kwargs):
        number = kwargs['round_number']
        histories.append(json.loads(kwargs['execution_history']))
        if number in (3, 5, 6):
            return dict(is_done=True, final_result='old', dispatches=[])
        role = 'orchestra_analyst' if number == 2 else 'orchestra_synthesizer'
        return dict(is_done=False, dispatches=[dict(agent_name=role, task='Same task', files=[])])
    async def dispatch(name, message, ctx=None):
        calls.append(name)
        item = dict(evidence_id=E) if name == 'orchestra_analyst' else dict(insights=['old'], summary='old')
        return SimpleNamespace(payload=dict(success=True, result=json.dumps(item)))
    async def run():
        with patch.object(bus, '_call_planner_raw', new=planner), patch.object(bus, '_call_agent', new=dispatch):
            response = await bus.submit(Task(content='Analysis'), max_rounds=5, planner_context_max_chars=6000)
        await bus.shutdown()
        assert not response.payload['success']
    asyncio.run(run())
    assert calls == ['orchestra_synthesizer', 'orchestra_analyst']
    assert 'synthesis_status' in histories[4] and 'latest_synthesis' not in histories[4]


def test_failed_handoff_is_rechecked_when_evidence_becomes_available():
    bus = AgentBus()
    for role in ('orchestra_analyst', 'orchestra_interpreter'):
        bus.register_agent(role, role)
    interpreter_calls = 0
    async def planner(**kwargs):
        number = kwargs['round_number']
        role = 'orchestra_analyst' if number == 2 else 'orchestra_interpreter'
        return dict(is_done=number == 4, final_result='done', dispatches=[] if number == 4 else [
            dict(agent_name=role, task='Same interpretation' if number != 2 else 'Compute', files=[])])
    async def dispatch(name, message, ctx=None):
        nonlocal interpreter_calls
        if name == 'orchestra_analyst':
            return SimpleNamespace(payload=dict(success=True, result=json.dumps(dict(evidence_id=E))))
        interpreter_calls += 1
        payload = dict(detail='Unknown evidence ID in this case: '+E) if interpreter_calls == 1 else dict(result_id=R, count=1)
        return SimpleNamespace(payload=dict(success=interpreter_calls > 1, result=json.dumps(payload)))
    async def run():
        with patch.object(bus, '_call_planner_raw', new=planner), patch.object(bus, '_call_agent', new=dispatch):
            response = await bus.submit(Task(content='Analysis'), max_rounds=4, planner_context_max_chars=6000)
        await bus.shutdown()
        assert response.payload['success']
    asyncio.run(run())
    assert interpreter_calls == 2
