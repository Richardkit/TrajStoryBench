"""Offline checks of the real model/bus boundary, including parallel dispatch."""
import asyncio
import importlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.model.manager import ModelManager
from src.model.request_budget import (
    ModelRequestBudget, ModelRequestBudgetExceeded, SampleNetworkFailureLimitExceeded, current_request_budget,
    invoke_with_request_budget, request_budget_scope, request_role,
)


def manager_with(client):
    manager = ModelManager()
    manager.models['test'] = SimpleNamespace(model_type='chat/completions',
                                            fallback_model=None, provider='openai')
    manager.model_clients['test'] = client
    return manager


@pytest.mark.parametrize('limit', [0, -1, True, 1.5])
def test_invalid_limits_are_rejected(limit):
    with pytest.raises(ValueError, match='positive integer'):
        ModelRequestBudget(limit)


def test_parallel_roles_share_83_slots_and_nonnetwork_failures_are_counted():
    sent = []

    async def client(**kwargs):
        sent.append(kwargs['messages'][0])
        await asyncio.sleep(0)
        return SimpleNamespace(success=kwargs['messages'][0] % 3 != 0, extra=None)

    manager = manager_with(client)
    budget = ModelRequestBudget(83)

    async def role(name, number):
        with request_role(name):
            return await manager(model='test', messages=[number])

    async def run():
        with request_budget_scope(budget):
            return await asyncio.gather(*(role('analyst' if n % 2 else 'planner', n)
                                          for n in range(120)), return_exceptions=True)

    results = asyncio.run(run())
    assert len(sent) == 83
    assert sum(isinstance(r, ModelRequestBudgetExceeded) for r in results) == 37
    usage = budget.snapshot()
    assert usage['requests_started'] == 83 and usage['requests_remaining'] == 0
    assert usage['requests_succeeded'] + usage['requests_failed'] == 83
    assert usage['requests_failed'] > 0 and usage['requests_in_flight'] == 0
    assert sum(usage['by_role'].values()) == 83
    assert current_request_budget() is None


@pytest.mark.parametrize('limit', [1, 2])
def test_fallback_is_counted_and_cannot_bypass_exhaustion(limit):
    calls = []

    async def primary(**kwargs):
        calls.append('primary')
        raise RuntimeError('connection failed')

    async def fallback(**kwargs):
        calls.append('fallback')
        return SimpleNamespace(success=True, extra=None)

    manager = manager_with(primary)
    manager.models['test'].fallback_model = 'fallback'
    manager.models['fallback'] = SimpleNamespace(model_type='chat/completions')
    manager.model_clients['fallback'] = fallback
    budget = ModelRequestBudget(limit)

    async def run():
        with request_budget_scope(budget):
            return await manager(model='test', messages=['test'])

    if limit == 1:
        with pytest.raises(ModelRequestBudgetExceeded):
            asyncio.run(run())
        assert calls == ['primary']
    else:
        assert asyncio.run(run()).success
        assert calls == ['primary', 'fallback']
    assert budget.snapshot()['requests_started'] == limit


def test_sdk_retries_disabled_only_inside_budgeted_case():
    from src.model.openai.chat import ChatOpenAI
    client = ChatOpenAI(model='gpt-6-luna', api_key='offline-test', max_retries=5)
    with patch('src.model.openai.chat.AsyncOpenAI', side_effect=lambda **kwargs: kwargs):
        assert client.get_client()['max_retries'] == 5
        with request_budget_scope(ModelRequestBudget(83)):
            assert client.get_client()['max_retries'] == 0
        assert client.get_client()['max_retries'] == 5


def run_bus_cases(analyst_calls, *, case_count=1, max_rounds=3):
    module = importlib.import_module('src.interaction.bus')
    from src.interaction.bus import AgentBus
    from src.session import SessionContext
    from src.task import Task
    bus = AgentBus()
    roles = ['orchestra_analyst', 'orchestra_interpreter',
             'orchestra_verifier', 'orchestra_synthesizer']
    for role in roles:
        bus.register_agent(role)
    sent = []
    planner_rounds = {}

    async def client(**kwargs):
        sent.append(kwargs['messages'][0])
        await asyncio.sleep(0)
        return SimpleNamespace(success=True, message='ok', extra=None)

    manager = manager_with(client)

    async def acp(name, input, ctx, **kwargs):
        if name == 'planning':
            await manager(model='test', messages=[name])
            key = kwargs['task_id']
            planner_rounds[key] = planner_rounds.get(key, 0) + 1
            decision = (dict(is_done=True, final_result='finished', dispatches=[])
                        if planner_rounds[key] > 1 else
                        dict(is_done=False, dispatches=[dict(agent_name=r, task=r) for r in roles]))
            return SimpleNamespace(extra=SimpleNamespace(data={'decision': decision}))
        for _ in range(analyst_calls if name == 'orchestra_analyst' else 1):
            await manager(model='test', messages=[name])
        return SimpleNamespace(success=True, message='role complete')

    async def run():
        ctx = SessionContext()
        responses = []
        with patch.object(module, 'acp', new=acp):
            try:
                # Reuse one queue worker to check that its next task starts
                # with a new counter rather than inheriting the previous one.
                for _ in range(case_count):
                    responses.append(await bus.submit(Task(content='offline', session_id=ctx.id),
                        session_ctx=ctx, max_rounds=max_rounds, max_model_requests=83))
                await asyncio.sleep(0)
            finally:
                await bus.shutdown()
        return responses

    return asyncio.run(run()), sent


def test_last_allowed_request_can_complete_and_next_case_gets_fresh_budget():
    responses, sent = run_bus_cases(78, case_count=2)
    assert len(sent) == 166
    for response in responses:
        assert response.payload['success']
        usage = response.payload['request_budget']
        assert usage['limit'] == usage['requests_started'] == 83
        assert usage['requests_succeeded'] == 83
        assert usage['requests_in_flight'] == usage['blocked_requests'] == 0
        assert usage['by_role'] == dict(planning=2, orchestra_analyst=78,
            orchestra_interpreter=1, orchestra_verifier=1, orchestra_synthesizer=1)


@pytest.mark.parametrize('analyst_calls', [79, 120])
def test_planner_or_parallel_analyst_cannot_send_request_84(analyst_calls):
    responses, sent = run_bus_cases(analyst_calls)
    response = responses[0]
    assert len(sent) == 83
    assert not response.payload['success']
    assert response.payload['error_type'] == 'ModelRequestBudgetExceeded'
    assert response.payload['request_budget']['requests_started'] == 83
    assert response.payload['request_budget']['requests_in_flight'] == 0


def test_round_limit_does_not_make_unreported_finalization_request():
    responses, sent = run_bus_cases(1, max_rounds=1)
    assert not responses[0].payload['success']
    assert len(sent) == responses[0].payload['request_budget']['requests_started'] == 5


def test_concurrent_case_scopes_are_isolated():
    async def client(**kwargs):
        await asyncio.sleep(0)
        return SimpleNamespace(success=True)

    async def case(limit):
        budget = ModelRequestBudget(limit)
        with request_budget_scope(budget):
            results = await asyncio.gather(*(invoke_with_request_budget(client, 'test')
                                              for _ in range(100)), return_exceptions=True)
        return budget.snapshot(), results

    async def run():
        return await asyncio.gather(case(83), case(7))

    for (usage, results), expected in zip(asyncio.run(run()), [83, 7]):
        assert usage['requests_started'] == expected
        assert sum(not isinstance(r, BaseException) for r in results) == expected


def test_network_failures_are_excluded_and_success_resets_failure_streak():
    attempts = []
    budget = ModelRequestBudget(2)

    async def client(**kwargs):
        attempts.append(1)
        if len(attempts) % 5:
            raise ConnectionError('offline simulated connection drop')
        return SimpleNamespace(success=True, extra=None)

    async def run():
        with request_budget_scope(budget):
            for _ in range(2):
                assert (await invoke_with_request_budget(client, 'test')).success
            with pytest.raises(ModelRequestBudgetExceeded):
                await invoke_with_request_budget(client, 'test')

    asyncio.run(run())
    usage = budget.snapshot()
    assert len(attempts) == usage['requests_started'] == 10
    assert usage['requests_used'] == usage['requests_succeeded'] == 2
    assert usage['network_failures'] == 8
    assert usage['consecutive_network_failures'] == 0
    assert not usage['network_failure_stop']


@pytest.mark.parametrize('wrapped_response', [False, True])
def test_five_consecutive_network_failures_stop_without_sixth_attempt(wrapped_response):
    attempts = []
    budget = ModelRequestBudget(83)

    async def client(**kwargs):
        attempts.append(1)
        if wrapped_response:
            return SimpleNamespace(success=False, extra=SimpleNamespace(data={'network_failure': True}))
        raise TimeoutError('offline simulated timeout')

    async def run():
        with request_budget_scope(budget):
            with pytest.raises(SampleNetworkFailureLimitExceeded):
                await invoke_with_request_budget(client, 'test')
            with pytest.raises(SampleNetworkFailureLimitExceeded):
                await invoke_with_request_budget(client, 'test')

    asyncio.run(run())
    usage = budget.snapshot()
    assert len(attempts) == usage['network_failures'] == 5
    assert usage['requests_used'] == 0 and usage['requests_remaining'] == 83
    assert usage['network_failure_stop'] and usage['consecutive_network_failures'] == 5


def test_inflight_network_failure_frees_last_slot_for_parallel_request():
    budget = ModelRequestBudget(1)
    attempts = []

    async def client(**kwargs):
        attempts.append(1)
        n = len(attempts)
        await asyncio.sleep(0)
        if n == 1:
            raise ConnectionError('offline first attempt failed')
        return SimpleNamespace(success=True, extra=None)

    async def run():
        with request_budget_scope(budget):
            return await asyncio.gather(*(invoke_with_request_budget(client, 'test')
                                          for _ in range(2)), return_exceptions=True)

    results = asyncio.run(run())
    assert len(attempts) == 2
    assert sum(isinstance(r, ModelRequestBudgetExceeded) for r in results) == 1
    usage = budget.snapshot()
    assert usage['requests_used'] == 1 and usage['network_failures'] == 1


def test_bus_reports_network_stop_and_next_sample_can_run():
    module = importlib.import_module('src.interaction.bus')
    from src.interaction.bus import AgentBus
    from src.session import SessionContext
    from src.task import Task
    bus = AgentBus()
    attempts = []

    async def client(**kwargs):
        attempts.append(1)
        if len(attempts) <= 5:
            raise ConnectionError('offline connection failure')
        return SimpleNamespace(success=True, extra=None)

    manager = manager_with(client)

    async def acp(name, input, ctx, **kwargs):
        await manager(model='test', messages=['offline'])
        return SimpleNamespace(extra=SimpleNamespace(data={'decision': dict(
            is_done=True, final_result='finished', dispatches=[])}))

    async def run():
        ctx = SessionContext()
        with patch.object(module, 'acp', new=acp):
            try:
                return [await bus.submit(Task(content='offline', session_id=ctx.id),
                    session_ctx=ctx, max_model_requests=83) for _ in range(2)]
            finally:
                await bus.shutdown()

    first, second = asyncio.run(run())
    assert first.payload['error_type'] == 'SampleNetworkFailureLimitExceeded'
    assert first.payload['request_budget']['requests_used'] == 0
    assert first.payload['request_budget']['network_failures'] == 5
    assert second.payload['success']
    assert second.payload['request_budget']['requests_used'] == 1
    assert len(attempts) == 6
