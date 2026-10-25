import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
from pandas_multi_role.network_requests import (
    ConsecutiveNetworkFailures, NetworkRequestPolicy, NetworkRetryChatOpenAI,
)
from pandas_multi_role.workflow import PandasRoleWorkflow
from pandas_multi_role.role_runner import offline_factory
from baselines.scripts.pandas_usage_logging import ProviderUsageRecorder


class NetworkRequestTests(unittest.TestCase):
    def test_network_failures_retry_without_charging_request_and_reset_streak(self):
        policy = NetworkRequestPolicy(delay_seconds=0)
        attempts = []
        def operation():
            attempts.append(1)
            if len(attempts) < 5:
                raise ConnectionError()
            return 'ok'
        self.assertEqual(policy.run(operation), 'ok')
        self.assertEqual(policy.successful_requests, 1)
        self.assertEqual(policy.network_failures, 4)
        self.assertEqual(policy.consecutive_failures, 0)
        policy.run(lambda: 'next')
        self.assertEqual(policy.successful_requests, 2)

    def test_fifth_failure_stops_case_and_no_further_attempts(self):
        policy = NetworkRequestPolicy(delay_seconds=0)
        attempts = []
        def fail():
            attempts.append(1)
            raise TimeoutError()
        for _ in range(2):
            with self.assertRaises(ConsecutiveNetworkFailures):policy.run(fail)
        self.assertEqual(len(attempts), 5)
        self.assertEqual(policy.successful_requests, 0)
        with tempfile.TemporaryDirectory() as tmp:
            recorder = ProviderUsageRecorder(Path(tmp))
            recorder.on_chat_model_start({}, [], run_id='failed')
            recorder.on_llm_error(ConsecutiveNetworkFailures(), run_id='failed')
            self.assertEqual(recorder.summary()['requests_used'], 0)

    def test_shared_policy_for_role_copies(self):
        base = NetworkRetryChatOpenAI(api_key='offline-test-only', max_retries=0)
        self.assertIs(copy.copy(base).network_policy, base.network_policy)

    def test_branch_network_stop_is_not_swallowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            factory = offline_factory(3, 'df')
            visited = []
            def stop_analyst(role, recorder):
                visited.append(role)
                if role.startswith('analyst'):
                    raise ConsecutiveNetworkFailures()
                return factory(role, recorder)
            workflow = PandasRoleWorkflow(stop_analyst, offline=True)
            with self.assertRaises(ConsecutiveNetworkFailures):
                workflow.run(SimpleNamespace(goal='Count',context=''),
                    [pd.DataFrame({'x':[1,2,3]})],
                    [dict(dataframe='df',file_id='main',role='data')],Path(tmp))
            self.assertEqual(visited,['planner_1','analyst_Q1'])
