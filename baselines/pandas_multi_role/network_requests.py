"""Retry transport failures within one logical model request, per case."""
import time
from openai import APIConnectionError, APITimeoutError
from langchain_core.pydantic_v1 import Field
from baselines.scripts.pandas_usage_logging import UsageChatOpenAI


class ConsecutiveNetworkFailures(RuntimeError):
    """Five consecutive transport failures terminate the whole case."""


class NetworkRequestPolicy:
    def __init__(self, delay_seconds=1):
        self.delay_seconds = delay_seconds
        self.consecutive_failures = 0
        self.network_failures = 0
        self.successful_requests = 0

    def run(self, operation):
        if self.consecutive_failures >= 5:
            raise ConsecutiveNetworkFailures('Network failure limit reached')
        while True:
            try:
                result = operation()
            except (APIConnectionError, APITimeoutError, ConnectionError, TimeoutError):
                self.network_failures += 1
                self.consecutive_failures += 1
                if self.consecutive_failures >= 5:
                    raise ConsecutiveNetworkFailures('Five consecutive network failures') from None
                time.sleep(self.delay_seconds)
            except Exception:
                # A provider response (e.g. invalid request) breaks a network streak.
                self.consecutive_failures = 0
                raise
            else:
                self.consecutive_failures = 0
                self.successful_requests += 1
                return result

    def summary(self):
        return dict(network_failed_attempts=self.network_failures,
                    consecutive_network_failures=self.consecutive_failures,
                    network_failure_limit=5)


class NetworkRetryChatOpenAI(UsageChatOpenAI):
    network_policy: NetworkRequestPolicy = Field(default_factory=NetworkRequestPolicy, exclude=True)

    def _generate(self, *args, **kwargs):
        generate = super()._generate
        return self.network_policy.run(lambda: generate(*args, **kwargs))
