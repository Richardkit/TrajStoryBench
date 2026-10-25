"""Per-case generation budget, independent of transport retries and judging."""
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Condition

import httpx


RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
_current = ContextVar("insightbench_request_budget", default=None)


class ModelRequestBudgetExceeded(RuntimeError):
    pass


class SampleNetworkFailureLimitExceeded(RuntimeError):
    pass


def is_network_error(exc):
    if isinstance(exc, (ConnectionError, TimeoutError, httpx.TransportError)):
        return True
    if getattr(exc, "status_code", None) in RETRYABLE_STATUS_CODES:
        return True
    from openai import APIConnectionError
    return isinstance(exc, APIConnectionError)


class ModelRequestBudget:
    def __init__(self, limit=83, network_failure_limit=5):
        if type(limit) is not int or limit < 1:
            raise ValueError("max_model_requests must be a positive integer")
        if type(network_failure_limit) is not int or network_failure_limit < 1:
            raise ValueError("network_failure_limit must be a positive integer")
        self.limit = limit
        self.network_failure_limit = network_failure_limit
        self._condition = Condition()
        self._attempts = []
        self._streak = 0
        self._network_stopped = False
        self._blocked = 0

    def _used(self):
        return sum(a != "network_failure" for a in self._attempts)

    def _reserve(self):
        with self._condition:
            while True:
                if self._network_stopped:
                    raise SampleNetworkFailureLimitExceeded(
                        f"Sample stopped after {self.network_failure_limit} consecutive network failures")
                if self._used() < self.limit:
                    self._attempts.append("pending")
                    return len(self._attempts) - 1
                if "pending" not in self._attempts:
                    self._blocked += 1
                    raise ModelRequestBudgetExceeded(
                        f"Case model request budget exhausted ({self.limit}/{self.limit})")
                self._condition.wait()

    def _finish(self, ticket, outcome):
        with self._condition:
            self._attempts[ticket] = outcome
            if not self._network_stopped:
                self._streak = self._streak + 1 if outcome == "network_failure" else 0
                self._network_stopped = self._streak >= self.network_failure_limit
            self._condition.notify_all()

    def invoke(self, call):
        while True:
            ticket = self._reserve()
            try:
                response = call()
            except BaseException as exc:
                network = is_network_error(exc)
                self._finish(ticket, "network_failure" if network else "failed")
                if network:
                    continue
                raise
            self._finish(ticket, "succeeded")
            return response

    def snapshot(self):
        with self._condition:
            return dict(
                limit=self.limit, requests_used=self._used(),
                requests_started=len(self._attempts),
                requests_succeeded=self._attempts.count("succeeded"),
                requests_failed=self._attempts.count("failed"),
                network_failures=self._attempts.count("network_failure"),
                consecutive_network_failures=self._streak,
                network_failure_limit=self.network_failure_limit,
                network_failure_stop=self._network_stopped,
                requests_in_flight=self._attempts.count("pending"),
                blocked_requests=self._blocked,
                unit="generation model requests excluding network failures; judging excluded",
            )


@contextmanager
def request_budget_scope(budget):
    token = _current.set(budget)
    try:
        yield budget
    finally:
        _current.reset(token)


def request_completion(client, **kwargs):
    budget = _current.get()
    if budget is None:
        return client.chat.completions.create(**kwargs)
    # Count every transport attempt ourselves; SDK retries must not hide failures.
    direct_client = client.with_options(max_retries=0)
    return budget.invoke(lambda: direct_client.chat.completions.create(**kwargs))
