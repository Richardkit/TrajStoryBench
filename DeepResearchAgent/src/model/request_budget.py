"""Opt-in case budget; network failures are retried separately from model calls."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock

import httpx


RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


class ModelRequestBudgetExceeded(RuntimeError):
    """The next model invocation would exceed the case request limit."""


class SampleNetworkFailureLimitExceeded(RuntimeError):
    """Five consecutive network failures stopped this sample."""


def is_network_error(exc):
    if isinstance(exc, (ConnectionError, TimeoutError, httpx.TransportError)):
        return True
    if getattr(exc, 'status_code', None) in RETRYABLE_STATUS_CODES:
        return True
    try:
        from openai import APIConnectionError
        return isinstance(exc, APIConnectionError)
    except ImportError:
        return False


class ModelRequestBudget:
    def __init__(self, limit: int):
        if type(limit) is not int or limit < 1:
            raise ValueError("max_model_requests must be a positive integer")
        self.limit = limit
        self.network_failure_limit = 5
        self._lock = Lock()
        self._requests = []
        self._blocked = 0
        self._consecutive_network_failures = 0
        self._network_stopped = False
        self._changed = asyncio.Event()

    def check_network_stop(self):
        if self._network_stopped:
            raise SampleNetworkFailureLimitExceeded(
                "Sample stopped after 5 consecutive network request failures; "
                "these failures do not consume the model request budget."
            )

    def _used(self):
        return sum(r['outcome'] != 'network_failure' for r in self._requests)

    def _exhausted(self):
        self._blocked += 1
        raise ModelRequestBudgetExceeded(
            f"Case model request budget exhausted ({self._used()}/{self.limit}); "
            "no further model request was sent."
        )

    def check_available(self):
        with self._lock:
            self.check_network_stop()
            if self._used() >= self.limit and not any(
                    r['outcome'] == 'pending' for r in self._requests):
                self._exhausted()

    async def reserve(self, model: str) -> int:
        while True:
            with self._lock:
                self.check_network_stop()
                if self._used() < self.limit:
                    self._requests.append(dict(model=model, role=_role.get(), outcome='pending'))
                    return len(self._requests) - 1
                if not any(r['outcome'] == 'pending' for r in self._requests):
                    self._exhausted()
                changed = self._changed
            # In-flight network failures can free a slot. Wait for their
            # outcome instead of prematurely stopping parallel workers.
            await changed.wait()

    def finish(self, ticket: int, success: bool, *, network_failure=False):
        with self._lock:
            self._requests[ticket]['outcome'] = (
                'network_failure' if network_failure else 'succeeded' if success else 'failed')
            if not self._network_stopped:
                if network_failure:
                    self._consecutive_network_failures += 1
                    self._network_stopped = (
                        self._consecutive_network_failures >= self.network_failure_limit)
                else:
                    self._consecutive_network_failures = 0
            self._changed.set()
            self._changed = asyncio.Event()

    def snapshot(self):
        with self._lock:
            by_role, attempts_by_role = {}, {}
            for request in self._requests:
                role = request['role']
                attempts_by_role[role] = attempts_by_role.get(role, 0) + 1
                if request['outcome'] != 'network_failure':
                    by_role[role] = by_role.get(role, 0) + 1
            used = self._used()
            network_failed = sum(r['outcome'] == 'network_failure' for r in self._requests)
            return dict(
                limit=self.limit,
                requests_used=used,
                requests_started=len(self._requests),
                requests_succeeded=sum(r['outcome'] == 'succeeded' for r in self._requests),
                requests_failed=sum(r['outcome'] in ('failed', 'network_failure') for r in self._requests),
                network_failures=network_failed,
                consecutive_network_failures=self._consecutive_network_failures,
                network_failure_limit=self.network_failure_limit,
                network_failure_stop=self._network_stopped,
                requests_in_flight=sum(r['outcome'] == 'pending' for r in self._requests),
                requests_remaining=self.limit - used,
                limit_reached=used >= self.limit,
                blocked_requests=self._blocked,
                by_role=by_role,
                attempts_by_role=attempts_by_role,
                unit="model invocations excluding network failures; transport attempts recorded separately",
            )


_budget: ContextVar[ModelRequestBudget | None] = ContextVar("model_request_budget", default=None)
_role: ContextVar[str] = ContextVar("model_request_role", default="unspecified")


def current_request_budget():
    return _budget.get()


@contextmanager
def request_budget_scope(budget):
    token = _budget.set(budget)
    try:
        yield
    finally:
        _budget.reset(token)


@contextmanager
def request_role(role):
    token = _role.set(role)
    try:
        yield
    finally:
        _role.reset(token)


async def invoke_with_request_budget(client, model, **kwargs):
    budget = current_request_budget()
    if budget is None:
        return await client(**kwargs)
    while True:
        ticket = await budget.reserve(model)
        try:
            result = await client(**kwargs)
        except BaseException as exc:
            network_failure = is_network_error(exc)
            budget.finish(ticket, False, network_failure=network_failure)
            if network_failure:
                budget.check_network_stop()
                continue
            raise
        extra = getattr(result, 'extra', None)
        network_failure = bool(extra and (extra.data or {}).get('network_failure'))
        budget.finish(ticket, bool(result.success), network_failure=network_failure)
        if network_failure:
            budget.check_network_stop()
            continue
        return result
