"""Per-task budget for adapters whose chat factory returns callable models."""
from functools import wraps


class ModelCallBudgetExceeded(RuntimeError):
    pass


class ModelCallBudget:
    """Share one invocation counter across all chat callables in a task.

    This scope is for the sequential InsightBench adapter. Provider-internal
    transport retries belong to the same logical invocation. Failed invocations
    still consume a slot, and an exhausted budget never calls the provider.
    """

    def __init__(self, factory_owner, limit=83):
        if type(limit) is not int or limit < 1:
            raise ValueError("model-call budget must be a positive integer")
        self.factory_owner = factory_owner
        self.limit = limit
        self.used = 0
        self.exceeded = False

    def __enter__(self):
        self.original = self.factory_owner.get_chat_model

        @wraps(self.original)
        def factory(*args, **kwargs):
            chat = self.original(*args, **kwargs)

            @wraps(chat)
            def complete(*args, **kwargs):
                if self.used >= self.limit:
                    self.exceeded = True
                    self.raise_if_exceeded()
                self.used += 1
                return chat(*args, **kwargs)

            return complete

        self.factory_owner.get_chat_model = factory
        return self

    def raise_if_exceeded(self):
        if self.exceeded:
            raise ModelCallBudgetExceeded(
                f"Task exceeded its {self.limit}-call generation budget"
            )

    def __exit__(self, *exc):
        self.factory_owner.get_chat_model = self.original

    def as_dict(self):
        return {"limit": self.limit, "calls_started": self.used,
                "exceeded": self.exceeded,
                "unit": "logical model invocations, including failed invocations"}
