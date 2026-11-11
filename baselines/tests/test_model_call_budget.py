"""Offline checks for the shared AgentPoirot invocation limit."""
import unittest
from types import SimpleNamespace

from baselines.baseline_support.model_call_budget import (
    ModelCallBudget, ModelCallBudgetExceeded,
)


class ModelCallBudgetTest(unittest.TestCase):
    def setUp(self):
        self.calls = []

        def factory(model):
            def complete(content):
                self.calls.append((model, content))
                if content == "fail":
                    raise RuntimeError("simulated provider failure")
                return content
            return complete

        self.owner = SimpleNamespace(get_chat_model=factory)
        self.original = factory

    def test_shared_ceiling_blocks_84th_invocation_and_resets_per_task(self):
        for _ in range(2):
            before = len(self.calls)
            with ModelCallBudget(self.owner) as budget:
                planner = self.owner.get_chat_model("planner")
                analyst = self.owner.get_chat_model("analyst")
                for index in range(83):
                    (planner if index % 2 else analyst)("ok")
                self.assertEqual(budget.used, 83)
                budget.raise_if_exceeded()  # Finishing on call 83 is valid.
                with self.assertRaises(ModelCallBudgetExceeded):
                    analyst("blocked")
                self.assertEqual(len(self.calls) - before, 83)
            self.assertIs(self.owner.get_chat_model, self.original)

    def test_failed_invocation_counts_and_factory_restores_on_exception(self):
        with self.assertRaisesRegex(RuntimeError, "simulated provider"):
            with ModelCallBudget(self.owner, 1) as budget:
                self.owner.get_chat_model("analyst")("fail")
        self.assertEqual(budget.used, 1)
        self.assertIs(self.owner.get_chat_model, self.original)

    def test_swallowed_exhaustion_is_detectable(self):
        with ModelCallBudget(self.owner, 1) as budget:
            chat = self.owner.get_chat_model("analyst")
            chat("ok")
            for _ in range(3):
                try:
                    chat("retry")
                except ModelCallBudgetExceeded:
                    pass
            with self.assertRaises(ModelCallBudgetExceeded):
                budget.raise_if_exceeded()
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(budget.as_dict()["exceeded"])

    def test_invalid_budget(self):
        for value in (0, -1, None, True):
            with self.assertRaises(ValueError):
                ModelCallBudget(self.owner, value)


if __name__ == "__main__":
    unittest.main()
