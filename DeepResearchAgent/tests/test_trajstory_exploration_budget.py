"""Offline checks for the controlled six-dispatch exploration budget."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.trajstory_exploration_budget import TrajstoryExplorationBudget


def test_initial_follow_up_and_repair_limits():
    budget = TrajstoryExplorationBudget()
    assert budget.authorize({})[0]
    budget.record_result(False)
    for _ in range(2):
        assert budget.authorize({})[0]
        budget.record_result(False)
    assert not budget.authorize({})[0]

    budget = TrajstoryExplorationBudget()
    assert budget.authorize({})[0]
    budget.record_result(True)
    assert budget.authorize({})[0]
    budget.record_result(False)
    assert budget.authorize({})[0]
    budget.record_result(True)
    assert budget.successful_analyses == 2
    assert not budget.authorize({})[0]


def test_tool_loop_blocks_batching_and_third_analysis(monkeypatch):
    from src.agent import tool_calling_agent as module

    class Action:
        def __init__(self, name, args):
            self.type = "tool"
            self.name = name
            self.args = json.dumps(args)

        def model_dump(self):
            return {"type": self.type, "name": self.name, "args": self.args}

    answers = [
        [
            Action("trajstory_analysis", {
                "spec_path": "/safe/case_input.json", "code": "print(1)",
            }),
            Action("trajstory_analysis", {
                "spec_path": "/safe/case_input.json", "code": "print(2)",
            }),
            Action("done", {"result": "too early"}),
        ],
        [Action("trajstory_analysis", {
            "spec_path": "/safe/case_input.json", "code": "print(2)",
        })],
        [Action("trajstory_analysis", {
            "spec_path": "/safe/case_input.json", "code": "print(3)",
        })],
        [Action("done", {"result": "supported result"})],
    ]
    calls = []

    async def fake_model_manager(**kwargs):
        actions = answers.pop(0)
        return SimpleNamespace(extra=SimpleNamespace(parsed_model=SimpleNamespace(
            thinking="test", evaluation_previous_goal="", memory="",
            next_goal="test", actions=actions,
        )))

    async def fake_tcp(*, name, input, ctx):
        calls.append(name)
        if name == "done":
            class Extra:
                data = {"reasoning": ""}

                def model_dump(self):
                    return {"data": self.data}

            return SimpleNamespace(success=True, message=input["result"],
                                   extra=Extra())
        return SimpleNamespace(success=True, message="Stdout:\n1", extra=None)

    monkeypatch.setattr(module, "model_manager", fake_model_manager)
    monkeypatch.setattr(module, "tcp", fake_tcp)
    agent = SimpleNamespace(model_name="fake", memory_name=None, use_memory=False)
    budget = TrajstoryExplorationBudget()

    def step(index):
        record = SimpleNamespace(tool=None)
        result = asyncio.run(module.ToolCallingAgent._think_and_tool(
            agent, [], "test", index, record=record, ctx=None,
            analysis_budget=budget,
        ))
        return result, record

    first, record = step(0)
    assert not first["done"]
    assert budget.successful_analyses == 1
    assert calls == ["trajstory_analysis"]
    assert len(record.tool["actions"]) == 3
    second, _ = step(1)
    assert not second["done"]
    assert budget.successful_analyses == 2
    third, _ = step(2)
    assert not third["done"]
    assert calls == ["trajstory_analysis", "trajstory_analysis"]
    fourth, _ = step(3)
    assert fourth["done"]
    assert fourth["result"] == "supported result"


def test_missing_arguments_return_feedback_and_allow_empty_exit(monkeypatch):
    from src.agent import tool_calling_agent as module

    class Action:
        type = "tool"

        def __init__(self, name, args):
            self.name = name
            self.args = json.dumps(args)

        def model_dump(self):
            return {"type": self.type, "name": self.name, "args": self.args}

    answers = [Action("trajstory_analysis", {}) for _ in range(3)]
    answers.append(Action("done", {}))
    answers.append(Action("done", {"result": "no verified findings"}))
    calls = []
    done_inputs = []

    async def fake_model_manager(**kwargs):
        return SimpleNamespace(extra=SimpleNamespace(parsed_model=SimpleNamespace(
            thinking="test", evaluation_previous_goal="", memory="",
            next_goal="test", actions=[answers.pop(0)],
        )))

    async def fake_tcp(*, name, input, ctx):
        calls.append(name)
        done_inputs.append(input)
        return SimpleNamespace(success=True, message=input["result"], extra=None)

    monkeypatch.setattr(module, "model_manager", fake_model_manager)
    monkeypatch.setattr(module, "tcp", fake_tcp)
    agent = SimpleNamespace(model_name="fake", memory_name=None, use_memory=False)
    budget = TrajstoryExplorationBudget()
    for index in range(3):
        record = SimpleNamespace(tool=None)
        response = asyncio.run(module.ToolCallingAgent._think_and_tool(
            agent, [], "test", index, record=record, ctx=None,
            analysis_budget=budget,
        ))
        assert not response["done"]
        assert "malformed or missing arguments" in record.tool["actions"][0]["output"]
    assert calls == []
    assert budget.failed_attempts_in_phase == 3

    record = SimpleNamespace(tool=None)
    response = asyncio.run(module.ToolCallingAgent._think_and_tool(
        agent, [], "test", 3, record=record, ctx=None,
        analysis_budget=budget,
    ))
    assert not response["done"]
    assert "missing result" in record.tool["actions"][0]["output"]

    record = SimpleNamespace(tool=None)
    response = asyncio.run(module.ToolCallingAgent._think_and_tool(
        agent, [], "test", 4, record=record, ctx=None,
        analysis_budget=budget,
    ))
    assert response["done"]
    assert response["result"] == "no verified findings"
    assert calls == ["done"]
    assert done_inputs == [{"result": "no verified findings", "reasoning": ""}]
