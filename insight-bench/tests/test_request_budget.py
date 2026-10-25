"""Offline checks: no generation, judging, or network request is performed."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "insight-bench")]
from insightbench.request_budget import (
    ModelRequestBudget, ModelRequestBudgetExceeded,
    SampleNetworkFailureLimitExceeded, request_budget_scope, request_completion,
)


def test_83rd_response_allowed_84th_never_sent():
    budget = ModelRequestBudget()
    calls = []
    for _ in range(83):
        budget.invoke(lambda: calls.append(1))
    with pytest.raises(ModelRequestBudgetExceeded):
        budget.invoke(lambda: calls.append(1))
    assert len(calls) == 83
    assert budget.snapshot()["requests_used"] == 83


def test_network_failures_do_not_spend_budget_and_fifth_stops_case():
    budget = ModelRequestBudget(1)
    calls = []
    def offline():
        calls.append(1)
        raise ConnectionError("offline")
    with pytest.raises(SampleNetworkFailureLimitExceeded):
        budget.invoke(offline)
    assert len(calls) == 5
    assert budget.snapshot()["requests_used"] == 0
    with pytest.raises(SampleNetworkFailureLimitExceeded):
        budget.invoke(lambda: calls.append(1))
    assert len(calls) == 5


def test_response_resets_failure_streak():
    budget = ModelRequestBudget(2)
    outcomes = iter([ConnectionError()] * 4 + ["ok"] + [ConnectionError()] * 4 + ["ok"])
    def call():
        result = next(outcomes)
        if isinstance(result, Exception):
            raise result
        return result
    assert budget.invoke(call) == budget.invoke(call) == "ok"
    snap = budget.snapshot()
    assert snap["requests_used"] == 2
    assert snap["network_failures"] == 8
    assert snap["consecutive_network_failures"] == 0


@pytest.mark.parametrize("code,network", [(429, True), (503, True), (400, False), (401, False)])
def test_status_failures_match_other_baseline(code, network):
    class Error(Exception):
        status_code = code
    budget = ModelRequestBudget()
    def fail():
        raise Error()
    with pytest.raises(SampleNetworkFailureLimitExceeded if network else Error):
        budget.invoke(fail)
    assert budget.snapshot()["requests_used"] == (0 if network else 1)


def test_parallel_workers_cannot_exceed_case_limit():
    budget = ModelRequestBudget()
    def worker(_):
        try:
            budget.invoke(lambda: "ok")
            return True
        except ModelRequestBudgetExceeded:
            return False
    with ThreadPoolExecutor(max_workers=12) as pool:
        assert sum(pool.map(worker, range(120))) == 83


def test_multiple_model_clients_share_case_budget_and_sdk_retries_disabled(monkeypatch):
    from insightbench.utils import agent_utils
    monkeypatch.delenv("TRAJSTORY_LLM_AUDIT_PATH", raising=False)
    options, calls = [], []
    class Client:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        def with_options(self, **kwargs):
            options.append(kwargs)
            return self
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="invalid-format"))])
    monkeypatch.setattr(agent_utils, "OpenAI", Client)
    with request_budget_scope(ModelRequestBudget(2)):
        a = agent_utils.get_chat_model("gpt-6-luna")
        b = agent_utils.get_chat_model("gpt-6-luna")
        assert a("question") == b("summary") == "invalid-format"
        with pytest.raises(ModelRequestBudgetExceeded):
            a("format repair")
    assert len(calls) == 2  # Bad output still spends one request.
    assert all(option == {"max_retries": 0} for option in options)
    # New case starts fresh; outside generation scope judging remains uncapped.
    with request_budget_scope(ModelRequestBudget(1)):
        assert a("new case") == "invalid-format"
    assert a("external judging") == "invalid-format"
    assert len(calls) == 4


@pytest.mark.parametrize("name", ["run_v5_gpt_insightbench125", "run_v5_qwen_insightbench125", "run_insightbench_full125"])
def test_terminal_stop_prevents_whole_case_automatic_retry(monkeypatch, tmp_path, name):
    from importlib import import_module
    module = import_module("scripts." + name)
    cid = "offline_case"
    case = {"case_id": cid, "json_path": "unused"}
    if name == "run_insightbench_full125":
        monkeypatch.setattr(module, "OUT", tmp_path)
        folder = tmp_path / "generation/gpt" / cid
    else:
        monkeypatch.setattr(module, "directory", lambda _: tmp_path)
        monkeypatch.setattr(module.shared, "verify_current", lambda _: None)
        folder = tmp_path / "generation" / cid
    saved = folder / "attempt_1/trajstory_case_results.json"
    saved.parent.mkdir(parents=True)
    saved.write_text(json.dumps([{"status": "failed", "stop_reason": "consecutive_network_failures"}]))
    # The legacy entry uses this marker to avoid dispatching an existing run.
    (saved.parent / "started.json").write_text("{}")
    if name == "run_insightbench_full125":
        with pytest.raises(RuntimeError, match="without whole-case retry"):
            module.generate_case("gpt", case)
    else:
        with pytest.raises(RuntimeError, match="without whole-case retry"):
            module.generate("insightbench", case, {})
    assert not (folder / "attempt_2").exists()


def test_runner_isolates_case_and_excludes_external_judging(monkeypatch, tmp_path):
    from scripts import run_trajstory_insightbench as runner
    clients = []
    class Client:
        def __init__(self):
            self.calls = 0
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
            clients.append(self)
        def with_options(self, **kwargs):
            assert kwargs == {"max_retries": 0}
            return self
        def create(self, **kwargs):
            self.calls += 1
            return "mock"
    class Agent:
        def __init__(self, **kwargs):
            self.count = 84 if kwargs["goal"] == "first" else 1
            self.client = Client()
        def get_insights(self, **kwargs):
            for _ in range(self.count):
                request_completion(self.client)
            return ["insight"], "summary"
    cases = [dict(case_id=name, data={"metadata": {"goal": name}, "insights": ["ref"], "summary": "ref"},
                  dataset_csv_path="unused.csv", user_dataset_csv_path=None) for name in ["first", "second"]]
    monkeypatch.setattr(runner, "load_cases", lambda _: cases)
    monkeypatch.setattr(runner, "validate_case", lambda _: [])
    monkeypatch.setattr(runner, "schema_report", lambda _: {"tables": []})
    monkeypatch.setattr(runner.agents, "Agent", Agent)
    judge = Client()
    monkeypatch.setattr(runner.benchmarks, "evaluate_insights", lambda *a, **k: request_completion(judge))
    monkeypatch.setattr(runner.benchmarks, "evaluate_summary", lambda *a, **k: request_completion(judge))
    args = SimpleNamespace(model_name="gpt-6-luna", seed=0, reasoning_effort="none", manifest="unused",
                           case_ids=None, results_dir=str(tmp_path), shared_analysis_tools=False,
                           keep_artifacts=True, dry_run=False, max_questions=3, branch_depth=2,
                           n_retries=2, temperature=0, score_name="g_eval")
    runner.run(args)
    rows = json.loads((tmp_path / "trajstory_case_results.json").read_text())
    assert rows[0]["stop_reason"] == "model_request_budget_exhausted"
    assert rows[0]["model_request_budget"]["requests_used"] == 83
    assert rows[1]["status"] == "completed"
    assert rows[1]["model_request_budget"]["requests_used"] == 1
    assert judge.calls == 2  # Both judges are outside generation scope.
