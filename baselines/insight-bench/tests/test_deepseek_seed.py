"""Offline request checks for InsightBench DeepSeek generation."""

from argparse import Namespace
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "insight-bench"))

from insightbench.utils import agent_utils
from baselines.scripts import run_trajstory_insightbench as runner


@pytest.mark.parametrize("seed", [None, 0, 42, 2**31 - 1])
def test_dashscope_deepseek_request_seed_and_nonthinking(monkeypatch, seed):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "offline")
    monkeypatch.setenv("TRAJSTORY_DASHSCOPE_DEEPSEEK_THINKING", "disabled")
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))])

    monkeypatch.setattr(agent_utils, "OpenAI", lambda **kwargs:
                        SimpleNamespace(chat=SimpleNamespace(
                            completions=SimpleNamespace(create=create))))
    assert agent_utils.get_chat_model(
        "deepseek-v4.1-flash", temperature=0, seed=seed,
    )("Test") == "OK"
    assert captured["temperature"] == 0
    assert captured["extra_body"] == {"enable_thinking": False}
    if seed is None:
        assert "seed" not in captured
    else:
        assert captured["seed"] == seed


@pytest.mark.parametrize("seed", [-1, 2**31, 1.5, True])
def test_dashscope_deepseek_rejects_invalid_seed(monkeypatch, seed):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "offline")
    monkeypatch.setenv("TRAJSTORY_DASHSCOPE_DEEPSEEK_THINKING", "disabled")
    monkeypatch.setattr(agent_utils, "OpenAI", lambda **kwargs:
                        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace())))
    with pytest.raises(ValueError, match="seed"):
        agent_utils.get_chat_model("deepseek-v4.1-flash", seed=seed)


def test_official_deepseek_sends_fixed_seed(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline")
    monkeypatch.setenv("TRAJSTORY_OPENAI_REASONING_EFFORT", "none")
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))])

    monkeypatch.setattr(agent_utils, "OpenAI", lambda **kwargs:
                        SimpleNamespace(chat=SimpleNamespace(
                            completions=SimpleNamespace(create=create))))
    assert agent_utils.get_chat_model("deepseek-flash", temperature=0, seed=42)("Test") == "OK"
    assert captured["temperature"] == 0
    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}
    assert captured["seed"] == 42


@pytest.mark.parametrize("model,expected_api_seed", [
    ("deepseek-v4.1-flash", 42),
    ("deepseek-flash", 42),
])
def test_runner_records_effective_deepseek_seed(
    tmp_path, monkeypatch, model, expected_api_seed,
):
    case = {"case_id": "offline_case", "dataset_csv_path": tmp_path / "data.csv",
            "user_dataset_csv_path": None, "reference_tables": [],
            "data": {"insights": ["offline"], "summary": "offline"}}
    monkeypatch.setattr(runner, "load_cases", lambda _: [case])
    monkeypatch.setattr(runner, "validate_case", lambda _: [])
    monkeypatch.setattr(runner, "schema_report", lambda _: {"tables": []})
    monkeypatch.setenv("TRAJSTORY_OPENAI_REASONING_EFFORT", "pretest")
    monkeypatch.setenv("TRAJSTORY_DASHSCOPE_DEEPSEEK_THINKING", "pretest")
    args = Namespace(
        manifest="unused.json", results_dir=str(tmp_path / "results"),
        model_name=model, reasoning_effort="none", case_ids=None,
        shared_analysis_tools=False, dry_run=True, keep_artifacts=True,
        seed=42,
    )
    runner.run(args)
    row = json.loads((tmp_path / "results/trajstory_case_results.json").read_text())[0]
    assert row["seed"] == 42
    assert row["api_seed"] == expected_api_seed
    if model == "deepseek-v4.1-flash":
        assert os.getenv("TRAJSTORY_DASHSCOPE_DEEPSEEK_THINKING") == "disabled"
