"""Offline integration check for the six-exploration TrajStory adapter."""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from trajstory_six_explorations import load_cases, provider_settings, run_case


class FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **request):
        self.calls.append(request)
        latest_user = next(message["content"] for message in reversed(request["messages"])
                           if message["role"] == "user"
                           and not str(message["content"]).startswith(
                               ("The tool phase is over.", "Executed Jupyter output:")))
        if request.get("tools"):
            question = request["tools"][0]["function"]["parameters"]["properties"]["question"]["enum"][0]
            tool_call = types.SimpleNamespace(
                id=f"tool-{len(self.calls)}", type="function",
                function=types.SimpleNamespace(name="jupyter_code_interpreter",
                                               arguments=json.dumps({"question": question, "code": "print('measured')"})),
            )
            message = types.SimpleNamespace(role="assistant", content=None, tool_calls=[tool_call])
        else:
            if latest_user.startswith("Summarize"):
                content = json.dumps({"summary": "Six measured trajectory findings."})
            elif latest_user.startswith("Register question"):
                index = int(latest_user.split("Register question ")[1].split("/")[0])
                content = json.dumps({"question": f"What is measured quantity {index}?"})
            else:
                index = int(latest_user.split("Exploration ")[1].split("/")[0])
                content = json.dumps({"question": f"What is measured quantity {index}?",
                                      "insight": f"Finding {index}: measured value {index}.",
                                      "evidence": "Python output: measured"})
            message = types.SimpleNamespace(role="assistant", content=content, tool_calls=None)
        usage = types.SimpleNamespace(prompt_tokens=100, completion_tokens=10, total_tokens=110)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)], usage=usage)


class FakeSandbox:
    instances = []

    def __init__(self, timeout, kernel_name):
        self.timeout = timeout
        self.kernel_name = kernel_name
        self.executions = []
        self.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def execute_code(self, code):
        self.executions.append(code)
        self.last_execution_error = False
        return "measured\n", []


class ErrorSandbox(FakeSandbox):
    def execute_code(self, code):
        self.executions.append(code)
        self.last_execution_error = True
        return "[Error] NameError: calculation failed\n", []


class AdapterTest(unittest.TestCase):
    def run_locked(self, responses, sandbox=None, limit=6):
        from data_agent import DataScienceAgent
        calls = []
        def create(**request):
            calls.append(request)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=responses.pop(0))])
        client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
        agent = DataScienceAgent(client, "qwen3.8-flash")
        sandbox = sandbox or FakeSandbox(180, "python3")
        with tempfile.TemporaryDirectory() as path:
            log, _ = agent.run("Answer the locked question", path, sandbox, verbose=False,
                               max_iterations=limit, final_answer_on_last_step=True,
                               locked_question="What is the mean distance?")
        return log, calls, sandbox

    @staticmethod
    def tool_message(question="What is the mean distance?", count=1):
        return types.SimpleNamespace(role="assistant", content=None, tool_calls=[
            types.SimpleNamespace(id=f"call-{i}", type="function", function=types.SimpleNamespace(
                name="jupyter_code_interpreter",
                arguments=json.dumps({"question": question, "code": "print('measured')"})))
            for i in range(count)
        ])

    @staticmethod
    def final_message():
        return types.SimpleNamespace(role="assistant", content='{"insight":"measured"}', tool_calls=None)

    def test_multi_call_and_changed_question_are_blocked_before_execution(self):
        log, calls, sandbox = self.run_locked([
            self.tool_message(count=2), self.tool_message("What is the maximum speed?"),
            self.tool_message(), self.final_message(),
        ])
        self.assertTrue(log["finished"])
        self.assertEqual(log["question_execution"]["blocked_tool_calls"], 3)
        self.assertEqual(log["question_execution"]["execution_attempts"], 1)
        self.assertEqual(sandbox.executions.count("print('measured')"), 1)
        self.assertNotIn("tools", calls[-1])

    def test_only_execution_error_allows_repair(self):
        class FailOnceSandbox(FakeSandbox):
            def execute_code(self, code):
                result = super().execute_code(code)
                self.last_execution_error = code == "print('measured')" and self.executions.count(code) == 1
                return result
        log, calls, sandbox = self.run_locked([
            self.tool_message(), self.tool_message(), self.final_message(),
        ], sandbox=FailOnceSandbox(180, "python3"))
        self.assertTrue(log["finished"])
        self.assertEqual(log["question_execution"]["execution_attempts"], 2)
        self.assertEqual(log["question_execution"]["successful_executions"], 1)
        self.assertIn("tools", calls[1])
        self.assertNotIn("tools", calls[2])

    def test_provider_cannot_execute_after_success_even_if_it_ignores_tool_omission(self):
        log, calls, sandbox = self.run_locked([self.tool_message(), self.tool_message()])
        self.assertFalse(log["finished"])
        self.assertNotIn("tools", calls[1])
        self.assertEqual(sandbox.executions.count("print('measured')"), 1)

    def test_answer_without_current_question_execution_is_rejected(self):
        log, calls, sandbox = self.run_locked([
            self.final_message(), self.tool_message(), self.final_message(),
        ])
        self.assertTrue(log["finished"])
        self.assertEqual(log["question_execution"]["execution_attempts"], 1)
        self.assertEqual(sandbox.executions.count("print('measured')"), 1)
        self.assertIn("rejected answer without execution", log["reasoning"])

    def test_legacy_agent_can_still_execute_multiple_calls_without_question_lock(self):
        from data_agent import DataScienceAgent
        responses = [self.tool_message(count=2), self.final_message()]
        client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(
            create=lambda **kwargs: types.SimpleNamespace(choices=[types.SimpleNamespace(message=responses.pop(0))]))))
        agent = DataScienceAgent(client, "qwen3.8-flash")
        sandbox = FakeSandbox(180, "python3")
        with tempfile.TemporaryDirectory() as path:
            log, _ = agent.run("legacy request", path, sandbox, verbose=False, max_iterations=3)
        self.assertTrue(log["finished"])
        self.assertEqual(sandbox.executions.count("print('measured')"), 2)

    def test_qwen_default_endpoint_matches_previous_comparison(self):
        with patch.dict("os.environ", {"DASHSCOPE_BASE_URL": ""}):
            base_url, key_env, kwargs = provider_settings(
                "qwen3.8-flash", None, None, 0.0, "none", 16000
            )
        self.assertEqual(base_url, "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.assertEqual(key_env, "DASHSCOPE_API_KEY")
        self.assertEqual(kwargs["extra_body"], {"enable_thinking": False, "top_k": 1})
        self.assertEqual(kwargs["seed"], 0)

    def test_six_serial_explorations_and_no_gold_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv_path = root / "data.csv"
            csv_path.write_text("trip_id,distance\n1,3\n", encoding="utf-8")
            case_path = root / "case.json"
            case_path.write_text(json.dumps({
                "dataset_csv_path": str(csv_path),
                "metadata": {"goal": "Investigate trip distances."},
                "insights": ["TOP_SECRET_GOLD_INSIGHT"],
                "summary": "TOP_SECRET_GOLD_SUMMARY",
            }), encoding="utf-8")
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({"cases": [{"case_id": "demo", "json_path": str(case_path)}]}),
                                     encoding="utf-8")
            case = load_cases(manifest_path, ["demo"])[0]
            self.assertNotIn("TOP_SECRET", json.dumps(case))

            completions = FakeCompletions()
            client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))
            args = types.SimpleNamespace(model="qwen3.8-flash", is_multimodal=False,
                                         max_iterations_per_exploration=4, code_timeout=180,
                                         kernel_name="python3", verbose=False)
            FakeSandbox.instances.clear()
            with patch.dict(sys.modules, {"jupyter_sandbox": types.SimpleNamespace(JupyterSandbox=FakeSandbox)}):
                result = run_case(case, root / "output", args, client, {"temperature": 0})

            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["exploration_count"], 6)
            self.assertEqual((result["initial_count"], result["follow_up_count"]), (3, 3))
            self.assertEqual(len(result["pred_insights"]), 6)
            self.assertEqual(result["token_usage"]["llm_calls"], 19)
            self.assertEqual(len(FakeSandbox.instances), 1)
            self.assertEqual(len([c for c in FakeSandbox.instances[0].executions if c == "print('measured')"]), 6)
            self.assertEqual(result["exploration_protocol"], "locked_single_question_v1")
            self.assertEqual(result["code_execution_count"], 6)
            self.assertEqual(result["successful_analysis_count"], 6)
            self.assertEqual(result["repair_execution_count"], 0)
            self.assertTrue((root / "output" / "analysis.ipynb").is_file())
            self.assertEqual(sum(bool(request.get("tools")) for request in completions.calls), 6)
            self.assertEqual(sum("tools" not in request for request in completions.calls), 13)
            self.assertTrue(all(
                message["role"] != "tool" and not message.get("tool_calls")
                for request in completions.calls if "tools" not in request
                for message in request["messages"]
            ))

    def test_exploration_rejects_code_that_raises(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv_path = root / "data.csv"
            csv_path.write_text("trip_id,distance\n1,3\n", encoding="utf-8")
            case = {"case_id": "demo", "goal": "Investigate trip distances.",
                    "sources": [{"source": str(csv_path), "role": "primary trajectory table"}]}
            completions = FakeCompletions()
            client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))
            args = types.SimpleNamespace(model="qwen3.8-flash", is_multimodal=False,
                                         max_iterations_per_exploration=4, code_timeout=180,
                                         kernel_name="python3", verbose=False)
            with patch.dict(sys.modules, {"jupyter_sandbox": types.SimpleNamespace(JupyterSandbox=ErrorSandbox)}):
                result = run_case(case, root / "output", args, client, {"temperature": 0})

            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["exploration_count"], 1)
            stage = json.loads((root / "output" / "exploration_01.json").read_text())
            self.assertEqual(stage["code_cells"], 3)
            self.assertEqual(stage["successful_code_cells"], 0)
            self.assertIn("successful code execution", result["error"]["message"])


if __name__ == "__main__":
    unittest.main()
