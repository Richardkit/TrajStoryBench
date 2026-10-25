"""Verify native exploration remains unrestricted and bounded after I/O adaptation."""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from prompts import DSAgent_PROMPT_LLM, DSAgent_PROMPT_VLM
from trajstory_native import load_cases, run_case, system_prompt


def response(code=None):
    if code is None:
        payload = {"insights": [f"Measured finding {i}" for i in range(6)], "summary": "Measured summary"}
        message = types.SimpleNamespace(role="assistant", content=(
            "## Final Output\n**1. Task Understanding**\nGoal\n**2. Approach Summary**\nAnalysis\n"
            "**3. Key Implementation**\nPython\n**4. Results & Explanation**\n```json\n"
            + json.dumps(payload) + "\n```"), tool_calls=None)
    else:
        message = types.SimpleNamespace(role="assistant", content=None, tool_calls=[
            types.SimpleNamespace(id="code", type="function", function=types.SimpleNamespace(
                name="jupyter_code_interpreter", arguments=json.dumps({"code": code})))])
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)],
                                 usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15))


class Sandbox:
    instances = []
    def __init__(self, timeout, kernel_name):
        self.executions = []
        self.last_execution_error = False
        self.instances.append(self)
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def execute_code(self, code):
        self.executions.append(code)
        self.last_execution_error = code == "bad calculation"
        return ("[Error] calculation failed" if self.last_execution_error else "measured"), []


class NativeTests(unittest.TestCase):
    def execute(self, responses, limit):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data.csv").write_text("x\n1\n2\n", encoding="utf-8")
            (root / "case.json").write_text(json.dumps({"dataset_csv_path": str(root / "data.csv"),
                "metadata": {"goal": "Analyze x"}, "insights": ["SECRET_GOLD"], "summary": "SECRET_SUMMARY"}), encoding="utf-8")
            (root / "manifest.json").write_text(json.dumps({"cases": [{"case_id": "demo", "json_path": str(root / "case.json") }]}), encoding="utf-8")
            case = load_cases(root / "manifest.json", ["demo"])[0]
            calls = []
            def create(**request):
                # Snapshot messages; the real loop mutates its history after requests.
                calls.append(json.loads(json.dumps(request)))
                return responses.pop(0)
            client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
            args = types.SimpleNamespace(model="gpt-6-luna", max_iterations=limit,
                code_timeout=180, kernel_name="dsa_eval", verbose=False, is_multimodal=False)
            Sandbox.instances.clear()
            with patch.dict(sys.modules, {"jupyter_sandbox": types.SimpleNamespace(JupyterSandbox=Sandbox)}):
                row = run_case(case, root / "output", args, client, {})
            self.assertTrue((root / "output" / "analysis.ipynb").is_file())
            self.assertTrue((root / "output" / "final_report.md").is_file())
            self.assertEqual([p.name for p in (root / "output" / "input").iterdir()], ["00_data.csv"])
            self.assertNotIn("SECRET_", json.dumps(calls))
            self.assertEqual(len(Sandbox.instances), 1)
            return row, calls, Sandbox.instances[0].executions

    def test_multiple_successful_calculations_keep_tools_and_history(self):
        row, calls, executions = self.execute([response("load data"), response("verify earlier result"), response()], 3)
        self.assertEqual(row["status"], "completed")
        self.assertEqual(row["code_execution_count"], 2)
        self.assertEqual(row["token_usage"]["llm_calls"], 3)
        self.assertTrue(all("tools" in c for c in calls))
        self.assertNotIn("question", calls[0]["tools"][0]["function"]["parameters"]["properties"])
        self.assertTrue(any(m["role"] == "tool" for m in calls[-1]["messages"]))
        self.assertIn("verify earlier result", executions)
        self.assertEqual(sum(m["role"] == "user" for m in calls[-1]["messages"]), 1)

    def test_budget_has_no_extra_summary_or_forced_answer_call(self):
        row, calls, _ = self.execute([response("load data"), response("continue analysis")], 2)
        self.assertEqual(row["status"], "failed")
        self.assertEqual(len(calls), 2)
        self.assertIn("IterationLimitExceeded", row["error"]["message"])
        self.assertIn("tools", calls[-1])

    def test_execution_error_can_be_repaired_then_further_verified(self):
        row, _, _ = self.execute([response("bad calculation"), response("repair"), response("verify"), response()], 4)
        self.assertEqual(row["status"], "completed")
        self.assertEqual(row["code_execution_count"], 3)
        self.assertEqual(row["successful_analysis_count"], 2)

    def test_native_report_can_finish_early(self):
        row, calls, _ = self.execute([response()], 20)
        self.assertEqual(row["status"], "completed")
        self.assertEqual(len(calls), 1)

    def test_both_original_templates_survive_except_environment_lines(self):
        for visual, template in [(False, DSAgent_PROMPT_LLM), (True, DSAgent_PROMPT_VLM)]:
            result = system_prompt([{"path": "input/table.csv", "role": "data"}], Path("out"), visual, 20, 180)
            for line in template.splitlines():
                if (not line.strip() or "{data_path}" in line or "{working_path}" in line
                    or line.startswith(("- There are 4", "Under no circumstances", "and you MUST not terminate"))
                    or "1 hour" in line):
                    continue
                self.assertIn(line, result)
            self.assertNotIn("4 * NVIDIA", result)
            self.assertIn("180 seconds", result)


if __name__ == "__main__": unittest.main()
