"""Paper InsightBench generation protocol; scoring is a separate GPT two-round step.

Defaults reproduce the frozen GPT/Qwen settings: three questions, depth two,
two agent retries, temperature zero, reasoning disabled, projected inputs and
the original shared input-policy prompt. The public entry additionally enforces
a per-task ceiling of 83 model invocations; historical outputs remain unchanged.
Choose --model-name gpt-6-luna or qwen3.8-flash. --dry-run makes no API calls.
"""
import sys
import os
import argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import baselines
sys.path.insert(0, str(ROOT / "baselines/insight-bench"))
from insightbench import benchmarks
from insightbench.utils import agent_utils as au
from baselines.scripts import run_trajstory_insightbench as adapter
INPUT_POLICY = 'INPUT POLICY:\nOnly the named projected input tables and visible columns are available.\nPreviously supplied derived columns may be absent by design. Compute needed\nderived quantities from visible fields where possible; do not recover them\nfrom original CSVs, unlisted files, reference insights or prior experiment outputs.\nIf visible inputs cannot support a quantity or conclusion, state that limitation.\n'

def main():
    selection = argparse.ArgumentParser(add_help=False)
    selection.add_argument("--model-name", default="gpt-6-luna")
    selected, _ = selection.parse_known_args()
    qwen = selected.model_name.lower() == "qwen3.8-flash"
    previous_urls = {name: os.environ.get(name) for name in ("DASHSCOPE_BASE_URL", "QWEN_BASE_URL")}
    if qwen:
        for name in previous_urls:
            os.environ[name] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    original = au.get_chat_model
    def model(*args, **kwargs):
        chat = original(*args, **kwargs)
        def complete(content):
            return chat(INPUT_POLICY + "\n" + content)
        return complete
    au.get_chat_model = model
    old_insights, old_summary = benchmarks.evaluate_insights, benchmarks.evaluate_summary
    benchmarks.evaluate_insights = lambda *args, **kwargs: None
    benchmarks.evaluate_summary = lambda *args, **kwargs: None
    previous = sys.argv
    defaults = ["--model-name", "gpt-6-luna", "--max-questions", "3",
                "--branch-depth", "2", "--n-retries", "2", "--temperature", "0",
                "--reasoning-effort", "none", "--shared-analysis-tools", "--keep-artifacts"]
    if qwen:
        defaults.extend(["--seed", "0"])
    sys.argv = [previous[0], *defaults, *previous[1:]]
    try:
        adapter.main()
    finally:
        sys.argv = previous
        for name, value in previous_urls.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        au.get_chat_model = original
        benchmarks.evaluate_insights, benchmarks.evaluate_summary = old_insights, old_summary

if __name__ == "__main__":
    main()
