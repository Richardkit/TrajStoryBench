"""Minimal TrajStory input/output adaptation of DSAEval's native analysis loop."""
from __future__ import annotations

import sys as _baseline_sys
from pathlib import Path as _BaselinePath
_baseline_repo = next(p for p in _BaselinePath(__file__).resolve().parents if (p / "baselines" / "__init__.py").is_file())
_baseline_sys.path.insert(0, str(_baseline_repo))
import baselines  # Register the shared baseline module location.


import argparse
import hashlib
import json
import os
import re
import time
import traceback
from pathlib import Path

from prompts import DSAgent_PROMPT_LLM, DSAgent_PROMPT_VLM
from trajstory_six_explorations import (
    DEFAULT_MANIFEST, load_cases, provider_settings, resolve_path, stage_inputs, write_json,
)

PROTOCOL = "native_free_analysis_v1"


def system_prompt(staged, output_dir, is_multimodal, max_iterations, code_timeout):
    """Keep the upstream template; correct only paths and runtime capabilities."""
    template = DSAgent_PROMPT_VLM if is_multimodal else DSAgent_PROMPT_LLM
    lines = []
    for line in template.splitlines():
        if line.startswith("- There are 4 * NVIDIA"):
            lines.append("- This Windows environment provides CPU-based scientific computing; no GPU is allocated.")
        elif line.startswith("Under no circumstances") or line.startswith("and you MUST not terminate"):
            continue  # Continuation of the upstream A100 environment description.
        else:
            lines.append(line)
    template = "\n".join(lines)
    template = template.replace("20 steps/iteration opportunities (from 0-19)",
                                f"{max_iterations} steps/iteration opportunities (from 0-{max_iterations - 1})")
    template = template.replace("1 hour", f"{code_timeout} seconds")
    paths = "\n" + "\n".join(f"- {item['path']} ({item['role']})" for item in staged)
    return template.format(data_path=paths, working_path=str(output_dir))


def task_prompt(goal):
    return (
        f"{goal}\n\n"
        "Discover SIX factual insights supported by the executed data analysis, "
        "and provide a coherent summary. In section 4 (Results & Explanation) of your "
        "Markdown report, include one fenced JSON block with keys "
        '\"insights\" (six nonempty strings) and \"summary\" (a nonempty string). '
        "Include the supporting numerical results in the report."
    )


def parse_report(report):
    candidates = re.findall(r"```json\s*(.*?)```", report, flags=re.DOTALL | re.IGNORECASE)
    if len(candidates) != 1:
        raise ValueError("Report must contain one JSON block for insights and summary")
    obj = json.loads(candidates[0])
    if not isinstance(obj, dict):
        raise ValueError("Report JSON must be an object")
    insights, summary = obj.get("insights"), obj.get("summary") or ""
    if not (isinstance(insights, list) and len(insights) > 0
            and all(isinstance(x, str) and x.strip() for x in insights)
            and isinstance(summary, str)):
        raise ValueError("Expected a nonempty list of insight strings; summary is optional")
    return insights, summary


def run_case(case, case_dir, args, client, generation_kwargs):
    from data_agent import DataScienceAgent
    from jupyter_sandbox import JupyterSandbox

    started = time.monotonic()
    staged = stage_inputs(case, case_dir)  # Copy files only; do not preload DataFrames.
    artifacts = case_dir / "artifacts"
    artifacts.mkdir(exist_ok=True)
    prompt = system_prompt(staged, artifacts, args.is_multimodal,
                           args.max_iterations, args.code_timeout)
    task = task_prompt(case["goal"])
    (case_dir / "system_prompt.txt").write_text(prompt, encoding="utf-8")
    (case_dir / "task.txt").write_text(task, encoding="utf-8")
    agent = DataScienceAgent(client=client, model=args.model, system_prompt=prompt,
                             max_iterations=args.max_iterations,
                             is_multimodal=args.is_multimodal, generation_kwargs=generation_kwargs)
    log, report, insights, summary = {}, "", [], ""
    error, status = None, "failed"
    try:
        with JupyterSandbox(timeout=args.code_timeout, kernel_name=args.kernel_name) as sandbox:
            # One native invocation. No question lock, checkpoints, tool removal or recovery call.
            log, report = agent.run(task, str(artifacts), sandbox, verbose=args.verbose)
        write_json(case_dir / "agent_log.json", log)
        (case_dir / "final_report.md").write_text(report, encoding="utf-8")
        if not log.get("finished"):
            api_error = "API Error:" in log.get("reasoning", "")
            raise RuntimeError("APIError" if api_error else "IterationLimitExceeded: no native final report")
        insights, summary = parse_report(report)
        status = "completed" if len(insights) == 6 and summary.strip() else "partial"
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        agent.save_notebook(str(case_dir / "analysis.ipynb"))
    cells = [c for c in agent.notebook_cells if c["cell_type"] == "code"]
    row = dict(case_id=case["case_id"], status=status, pred_insights=insights, pred_summary=summary,
               exploration_protocol=PROTOCOL, execution_mode="single_native_persistent_session",
               agent_invocations=1, max_model_calls_per_case=args.max_iterations,
               native_finished=bool(log.get("finished")), structure_matches_request=status == "completed",
               code_execution_count=len(cells),
               successful_analysis_count=sum(not c.get("metadata", {}).get("execution_error", False) for c in cells),
               model=args.model, token_usage=log.get("token_usage", {}),
               elapsed_seconds=round(time.monotonic() - started, 3),
               notebook_path=str(case_dir / "analysis.ipynb"), error=error)
    write_json(case_dir / "result.json", row)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--case-ids", nargs="*")
    p.add_argument("--case-ids-file")
    p.add_argument("--results-dir", required=True)
    p.add_argument("--model", default="gpt-6-luna")
    p.add_argument("--base-url")
    p.add_argument("--api-key-env")
    p.add_argument("--reasoning-effort", default="none", choices=("none", "low", "medium", "high", "xhigh", "max"))
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=16000)
    p.add_argument("--max-iterations", type=int, default=60, help="Total native model decisions per case, including final report")
    p.add_argument("--code-timeout", type=int, default=180)
    p.add_argument("--kernel-name", default="python3")
    p.add_argument("--is-multimodal", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    ids = list(args.case_ids or [])
    if args.case_ids_file:
        selected = json.loads(resolve_path(args.case_ids_file).read_text(encoding="utf-8"))
        ids.extend(selected["case_ids"] if isinstance(selected, dict) else selected)
    if not ids or not all(isinstance(i, str) for i in ids) or len(ids) != len(set(ids)):
        p.error("provide unique case IDs")
    if args.max_iterations < 1 or args.code_timeout < 1:
        p.error("iteration and code-timeout budgets must be positive")
    cases = load_cases(resolve_path(args.manifest), ids)
    out = resolve_path(args.results_dir)
    url, key_env, kwargs = provider_settings(args.model, args.base_url, args.api_key_env,
                                            args.temperature, args.reasoning_effort, args.max_tokens)
    sources = [Path(__file__), Path(__file__).with_name("trajstory_six_explorations.py"),
               Path(__file__).with_name("prompts.py"), Path(__file__).with_name("data_agent.py"),
               Path(__file__).with_name("jupyter_sandbox.py")]
    config = dict(framework="DSAEval native agent with TrajStory I/O adaptation",
                  upstream_commit="160f3d577e8a1b6a7e4b6ae8c5db168ba9d188f1",
                  exploration_protocol=PROTOCOL, model=args.model, base_url=url, api_key_env=key_env,
                  manifest=str(resolve_path(args.manifest)), case_ids=ids, generation_kwargs=kwargs,
                  max_model_calls_per_case=args.max_iterations, code_timeout=args.code_timeout,
                  kernel_name=args.kernel_name, is_multimodal=args.is_multimodal,
                  input_mode="file_paths_agent_loads_data", output_mode="native_markdown_with_result_json",
                  forced_final_call=False, question_lock=False,
                  source_sha256={s.name: hashlib.sha256(s.read_bytes()).hexdigest() for s in sources},
                  input_sha256={c["case_id"]: {i["source"]: hashlib.sha256(Path(i["source"]).read_bytes()).hexdigest()
                                             for i in c["sources"]} for c in cases})
    if args.dry_run:
        print(json.dumps(dict(config=config, cases=cases), ensure_ascii=False, indent=2))
        return
    # A fresh directory is mandatory: never mix/relabel old runs or overwrite prior outputs.
    if out.exists() and any(out.iterdir()):
        p.error("use a fresh empty results directory")
    key = os.environ.get(key_env)
    if not key:
        p.error(f"required API key environment variable is unset: {key_env}")
    from openai import OpenAI
    client = OpenAI(api_key=key, base_url=url)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "experiment_config.json", config)
    rows = []
    for case in cases:
        case_dir = out / case["case_id"]
        case_dir.mkdir()
        rows.append(run_case(case, case_dir, args, client, kwargs))
        write_json(out / "trajstory_case_results.json", rows)
        print(json.dumps({"case_id": case["case_id"], "status": rows[-1]["status"],
                          "token_usage": rows[-1]["token_usage"]}))


if __name__ == "__main__":
    main()
