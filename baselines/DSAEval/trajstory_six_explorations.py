#!/usr/bin/env python3
"""Run the DSAEval data agent on TrajStory cases with six serial explorations.

The upstream DSAEval agent has one persistent conversation and Jupyter kernel
per dataset session. This adapter keeps that behavior. Explorations 1-3 seek
distinct directions; explorations 4-6 revisit the evidence. They are serial,
not three independent branches followed by three parallel follow-ups.
"""

from __future__ import annotations

import sys as _baseline_sys
from pathlib import Path as _BaselinePath
_baseline_repo = next(p for p in _BaselinePath(__file__).resolve().parents if (p / "baselines" / "__init__.py").is_file())
_baseline_sys.path.insert(0, str(_baseline_repo))
import baselines  # Register the shared baseline module location.


import argparse
import json
import os
import shutil
import time
import traceback
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "trajstory_examples" / "insight" / "manifest.json"
EXPLORATIONS = 6
EXPLORATION_PROTOCOL = "locked_single_question_v1"


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def load_cases(manifest_path: Path, case_ids: list[str]) -> list[dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_id = {entry["case_id"]: entry for entry in manifest["cases"]}
    unknown = sorted(set(case_ids) - set(by_id))
    if unknown:
        raise ValueError(f"Unknown case IDs: {unknown}")
    cases = []
    for case_id in case_ids:
        case_json = resolve_path(by_id[case_id]["json_path"])
        payload = json.loads(case_json.read_text(encoding="utf-8"))
        # Deliberately do not pass payload['insights'] or payload['summary'].
        sources = []
        seen = set()
        declarations = [
            (payload.get("dataset_csv_path"), "primary trajectory table"),
            (payload.get("user_dataset_csv_path"), "secondary table"),
        ] + [
            (item.get("path"), item.get("role", "reference table"))
            for item in payload.get("input_references", [])
        ]
        for raw_path, role in declarations:
            if not raw_path:
                continue
            source = resolve_path(raw_path)
            if source in seen:
                continue
            if not source.is_file() or source.suffix.lower() != ".csv":
                raise FileNotFoundError(f"{case_id}: input CSV missing: {source}")
            seen.add(source)
            sources.append({"source": str(source), "role": role})
        if not sources:
            raise ValueError(f"{case_id}: no input CSVs")
        cases.append({
            "case_id": case_id,
            "goal": payload.get("metadata", {}).get("goal") or "Discover meaningful trajectory patterns.",
            "sources": sources,
        })
    return cases


def provider_settings(model: str, base_url: str | None, api_key_env: str | None,
                      temperature: float, reasoning_effort: str, max_tokens: int) -> tuple[str, str, dict]:
    lower = model.lower()
    if lower.startswith("qwen"):
        # Match the endpoint used by the existing TrajStory/InsightBench Qwen runs.
        base_url = base_url or os.environ.get("DASHSCOPE_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        api_key_env = api_key_env or "DASHSCOPE_API_KEY"
        kwargs = {
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            "reasoning_effort": reasoning_effort,
            "extra_body": {"enable_thinking": reasoning_effort != "none"},
            "seed": 0,
        }
        if temperature == 0 and reasoning_effort == "none":
            kwargs["extra_body"]["top_k"] = 1
    elif lower.startswith("deepseek"):
        base_url = base_url or "https://api.deepseek.com"
        api_key_env = api_key_env or "DEEPSEEK_API_KEY"
        kwargs = {"temperature": temperature, "max_tokens": max_tokens}
        if reasoning_effort == "none":
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    else:
        base_url = base_url or "https://api.openai.com/v1"
        api_key_env = api_key_env or "OPENAI_API_KEY"
        kwargs = {"max_completion_tokens": max_tokens, "reasoning_effort": reasoning_effort}
        if reasoning_effort == "none":
            kwargs["temperature"] = temperature
    return base_url.rstrip("/"), api_key_env, kwargs


def stage_inputs(case: dict, case_dir: Path) -> list[dict]:
    input_dir = case_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    staged = []
    for index, item in enumerate(case["sources"]):
        source = Path(item["source"])
        destination = input_dir / f"{index:02d}_{source.name}"
        shutil.copy2(source, destination)
        staged.append({"path": str(destination), "role": item["role"]})
    return staged


def system_prompt(staged: list[dict], output_dir: Path, is_multimodal: bool) -> str:
    files = "\n".join(f"- {item['path']}: {item['role']}" for item in staged)
    visual = ("Plots returned by the Jupyter tool are visible to you."
              if is_multimodal else
              "You cannot inspect plot images directly; print numeric summaries for every conclusion.")
    return (
        "You are the data-analysis agent from DSAEval, using a persistent Jupyter kernel. "
        "Write and execute Python with the jupyter_code_interpreter tool to investigate each request. "
        "You have six sequential explorations of one trajectory-analysis goal. "
        "Use the observed data to choose the next direction. Do not invent findings. "
        "Each exploration must execute code and end with one distinct, numerically supported finding. "
        "ONE exploration means ONE atomic question, not a bundle of subquestions. "
        "First register that question without executing code. It is then immutable for this exploration. "
        "Do not combine multiple hypotheses, screen candidate directions, or pivot after seeing results. "
        "Use one self-contained code call to answer only the registered question; include necessary loading, "
        "cleaning, calculation and any relevant plot in that call. A successful call ends the tool phase. "
        "After an execution error, only fix that same calculation; do not expand or change the question. "
        "The first three should cover different aspects; the next three should deepen, verify, or challenge "
        "earlier findings. Avoid restating the same result. "
        f"{visual}\n"
        "Available input CSV files (and no reference answers):\n"
        f"{files}\n"
        f"Write optional artifacts under {output_dir}. Read data only from the listed CSVs. "
        "Follow the JSON schema for the current phase: registration returns 'question'; "
        "an exploration answer returns 'question', 'insight' and 'evidence'; the final summary returns 'summary'. "
        "Put the quantitative conclusion in 'insight'; use 'evidence' for the supporting calculation and caveat."
    )


def planning_prompt(goal: str, index: int, prior_questions: list[str], retry: bool = False) -> str:
    phase = ("Find a distinct initial direction." if index <= 3 else
             "Use findings and executed evidence from earlier explorations to deepen, verify, or challenge a direction.")
    prior = json.dumps(prior_questions, ensure_ascii=False)
    return (
        f"Register question {index}/6. Goal: {goal}\n{phase} "
        "Choose exactly ONE specific, atomic question from the available inputs and prior observations. "
        "Do not bundle separate comparisons, hypotheses or exploratory directions. "
        f"Already registered questions: {prior}. You must ask a meaningfully different question. "
        + ("Your previous proposal repeated a registered question. Select a different variable or relationship. " if retry else "")
        +
        "Do not execute code or claim a result. This question will be locked and consumes one slot. "
        'Return ONLY JSON: {"question": "..."}.'
    )


def exploration_prompt(goal: str, index: int, question: str) -> str:
    return (
        f"Exploration {index}/6. Goal: {goal}\nLocked question: {question}\n"
        "Answer ONLY this question. No additional subquestions, screening or direction changes. "
        "Submit exactly one jupyter_code_interpreter call per response, with the exact locked question "
        "in its 'question' field and one self-contained analysis in 'code'. "
        "Include necessary data loading and calculations in that call. Once it succeeds, "
        "no further code is allowed. If it errors, repair only that same calculation. "
        "Conclude one finding with supporting numbers, or explicitly report the absence of a pattern. "
        'Return ONLY JSON: {"question": "<exact locked question>", "insight": "...", "evidence": "..."}.'
    )


def parse_json_object(report: str, required_keys: tuple[str, ...]) -> dict:
    value = report.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            value = "\n".join(lines[1:-1]).strip()
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("final answer is not a JSON object")
    for key in required_keys:
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            raise ValueError(f"final answer requires a nonempty {key!r} string")
    return payload


def run_case(case: dict, case_dir: Path, args: argparse.Namespace, client, generation_kwargs: dict) -> dict:
    from data_agent import DataScienceAgent
    from jupyter_sandbox import JupyterSandbox

    started = time.monotonic()
    staged = stage_inputs(case, case_dir)
    artifact_dir = case_dir / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    agent = DataScienceAgent(
        client=client,
        model=args.model,
        system_prompt=system_prompt(staged, artifact_dir, args.is_multimodal),
        max_iterations=args.max_iterations_per_exploration,
        is_multimodal=args.is_multimodal,
        generation_kwargs=generation_kwargs,
    )
    stages = []
    insights = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "llm_calls": 0}
    notebook_path = case_dir / "analysis.ipynb"
    status = "completed"
    error = None
    try:
        with JupyterSandbox(timeout=args.code_timeout, kernel_name=args.kernel_name) as sandbox:
            for index in range(1, EXPLORATIONS + 1):
                prior_questions = [s["question"] for s in stages]
                plans = []
                for attempt in range(2):
                    plan_log, plan_report = agent.run(
                        user_input=planning_prompt(case["goal"], index, prior_questions, bool(attempt)),
                        working_path=str(artifact_dir), jupyter_sandbox=sandbox,
                        verbose=args.verbose, max_iterations=1, final_answer_on_last_step=True,
                    )
                    for key in usage:
                        usage[key] += plan_log.get("token_usage", {}).get(key, 0)
                    if not plan_log["finished"]:
                        raise RuntimeError(f"question {index} registration failed")
                    question = parse_json_object(plan_report, ("question",))["question"].strip()
                    plans.append({"question": question, "log": plan_log})
                    if question not in prior_questions:
                        break
                write_json(case_dir / f"question_{index:02d}.json", {
                    "index": index, "protocol": EXPLORATION_PROTOCOL,
                    "question": question, "planning_attempts": plans,
                    "repeats_prior_question": question in prior_questions,
                })
                before_cells = len(agent.notebook_cells)
                log, report = agent.run(
                    user_input=exploration_prompt(case["goal"], index, question),
                    working_path=str(artifact_dir),
                    jupyter_sandbox=sandbox,
                    verbose=args.verbose,
                    max_iterations=args.max_iterations_per_exploration,
                    final_answer_on_last_step=True,
                    locked_question=question,
                )
                for key in usage:
                    usage[key] += log.get("token_usage", {}).get(key, 0)
                stage = {
                    "index": index,
                    "question": question,
                    "repeats_prior_question": question in prior_questions,
                    "protocol": EXPLORATION_PROTOCOL,
                    "question_execution": log["question_execution"],
                    "phase": "initial" if index <= 3 else "follow_up",
                    "finished": log["finished"],
                    "code_cells": sum(cell["cell_type"] == "code" for cell in agent.notebook_cells[before_cells:]),
                    "successful_code_cells": sum(
                        cell["cell_type"] == "code"
                        and not cell.get("metadata", {}).get("execution_error", False)
                        for cell in agent.notebook_cells[before_cells:]
                    ),
                    "report": report,
                    "token_usage": log["token_usage"],
                    "reasoning": log["reasoning"],
                }
                stages.append(stage)
                write_json(case_dir / f"exploration_{index:02d}.json", stage)
                if not stage["finished"] or stage["successful_code_cells"] != 1:
                    raise RuntimeError(
                        f"exploration {index} did not finish with successful code execution"
                    )
                parsed = parse_json_object(report, ("question", "insight", "evidence"))
                if parsed["question"] != question:
                    raise RuntimeError("final answer changed the locked question")
                stage["parsed"] = parsed
                write_json(case_dir / f"exploration_{index:02d}.json", stage)
                insights.append(parsed["insight"].strip())

            log, report = agent.run(
                user_input=(
                    "Summarize the six completed findings for the original goal. "
                    "Use only the evidence already obtained; do not introduce a seventh exploration. "
                    "Return ONLY JSON: {\"summary\": \"...\"}."
                ),
                working_path=str(artifact_dir),
                jupyter_sandbox=sandbox,
                verbose=args.verbose,
                max_iterations=1,
                final_answer_on_last_step=True,
            )
            for key in usage:
                usage[key] += log.get("token_usage", {}).get(key, 0)
            write_json(case_dir / "summary_log.json", log)
            if not log["finished"]:
                raise RuntimeError("summary generation did not finish")
            summary = parse_json_object(report, ("summary",))["summary"].strip()
    except Exception as exc:
        status = "failed"
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        summary = ""
    finally:
        agent.save_notebook(str(notebook_path))

    row = {
        "case_id": case["case_id"],
        "status": status,
        "pred_insights": insights,
        "pred_summary": summary,
        "exploration_count": len(stages),
        "distinct_question_count": len({s["question"] for s in stages}),
        "exploration_protocol": EXPLORATION_PROTOCOL,
        "code_execution_count": sum(s["question_execution"]["execution_attempts"] for s in stages),
        "successful_analysis_count": sum(s["successful_code_cells"] for s in stages),
        "repair_execution_count": sum(max(0, s["question_execution"]["execution_attempts"] - 1) for s in stages),
        "initial_count": sum(s["phase"] == "initial" for s in stages),
        "follow_up_count": sum(s["phase"] == "follow_up" for s in stages),
        "execution_mode": "serial_persistent_session",
        "model": args.model,
        "token_usage": usage,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "notebook_path": str(notebook_path),
        "error": error,
    }
    write_json(case_dir / "result.json", row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--case-ids", nargs="*")
    parser.add_argument("--case-ids-file", help="JSON list, or an experiment_config.json containing case_ids")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env")
    parser.add_argument("--reasoning-effort", default="none", choices=("none", "low", "medium", "high", "xhigh", "max"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--max-iterations-per-exploration", type=int, default=6)
    parser.add_argument("--code-timeout", type=int, default=180)
    parser.add_argument("--kernel-name", default="python3")
    parser.add_argument("--is-multimodal", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    case_ids = list(args.case_ids or [])
    if args.case_ids_file:
        selected = json.loads(resolve_path(args.case_ids_file).read_text(encoding="utf-8"))
        if isinstance(selected, dict):
            selected = selected["case_ids"]
        if not isinstance(selected, list) or not all(isinstance(x, str) for x in selected):
            parser.error("--case-ids-file must contain a JSON string list or a case_ids field")
        case_ids.extend(selected)
    if not case_ids:
        parser.error("provide --case-ids or --case-ids-file")
    if len(case_ids) != len(set(case_ids)):
        parser.error("case IDs must be unique")
    if args.max_iterations_per_exploration < 2:
        parser.error("each exploration needs at least two iterations: one tool call and one final answer")

    cases = load_cases(resolve_path(args.manifest), case_ids)
    results_dir = resolve_path(args.results_dir)
    base_url, api_key_env, generation_kwargs = provider_settings(
        args.model, args.base_url, args.api_key_env,
        args.temperature, args.reasoning_effort, args.max_tokens,
    )
    config = {
        "framework": "DSAEval DataScienceAgent adapted for TrajStory",
        "upstream_commit": "160f3d577e8a1b6a7e4b6ae8c5db168ba9d188f1",
        "manifest": str(resolve_path(args.manifest)),
        "case_ids": case_ids,
        "model": args.model,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "generation_kwargs": generation_kwargs,
        "explorations": EXPLORATIONS,
        "exploration_protocol": EXPLORATION_PROTOCOL,
        "successful_analyses_per_question": 1,
        "execution_mode": "serial_persistent_session",
        "max_iterations_per_exploration": args.max_iterations_per_exploration,
        "max_model_calls_per_case": EXPLORATIONS * (2 + args.max_iterations_per_exploration) + 1,
        "code_timeout": args.code_timeout,
        "kernel_name": args.kernel_name,
        "is_multimodal": args.is_multimodal,
    }
    if args.dry_run:
        print(json.dumps({"config": config, "cases": cases}, indent=2, ensure_ascii=False))
        return

    prior_config = results_dir / "experiment_config.json"
    if prior_config.exists():
        previous = json.loads(prior_config.read_text(encoding="utf-8"))
        if previous.get("exploration_protocol") != EXPLORATION_PROTOCOL:
            parser.error("exploration protocol changed; use a fresh results directory")

    key = os.environ.get(api_key_env)
    if not key:
        parser.error(f"required API key environment variable is unset: {api_key_env}")
    from openai import OpenAI
    client = OpenAI(api_key=key, base_url=base_url)
    results_dir.mkdir(parents=True, exist_ok=True)
    write_json(results_dir / "experiment_config.json", config)
    rows = []
    for case in cases:
        case_dir = results_dir / case["case_id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        row = run_case(case, case_dir, args, client, generation_kwargs)
        rows.append(row)
        write_json(results_dir / "trajstory_case_results.json", rows)
        print(json.dumps({"case_id": case["case_id"], "status": row["status"],
                          "explorations": row["exploration_count"], "llm_calls": row["token_usage"]["llm_calls"]}))


if __name__ == "__main__":
    main()
