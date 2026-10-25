#!/usr/bin/env python3
"""Run TrajStory data through the native Planner/AgentBus workflow.

Use independently scoped computation, interpretation, review and synthesis roles.
Native planning stays dynamic; indexed handoffs avoid copying execution history.
There is no fixed question count or two-round dispatch schedule.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any


FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAJSTORY_ROOT = FRAMEWORK_ROOT.parent
sys.path.insert(0, str(FRAMEWORK_ROOT))
os.environ["TRAJSTORY_MINIMAL_IMPORTS"] = "1"

from src.trajstory import (  # noqa: E402
    build_case_spec,
    load_manifest_cases,
    parse_trajstory_result,
    write_case_spec,
)


AGENT_CONFIG_KEYS = ("planning_agent", "orchestra_analyst_agent", "orchestra_interpreter_agent",
                     "orchestra_verifier_agent", "orchestra_synthesizer_agent")
PROTOCOL = "native_planner_bounded_indexed_multi_role"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AgentOrchestra on TrajStory cases")
    parser.add_argument("--config", default=str(FRAMEWORK_ROOT / "configs" / "trajstory.py"))
    parser.add_argument("--trajstory-root", default=str(DEFAULT_TRAJSTORY_ROOT))
    parser.add_argument(
        "--manifest",
        default="trajstory_examples/insight/manifest.json",
        help="Absolute path or a path relative to --trajstory-root.",
    )
    parser.add_argument("--case-ids", nargs="+", default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum selected cases. Without --case-ids, one case is used by default.",
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_TRAJSTORY_ROOT / "results" / "agentorchestra_indexed_multi_role_v4"),
    )
    parser.add_argument("--model-name", default="openai/gpt-6-luna")
    parser.add_argument("--max-rounds", type=int, default=10,
                        help="Native AgentBus maximum planner rounds, not a required number")
    parser.add_argument("--max-model-requests", type=int, default=83,
                        help="Per-case request limit shared by Planner and all task roles")
    parser.add_argument("--schema-sample-rows", type=int, default=200)
    parser.add_argument("--input-csvs", action="store_true",
                        help="Use strict input_ CSV projections, matching the v5 data adapter")
    parser.add_argument("--history-result-max-chars", type=int, default=5000,
                        help="Characters retained per sub-agent result in planner history")
    parser.add_argument('--planner-context-max-chars', type=int, default=6000,
                        help='Bounded, deduplicated planner state; full history stays in logs')
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_model_requests < 1:
        parser.error("--max-model-requests must be a positive integer")
    return args


def build_task(spec: dict[str, Any], spec_path: Path) -> str:
    table_catalog = [
        {
            "name": table["name"],
            "role": table["role"],
            "path": table["path"],
            "columns": [column["name"] for column in table["schema"]["columns"]],
        }
        for table in spec["tables"]
    ]
    return (
        f"Case ID: {spec['case_id']}\n"
        f"Analyst role: {spec['role']}\n"
        f"Goal: {spec['goal']}\n"
        f"Dataset description: {spec.get('dataset_description') or 'not provided'}\n"
        f"Sanitised spec_path: {spec_path}\n"
        f"Input table catalog: {json.dumps(table_catalog, ensure_ascii=False)}\n\n"
        "Use trajstory_analysis with the exact spec_path for quantitative work. "
        "The source insight.json is an answer file and is strictly forbidden. "
        "Use separate roles: orchestra_analyst computes assigned subtasks; "
        "orchestra_interpreter interprets their execution outputs; orchestra_verifier "
        "independently reviews candidates; orchestra_synthesizer writes the final report. "
        "Assign each analyst one focused computational question, not the whole goal or six insights. "
        "Code, stdout and repair records are saved automatically. Do not ask analysts to repeat them. "
        "Planner results contain evidence IDs and small indexes; this is a complete handoff, not a truncated analysis. "
        "For interpreter/verifier/synthesizer, dispatch a JSON task string containing only "
        "instruction and evidence_ids. Pass E_ IDs to interpreters, interpreter R_ IDs to verifiers, "
        "and verifier R_ IDs to synthesizers. The adapter retrieves the exact corresponding records. "
        "Never paste code, stdout, full candidate packets or repair history into dispatch tasks. "
        "If a batch is too large, split its /S#/C# evidence IDs; do not rerun the analysis to retrieve evidence. "
        "Choose the number of subtasks, parallelism and follow-ups dynamically. "
        "All four roles must successfully participate before completion. "
        "The planner coordinates and reports the synthesizer's answer, not its own "
        "end-to-end analysis or invented findings. "
        "Find six factual insights supported by the data and relevant to the goal. "
        "Return one JSON object with `insights`: [six insight strings] and "
        "`summary`: a coherent summary."
    )


def _configure_runtime(args: argparse.Namespace, results_dir: Path) -> None:
    from src.config import config

    config.initialize(
        config_path=args.config,
        args=Namespace(config=args.config, cfg_options={"model_name": args.model_name}),
    )
    runtime_dir = (results_dir / "_framework").resolve()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    config.workdir = str(runtime_dir)
    config.log_path = str(runtime_dir / "trajstory.log")
    config.trajstory_analysis_tool["trajstory_root"] = str(Path(args.trajstory_root).resolve())
    for key in AGENT_CONFIG_KEYS:
        agent_config = config.get(key)
        if agent_config is None:
            continue
        agent_config["model_name"] = args.model_name
        agent_config["workdir"] = str(runtime_dir / "agent" / agent_config["name"])


async def _initialize_framework():
    from src.agent import acp
    from src.environment import ecp
    from src.interaction import bus
    from src.memory import memory_manager
    from src.model import model_manager
    from src.prompt import prompt_manager
    from src.skill import scp
    from src.tool import tcp
    from src.version import version_manager
    from src.config import config

    await model_manager.initialize()
    await prompt_manager.initialize()
    await memory_manager.initialize(memory_names=config.memory_names)
    await tcp.initialize(tool_names=config.tool_names)
    await scp.initialize(skill_names=config.skill_names)
    await ecp.initialize(config.env_names)
    await acp.initialize(agent_names=config.agent_names)
    await version_manager.initialize()
    await bus.initialize()
    return bus


async def run(args: argparse.Namespace) -> None:
    trajstory_root = Path(args.trajstory_root).resolve()
    results_dir = Path(args.results_dir).resolve()
    effective_limit = args.limit if args.limit is not None else (None if args.case_ids else 1)
    entries = load_manifest_cases(
        args.manifest,
        trajstory_root=trajstory_root,
        case_ids=args.case_ids,
        limit=effective_limit,
    )
    prepared = []
    for entry in entries:
        spec = build_case_spec(
            entry,
            trajstory_root=trajstory_root,
            sample_rows=args.schema_sample_rows,
            input_csvs=args.input_csvs,
        )
        prepared.append((entry, spec))

    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "ready",
                    "cases": [
                        {
                            "case_id": spec["case_id"],
                            "goal": spec["goal"],
                            "tables": [table["name"] for table in spec["tables"]],
                            "gold_fields_exposed": False,
                        }
                        for _, spec in prepared
                    ],
                    "model": args.model_name,
                    "max_rounds": args.max_rounds,
                    "max_model_requests": args.max_model_requests,
                    "history_result_max_chars": args.history_result_max_chars,
                    "protocol": PROTOCOL,
                    "role_names": [key.removesuffix('_agent') for key in AGENT_CONFIG_KEYS],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    results_dir.mkdir(parents=True, exist_ok=True)
    _configure_runtime(args, results_dir)

    from src.interaction import bus
    from src.session import SessionContext
    from src.task import Task

    await _initialize_framework()
    results: list[dict[str, Any]] = []
    result_file = results_dir / "trajstory_case_results.json"
    try:
        for _, spec in prepared:
            case_dir = results_dir / spec["case_id"]
            spec_path = write_case_spec(spec, case_dir / "case_input.json")
            ctx = SessionContext()
            from types import SimpleNamespace
            from src.agent.orchestra_handoffs import store_for, write as write_handoff
            bound_store=store_for(SimpleNamespace(workdir=str(results_dir/'_framework/agent/planning')),ctx)
            write_handoff(bound_store/'case_context.json',dict(spec_path=str(spec_path)))
            task = Task(content=build_task(spec, spec_path), files=[], session_id=ctx.id)
            try:
                response = await bus.submit(
                    task,
                    session_ctx=ctx,
                    max_rounds=args.max_rounds,
                    history_result_max_chars=args.history_result_max_chars,
                    planner_context_max_chars=args.planner_context_max_chars,
                    max_model_requests=args.max_model_requests,
                )
                success = bool(response.payload.get("success", False))
                raw_result = str(
                    response.payload.get("result")
                    or response.payload.get("error")
                    or ""
                )
                record: dict[str, Any] = {
                    "case_id": spec["case_id"],
                    "protocol": PROTOCOL,
                    "status": "completed" if success else "failed",
                    "model": args.model_name,
                    "session_id": ctx.id,
                    "case_input": str(spec_path),
                    "raw_result": raw_result,
                    "request_budget": response.payload.get("request_budget"),
                }
                if success:
                    try:
                        record.update(parse_trajstory_result(raw_result))
                        record["structure_matches_request"] = (
                            len(record["pred_insights"]) == 6 and bool(record["pred_summary"])
                        )
                        record["output_status"] = (
                            "complete" if record["structure_matches_request"]
                            else "count_mismatch" if len(record["pred_insights"]) > 6
                            else "partial"
                        )
                    except ValueError as exc:
                        record["status"] = "failed"
                        record["error_type"] = "OutputValidationError"
                        record["error"] = str(exc)
                    from src.agent.orchestra_roles import role_audit
                    audit = role_audit(results_dir / '_framework', ctx.id)
                    record['role_audit'] = audit
                    if record['status'] == 'completed' and (audit['missing_roles'] or not audit['separate_role_contexts']):
                        record['status'] = 'failed'
                        record['error_type'] = 'RoleWorkflowIncomplete'
                        record['error'] = 'Required independent roles did not all complete successfully'
                    elif record['status'] == 'completed':
                        synthesized = []
                        for text in audit['synthesized_results']:
                            try:
                                synthesized.append(parse_trajstory_result(text))
                            except ValueError:
                                continue
                        if not any(item['pred_insights'] == record['pred_insights']
                                   and item['pred_summary'] == record['pred_summary']
                                   for item in synthesized):
                            record['status'] = 'failed'
                            record['error_type'] = 'PlannerBypassedSynthesizer'
                            record['error'] = 'Final answer differs from every successful synthesis output'
                else:
                    record["error"] = raw_result
                    if response.payload.get("error_type"):
                        record["error_type"] = response.payload["error_type"]
            except Exception as exc:
                record = {
                    "case_id": spec["case_id"],
                    "protocol": PROTOCOL,
                    "status": "failed",
                    "model": args.model_name,
                    "session_id": ctx.id,
                    "case_input": str(spec_path),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            results.append(record)
            result_file.write_text(
                json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    finally:
        await bus.shutdown()

    print(json.dumps({"cases": len(results), "results": str(result_file)}, indent=2))


def main() -> None:
    try:
        asyncio.run(run(parse_args()))
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing DeepResearchAgent dependency '{exc.name}'. "
            "Install the repository requirements before a non-dry run."
        ) from exc


if __name__ == "__main__":
    main()
