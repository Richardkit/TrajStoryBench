#!/usr/bin/env python3
"""Run TrajStory InsightBench cases through the local AgentPoirot interface.

This adapter intentionally does not use the official flag-*.json benchmark
loader. It preserves the InsightBench execution/evaluation interface while
adding TrajStory-specific case discovery, absolute paths, secondary-table
selection, per-case goals, and per-case Agent isolation.
"""

from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
import baselines


import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "trajstory_examples" / "insight" / "manifest.json"
INSIGHT_BENCH_ROOT = REPO_ROOT / "baselines/insight-bench"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(INSIGHT_BENCH_ROOT))

from insightbench import agents, benchmarks
from insightbench.utils import agent_utils as au
from baseline_support.input_paths import resolve_input_csv


def resolve_path(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def load_cases(manifest_path: Path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = []
    for entry in manifest["cases"]:
        json_path = resolve_path(entry["json_path"])
        data = json.loads(json_path.read_text(encoding="utf-8"))
        main_path = resolve_input_csv(resolve_path(data["dataset_csv_path"]))
        if main_path is None:
            raise ValueError(f"Primary dataset has no retained input columns: {entry['case_id']}")
        user_path = resolve_input_csv(resolve_path(data.get("user_dataset_csv_path")))
        references = data.get("input_references", [])
        omitted_references = [reference for reference in references
                              if resolve_input_csv(resolve_path(reference.get("path"))) is None]

        # Prefer an explicitly declared secondary table. Otherwise select the
        # timestamped trajectory reference for order/trip cases.
        selected_role = None
        if user_path is None:
            for reference in references:
                ref_path = reference.get("path", "")
                role = reference.get("role", "").lower()
                if "trajectory" in ref_path.lower() or "trajectory" in role:
                    user_path = resolve_input_csv(resolve_path(ref_path))
                    if user_path is None:
                        continue
                    selected_role = reference.get("role")
                    break
        else:
            for reference in references:
                if resolve_input_csv(resolve_path(reference.get("path"))) == user_path:
                    selected_role = reference.get("role")
                    break

        # Pass every non-primary reference table to the agent. The official
        # InsightBench API has a primary/secondary convention, while
        # TrajStory cases may also contain courier-day, station, vessel, or
        # other reference tables. These are named from their CSV stem so the
        # model can select them by name in the schema and code prompt.
        reference_tables = []
        used_paths = {p for p in (main_path, user_path) if p is not None}
        used_names = {"main", "secondary"}
        for index, reference in enumerate(references, start=1):
            reference_path = resolve_input_csv(resolve_path(reference.get("path")))
            if reference_path is None or reference_path in used_paths:
                continue
            base_name = reference_path.stem.removeprefix("input_").replace("-", "_").replace(" ", "_")
            name = base_name or f"reference_{index}"
            if name in used_names:
                name = f"reference_{index}_{name}"
            used_names.add(name)
            reference_tables.append({
                "name": name,
                "path": str(reference_path),
                "role": reference.get("role", "reference table"),
            })
            used_paths.add(reference_path)

        cases.append({
            "case_id": entry["case_id"],
            "json_path": json_path,
            "data": data,
            "dataset_csv_path": main_path,
            "user_dataset_csv_path": user_path,
            "selected_secondary_role": selected_role,
            "reference_tables": reference_tables,
            "omitted_input_references": omitted_references,
            "input_csv_policy": "input_ siblings only; no original fallback",
            "ignored_references": [
                reference for reference in references
                if resolve_input_csv(resolve_path(reference.get("path"))) != user_path
            ],
        })
    return cases


def validate_case(case):
    errors = []
    for label, path in [("dataset_csv_path", case["dataset_csv_path"]), ("user_dataset_csv_path", case["user_dataset_csv_path"])]:
        if path is not None and not path.exists():
            errors.append(f"{label} does not exist: {path}")
    if not case["data"].get("insights"):
        errors.append("missing ground-truth insights")
    if not case["data"].get("summary"):
        errors.append("missing ground-truth summary")
    return errors


def schema_report(case):
    """Return the complete named schema catalog passed to AgentPoirot."""
    import pandas as pd

    main_df = pd.read_csv(case["dataset_csv_path"])
    user_df = pd.read_csv(case["user_dataset_csv_path"]) if case["user_dataset_csv_path"] else None
    main_schema = au.get_schema(main_df)
    user_schema = au.get_schema(user_df) if user_df is not None else None
    tables = [{
        "name": "main",
        "role": "primary dataset",
        "path": str(case["dataset_csv_path"]),
        "schema": main_schema,
    }]
    if user_df is not None:
        tables.append({
            "name": "secondary",
            "role": case.get("selected_secondary_role") or "secondary dataset",
            "path": str(case["user_dataset_csv_path"]),
            "schema": user_schema,
        })
    for reference in case.get("reference_tables", []):
        ref_df = pd.read_csv(reference["path"])
        tables.append({
            "name": reference["name"],
            "role": reference["role"],
            "path": reference["path"],
            "schema": au.get_schema(ref_df),
        })
    return {
        "tables": tables,
        "main_table": str(case["dataset_csv_path"]),
        "main_schema": main_schema,
        "secondary_table": str(case["user_dataset_csv_path"]) if case["user_dataset_csv_path"] else None,
        "secondary_schema": user_schema,
        "ignored_references": case["ignored_references"],
        "omitted_input_references": case.get("omitted_input_references", []),
        "input_csv_policy": case.get("input_csv_policy"),
    }


def run(args):
    dashscope_deepseek = args.model_name.lower() == "deepseek-v4.1-flash"
    official_deepseek = args.model_name.lower() in {
        "deepseek-flash", "deepseek_flash", "deepseek_v4_1_flash",
    }
    deepseek_seed_metadata = (
        {"api_seed": args.seed}
        if dashscope_deepseek or official_deepseek else {}
    )
    reasoning_env = ("TRAJSTORY_QWEN_REASONING_EFFORT" if args.model_name.lower() == "qwen3.8-flash"
                     else "TRAJSTORY_OPENAI_REASONING_EFFORT")
    if args.reasoning_effort is None:
        os.environ.pop(reasoning_env, None)
    else:
        os.environ[reasoning_env] = args.reasoning_effort
    if dashscope_deepseek and args.reasoning_effort is not None:
        os.environ["TRAJSTORY_DASHSCOPE_DEEPSEEK_THINKING"] = (
            "disabled" if args.reasoning_effort == "none" else "enabled"
        )
    cases = load_cases(resolve_path(args.manifest))
    if args.case_ids:
        requested = set(args.case_ids)
        cases = [case for case in cases if case["case_id"] in requested]
        missing = requested - {case["case_id"] for case in cases}
        if missing:
            raise ValueError(f"Unknown case IDs: {sorted(missing)}")
    results_dir = resolve_path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    analysis_toolkit = {"enabled": False}
    if args.shared_analysis_tools:
        from baseline_support.analysis_tools import SHARED_TOOL_REFERENCE, analysis_tool_metadata
        analysis_toolkit = analysis_tool_metadata()
        (results_dir / "shared_analysis_tools.txt").write_text(SHARED_TOOL_REFERENCE, encoding="utf-8")
    (results_dir / "analysis_toolkit.json").write_text(
        json.dumps(analysis_toolkit, indent=2), encoding="utf-8"
    )
    results = []

    for case in cases:
        errors = validate_case(case)
        if errors:
            raise FileNotFoundError(f"{case['case_id']}: " + "; ".join(errors))

        case_dir = results_dir / case["case_id"]
        if case_dir.exists() and not args.keep_artifacts:
            shutil.rmtree(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)

        report = schema_report(case)
        (case_dir / "input_schema_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

        if args.dry_run:
            results.append({
                "case_id": case["case_id"],
                "status": "ready",
                "schema_report": str(case_dir / "input_schema_report.json"),
                "schema_table_count": len(report["tables"]),
                "schema_table_names": [table["name"] for table in report["tables"]],
                "seed": args.seed,
                **deepseek_seed_metadata,
                "analysis_toolkit": analysis_toolkit,
            })
            continue

        # A fresh Agent is intentional: insights_history must not leak between
        # different TrajStory cases. A malformed model response in one case is
        # recorded and isolated so it cannot abort the remaining cases/seed.
        try:
            agent = agents.Agent(
                model_name=args.model_name,
                goal=case["data"].get("metadata", {}).get("goal", "I want to find interesting trends in this dataset"),
                max_questions=args.max_questions,
                branch_depth=args.branch_depth,
                n_retries=args.n_retries,
                savedir=str(case_dir),
                additional_tables=case.get("reference_tables", []),
                temperature=args.temperature,
                seed=args.seed,
                shared_analysis_tools=args.shared_analysis_tools,
            )
            pred_insights, pred_summary = agent.get_insights(
                dataset_csv_path=str(case["dataset_csv_path"]),
                user_dataset_csv_path=str(case["user_dataset_csv_path"]) if case["user_dataset_csv_path"] else None,
                additional_tables=case.get("reference_tables", []),
            )
            result = {
                "case_id": case["case_id"],
                "status": "completed",
                "pred_insights": pred_insights,
                "pred_summary": pred_summary,
                "score_insights": benchmarks.evaluate_insights(pred_insights, case["data"]["insights"], score_name=args.score_name),
                "score_summary": benchmarks.evaluate_summary(pred_summary, case["data"]["summary"], score_name=args.score_name),
                "model": args.model_name,
                "temperature": args.temperature,
                "reasoning_effort": args.reasoning_effort,
            }
        except Exception as exc:
            result = {
                "case_id": case["case_id"],
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        result.update({
            "model": args.model_name,
            "temperature": args.temperature,
            "secondary_table": str(case["user_dataset_csv_path"]) if case["user_dataset_csv_path"] else None,
            "schema_table_count": len(report["tables"]),
            "schema_table_names": [table["name"] for table in report["tables"]],
            "seed": args.seed,
            **deepseek_seed_metadata,
            "reasoning_effort": args.reasoning_effort,
            "analysis_toolkit": analysis_toolkit,
        })
        results.append(result)
        # Persist partial progress so a long run remains inspectable.
        (results_dir / "trajstory_case_results.json").write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8"
        )

        if not args.keep_artifacts:
            shutil.rmtree(case_dir)

    (results_dir / "trajstory_case_results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"cases": len(results), "results": str(results_dir / "trajstory_case_results.json"), "dry_run": args.dry_run}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--results-dir", default="results/trajstory_insightbench")
    parser.add_argument("--model-name", default="gpt-4o-mini")
    parser.add_argument("--max-questions", type=int, default=2)
    parser.add_argument("--branch-depth", type=int, default=1)
    parser.add_argument("--n-retries", type=int, default=2)
    parser.add_argument("--score-name", choices=["rouge1", "g_eval"], default="rouge1")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh", "max"],
        default=None,
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Generation API seed; supported for DashScope DeepSeek, not sent to the official DeepSeek endpoint.",
    )
    parser.add_argument(
        "--shared-analysis-tools", action=argparse.BooleanOptionalAction, default=True,
        help="Provide the same trajectory computation toolkit and API docs as TrajStoryAgent (default: enabled).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate paths and schemas without calling the Agent or LLM.")
    parser.add_argument("--keep-artifacts", action="store_true")
    parser.add_argument("--case-ids", nargs="+", default=None, help="Run only the listed case IDs.")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
