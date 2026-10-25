from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from trajstoryagent_v5.v5_framework.code_generation import AnalysisCodeGenerator
from trajstoryagent_v5.v5_framework.execution import PythonCodeExecutor, load_raw_analysis_result
from trajstoryagent_v5.v5_framework.json_utils import write_json
from trajstoryagent_v5.v5_framework.llm import LLMClient
from trajstoryagent_v5.v5_framework.models import (
    AnalysisGranularity,
    AnalysisQuestion,
    DataBundle,
    FollowUpQuestion,
    FrameworkConfig,
    TrajectoryDataProfile,
)
from trajstoryagent_v5.v5_framework.result_interpretation import AnalysisResultInterpreter

from trajstoryagent_v5.v5_framework.models import EvidencePackage

LOGGER = logging.getLogger(__name__)


class AnalysisEngine:
    """Generate Python directly from questions, execute it, and interpret results."""

    def __init__(self, llm: LLMClient, config: FrameworkConfig) -> None:
        self.llm = llm
        self.config = config
        self.code_generator = AnalysisCodeGenerator(llm)
        self.executor = PythonCodeExecutor(config.code_timeout_seconds)
        self.result_interpreter = AnalysisResultInterpreter(
            llm, max_repairs=config.max_interpretation_repairs
        )

    @staticmethod
    def _is_visualization_failure(failure: str) -> bool:
        return any(marker in failure for marker in (
            "in save_figure", "y_axis.value", "x_axis.value",
            "plot_group_comparison", "plot_heatmap", "plot_profile",
            "plot_trajectories", "structured visualization",
            "visualization references unknown statistic",
        ))

    def analyze_initial(
        self,
        *,
        bundle: DataBundle,
        profile: TrajectoryDataProfile,
        question: AnalysisQuestion,
        output_dir: Path,
        revision_instruction: str = "",
    ) -> EvidencePackage:
        return self._analyze(
            bundle=bundle,
            profile=profile,
            question_id=question.question_id,
            question=question.question,
            granularity=question.analysis_granularity,
            evidence_id=f"E{question.question_id.removeprefix('Q')}",
            branch_kind="initial",
            output_dir=output_dir,
            parent_evidence_ids=[],
            parent_evidence=[],
            revision_instruction=revision_instruction,
            coding_errors=None,
        )

    def analyze_follow_up(
        self,
        *,
        bundle: DataBundle,
        profile: TrajectoryDataProfile,
        follow_up: FollowUpQuestion,
        output_dir: Path,
        parent_evidence: list[EvidencePackage] | None = None,
        revision_instruction: str = "",
        coding_errors: list[str] | None = None,
    ) -> EvidencePackage:
        parent_ids = list(follow_up.related_evidence_ids)
        return self._analyze(
            bundle=bundle,
            profile=profile,
            question_id=follow_up.follow_up_id,
            question=follow_up.question,
            granularity=follow_up.analysis_granularity,
            evidence_id=f"E_{follow_up.follow_up_id}",
            branch_kind="follow_up",
            output_dir=output_dir,
            parent_evidence_ids=list(dict.fromkeys(parent_ids)),
            parent_evidence=parent_evidence or [],
            revision_instruction=revision_instruction,
            coding_errors=coding_errors,
        )

    def _analyze(
        self,
        *,
        bundle: DataBundle,
        profile: TrajectoryDataProfile,
        question_id: str,
        question: str,
        granularity: AnalysisGranularity,
        evidence_id: str,
        branch_kind: str,
        output_dir: Path,
        parent_evidence_ids: list[str],
        parent_evidence: list[EvidencePackage],
        revision_instruction: str,
        coding_errors: list[str] | None,
    ) -> EvidencePackage:
        output_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("A2-code %s generating analysis program", question_id)
        code = self.code_generator.generate(
            bundle=bundle,
            profile=profile,
            question=question,
            granularity=granularity,
            branch_kind=branch_kind,
            output_dir=output_dir / "attempt_1",
            parent_evidence=parent_evidence,
            revision_instruction=revision_instruction,
            coding_errors=coding_errors,
        )

        raw_result: dict[str, Any] | None = None
        successful_dir: Path | None = None
        failure = ""
        failure_history: list[dict[str, object]] = []
        for attempt in range(self.config.max_code_repairs + 1):
            attempt_dir = output_dir / f"attempt_{attempt + 1}"
            if attempt:
                LOGGER.info("A2-code %s repairing code, attempt %d", question_id, attempt + 1)
                code = self.code_generator.repair_code(
                    question=question,
                    profile=profile,
                    output_dir=attempt_dir,
                    granularity=granularity,
                    branch_kind=branch_kind,
                    parent_evidence=parent_evidence,
                    previous_code=code,
                    failure=failure,
                    failure_history=failure_history,
                    coding_errors=coding_errors,
                )
            LOGGER.info("A2-execute %s executing attempt %d", question_id, attempt + 1)
            try:
                execution = self.executor.run(
                    code,
                    attempt_dir,
                    input_files={item.file_id: str(item.path) for item in bundle.files},
                )
                if not execution.success:
                    raise ValueError(execution.diagnostic)
                raw_result = load_raw_analysis_result(attempt_dir)
                self._validate_scope(raw_result, profile)
                notes = str(raw_result.get("execution_notes", ""))
                if execution.stdout.strip():
                    raw_result["execution_notes"] = (
                        notes + "\n" + execution.stdout.strip()
                    ).strip()
                successful_dir = attempt_dir
                break
            except Exception as exc:
                failure = f"{type(exc).__name__}: {exc}"
                failure_history.append({
                    "attempt": attempt + 1,
                    "failure": failure,
                })
                LOGGER.warning(
                    "A2-execute %s attempt %d failed: %s", question_id, attempt + 1, failure
                )

        chart_failures = sum(
            self._is_visualization_failure(str(item["failure"]))
            for item in failure_history
        )
        fallback_attempted = False
        if (
            raw_result is None and chart_failures >= 2 and failure_history
            and self._is_visualization_failure(str(failure_history[-1]["failure"]))
        ):
            fallback_attempted = True
            fallback_dir = output_dir / "metadata_fallback"
            last_error = str(failure_history[-1]["failure"]).strip().splitlines()[-1][:300]
            LOGGER.warning(
                "A2-execute %s visualization failed %d times; trying metadata-only fallback",
                question_id, chart_failures,
            )
            try:
                fallback_code = self.code_generator.repair_code(
                    question=question,
                    profile=profile,
                    output_dir=fallback_dir,
                    granularity=granularity,
                    branch_kind=branch_kind,
                    parent_evidence=parent_evidence,
                    previous_code=code,
                    failure=failure,
                    failure_history=failure_history,
                    coding_errors=coding_errors,
                    metadata_only=True,
                )
                execution = self.executor.run(
                    fallback_code,
                    fallback_dir,
                    input_files={item.file_id: str(item.path) for item in bundle.files},
                    metadata_only_fallback=True,
                )
                if not execution.success:
                    raise ValueError(execution.diagnostic)
                raw_result = load_raw_analysis_result(fallback_dir)
                self._validate_scope(raw_result, profile)
                if raw_result.get("analysis_mode") == "metadata_only":
                    raw_result["visualization_fallback"] = {
                        "failed_chart_attempts": chart_failures,
                        "last_chart_error": last_error,
                    }
                    notes = str(raw_result.get("execution_notes", ""))
                    raw_result["execution_notes"] = (
                        notes + "\nVisualization unavailable after repeated chart failures; "
                        "interpret computed statistics and table only. " + last_error
                    ).strip()
                if execution.stdout.strip():
                    raw_result["execution_notes"] = (
                        str(raw_result.get("execution_notes", "")) + "\n"
                        + execution.stdout.strip()
                    ).strip()
                write_json(fallback_dir / "raw_analysis_result.json", raw_result)
                successful_dir = fallback_dir
                LOGGER.warning(
                    "A2-execute %s recovered with %s evidence",
                    question_id, raw_result.get("analysis_mode", "charted"),
                )
            except Exception as exc:
                fallback_failure = f"{type(exc).__name__}: {exc}"
                write_json(fallback_dir / "fallback_failure.json", {
                    "chart_failures": chart_failures,
                    "last_chart_error": last_error,
                    "fallback_failure": fallback_failure,
                })
                LOGGER.warning(
                    "A2-execute %s metadata-only fallback failed: %s",
                    question_id, fallback_failure,
                )
                failure += f"; metadata-only fallback failed: {fallback_failure}"

        if raw_result is None or successful_dir is None:
            attempt_count = self.config.max_code_repairs + 1
            attempts = f"{attempt_count} attempt{'s' if attempt_count != 1 else ''}"
            if fallback_attempted:
                attempts += " and one metadata-only fallback"
            raise RuntimeError(
                f"A2 code execution failed for {question_id} after {attempts}: {failure}"
            )

        LOGGER.info("A2-interpret %s interpreting computed result", question_id)
        agreement_context = next((e.agreement_context for e in parent_evidence
                                  if e.agreement_context), {})
        interpretation = self.result_interpreter.interpret(
            goal=bundle.goal,
            question=question,
            granularity=granularity,
            raw_result=raw_result,
            output_dir=successful_dir,
            agreement_context=agreement_context,
        )
        return EvidencePackage(
            evidence_id=evidence_id,
            question_id=question_id,
            question=question,
            analysis_granularity=granularity,
            insight=interpretation.insight,
            statistics=raw_result["statistics"],
            visualizations=raw_result["visualizations"],
            supporting_scope=raw_result["supporting_scope"],
            tables=raw_result["tables"],
            justification=interpretation.justification,
            limitations=interpretation.limitations,
            evidence_strength=interpretation.evidence_strength,
            execution_notes=str(raw_result.get("execution_notes", "")),
            code_path=str((successful_dir / "analysis.py").resolve()),
            raw_result_path=str((successful_dir / "raw_analysis_result.json").resolve()),
            interpretation_path=str((successful_dir / "interpretation.json").resolve()),
            parent_evidence_ids=parent_evidence_ids,
            agreement_context=agreement_context,
            comparison_records=raw_result.get("comparison_records", []),
        )

    @staticmethod
    def _validate_scope(result: dict[str, Any], profile: TrajectoryDataProfile) -> None:
        scope = result["supporting_scope"]
        files = scope.get("files", [])
        fields = scope.get("fields", [])
        ids = scope.get("trajectory_or_user_ids", [])
        if not all(isinstance(value, str) for value in files) or not set(files) <= profile.file_ids():
            raise ValueError("supporting_scope.files contains unknown file IDs")
        fields_by_file = {
            item.file_id: {column.name for column in item.columns}
            for item in profile.files
        }
        aliases = {
            Path(item.path).stem: item.file_id
            for item in profile.files
        }
        known_fields = {field for values in fields_by_file.values() for field in values}
        if not isinstance(fields, list) or not all(isinstance(value, str) for value in fields):
            raise ValueError("supporting_scope.fields must be a list of field names")
        normalized_fields: list[str] = []
        for value in fields:
            if value in known_fields:
                normalized_fields.append(value)
                continue
            if "." in value:
                prefix, field = value.rsplit(".", 1)
                file_id = prefix if prefix in fields_by_file else aliases.get(prefix)
                if file_id and field in fields_by_file[file_id]:
                    normalized_fields.append(field)
                    continue
            raise ValueError(f"supporting_scope.fields contains unknown field: {value}")
        scope["fields"] = list(dict.fromkeys(normalized_fields))
        if not isinstance(ids, list):
            raise ValueError("supporting_scope.trajectory_or_user_ids must be a list")
