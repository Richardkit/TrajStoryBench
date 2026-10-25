from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .analysis_tools import TOOL_LAYER_VERSION


FORBIDDEN_IMPORT_ROOTS = {
    "ftplib",
    "httpx",
    "requests",
    "socket",
    "subprocess",
    "urllib",
}
FORBIDDEN_CALLS = {"compile", "eval", "exec", "__import__"}
FORBIDDEN_ATTRIBUTES = {("os", "system"), ("os", "popen"), ("shutil", "rmtree")}


class UnsafeCodeError(ValueError):
    pass


def validate_generated_code(code: str) -> None:
    """Reject obvious network, process-spawning, and destructive operations.

    This is a guardrail, not an OS sandbox. Production deployments should run the
    executor in a container or similarly isolated worker with read-only inputs.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"generated code is not valid Python: {exc}") from exc
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", 1)[0] in FORBIDDEN_IMPORT_ROOTS:
                    raise UnsafeCodeError(f"forbidden import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root in FORBIDDEN_IMPORT_ROOTS:
                raise UnsafeCodeError(f"forbidden import: {node.module}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                raise UnsafeCodeError(f"forbidden call: {node.func.id}")
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                pair = (node.func.value.id, node.func.attr)
                if pair in FORBIDDEN_ATTRIBUTES:
                    raise UnsafeCodeError(f"forbidden call: {pair[0]}.{pair[1]}")


@dataclass(frozen=True)
class ExecutionResult:
    success: bool
    stdout: str
    stderr: str
    return_code: int | None
    timed_out: bool = False

    @property
    def diagnostic(self) -> str:
        return (
            f"return_code={self.return_code}; timed_out={self.timed_out}\n"
            f"STDOUT:\n{self.stdout[-6000:]}\nSTDERR:\n{self.stderr[-6000:]}"
        )


class PythonCodeExecutor:
    def __init__(self, timeout_seconds: int = 120) -> None:
        self.timeout_seconds = timeout_seconds

    def run(
        self,
        code: str,
        output_dir: Path,
        *,
        input_files: dict[str, str] | None = None,
        metadata_only_fallback: bool = False,
    ) -> ExecutionResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        code_path = output_dir / "analysis.py"
        code_path.write_text(code, encoding="utf-8")
        # Persist the exact generated source before validation so failed
        # attempts remain auditable. Validation still occurs before execution.
        validate_generated_code(code)
        environment = os.environ.copy()
        environment.update(
            {
                "MPLBACKEND": "Agg",
                "TRAJSTORY_OUTPUT_DIR": str(output_dir.resolve()),
                "TRAJSTORY_INPUT_FILES": json.dumps(input_files or {}),
            }
        )
        if metadata_only_fallback:
            environment["TRAJSTORY_METADATA_ONLY_FALLBACK"] = "1"
        else:
            environment.pop("TRAJSTORY_METADATA_ONLY_FALLBACK", None)
        package_root = str(Path(__file__).resolve().parent.parent)
        existing_pythonpath = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = (
            package_root
            if not existing_pythonpath
            else package_root + os.pathsep + existing_pythonpath
        )
        try:
            process = subprocess.run(
                [sys.executable, str(code_path.resolve())],
                cwd=output_dir,
                env=environment,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecutionResult(
                False,
                exc.stdout or "",
                exc.stderr or "",
                None,
                timed_out=True,
            )
        return ExecutionResult(
            process.returncode == 0,
            process.stdout,
            process.stderr,
            process.returncode,
        )


def load_raw_analysis_result(output_dir: Path) -> dict[str, Any]:
    """Load deterministic execution output before any LLM interpretation."""
    result_path = output_dir / "raw_analysis_result.json"
    if not result_path.is_file():
        raise ValueError("analysis did not create raw_analysis_result.json")
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"raw_analysis_result.json is invalid: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("raw_analysis_result.json must contain an object")
    if payload.get("producer") != TOOL_LAYER_VERSION:
        raise ValueError(
            "raw_analysis_result.json must be emitted by AnalysisSession.emit_result()"
        )
    if "claim" in payload or "justification" in payload:
        raise ValueError(
            "raw_analysis_result.json must not contain interpreted claims or justifications"
        )
    for field in ("statistics", "tables", "visualizations"):
        if not isinstance(payload.get(field), list):
            raise ValueError(f"raw_analysis_result.json field {field} must be a list")
    if not isinstance(payload.get("supporting_scope"), dict):
        raise ValueError("raw_analysis_result.json field supporting_scope must be an object")
    if not isinstance(payload.get("execution_notes", ""), str):
        raise ValueError("raw_analysis_result.json field execution_notes must be a string")
    metadata_only = payload.get("analysis_mode") == "metadata_only"
    count = len(payload["visualizations"])
    if (metadata_only and count != 0) or (not metadata_only and count not in (1, 2)):
        raise ValueError("every analysis branch must emit one or two structured visualizations unless metadata-only fallback was used")
    if metadata_only:
        has_statistic = any(
            isinstance(item, dict) and item.get("value") is not None
            for item in payload["statistics"]
        )
        has_table = any(
            isinstance(item, dict) and bool(item.get("rows"))
            for item in payload["tables"]
        )
        if not (has_statistic or has_table):
            raise ValueError(
                "metadata-only fallback requires a computed value or nonempty result table"
            )
    for visualization in payload["visualizations"]:
        if not isinstance(visualization, dict) or not visualization.get("path"):
            raise ValueError("every visualization requires a path")
        path = (output_dir / str(visualization["path"])).resolve()
        if output_dir.resolve() not in path.parents or not path.is_file():
            raise ValueError(f"visualization does not exist inside output directory: {path}")
        observation = visualization.get("observation")
        if not isinstance(observation, dict):
            raise ValueError("every visualization requires a structured observation")
        if not observation.get("chart_type"):
            raise ValueError("visualization observation requires chart_type")
        for axis_name in ("x_axis", "y_axis"):
            axis = observation.get(axis_name)
            if not isinstance(axis, dict) or not axis.get("description") or "value" not in axis:
                raise ValueError(f"visualization observation requires structured {axis_name}")
        visualization["path"] = str(path)
    return payload


def load_evidence_result(output_dir: Path) -> dict[str, Any]:
    """Backward-compatible name for callers; the on-disk contract is now raw-only."""
    return load_raw_analysis_result(output_dir)
