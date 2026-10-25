"""Load TrajStory InsightBench cases without exposing reference answers.

The source ``insight.json`` files contain both model inputs and gold insights.
Agents must never receive those files directly.  This module creates a
sanitised case specification containing only the goal, metadata, table paths,
roles, and bounded schema samples.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


_SAFE_NAME = re.compile(r"[^A-Za-z0-9_]+")
_FORBIDDEN_OUTPUT_KEYS = {"insight_list", "insights", "summary"}


def _resolve_path(value: str | Path, trajstory_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (trajstory_root / path).resolve()


def _table_name(path: Path, used: set[str]) -> str:
    base = _SAFE_NAME.sub("_", path.stem).strip("_").lower() or "reference"
    name = base
    index = 2
    while name in used:
        name = f"{base}_{index}"
        index += 1
    used.add(name)
    return name


def _schema_preview(path: Path, sample_rows: int = 200) -> dict[str, Any]:
    frame = pd.read_csv(path, nrows=sample_rows, low_memory=False)
    columns = []
    for name in frame.columns:
        values = frame[name].dropna().head(3).tolist()
        columns.append(
            {
                "name": str(name),
                "sample_dtype": str(frame[name].dtype),
                "sample_non_null": int(frame[name].notna().sum()),
                "sample_values": [str(value)[:120] for value in values],
            }
        )
    return {
        "file_size_bytes": path.stat().st_size,
        "sample_rows_read": len(frame),
        "columns": columns,
    }


def load_manifest_cases(
    manifest_path: str | Path,
    *,
    trajstory_root: str | Path,
    case_ids: Iterable[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Return selected manifest entries, preserving manifest order."""

    root = Path(trajstory_root).resolve()
    manifest = Path(manifest_path)
    if not manifest.is_absolute():
        manifest = (root / manifest).resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    entries = payload.get("cases")
    if not isinstance(entries, list):
        raise ValueError(f"Manifest has no cases list: {manifest}")

    requested = set(case_ids or [])
    selected = [entry for entry in entries if not requested or entry.get("case_id") in requested]
    missing = requested - {entry.get("case_id") for entry in selected}
    if missing:
        raise ValueError(f"Unknown TrajStory case IDs: {sorted(missing)}")
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        selected = selected[:limit]
    return selected


def build_case_spec(
    entry: dict[str, Any],
    *,
    trajstory_root: str | Path,
    sample_rows: int = 200,
    input_csvs: bool = False,
) -> dict[str, Any]:
    """Build a gold-answer-free specification for one TrajStory case."""

    root = Path(trajstory_root).resolve()
    case_json = _resolve_path(entry["json_path"], root)
    source = json.loads(case_json.read_text(encoding="utf-8"))
    metadata = source.get("metadata") or {}

    used_paths: set[Path] = set()
    used_names = {"main", "secondary"}
    tables: list[dict[str, Any]] = []

    def add_table(name: str, value: str | None, role: str) -> None:
        if not value:
            return
        path = _resolve_path(value, root)
        if path in used_paths:
            return
        if not path.is_file():
            raise FileNotFoundError(f"Missing table for {entry['case_id']}: {path}")
        used_paths.add(path)
        tables.append(
            {
                "name": name,
                "role": role,
                "path": str(path),
                "schema": _schema_preview(path, sample_rows=sample_rows),
            }
        )

    if input_csvs:
        sys.path.insert(0, str(root))
        from baseline_support.case_adapter import load_trajstory_case
        bundle = load_trajstory_case(case_json, root, input_csvs=True)
        for file in bundle.files:
            add_table(file.file_id, str(file.path), file.role)
    else:
        add_table("main", source.get("dataset_csv_path"), "primary trajectory/task table")
        add_table("secondary", source.get("user_dataset_csv_path"), "secondary/user-level table")
        for reference in source.get("input_references") or []:
            ref_path = _resolve_path(reference.get("path", ""), root)
            if ref_path in used_paths:
                continue
            add_table(
                _table_name(ref_path, used_names),
                reference.get("path"),
                reference.get("role") or "reference table",
            )

    if not tables:
        raise ValueError(f"Case {entry['case_id']} contains no input tables")

    spec = {
        "format": "trajstory-agent-input-v1",
        "case_id": entry["case_id"],
        "goal": metadata.get("goal") or "Find important and well-supported trajectory insights.",
        "role": metadata.get("role") or "trajectory data analyst",
        "category": metadata.get("category"),
        "dataset_description": metadata.get("dataset_description"),
        "tables": tables,
    }
    leaked = _FORBIDDEN_OUTPUT_KEYS.intersection(spec)
    if leaked:
        raise AssertionError(f"Gold-answer fields leaked into case spec: {sorted(leaked)}")
    return spec


def write_case_spec(spec: dict[str, Any], path: str | Path) -> Path:
    """Persist a sanitised case specification used by the analysis tool."""

    leaked = _FORBIDDEN_OUTPUT_KEYS.intersection(spec)
    if leaked:
        raise ValueError(f"Refusing to write gold-answer fields: {sorted(leaked)}")
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    return output

