from __future__ import annotations

import json
import re
from pathlib import Path

from .models import DataBundle, DataFile


SUPPORTED_SUFFIXES = {".csv", ".parquet", ".pq"}


def _file_id(value: str, used: set[str]) -> str:
    base = re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower() or "data"
    candidate = base
    index = 2
    while candidate in used:
        candidate = f"{base}_{index}"
        index += 1
    used.add(candidate)
    return candidate


def load_trajstory_case(case_json: str | Path, repo_root: str | Path | None = None,
                        *, input_csvs: bool = False) -> DataBundle:
    """Load only model-facing inputs from a TrajStory/InsightBench-style case JSON.

    Ground-truth insights, summaries, notebook analysis, and visualization instructions
    are intentionally ignored to prevent evaluation leakage.
    input_csvs selects projected CSVs before constructing any DataFile; existing
    callers retain their original input policy unless they opt in.
    """
    case_path = Path(case_json).expanduser().resolve()
    root = Path(repo_root).expanduser().resolve() if repo_root else Path.cwd().resolve()
    payload = json.loads(case_path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    goal = metadata.get("goal")
    if not goal:
        raise ValueError(f"case has no metadata.goal: {case_path}")

    def resolve(value: str) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    used_ids: set[str] = set()
    used_paths: set[Path] = set()
    files: list[DataFile] = []

    def add(path_value: str | None, preferred_id: str, role: str) -> None:
        if not path_value:
            return
        path = resolve(path_value)
        if input_csvs:
            from .input_paths import resolve_input_csv
            path = resolve_input_csv(path)
            if path is None:
                if preferred_id == "main":
                    raise ValueError(f"Primary dataset has no retained input columns: {path_value}")
                return
        if path in used_paths or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            return
        used_paths.add(path)
        files.append(DataFile(_file_id(preferred_id, used_ids), path, role))

    add(payload.get("dataset_csv_path"), "main", "primary dataset")
    add(payload.get("user_dataset_csv_path"), "secondary", "secondary trajectory/context dataset")
    references = payload.get("input_references", [])
    if not isinstance(references, list):
        references = []
    for reference in references:
        path_value = reference.get("path")
        preferred = Path(path_value).stem if path_value else "reference"
        add(path_value, preferred, reference.get("role", "reference dataset"))

    return DataBundle(
        goal=str(goal),
        files=tuple(files),
        context=str(metadata.get("dataset_description", "")),
        role=str(metadata.get("role", "trajectory analyst")),
    )
