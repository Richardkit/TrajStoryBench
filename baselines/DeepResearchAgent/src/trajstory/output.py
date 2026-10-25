"""Validation and conversion for AgentOrchestra TrajStory outputs."""

from __future__ import annotations

import json
from typing import Any


def _json_candidates(text: str):
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        yield "\n".join(lines).strip()
    yield stripped

    start = None
    depth = 0
    quoted = False
    escaped = False
    for index, char in enumerate(text):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start : index + 1]
                start = None


def parse_trajstory_result(text: str) -> dict[str, Any]:
    """Parse the final JSON and expose InsightBench-compatible fields."""

    payload = None
    last_error: Exception | None = None
    for candidate in _json_candidates(text):
        try:
            value = json.loads(candidate)
        except (TypeError, json.JSONDecodeError) as exc:
            last_error = exc
            continue
        if isinstance(value, dict) and "insights" in value:
            payload = value
            break
    if payload is None:
        raise ValueError(f"No valid TrajStory result JSON found: {last_error}")

    raw_insights = payload.get("insights")
    summary = payload.get("summary", "")
    if not isinstance(raw_insights, list) or not raw_insights:
        raise ValueError("Result must contain a non-empty insights list")
    if not isinstance(summary, str):
        raise ValueError("Result summary must be a string when present")

    structured = []
    statements = []
    for index, item in enumerate(raw_insights, start=1):
        if isinstance(item, str):
            item = {"statement": item, "evidence": [], "dimensions": []}
        if not isinstance(item, dict):
            raise ValueError(f"Insight {index} must be a string or object")
        statement = item.get("statement") or item.get("insight")
        if not isinstance(statement, str) or not statement.strip():
            raise ValueError(f"Insight {index} has no statement")
        normalised = dict(item)
        normalised["statement"] = statement.strip()
        normalised.setdefault("evidence", [])
        normalised.setdefault("dimensions", [])
        structured.append(normalised)
        statements.append(statement.strip())

    return {
        "pred_insights": statements,
        "pred_summary": summary.strip(),
        "structured_insights": structured,
        "raw_payload": payload,
    }
