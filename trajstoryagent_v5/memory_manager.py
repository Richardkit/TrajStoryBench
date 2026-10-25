"""A post-A3, case-local Memory Manager for the current TrajStory pipeline.

Question and Finding prepare planning context; coding errors prepare follow-up
code-writing context. Memory never replaces the original evidence used by A2,
A3, A5, or external evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from trajstoryagent_v5.llm import LLMClient
from trajstoryagent_v5.models import AnalysisQuestion, EvidencePackage
from trajstoryagent_v5.reasoning_agents import _structured_call


MEMORY_KEYS = ("Question", "Finding", "Errors appear in coding")
ROLE_PATH = Path(__file__).parent / "roles" / "memory_manager.md"
NUMERIC_LITERAL = re.compile(
    r"(?<![A-Za-z_])[-+]?\d+(?:[.,]\d+)?(?:\s*[–—-]\s*\d+(?:[.,]\d+)?)?"
    r"(?:\s*(?:%|km|meters?|kilometers?|minutes?|mins?|hours?|days?))?",
    flags=re.IGNORECASE,
)
NUMBER_WORD = re.compile(
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty|half|quarter|dozen|hundred|thousand|million)\b",
    flags=re.IGNORECASE,
)
CATEGORY_LABEL = re.compile(
    r"\b(day|route|track|cluster|class|group|type|zone|area|segment|period|"
    r"phase|trip|trajectory|vessel)[\s_-]+(\d+)\b",
    flags=re.IGNORECASE,
)


def _entry_ids(entries: list[str]) -> list[str]:
    return [entry.split(":", 1)[0].strip() for entry in entries]


def _mask_numeric_details(value: str) -> str:
    """Hide measured values in the Manager's input copy, never in evidence."""
    # Day 1 and route_2 are categorical labels, not measured result values.
    # Normalize the former so NUMERIC_LITERAL does not redact the label.
    categories = CATEGORY_LABEL.sub(lambda match: f"{match.group(1)}_{match.group(2)}", value)
    return NUMBER_WORD.sub("[quantity omitted]", NUMERIC_LITERAL.sub("[value omitted]", categories))


def _numeric_result_token(value: str) -> str | None:
    """Find a measured number while allowing categorical labels and IDs."""
    without_categories = CATEGORY_LABEL.sub("", value)
    match = NUMERIC_LITERAL.search(without_categories)
    return match.group(0) if match else None


def _contains_numeric_result(value: str) -> bool:
    return _numeric_result_token(value) is not None


class MemoryManagerAgent:
    """Compress the completed first round into three consumer-specific fields."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self.role = ROLE_PATH.read_text(encoding="utf-8").strip()

    @staticmethod
    def fallback(
        *,
        questions: list[AnalysisQuestion],
        evidence: list[EvidencePackage],
        branch_failures: list[dict[str, object]],
    ) -> dict[str, list[str]]:
        """Keep A4 on the memory path if LLM compression exhausts its retries."""
        def compact(value: str, limit: int) -> str:
            masked = _mask_numeric_details(value)
            masked = re.sub(r"\[(?:value|quantity) omitted\]", "", masked)
            return " ".join(masked.split())[:limit].rstrip(" ,;:-")

        errors = {}
        for item in branch_failures:
            if str(item.get("stage", "")) in {"initial_analysis", "initial_repair"}:
                question_id = str(item.get("branch_id", ""))
                if question_id:
                    errors[question_id] = (
                        f"{question_id}: {compact(str(item.get('error', 'coding failed')), 160)}"
                    )
        return {
            "Question": [
                f"{item.question_id}: {compact(item.question, 280)}" for item in questions
            ],
            "Finding": [
                f"{item.evidence_id}: {item.question_id} — "
                f"{compact(item.claim, 420)}"
                + (f" Limitation: {compact(item.limitations[0], 100)}"
                   if item.limitations else "")
                for item in evidence
            ],
            "Errors appear in coding": list(errors.values()),
        }

    def compress(
        self,
        *,
        goal: str,
        questions: list[AnalysisQuestion],
        evidence: list[EvidencePackage],
        branch_failures: list[dict[str, object]],
    ) -> dict[str, list[str]]:
        """Run only after initial A3 and optional agreement have finished."""
        if not evidence:
            raise ValueError("memory requires A3-validated first-round evidence")
        question_ids = {item.question_id for item in questions}
        evidence_ids = {item.evidence_id for item in evidence}
        origin_by_evidence_id = {item.evidence_id: item.question_id for item in evidence}
        if len(question_ids) != len(questions) or len(evidence_ids) != len(evidence):
            raise ValueError("memory inputs contain duplicate question or evidence IDs")
        if any(item.question_id not in question_ids for item in evidence):
            raise ValueError("validated evidence has no matching initial question")

        source_questions = [
            {"question_id": item.question_id, "question": item.question,
             "analysis_granularity": item.analysis_granularity.to_dict()}
            for item in questions
        ]
        source_findings = [
            {"evidence_id": item.evidence_id, "question_id": item.question_id,
             "claim": _mask_numeric_details(item.claim),
             "limitations": [_mask_numeric_details(text) for text in item.limitations],
             "evidence_strength": item.evidence_strength,
             "agreement_use_rules": item.agreement_context.get("use_rules")}
            for item in evidence
        ]
        coding_failures = [
            {"question_id": str(item.get("branch_id", "")),
             "stage": str(item.get("stage", "")),
             "error_type": str(item.get("error_type", "")),
             "error": str(item.get("error", ""))[:1200]}
            for item in branch_failures
            if str(item.get("stage", "")) in {"initial_analysis", "initial_repair"}
        ]
        coding_failure_ids = {item["question_id"] for item in coding_failures}

        def validate(payload: dict[str, Any]) -> dict[str, list[str]]:
            if set(payload) != set(MEMORY_KEYS):
                raise ValueError("memory must contain exactly the three required fields")
            for key in MEMORY_KEYS:
                entries = payload[key]
                if (not isinstance(entries, list)
                        or any(not isinstance(entry, str) or not entry.strip() for entry in entries)):
                    raise ValueError(f"{key} must be a list of non-empty strings")
            if set(_entry_ids(payload["Question"])) != question_ids or len(payload["Question"]) != len(questions):
                raise ValueError("Question must contain every initial question ID exactly once")
            if set(_entry_ids(payload["Finding"])) != evidence_ids or len(payload["Finding"]) != len(evidence):
                raise ValueError("Finding must contain every validated evidence ID exactly once")
            if any(not re.match(r"^Q\d+:\s*\S", entry) for entry in payload["Question"]):
                raise ValueError("Question entries must begin with Q-id: text")
            if any(not re.match(r"^[A-Za-z][A-Za-z0-9_]*:\s*\S", entry)
                   for entry in payload["Finding"]):
                raise ValueError("Finding entries must begin with evidence-id: text")
            for entry in payload["Question"]:
                if len(entry) > 400:
                    raise ValueError("each Question entry must be at most 400 characters")
            for entry in payload["Finding"]:
                match = re.match(
                    r"^([A-Za-z][A-Za-z0-9_]*):\s*(Q\d+)\s*[—:-]\s*(.+)$", entry
                )
                if not match or origin_by_evidence_id.get(match.group(1)) != match.group(2):
                    raise ValueError("Finding must name its matching originating Q-id")
                qualitative_text = match.group(3)
                numeric_token = _numeric_result_token(qualitative_text)
                if numeric_token is not None:
                    raise ValueError(
                        "Finding must omit numerical result values: "
                        f"{match.group(1)} contains {numeric_token!r}"
                    )
                if len(entry) > 900:
                    raise ValueError("each Finding entry must be at most 900 characters")
            error_entries = payload["Errors appear in coding"]
            if (set(_entry_ids(error_entries)) != coding_failure_ids
                    or len(error_entries) != len(coding_failure_ids)):
                raise ValueError(
                    "Errors appear in coding must summarize each failed first-round coding branch exactly once"
                )
            if any(len(entry) > 400 for entry in error_entries):
                raise ValueError("each coding error summary must be at most 400 characters")
            memory_chars = len(json.dumps(payload, ensure_ascii=False))
            if memory_chars > 4000:
                raise ValueError("memory exceeds the 4000-character planning maximum")
            return {key: payload[key] for key in MEMORY_KEYS}

        return _structured_call(
            self.llm,
            system=self.role,
            user=(
                f"HIGH-LEVEL GOAL:\n{goal}\n\n"
                f"FIRST-ROUND QUESTIONS (including failed branches):\n"
                f"{json.dumps(source_questions, ensure_ascii=False)}\n\n"
                f"VERIFIED FINDINGS ONLY:\n"
                f"{json.dumps(source_findings, ensure_ascii=False)}\n\n"
                f"EXHAUSTED CODING FAILURES:\n"
                f"{json.dumps(coding_failures, ensure_ascii=False)}\n\n"
                "Produce a compressed planning memory, not a transcript. "
                "Return JSON with exactly three keys: Question, Finding, and "
                "Errors appear in coding. Each Question entry must start with its Q-id, "
                "and each Finding entry with its evidence ID and originating Q-id. "
                "Cover every supplied Q-id "
                "and every validated evidence ID exactly once. Preserve qualitative "
                "conclusions, necessary comparison conditions, limitations, and any "
                "applicable agreement restrictions. [value omitted] and "
                "[quantity omitted] in the supplied claims are redactions, not text to "
                "copy into memory. Finding must omit measured result "
                "values in digits or words: counts, percentages, averages, distances, "
                "durations, sample sizes, and ranges. Original evidence keeps them. "
                "Categorical labels such as day_1, Day 1, and route_2 may remain. "
                "A brief count of compared categories is allowed only when essential "
                "to describe the finding's scope. "
                "Retain a numeric question condition only if needed to distinguish its "
                "objective. Use one terse entry per ID: a short analytical objective "
                "for Question, and the qualitative result plus indispensable context "
                "for Finding. Aim for at most 260 characters per Question, 450 per "
                "Finding, and about 2,000 overall; hard limits are 400 per Question, "
                "900 per Finding, and 4,000 total. Do not copy source paragraphs, "
                "statistics, raw code, or repeated goal/profile details. "
                "Do not add unverified findings. For each supplied exhausted coding "
                "failure, keep one concise, actionable entry beginning with its Q-id; "
                "these entries guide the follow-up code writer, not question planning. "
                "Use [] only when no exhausted coding failures were supplied. "
                "Do not propose next questions.\n"
                '{"Question":["Q1: ..."],"Finding":["E1: Q1 — ..."],'
                '"Errors appear in coding":[]}'
            ),
            validator=validate,
            attempts=5,
        )
