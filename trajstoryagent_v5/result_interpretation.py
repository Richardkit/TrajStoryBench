from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

from trajstoryagent_v5.json_utils import extract_json_object, write_json
from trajstoryagent_v5.llm import LLMClient
from trajstoryagent_v5.models import AnalysisGranularity


@dataclass(frozen=True)
class ResultInterpretation:
    insight: str
    justification: str
    limitations: list[str]
    evidence_strength: Literal["strong", "moderate", "weak"]


class AnalysisResultInterpreter:
    """Interpret already-computed raw results; never generate or repair code."""

    def __init__(self, llm: LLMClient, max_repairs: int = 2) -> None:
        self.llm = llm
        self.max_repairs = max_repairs

    def interpret(
        self,
        *,
        goal: str,
        question: str,
        granularity: AnalysisGranularity,
        raw_result: dict[str, Any],
        output_dir: Path,
        agreement_context: dict[str, Any] | None = None,
    ) -> ResultInterpretation:
        metadata_only = raw_result.get("analysis_mode") == "metadata_only"
        fallback_instruction = (
            "The chart failed after the code repair budget. There is no structured "
            "visual observation. Base the conclusion only on computed statistics "
            "and result-table values, and acknowledge the missing visualization."
            if metadata_only else ""
        )
        prompt = f"""
HIGH-LEVEL GOAL:
{goal}

QUESTION:
{question}

ANALYSIS GRANULARITY:
{json.dumps(granularity.to_dict(), ensure_ascii=False)}

COMPUTED RAW RESULT:
{json.dumps(raw_result, ensure_ascii=False, default=str)}

PARENT AGREEMENT CONDITIONS (if any):
{json.dumps(agreement_context or {}, ensure_ascii=False)}
Respect the use_rules and distinguish deferred hypotheses from computed findings.
If the current result resolves or contradicts a prior argument, state its argument ID,
the matching scope, and the new computed basis in justification. Otherwise preserve
uncertainty. Parent agreement does not supply new measured facts.

Interpret only the supplied computed result. Treat each visualization.observation as
structured visual evidence: inspect its chart_type, x_axis values, y_axis values/series,
and referenced statistics to identify decision-relevant patterns, contrasts, extrema, or
exceptions. Do not infer from the image path or prose description alone. Do not write
Python, request another analysis, invent values, or use facts absent from the result.
State a negative/null result honestly. The justification must point to specific supplied
statistics, table values, or structured x/y visual-observation values.
Limitations must describe only material constraints visible in the result.
{fallback_instruction}

Return exactly:
{{"insight":"one coherent factual insight, at most 180 words","justification":"evidence-linked explanation",
  "limitations":["..."],"evidence_strength":"strong|moderate|weak"}}
"""
        previous = ""
        error = ""
        for _ in range(self.max_repairs + 1):
            repair = (
                f"\nYour prior interpretation was invalid: {error}\n"
                f"Previous response:\n{previous}\nRepair the interpretation only.\n"
                if error
                else ""
            )
            previous = self.llm.complete(
                system=(
                    "You are TrajStory's post-execution result interpreter. Interpret computed "
                    "evidence; never generate or repair code. Return JSON only."
                ),
                user=prompt + repair,
            )
            try:
                interpretation = self._parse(extract_json_object(previous))
                if metadata_only:
                    limitation = (
                        "The visualization failed; this conclusion uses computed "
                        "statistics and table values only."
                    )
                    if limitation not in interpretation.limitations:
                        interpretation = replace(
                            interpretation,
                            limitations=[*interpretation.limitations, limitation],
                        )
                write_json(output_dir / "interpretation.json", interpretation)
                return interpretation
            except (TypeError, ValueError) as exc:
                error = str(exc)
        raise RuntimeError(f"result interpretation failed after {self.max_repairs + 1} attempts: {error}")

    @staticmethod
    def _parse(payload: dict[str, Any]) -> ResultInterpretation:
        insight = payload.get("insight")
        justification = payload.get("justification")
        limitations = payload.get("limitations", [])
        strength = str(payload.get("evidence_strength", "")).lower()
        if not isinstance(insight, str) or not insight.strip():
            raise ValueError("insight must be a non-empty string")
        if len(insight.split()) > 180:
            raise ValueError("insight exceeds 180-word cap")
        if not isinstance(justification, str) or not justification.strip():
            raise ValueError("justification must be a non-empty string")
        if not isinstance(limitations, list) or not all(isinstance(item, str) for item in limitations):
            raise ValueError("limitations must be a list of strings")
        if strength not in {"strong", "moderate", "weak"}:
            raise ValueError("evidence_strength must be strong, moderate, or weak")
        return ResultInterpretation(
            insight.strip(),
            justification.strip(),
            [item.strip() for item in limitations if item.strip()],
            cast(Literal["strong", "moderate", "weak"], strength),
        )
