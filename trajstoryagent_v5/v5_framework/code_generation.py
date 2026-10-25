from __future__ import annotations

import json
import re
from pathlib import Path

from trajstoryagent_v5.v5_framework.analysis_tools import TOOL_REFERENCE
from trajstoryagent_v5.v5_framework.llm import LLMClient
from trajstoryagent_v5.v5_framework.models import (
    AnalysisGranularity,
    DataBundle,
    TrajectoryDataProfile,
)
from trajstoryagent_v5.v5_framework.models import EvidencePackage


def extract_python_source(response: str) -> str:
    """Accept raw Python or a Markdown Python fence, never a JSON code field."""
    blocks = re.findall(
        r"```(?:python)?\s*(.*?)```", response, flags=re.IGNORECASE | re.DOTALL
    )
    code = "\n".join(block.strip() for block in blocks) if blocks else response.strip()
    if not code:
        raise ValueError("code generator returned no Python source")
    return code


class AnalysisCodeGenerator:
    """Generate Python directly from a question and the available data."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def generate(
        self,
        *,
        bundle: DataBundle,
        profile: TrajectoryDataProfile,
        question: str,
        granularity: AnalysisGranularity,
        branch_kind: str,
        output_dir: Path,
        parent_evidence: list[EvidencePackage] | None = None,
        revision_instruction: str = "",
        coding_errors: list[str] | None = None,
    ) -> str:
        catalog = [
            {"file_id": item.file_id, "path": item.path, "role": item.role}
            for item in profile.files
        ]
        revision = (
            f"\nA3 REVISION INSTRUCTION (same question and granularity):\n"
            f"{revision_instruction}\n"
            if revision_instruction
            else ""
        )
        stage_contract = self._stage_contract(branch_kind)
        error_context = self._coding_error_context(branch_kind, coding_errors)
        prompt = f"""
HIGH-LEVEL GOAL:
{bundle.goal}

QUESTION (must remain unchanged):
{question}

FIXED ANALYSIS GRANULARITY PLANNED BY THE QUESTION AGENT:
{json.dumps(granularity.to_dict(), ensure_ascii=False)}

PARENT ANALYSIS CONTEXT (for follow-up questions):
{self._parent_context(parent_evidence)}
{error_context}

A2 BRANCH TYPE:
{branch_kind}

SIMPLE-FIRST ANALYSIS CONTRACT:
{stage_contract}

NAMED INPUT FILES:
{json.dumps(catalog, ensure_ascii=False)}

TRAJECTORY DATA PROFILE:
{json.dumps(profile.to_dict(), ensure_ascii=False)[:45000]}
{revision}
Generate Python that loads the full input files and computes the results needed to answer
the question at the fixed analysis granularity. Do not use profile samples as evidence.

STABLE ANALYSIS TOOL API:
{TOOL_REFERENCE}

The code must:
- perform the analysis rather than hard-code results;
- answer the fixed question's objective using all related metrics it requests; do not
  narrow a multi-metric comparison to one convenient field or add adjacent hypotheses;
- load only listed local inputs through AnalysisSession.load_table(), never direct paths;
- when combining tables, inspect the relevant fields and relationship, validate
  the chosen key and cardinality, and avoid unintended many-to-many expansion;
- prefer the supplied trajectory/statistics helpers over reimplementing routine operations;
- use the simplest sufficient method: counts/proportions, grouped summaries, direct
  ordering or geometry checks before considering statistical models;
- create one or two distinct decision-relevant visualizations through
  session.save_figure(); use separate filenames and figures when saving two;
  the session automatically includes every saved figure in emit_result(), so do
  not reconstruct their metadata from the images;
- choose the visualization type to match the question and attach its actual plotted data
  as the required structured observation: chart_type, x_axis, y_axis, and statistic_names;
  use at most 50 representative axis values and preserve multiple plotted series as a
  name-to-values mapping rather than collapsing them into a prose description;
- emit every decisive value encoded by the plot in the bounded result table or statistics
  as well, so the structured visual evidence remains independently auditable;
- emit at most one bounded result table and a small set of decisive statistics;
- finish with session.emit_result(), which writes the raw result and scope automatically;
- put computed values, including null/negative results, into statistics or tables;
- never write an insight, answer, justification, narrative, or interpretation;
- target 60--140 lines and avoid duplicate calculations or alternate methods.

Return Python source only. Do not return JSON and do not explain the code. A Markdown
```python fence is accepted, but raw Python is preferred.
"""
        response = self.llm.complete(
            system=(
                "Generate the analysis program. Output Python source only; never output "
                "JSON, analysis prose, or interpreted conclusions."
            ),
            user=prompt,
            json_mode=False,
        )
        return extract_python_source(response)

    def repair_code(
        self,
        *,
        question: str,
        profile: TrajectoryDataProfile,
        output_dir: Path,
        granularity: AnalysisGranularity,
        branch_kind: str,
        previous_code: str,
        failure: str,
        failure_history: list[dict[str, object]] | None = None,
        parent_evidence: list[EvidencePackage] | None = None,
        coding_errors: list[str] | None = None,
        metadata_only: bool = False,
    ) -> str:
        catalog = [
            {"file_id": item.file_id, "path": item.path, "role": item.role}
            for item in profile.files
        ]
        error_context = self._coding_error_context(branch_kind, coding_errors)
        visualization_contract = (
            "This is the final metadata-only fallback after repeated visualization failures. "
            "Do not call plotting helpers or session.save_figure(). Remove figure creation "
            "and save calls while preserving the original data calculations and question. "
            "Put the decisive computed values formerly represented by the chart, including "
            "all metrics requested by the question, into statistics or one bounded result "
            "table. Do not infer values from the failed image or error text. "
            "Use session.emit_result() with computed evidence. This attempt alone is "
            "allowed to omit the visualization; "
            "supporting_scope still comes from AnalysisSession.load_table()."
            if metadata_only else
            "Preserve one primary analysis, at most one table and one or two distinct "
            "structured visualizations; use unique filenames and repair missing "
            "chart_type/x_axis/y_axis/statistic_names metadata. Remove unrelated "
            "or duplicate analysis while repairing."
        )
        prompt = f"""
QUESTION (unchanged):
{question}

FIXED ANALYSIS GRANULARITY:
{json.dumps(granularity.to_dict(), ensure_ascii=False)}

PARENT ANALYSIS CONTEXT (for follow-up questions):
{self._parent_context(parent_evidence)}
{error_context}

A2 BRANCH TYPE AND SIMPLE-FIRST CONTRACT:
{branch_kind}
{self._stage_contract(branch_kind)}

STABLE ANALYSIS TOOL API:
{TOOL_REFERENCE}

VISUALIZATION CONTRACT FOR THIS ATTEMPT:
{visualization_contract}

NAMED INPUT FILES:
{json.dumps(catalog, ensure_ascii=False)}

EXECUTION OR VALIDATION FAILURE:
{failure}

FULL FAILURE HISTORY (oldest to newest):
{json.dumps(failure_history or [], ensure_ascii=False)}

Repeated failures mean an earlier repair was ineffective. Use this history to identify
the root cause, then inspect every occurrence of the implicated field, helper, shape, or
metadata pattern in the complete previous program before producing the next revision.

PREVIOUS CODE:
{previous_code}

Repair only the Python source. Preserve the question, analysis granularity, and intended calculation. The repaired
code must load approved inputs through AnalysisSession.load_table() and finish with
session.emit_result(). Supporting scope is derived automatically and must not be manually
constructed. Preserve one primary analysis and at most one table. It must not produce a
insight or justification.
For cross-table operations, check the chosen key and cardinality against the
available tables before merging; do not infer a key solely from similar names.

Return Python source only. Do not wrap it in JSON and do not explain it.
"""
        response = self.llm.complete(
            system=(
                "You are TrajStory's Python code repair worker. Diagnose and repair the "
                "supplied analysis program. Output the complete "
                "repaired Python source only; never output JSON or interpretation."
            ),
            user=prompt,
            json_mode=False,
        )
        return extract_python_source(response)

    @staticmethod
    def _parent_context(evidence: list[EvidencePackage] | None) -> str:
        parents = []
        for item in evidence or []:
            parents.append({
                "evidence_id": item.evidence_id,
                "question": item.question,
                "insight": item.insight,
                "statistics": item.statistics,
                "tables": item.tables,
                "visualizations": item.visualizations,
                "execution_notes": item.execution_notes,
                "agreement_context": item.agreement_context,
            })
        return json.dumps(parents, ensure_ascii=False, default=str)

    @staticmethod
    def _coding_error_context(branch_kind: str, errors: list[str] | None) -> str:
        if branch_kind != "follow_up" or not errors:
            return ""
        return (
            "\nFIRST-ROUND CODING ERROR SUMMARY (implementation cautions only; "
            "not findings or reasons to change this question):\n"
            + json.dumps(errors, ensure_ascii=False)
            + "\nAvoid repeating these code or output-contract mistakes.\n"
        )

    @staticmethod
    def _stage_contract(branch_kind: str) -> str:
        if branch_kind == "initial":
            return (
                "Address one clear descriptive or comparative objective using all related metrics "
                "requested by the question. Multiple metrics and joint grouping conditions "
                "are allowed within that objective; combine them in the bounded result table "
                "rather than dropping metrics or inventing additional objectives. "
                "Do not add robustness checks, causal controls, alternate "
                "baselines, clustering, or predictive models unless the fixed question "
                "explicitly requires one. Leave other goal-relevant directions, connections, "
                "or refinements to follow-up branches."
            )
        if branch_kind == "follow_up":
            return (
                "Answer the selected next question directly, whether it broadens coverage, "
                "connects existing evidence, or refines a prior result. Include all related metrics "
                "it requests. Multiple metrics and joint grouping conditions can serve the same "
                "objective. Reuse the "
                "parent computation only where relevant and do not introduce a second analysis objective or "
                "multiple competing baselines unless the question explicitly requests a method comparison. "
                "Follow parent agreement_context.use_rules: deferred insights are hypotheses to check, "
                "not premises. Preserve method/scope conditions and never mix incompatible denominators."
            )
        raise ValueError(f"unknown A2 branch kind: {branch_kind}")
