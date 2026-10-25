"""Specialist agents used by the TrajStory AgentBus configuration."""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional

from pydantic import ConfigDict, Field

from src.agent.tool_calling_agent import ToolCallingAgent
from src.agent.trajstory_exploration_budget import TrajstoryExplorationBudget
from src.agent.types import AgentResponse
from src.registry import AGENT


_COMMON_RULES = """
TrajStory analysis rules:
1. Use only the sanitised case specification and its CSV tables; never search
   for or open insight.json, reference insights, summaries, or answer files.
2. Use `trajstory_analysis` for every numerical claim. Report denominators,
   group sizes, units, and the exact comparison supporting each conclusion.
3. Separate observations from interpretations. Do not make causal claims from
   correlations or descriptive trajectory data.
4. Prefer several non-overlapping, decision-relevant findings over paraphrases
   of the same dominant pattern.
5. Return compact JSON through the `done` tool. Do not return Markdown fences.
6. Before calling `done`, call `trajstory_analysis` with the sanitised
   `spec_path` and inspect its output. Your next step receives the executed
   tool results in `<executed_tool_results>`. Do not claim the tool is
   unavailable merely because no table values were in the initial task.
7. A schema preview is not a numerical finding. Use the schema already in the
   assigned case specification; do not spend analysis calls searching for it.
"""

_EXPLORATION_BUDGET_RULES = """
Exploration budget for this ONE assigned question:
- First run one focused numerical analysis of the assigned question. Do not
  bundle unrelated new questions into its Python code.
- After inspecting that result, you may run at most ONE evidence-driven deeper
  analysis. It must clarify the same question, not open another topic. The
  follow-up is optional.
- Both calls use `trajstory_analysis` with the exact sanitised `spec_path` and
  Python `code`; no additional metadata fields are needed. If code fails,
  repair it without changing the question, metric, population, or hypothesis.
  At most two repairs are allowed for each analysis.
- At most one analysis action per model step. Never call `done` in the same
  step as an analysis; read its output first. After one or two successful
  analyses, call `done` with only supported findings. No third analysis.
"""


class _TrajstoryRoleAgent(ToolCallingAgent):
    """Tool-calling agent that prepends stable role and evidence rules."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    role_instructions: ClassVar[str] = ""
    minimum_analysis_calls: ClassVar[int] = 0
    exploration_budget_enabled: ClassVar[bool] = False

    async def __call__(
        self,
        task: str,
        files: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AgentResponse:
        budget_rules = _EXPLORATION_BUDGET_RULES if self.exploration_budget_enabled else ""
        scoped_task = f"{self.role_instructions}\n{_COMMON_RULES}\n{budget_rules}\nAssigned task:\n{task}"
        if self.exploration_budget_enabled:
            kwargs["analysis_budget"] = TrajstoryExplorationBudget()
        return await super().__call__(
            task=scoped_task,
            files=files,
            min_analysis_calls=self.minimum_analysis_calls,
            **kwargs,
        )


@AGENT.register_module(force=True)
class TrajstoryTemporalSpatialAgent(_TrajstoryRoleAgent):
    minimum_analysis_calls: ClassVar[int] = 1
    exploration_budget_enabled: ClassVar[bool] = True
    name: str = Field(default="trajstory_temporal_spatial")
    description: str = Field(
        default=(
            "TrajStory specialist for temporal rhythms, spatial organization, "
            "route geometry, movement progression, and time-by-space interactions."
        )
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
    role_instructions: ClassVar[str] = """
You are the temporal-spatial trajectory specialist. Address only the focused
temporal/spatial question assigned by the planner. Return
{"candidate_insights": [{"statement": ..., "evidence": [...],
"dimensions": [...], "confidence": 0.0-1.0}], "coverage": [...]}.
"""


@AGENT.register_module(force=True)
class TrajstoryBehaviorComparisonAgent(_TrajstoryRoleAgent):
    minimum_analysis_calls: ClassVar[int] = 1
    exploration_budget_enabled: ClassVar[bool] = True
    name: str = Field(default="trajstory_behavior_comparison")
    description: str = Field(
        default=(
            "TrajStory specialist for entity behavior, group comparisons, "
            "sequence/routine patterns, and relationships across input tables."
        )
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
    role_instructions: ClassVar[str] = """
You are the behavior and comparison specialist. Address only the focused
behavior/comparison question assigned by the planner.
Return {"candidate_insights": [{"statement": ..., "evidence": [...],
"dimensions": [...], "confidence": 0.0-1.0}], "coverage": [...]}.
"""


@AGENT.register_module(force=True)
class TrajstoryAnomalyAgent(_TrajstoryRoleAgent):
    minimum_analysis_calls: ClassVar[int] = 1
    exploration_budget_enabled: ClassVar[bool] = True
    name: str = Field(default="trajstory_anomaly")
    description: str = Field(
        default=(
            "TrajStory specialist for robust anomalies, rare regimes, change "
            "points, counter-patterns, and data-quality confounds."
        )
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
    role_instructions: ClassVar[str] = """
You are the anomaly and counter-pattern specialist. Address only the focused
anomaly question assigned by the planner, distinguishing real exceptions from
data-quality artifacts. Return
{"candidate_insights": [{"statement": ..., "evidence": [...],
"dimensions": [...], "confidence": 0.0-1.0}], "coverage": [...],
"data_quality_notes": [...]}.
"""


@AGENT.register_module(force=True)
class TrajstoryEvidenceVerifierAgent(_TrajstoryRoleAgent):
    minimum_analysis_calls: ClassVar[int] = 1
    name: str = Field(default="trajstory_evidence_verifier")
    description: str = Field(
        default=(
            "Reverse-checks candidate TrajStory insights against source tables, "
            "rejecting unsupported, duplicate, fragile, or overstated claims."
        )
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
    role_instructions: ClassVar[str] = """
You are the independent evidence verifier. Recompute the decisive quantities
behind every candidate supplied in prior-agent context. Reject claims that are
unsupported, depend on tiny groups without disclosure, duplicate another
claim, or overstate association. Return {"verified_insights": [...],
"rejected_insights": [{"statement": ..., "reason": ...}],
"coverage_gaps": [...]} where each verified insight preserves statement,
evidence, dimensions, and confidence.
"""


@AGENT.register_module(force=True)
class TrajstorySynthesisAgent(_TrajstoryRoleAgent):
    name: str = Field(default="trajstory_synthesis")
    description: str = Field(
        default=(
            "Selects a broad, non-redundant set of verified trajectory insights "
            "and produces the final InsightBench-compatible summary."
        )
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
    role_instructions: ClassVar[str] = """
You are the final TrajStory editor. Use verified findings from prior-agent
context. Select a broad but non-redundant set; do not restore rejected claims.
The final `done.result` MUST be exactly one JSON object:
{"insights": [{"statement": "...", "evidence": ["..."],
"dimensions": ["..."], "confidence": 0.0}], "summary": "..."}.
The summary must integrate the selected findings without adding new claims.
"""
