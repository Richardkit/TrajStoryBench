"""Planner instructions for a strictly scheduled TrajStory 3+3 run."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

from pydantic import ConfigDict, Field

from src.prompt.template.planning import AGENT_MESSAGE_PROMPT, SYSTEM_PROMPT
from src.prompt.types import Prompt
from src.registry import PROMPT


TRAJSTORY_PLANNING_RULES = """
<planning_rules>
Follow this staged protocol. Do not skip or reorder stages.

**Round 1 — breadth-first discovery**
Dispatch exactly these three agents concurrently, each with the sanitised
`spec_path`, the analytical goal, and ONE distinct, focused question. Do not
combine several independent metrics, comparisons, or hypotheses in one
specialist task; each specialist may perform one initial analysis and at most
one deeper check of that same question:
- `trajstory_temporal_spatial`
- `trajstory_behavior_comparison`
- `trajstory_anomaly`

**Round 2 — reverse evidence check**
Dispatch only `trajstory_evidence_verifier`. Ask it to recompute decisive
quantities, reject unsupported/duplicate/overstated candidates, and identify
coverage gaps for a second round. Prior-agent results are automatically
attached by the bus. This is validation, not a new exploration task.

**Round 3 — gap-directed second discovery**
Dispatch exactly the same three specialist agents concurrently, once each.
Give each a distinct, explicit question based on the first verifier's coverage
gaps and the original goal. Do not repeat or lightly paraphrase a question or
finding from round 1. Each specialist must use the same sanitised `spec_path`
and produce new quantitative evidence where supported. Each is again limited
to one initial analysis and at most one deeper check. This is the final
exploration round: six specialist dispatches total across rounds 1 and 3.

**Round 4 — final reverse evidence check**
Dispatch only `trajstory_evidence_verifier`. Recompute decisive quantities for
new candidates, consider both exploration rounds and the first verifier's
decisions, reject duplicates or unsupported claims, and return a consolidated
set of verified findings. Do not start new exploratory questions.

**Round 5 — synthesis**
Dispatch only `trajstory_synthesis`. Ask it to select a diverse,
non-redundant set from the consolidated verified findings and return the
required final JSON. Prior-agent results are automatically attached by the bus.

**Round 6 — completion**
Set `is_done=true`, with no dispatches. Copy the synthesis agent's JSON object
verbatim into `final_result`; do not wrap it in Markdown or add new claims.

If an agent fails, record the failure and continue with successful evidence.
The AgentBus enforces the exact dispatch roster for every round. Do not add,
remove, duplicate, or reorder stages, even if a specialist failed.
Never read or expose gold `insights`, `insight_list`, or `summary` fields.
Agent names must match exactly. Every dispatched task must remain explicit and
self-contained. The required final object is:
{"insights": [{"statement": "...", "evidence": ["..."],
"dimensions": ["..."], "confidence": 0.0}], "summary": "..."}
</planning_rules>
"""


TRAJSTORY_REASONING_RULES = """
<reasoning_rules>
Determine the current stage from the round number and execution history.
Check that claims have quantitative evidence, that dimensions are diverse,
and that rejected claims are not restored. Keep reasoning concise.
</reasoning_rules>
"""


TRAJSTORY_SYSTEM_PROMPT = deepcopy(SYSTEM_PROMPT)
TRAJSTORY_SYSTEM_PROMPT["name"] = "trajstory_planning_system_prompt"
TRAJSTORY_SYSTEM_PROMPT["description"] = "Planner policy for TrajStory multi-agent analysis"
TRAJSTORY_SYSTEM_PROMPT["variables"]["planning_rules"]["variables"] = TRAJSTORY_PLANNING_RULES
TRAJSTORY_SYSTEM_PROMPT["variables"]["reasoning_rules"]["variables"] = TRAJSTORY_REASONING_RULES

TRAJSTORY_AGENT_MESSAGE_PROMPT = deepcopy(AGENT_MESSAGE_PROMPT)
TRAJSTORY_AGENT_MESSAGE_PROMPT["name"] = "trajstory_planning_agent_message_prompt"
TRAJSTORY_AGENT_MESSAGE_PROMPT["description"] = "Per-round TrajStory orchestration context"


@PROMPT.register_module(force=True)
class TrajstoryPlanningSystemPrompt(Prompt):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    type: str = Field(default="system_prompt")
    name: str = Field(default="trajstory_planning")
    description: str = Field(default="System prompt for TrajStory planning")
    require_grad: bool = Field(default=False)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    prompt_config: Dict[str, Any] = Field(default=TRAJSTORY_SYSTEM_PROMPT)


@PROMPT.register_module(force=True)
class TrajstoryPlanningAgentMessagePrompt(Prompt):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    type: str = Field(default="agent_message_prompt")
    name: str = Field(default="trajstory_planning")
    description: str = Field(default="Per-round context for TrajStory planning")
    require_grad: bool = Field(default=False)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    prompt_config: Dict[str, Any] = Field(default=TRAJSTORY_AGENT_MESSAGE_PROMPT)
