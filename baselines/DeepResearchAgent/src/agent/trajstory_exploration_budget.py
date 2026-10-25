"""Per-dispatch execution budget for TrajStory discovery agents.

One initial analysis and one evidence-driven follow-up may succeed. Failed
code may be repaired twice per phase. Relatedness is handled by the planner
and specialist prompts, without brittle model-generated metadata fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TrajstoryExplorationBudget:
    max_successes: int = 2
    max_repairs_per_phase: int = 2
    successful_analyses: int = 0
    failed_attempts_in_phase: int = 0

    @property
    def phase(self) -> str:
        return "initial" if self.successful_analyses == 0 else "follow_up"

    def authorize(self, args: dict[str, Any]) -> tuple[bool, str]:
        if self.successful_analyses >= self.max_successes:
            return False, "Exploration budget exhausted: return findings with done; no third analysis."
        if self.failed_attempts_in_phase > self.max_repairs_per_phase:
            return False, "Code-repair budget exhausted for this analysis phase. Return supported findings."
        return True, ""

    def record_result(self, success: bool) -> None:
        if success:
            self.successful_analyses += 1
            self.failed_attempts_in_phase = 0
        else:
            self.failed_attempts_in_phase += 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "successful_analyses": self.successful_analyses,
            "max_successes": self.max_successes,
            "failed_attempts_in_phase": self.failed_attempts_in_phase,
            "max_repairs_per_phase": self.max_repairs_per_phase,
            "phase": self.phase if self.successful_analyses < self.max_successes else "complete",
        }
