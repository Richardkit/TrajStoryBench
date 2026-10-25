"""A5 writes only a summary of finalized branch insights; never new insights."""
import json
from trajstoryagent_v5.v5_framework.reasoning_agents import _structured_call
from trajstoryagent_v5.v5_framework.models import BranchInsight, TrajStoryReport


class SummarySynthesizer:
    def __init__(self, llm):
        self.llm = llm

    def synthesize(self, *, goal, questions, follow_ups, evidence):
        insights = [BranchInsight(f'I{i}', item.question_id, item.insight, (item.evidence_id,))
                    for i, item in enumerate(evidence, start=1)]
        def validate(payload):
            if set(payload) != {'summary'}:
                raise ValueError('Return only summary; do not generate a new insights list')
            if not isinstance(payload['summary'], str) or not payload['summary'].strip():
                raise ValueError('summary must be a nonempty string')
            return payload['summary'].strip()
        summary = _structured_call(
            self.llm,
            system='You are TrajStory A5. Summarize finalized branch insights. Return JSON only.',
            user=f'''HIGH-LEVEL GOAL:
{goal}

FINAL INSIGHTS (both rounds have completed conditional decomposition and compression):
{json.dumps([item.to_dict() for item in insights], ensure_ascii=False)}

EVIDENCE CONDITIONS AND LIMITATIONS:
{json.dumps([dict(evidence_id=e.evidence_id, question_id=e.question_id,
                 limitations=e.limitations, agreement_context=e.agreement_context)
             for e in evidence], ensure_ascii=False)}

Generate a coherent, goal-oriented summary based only on the supplied final insights.
Organize related findings, preserve conditions, exceptions, conflicts and uncertainty.
Do not add facts, numbers or causal interpretations absent from these insights.
Respect agreement_context.use_rules; deferred hypotheses are not established findings.
Do not generate, rewrite, merge or replace the insights list. It is fixed by the branches.
Return exactly {{"summary":"..."}}.''',
            validator=validate,
        )
        return TrajStoryReport(insights, summary, evidence, questions, follow_ups)
