"""Formal v5 planning implementation."""
from trajstoryagent_v5.reasoning_agents import CrossEvidenceExplorer


class _ReviewEvidence:
    """Bridge the shared reviewer's internal key filter; client emits insight."""
    def __init__(self, evidence):
        self.evidence = evidence

    def __getattr__(self, name):
        return getattr(self.evidence, name)

    def to_dict(self):
        data = self.evidence.to_dict()
        if 'insight' in data:
            data['claim'] = data.pop('insight')
        return data


class _PlanningContextClient:
    def __init__(self, client, context):
        self.client, self.context = client, context

    def complete(self, *, system, user, **kwargs):
        return self.client.complete(
            system=system, user=user + '\nDATA CONTEXT:\n' + self.context, **kwargs)


class RefinedContextExplorer(CrossEvidenceExplorer):
    """Supply accepted conditional facts throughout planning and review.

    Memory-mode prompts receive a request-scoped context supplement, including
    review and regenerated candidates. The goal and question-selection policy
    are preserved, and single-question repairs use the same context contract.
    """

    def explore(self, **kwargs):
        if kwargs.get('memory') is None:
            # The non-Memory review filters an internal claim key. Its prompt
            # passes through InsightClient, so the provider still sees insight.
            kwargs = {**kwargs, 'evidence': [_ReviewEvidence(e) for e in kwargs['evidence']]}
        context = kwargs.get('context', '')
        if kwargs.get('memory') is None or not context:
            return super().explore(**kwargs)
        delegate = CrossEvidenceExplorer(
            _PlanningContextClient(self.llm, context),
            coverage_review_enabled=self.coverage_review_enabled,
        )
        try:
            return delegate.explore(**kwargs)
        finally:
            self.last_candidates = delegate.last_candidates
            self.last_duplicate_reviews = delegate.last_duplicate_reviews
