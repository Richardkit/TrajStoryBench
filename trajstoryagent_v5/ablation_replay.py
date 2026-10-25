"""Frozen-first-round ablation adapters, migrated without legacy package imports."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import traceback
from . import models as legacy, export_support as refinement
from .models import FrameworkConfig, EvidencePackage
from .controller import TrajStoryAgent
from .conditional_math import conditional_partition, render_facts
from .interpretation import interpret
from .export_support import read, write

def load_profile(data):
    files = []
    for raw in data['files']:
        f = dict(raw)
        f['columns'] = [legacy.ColumnProfile(**c) for c in f['columns']]
        f['relations'] = [legacy.DataRelation(**r) for r in f['relations']]
        files.append(legacy.FileProfile(**f))
    return legacy.TrajectoryDataProfile(files, [legacy.DataRelation(**r) for r in data['cross_file_relations']])

def load_question(data):
    return legacy.AnalysisQuestion(**{**data, 'analysis_granularity': legacy.AnalysisGranularity(**data['analysis_granularity'])})

def load_evidence(data):
    data = dict(data)
    assert data.get('analysis_plan') is None, 'Replay loader requires the saved custom-code workflow'
    data['analysis_granularity'] = legacy.AnalysisGranularity(**data['analysis_granularity'])
    return EvidencePackage(**data)

class SummaryStageLabelClient:
    """Keep A5's task fixed, but do not claim a disabled stage was performed."""
    def __init__(self, client, arm):
        self.client, self.arm = client, arm

    def complete(self, *, system, user, **kwargs):
        label = ('both analysis rounds completed; no conditional refinement'
                 if self.arm == 'without_conditional_refinement' else
                 'both analysis rounds completed; conditional decomposition without compression')
        user = user.replace('both rounds have completed conditional decomposition and compression', label)
        return self.client.complete(system=system, user=user, **kwargs)

def conditional_branch(slug, evidence, bundle, profile, folder, *, cached_first=None):
    """Only conditional partitioning and interpretation; never call compression."""
    folder.mkdir(parents=True, exist_ok=True)
    write(folder / 'original_evidence.json', evidence)
    result = dict(question_id=evidence['question_id'], question=evidence['question'],
                  original_insight=evidence['insight'], status='pending',
                  insights={'conditional_only': evidence['insight']},
                  arm_status={'conditional_only': 'fallback'}, min_support=1,
                  compression_enabled=False)
    if cached_first is not None:
        previous = read(cached_first / 'branch.json')
        original = read(cached_first / 'original_evidence.json')
        assert original['insight'] == evidence['insight'] and original['question'] == evidence['question']
        result.update(status=previous['status'],
                      insights={'conditional_only': previous['insights']['conditional_only']},
                      arm_status={'conditional_only': previous['arm_status']['conditional_only']},
                      reused_first_round_conditional_only=str(cached_first))
        # Preserve only the matching uncompressed artifacts, never compressed outputs.
        for name in ['observations.json', 'conditional_only']:
            source = cached_first / name
            if source.exists():
                target = folder / name
                if source.is_dir():
                    shutil.copytree(source, target)
                else:
                    shutil.copy2(source, target)
        if (folder / 'observations.json').exists():
            obs = read(folder / 'observations.json')
            partitions = [conditional_partition(m, min_support=1) for m in obs['measures']]
            write(folder / 'conditional_partitions.json', partitions)
        write(folder / 'branch.json', result)
        return result
    try:
        obs = refinement.observations(slug, evidence, bundle, profile, folder)
        if not obs['measures']:
            result.update(status='unsupported', reason=obs.get('unsupported_reason'))
        else:
            partitions = [conditional_partition(m, min_support=1) for m in obs['measures']]
            facts = [fact for p in partitions for fact in render_facts(p, None)]
            write(folder / 'conditional_partitions.json', partitions)
            target = folder / 'conditional_only'
            write(target / 'facts.json', facts)
            result['insights']['conditional_only'] = interpret(slug, evidence, bundle, facts, target)
            result['arm_status']['conditional_only'] = 'applied' if read(target / 'interpretation.json')['use_refinement'] else 'declined'
            result['status'] = 'computed'
    except Exception:
        result.update(status='failed', error=traceback.format_exc())
        write(folder / 'error.json', {'traceback': result['error']})
    write(folder / 'branch.json', result)
    return result

class ReplayAblation(TrajStoryAgent):
    def __init__(self, llm, arm, source_run, generator="qwen"):
        self.arm, self.source_run = arm, Path(source_run)
        super().__init__(llm, generator=generator, config=FrameworkConfig(**read(self.source_run / '00_config.json')))
        self.synthesizer.llm = SummaryStageLabelClient(self.llm, arm)
        self.saved_profile = load_profile(read(self.source_run / '01_profile.json'))
        self.planned = [load_question(q) for q in read(self.source_run / '02_initial_questions.json')]
        self.validated = [load_evidence(e) for e in read(self.source_run / '03_validated_initial_evidence.json')]
        self.final_questions = [load_question(q) for q in read(self.source_run / '06_report.json')['questions']]
        self.first_failures = [f for f in read(self.source_run / 'branch_failures.json')
                               if str(f.get('stage', '')).startswith('initial_')]
        self.profiler.build = lambda bundle: copy.deepcopy(self.saved_profile)
        self.decomposer.decompose = lambda bundle, profile: copy.deepcopy(self.planned)

    def _parallel_map_best_effort(self, items, function, **kwargs):
        if kwargs.get('stage') == 'initial_analysis':
            for q in self.planned:
                self.history.add(q.question)
            return copy.deepcopy(self.validated), copy.deepcopy(self.first_failures)
        return super()._parallel_map_best_effort(items, function, **kwargs)

    def _verify_and_repair_initial(self, *, questions, **kwargs):
        questions[:] = copy.deepcopy(self.final_questions)
        audits = list(self.source_run.glob('03_initial_review_audit_round_*.json'))
        latest = max(audits, key=lambda p: int(p.stem.rsplit('_', 1)[1]))
        self.verifier.review_audit = read(latest)
        topics = self.source_run / latest.name.replace('review_audit', 'agreement_topics')
        self.verifier.agreement_topics = read(topics)
        # No A0/A1/first A2/A3 completion is made. Agreement has no admitted
        # topics in the frozen Qwen cohort; preserve its deterministic context.
        return copy.deepcopy(self.validated)

    def refine_initial(self, bundle, profile, initial, run_dir):
        if self.arm == 'without_conditional_refinement':
            self.initial_refinements = {}
            return bundle, copy.deepcopy(initial)
        records, supplement, enriched = [], [], copy.deepcopy(initial)
        for evidence, updated in zip(initial, enriched):
            folder = run_dir / 'early' / evidence.question_id
            result = conditional_branch(self.generator, evidence.to_dict(), bundle.to_dict(), profile.to_dict(), folder,
                                        cached_first=self.source_run / 'early' / evidence.question_id)
            records.append(result)
            updated.insight = result['insights']['conditional_only']
            accepted = result['arm_status']['conditional_only'] == 'applied'
            facts = read(folder / 'conditional_only/facts.json') if accepted else []
            supplement.append(dict(evidence_id=evidence.evidence_id, question_id=evidence.question_id,
                                   original_insight=evidence.insight, refined_insight=updated.insight,
                                   status=result['arm_status']['conditional_only'],
                                   accepted_conditional_facts=facts, limitations=evidence.limitations))
        self.initial_refinements = {r['question_id']: r for r in records}
        write(run_dir / 'early/predictions.json', dict(branches=records))
        addition = '\nFIRST-ROUND CONDITIONAL ANALYSIS SUPPLEMENT:\n' + json.dumps(supplement, ensure_ascii=False)
        addition += '\nThese are supplemental descriptive findings, not causal proof. Original evidence and its scope/conflict restrictions remain in force. Declined/failed refinements add no accepted facts. Use conditions and exceptions when choosing useful follow-ups.'
        return replace(bundle, context=bundle.context + addition), enriched

    def refine_final(self, evidence, bundle, profile, run_dir):
        enriched, results = copy.deepcopy(evidence), []
        if self.arm == 'without_compression':
            for item, updated in zip(evidence, enriched):
                result = self.initial_refinements.get(item.question_id)
                if result is None:
                    result = conditional_branch(self.generator, item.to_dict(), bundle.to_dict(), profile.to_dict(),
                                                run_dir / 'refinement' / item.question_id)
                results.append(result)
                updated.insight = result['insights']['conditional_only']
                updated.execution_notes += '\nConditional refinement status: ' + result['arm_status']['conditional_only']
                if result['arm_status']['conditional_only'] == 'applied':
                    folder = run_dir / ('early' if item.question_id in self.initial_refinements else 'refinement') / item.question_id
                    updated.execution_notes += '\nAccepted conditional facts: ' + json.dumps(read(folder / 'conditional_only/facts.json'), ensure_ascii=False)
        write(run_dir / 'predictions.json', dict(framework='v5_ablation', variant=self.arm,
                                               insights=[e.insight for e in enriched], branches=results))
        write(run_dir / '07_question_history.json', dict(questions=self.history.snapshot()))
        return enriched

    def run(self, bundle, output_dir):
        try:
            return super().run(bundle, output_dir)
        finally:
            p = Path(output_dir) / '00_framework.json'
            if p.exists():
                framework = read(p)
                framework.update(framework='v5_ablation', variant=self.arm,
                                 first_round_replayed_from=str(self.source_run),
                                 conditional_refinement=self.arm == 'without_compression',
                                 compression_enabled=False,
                                 compression_cost_model='disabled',
                                 summary_order='after both analysis rounds and enabled ablation stages')
                write(p, framework)
