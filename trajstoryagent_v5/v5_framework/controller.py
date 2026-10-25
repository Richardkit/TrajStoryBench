"""Formal v5 controller implementation."""
from __future__ import annotations
import copy
from dataclasses import replace
from functools import partial
import json
from trajstoryagent_v5.v5_framework.identity import FRAMEWORK_ID
import logging
from pathlib import Path
from trajstoryagent_v5.v5_framework.agreement import build_agreement, downstream_context
from trajstoryagent_v5.v5_framework.conflict_admission import admit_topics
from trajstoryagent_v5.v5_framework.json_utils import write_json
from trajstoryagent_v5.v5_framework.memory_manager import MemoryManagerAgent
from trajstoryagent_v5.v5_framework.models import DataBundle
from trajstoryagent_v5.v5_framework.config import FrameworkConfig
from trajstoryagent_v5.v5_framework.history import QuestionHistory, RecordingAnalysis
from trajstoryagent_v5.v5_framework.planning import RefinedContextExplorer
from trajstoryagent_v5.v5_framework.analysis_engine import AnalysisEngine
from trajstoryagent_v5.v5_framework.models import EvidencePackage, TrajStoryReport
from trajstoryagent_v5.v5_framework.summary import SummarySynthesizer
from trajstoryagent_v5.v5_framework.terminology import InsightClient

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, TypeVar
from trajstoryagent_v5.v5_framework.profiling import StructuredTrajectoryProfiler
from trajstoryagent_v5.v5_framework.reasoning_agents import GoalDecomposer, EvidenceVerifier
from trajstoryagent_v5.v5_framework.models import AnalysisQuestion, FollowUpQuestion, TrajectoryDataProfile
T = TypeVar("T")
R = TypeVar("R")

LOGGER = logging.getLogger(__name__)

class TrajStoryAgent:
    def __init__(self, llm, *, generator, refiner=None, config=None):
        config = config or FrameworkConfig()
        if not isinstance(config, FrameworkConfig):
            raise TypeError("Formal v5 requires trajstoryagent_v5.FrameworkConfig")
        llm = llm if isinstance(llm, InsightClient) else InsightClient(llm)
        self.llm, self.config = llm, config
        self.profiler = StructuredTrajectoryProfiler(llm, config)
        self.decomposer = GoalDecomposer(llm)
        self.verifier = EvidenceVerifier(llm)
        self.analysis = AnalysisEngine(llm, config)
        self.synthesizer = SummarySynthesizer(llm)
        self.explorer = RefinedContextExplorer(llm, coverage_review_enabled=True)
        self.history = QuestionHistory()
        self.analysis = RecordingAnalysis(self.analysis, self.history)
        if refiner is None:
            from trajstoryagent_v5.v5_framework.refinement import run_branch
            refiner = partial(run_branch, min_support=config.min_support)
        self.generator, self.refiner = generator, refiner
        self.initial_refinements = {}

    def build_initial_agreement(self, topics, initial, goal, output):
        return build_agreement(self.llm, topics, initial, goal, output)

    def refine_initial(self, bundle, profile, initial, run_dir):
        initial = list(initial)
        records, supplement = [], []
        enriched = copy.deepcopy(initial)
        for evidence, updated in zip(initial, enriched):
            dest = run_dir / 'early' / evidence.question_id
            result = self.refiner(self.generator, evidence.to_dict(), bundle.to_dict(), profile.to_dict(), dest)
            records.append(result)
            updated.insight = result['insights']['conditional_compressed']
            accepted = result['arm_status']['conditional_compressed'] == 'applied'
            facts = json.loads((dest/'conditional_compressed/facts.json').read_text(encoding='utf-8')) if accepted else []
            supplement.append(dict(evidence_id=evidence.evidence_id, question_id=evidence.question_id,
                original_insight=evidence.insight, refined_insight=updated.insight,
                status=result['arm_status']['conditional_compressed'],
                accepted_conditional_facts=facts, limitations=evidence.limitations))
        self.initial_refinements = {r['question_id']:r for r in records}
        write_json(run_dir/'early/predictions.json', dict(branches=records))
        addition = '\nFIRST-ROUND CONDITIONAL ANALYSIS SUPPLEMENT:\n' + json.dumps(supplement,ensure_ascii=False)
        addition += '\nThese are supplemental descriptive findings, not causal proof. Original evidence and its scope/conflict restrictions remain in force. Declined/failed refinements add no accepted facts. Use conditions and exceptions when choosing useful follow-ups.'
        return replace(bundle, context=bundle.context+addition), enriched

    def refine_final(self, evidence, bundle, profile, run_dir):
        results=[]
        original = list(evidence)
        enriched = copy.deepcopy(original)
        for item, updated in zip(original, enriched):
            if item.question_id in self.initial_refinements:
                result=self.initial_refinements[item.question_id]
            else:
                result=self.refiner(self.generator,item.to_dict(),bundle.to_dict(),profile.to_dict(),
                                    run_dir/'refinement'/item.question_id)
            results.append(result)
            updated.insight = result['insights']['conditional_compressed']
            accepted = result['arm_status']['conditional_compressed'] == 'applied'
            updated.execution_notes += ('\nConditional refinement status: '
                                       + result['arm_status']['conditional_compressed'])
            if accepted:
                directory = (run_dir/'early'/item.question_id if item.question_id in self.initial_refinements
                             else run_dir/'refinement'/item.question_id)
                updated.execution_notes += '\nAccepted conditional facts: ' + json.dumps(
                    json.loads((directory/'conditional_compressed/facts.json').read_text(encoding='utf-8')),
                    ensure_ascii=False)
        write_json(run_dir/'predictions.json',dict(
            original=[e.insight for e in original], branches=results,
            insights={arm:[r['insights'][arm] for r in results] for arm in ('conditional_only','conditional_compressed')},
            source_run=str(run_dir), framework=FRAMEWORK_ID, schema_version=2))
        write_json(run_dir/'07_question_history.json',dict(questions=self.history.snapshot()))
        return enriched

    def run(self, bundle: DataBundle, output_dir: str | Path) -> TrajStoryReport:
        run_dir = Path(output_dir).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        self.history.clear()
        self.initial_refinements = {}
        write_json(run_dir / "00_framework.json", dict(framework=FRAMEWORK_ID,initial_dpp=False,
            follow_up_dpp=False,rounds=2,questions_per_round=3,raw_evidence_preserved=True,
            follow_up_selection="CrossEvidenceExplorer",coverage_review_enabled=True,
            embedding=None,compression_cost_model="name_cost_only_no_fixed_value_cost",
            schema_version=2,min_support=self.config.min_support,
            summary_order="after both rounds conditional decomposition and compression",
            summary_only=True))
        LOGGER.info("TrajStory run started: %s", run_dir)
        write_json(run_dir / "00_input.json", bundle)
        write_json(run_dir / "00_config.json", self.config)

        LOGGER.info("A0 profiling %d input file(s)", len(bundle.files))
        profile = self.profiler.build(bundle)
        write_json(run_dir / "01_profile.json", profile)

        LOGGER.info("A1 decomposing the high-level goal")
        questions = self.decomposer.decompose(bundle, profile)
        planned_questions = list(questions)
        write_json(run_dir / "02_initial_questions.json", questions)
        branch_failures: list[dict[str, object]] = []

        LOGGER.info("A2 analyzing %d initial branches", len(questions))
        initial, failures = self._parallel_map_best_effort(
            questions,
            lambda question: self.analysis.analyze_initial(
                bundle=bundle,
                profile=profile,
                question=question,
                output_dir=run_dir / "initial" / question.question_id,
            ),
            stage="initial_analysis",
            id_getter=lambda question: question.question_id,
            question_getter=lambda question: question.question,
        )
        branch_failures.extend(failures)
        self._write_branch_failures(run_dir, branch_failures)
        successful_initial_ids = {item.question_id for item in initial}
        questions = [item for item in questions if item.question_id in successful_initial_ids]
        if not initial:
            raise RuntimeError(
                "A2 did not produce any usable initial evidence; the case cannot continue"
            )
        LOGGER.info("A3 verifying initial evidence")
        initial = self._verify_and_repair_initial(
            bundle=bundle,
            profile=profile,
            questions=questions,
            evidence=initial,
            run_dir=run_dir,
            branch_failures=branch_failures,
        )
        self._write_branch_failures(run_dir, branch_failures)
        write_json(run_dir / "03_validated_initial_evidence.json", initial)

        if self.config.agreement_enabled:
            topics = self.verifier.agreement_topics
            review = self.verifier.review_audit["agreement"]
            topics, admission = admit_topics(topics, initial)
            admission["review_status"] = review["status"]
            write_json(run_dir / "03_conflict_admission.json", admission)
            LOGGER.info("A3 admission: %d proposed, %d admitted topic(s)",
                        admission["proposed_topic_count"], len(topics))
            agreement = self.build_initial_agreement(
                topics, initial, bundle.goal, run_dir / "03_agreement")
            agreement["protocol"] = "bounded_partial_agreement_v2_execution_gate"
            agreement["admission"] = admission
            agreement["review"] = review
            agreement["use_rules"] += (
                " Candidate edges rejected by admission do not establish contradictions or invalidate "
                "their source evidence. Comparability_unconfirmed means missing verified context, "
                "not agreement or disagreement. No broader semantic comparison has been verified.")
            if review["status"] == "unreviewed":
                LOGGER.warning("A3 conflict review unavailable; retaining individual evidence with restrictions")
                agreement["use_rules"] += " " + review["use_rules"]
            write_json(run_dir / "03_agreement" / "commitment.json", agreement)
            context = downstream_context(agreement)
            for item in initial:
                item.agreement_context = context
            write_json(run_dir / "03_committed_initial_evidence.json", initial)

        reviewed_by_id = {item.question_id: item for item in questions}
        review_initial_questions = [
            reviewed_by_id.get(item.question_id, item) for item in planned_questions
        ]
        raw_bundle = bundle
        bundle, refined_initial = self.refine_initial(bundle, profile, initial, run_dir)
        write_json(run_dir / "03_actual_planning_bundle.json", bundle)
        memory = None
        if self.config.memory_manager_enabled:
            LOGGER.info("Memory Manager compressing A3-validated first-round evidence")
            try:
                memory = MemoryManagerAgent(self.llm).compress(
                    goal=bundle.goal,
                    questions=review_initial_questions,
                    evidence=refined_initial,
                    branch_failures=branch_failures,
                )
            except RuntimeError as exc:
                LOGGER.warning("Memory Manager failed; using a compact local memory for A4: %s", exc)
                memory = MemoryManagerAgent.fallback(
                    questions=review_initial_questions,
                    evidence=refined_initial,
                    branch_failures=branch_failures,
                )
                write_json(run_dir / "03_memory_fallback.json", {
                    "reason": str(exc),
                    "mode": "local_memory_summary",
                    "evidence_ids": [item.evidence_id for item in initial],
                    "question_ids": [item.question_id for item in review_initial_questions],
                })
            write_json(run_dir / "03_memory.json", memory)

        LOGGER.info("A4 generating three follow-ups with a bounded duplicate review")
        follow_ups = self.explorer.explore(
            goal=bundle.goal,
            initial_questions=questions,
            review_initial_questions=review_initial_questions,
            evidence=initial,
            profile=profile,
            context=bundle.context,
            **({"memory": memory} if memory is not None else {}),
        )
        write_json(run_dir / "04_duplicate_reviews.json", self.explorer.last_duplicate_reviews)
        write_json(run_dir / "04_follow_up_candidates.json", self.explorer.last_candidates)
        write_json(run_dir / "04_follow_up_questions.json", follow_ups)

        evidence_by_id = {item.evidence_id: item for item in initial}
        LOGGER.info("A2 analyzing %d follow-up branch(es)", len(follow_ups))
        follow_evidence, failures = self._parallel_map_best_effort(
            follow_ups,
            lambda follow_up: self.analysis.analyze_follow_up(
                bundle=bundle,
                profile=profile,
                follow_up=follow_up,
                output_dir=run_dir / "follow_up" / follow_up.follow_up_id,
                parent_evidence=[
                    evidence_by_id[evidence_id]
                    for evidence_id in follow_up.related_evidence_ids
                ],
                coding_errors=(memory["Errors appear in coding"] if memory is not None else None),
            ),
            stage="follow_up_analysis",
            id_getter=lambda follow_up: follow_up.follow_up_id,
            question_getter=lambda follow_up: follow_up.question,
        )
        branch_failures.extend(failures)
        successful_follow_up_ids = {item.question_id for item in follow_evidence}
        follow_ups = [
            item for item in follow_ups if item.follow_up_id in successful_follow_up_ids
        ]
        if follow_evidence:
            LOGGER.info("A3 verifying follow-up evidence")
            follow_evidence = self._verify_and_repair_follow_up(
                bundle=bundle,
                profile=profile,
                follow_ups=follow_ups,
                evidence=follow_evidence,
                parent_evidence=initial,
                run_dir=run_dir,
                branch_failures=branch_failures,
                memory=memory,
            )
        else:
            LOGGER.warning(
                "All follow-up branches failed; continuing with validated initial evidence"
            )
        self._write_branch_failures(run_dir, branch_failures)
        write_json(run_dir / "05_validated_follow_up_evidence.json", follow_evidence)

        final_evidence = self.refine_final([*initial, *follow_evidence], raw_bundle, profile, run_dir)
        write_json(run_dir / "05_final_insights_evidence.json", final_evidence)
        LOGGER.info("A5 summarizing finalized branch insights")
        report = self.synthesizer.synthesize(
            goal=bundle.goal,
            questions=questions,
            follow_ups=follow_ups,
            evidence=final_evidence,
        )
        write_json(run_dir / "06_report.json", report)
        LOGGER.info("TrajStory run completed: %s", run_dir / "06_report.json")
        return report


    def _verify_and_repair_initial(
        self,
        *,
        bundle: DataBundle,
        profile: TrajectoryDataProfile,
        questions: list[AnalysisQuestion],
        evidence: list[EvidencePackage],
        run_dir: Path,
        branch_failures: list[dict[str, object]] | None = None,
    ) -> list[EvidencePackage]:
        branch_failures = branch_failures if branch_failures is not None else []
        evidence_by_id = {item.question_id: item for item in evidence}
        question_by_id = {item.question_id: item for item in questions}
        for round_index in range(self.config.max_relevance_repairs + 1):
            ordered = [evidence_by_id[item.question_id] for item in questions]
            results = self._verify_with_audit(
                goal=bundle.goal, questions=questions, evidence=ordered,
                collect_agreement=self.config.agreement_enabled, profile=profile,
                audit_path=run_dir / f"03_initial_review_audit_round_{round_index + 1}.json")
            write_json(run_dir / f"03_initial_verification_round_{round_index + 1}.json", results)
            if self.config.agreement_enabled:
                write_json(run_dir / f"03_initial_agreement_topics_round_{round_index + 1}.json",
                           self.verifier.agreement_topics)
            revisions = [item for item in results if item.status == "REVISE"]
            if not revisions:
                return ordered
            LOGGER.info(
                "A3 requested initial repairs: %s",
                [(item.question_id, item.revision_target) for item in revisions],
            )
            if round_index == self.config.max_relevance_repairs:
                for result in revisions:
                    question = question_by_id[result.question_id]
                    branch_failures.append(self._failure_record(
                        stage="initial_validation",
                        branch_id=result.question_id,
                        question=question.question,
                        exc=RuntimeError(
                            "A3 did not validate the branch after the allowed repair rounds"
                        ),
                    ))
                    evidence_by_id.pop(result.question_id, None)
                questions[:] = [
                    item for item in questions if item.question_id in evidence_by_id
                ]
                remaining = [evidence_by_id[item.question_id] for item in questions]
                if not remaining:
                    raise RuntimeError("A3 did not validate any initial evidence")
                return remaining
            repaired = []
            question_revisions = []
            for result in revisions:
                question = question_by_id[result.question_id]
                if result.revision_target == "QUESTION":
                    question = self.decomposer.revise_question(
                        bundle=bundle,
                        profile=profile,
                        questions=questions,
                        failed_question_id=result.question_id,
                        revision_instruction=result.revision_instruction,
                    )
                    question_by_id[result.question_id] = question
                    question_index = next(
                        index for index, item in enumerate(questions)
                        if item.question_id == result.question_id
                    )
                    questions[question_index] = question
                    question_revisions.append({
                        "question_id": result.question_id,
                        "previous_question": evidence_by_id[result.question_id].question,
                        "replacement": question.to_dict(),
                        "issues": result.issues,
                        "revision_instruction": result.revision_instruction,
                    })
                try:
                    repaired.append(self.analysis.analyze_initial(
                        bundle=bundle,
                        profile=profile,
                        question=question,
                        output_dir=run_dir / "initial" / result.question_id / f"relevance_repair_{round_index + 1}",
                        revision_instruction=(
                            "" if result.revision_target == "QUESTION"
                            else result.revision_instruction
                        ),
                    ))
                except Exception as exc:
                    LOGGER.exception(
                        "Initial branch %s failed during A3 repair; dropping only this branch",
                        result.question_id,
                    )
                    branch_failures.append(self._failure_record(
                        stage="initial_repair",
                        branch_id=result.question_id,
                        question=question.question,
                        exc=exc,
                    ))
                    evidence_by_id.pop(result.question_id, None)
            if question_revisions:
                write_json(
                    run_dir / f"02_initial_question_revisions_round_{round_index + 1}.json",
                    question_revisions,
                )
                write_json(run_dir / "02_initial_questions.json", questions)
            evidence_by_id.update({item.question_id: item for item in repaired})
            questions[:] = [
                item for item in questions if item.question_id in evidence_by_id
            ]
            if not evidence_by_id:
                raise RuntimeError("A3 repair did not leave any usable initial evidence")
        raise AssertionError("unreachable")

    def _verify_and_repair_follow_up(
        self,
        *,
        bundle: DataBundle,
        profile: TrajectoryDataProfile,
        follow_ups: list[FollowUpQuestion],
        evidence: list[EvidencePackage],
        parent_evidence: list[EvidencePackage],
        run_dir: Path,
        branch_failures: list[dict[str, object]] | None = None,
        memory: dict[str, list[str]] | None = None,
    ) -> list[EvidencePackage]:
        branch_failures = branch_failures if branch_failures is not None else []
        questions = [
            AnalysisQuestion(item.follow_up_id, item.question, item.analysis_granularity)
            for item in follow_ups
        ]
        follow_by_id = {item.follow_up_id: item for item in follow_ups}
        evidence_by_id = {item.question_id: item for item in evidence}
        all_evidence_by_id = {item.evidence_id: item for item in parent_evidence}
        for round_index in range(self.config.max_relevance_repairs + 1):
            ordered = [evidence_by_id[item.question_id] for item in questions]
            results = self._verify_with_audit(
                goal=bundle.goal, questions=questions, evidence=ordered, profile=profile,
                audit_path=run_dir / f"05_follow_up_review_audit_round_{round_index + 1}.json",
            )
            write_json(run_dir / f"05_follow_up_verification_round_{round_index + 1}.json", results)
            revisions = [item for item in results if item.status == "REVISE"]
            if not revisions:
                return ordered
            LOGGER.info(
                "A3 requested follow-up repairs: %s",
                [(item.question_id, item.revision_target) for item in revisions],
            )
            if round_index == self.config.max_relevance_repairs:
                for result in revisions:
                    follow_up = follow_by_id[result.question_id]
                    branch_failures.append(self._failure_record(
                        stage="follow_up_validation",
                        branch_id=result.question_id,
                        question=follow_up.question,
                        exc=RuntimeError(
                            "A3 did not validate the branch after the allowed repair rounds"
                        ),
                    ))
                    evidence_by_id.pop(result.question_id, None)
                follow_ups[:] = [
                    item for item in follow_ups if item.follow_up_id in evidence_by_id
                ]
                return [evidence_by_id[item.follow_up_id] for item in follow_ups]
            repaired = []
            question_revisions = []
            for result in revisions:
                follow_up = follow_by_id[result.question_id]
                if result.revision_target == "QUESTION":
                    follow_up = self.explorer.revise_follow_up(
                        goal=bundle.goal,
                        initial_questions=[
                            AnalysisQuestion(
                                item.question_id,
                                item.question,
                                item.analysis_granularity,
                            )
                            for item in parent_evidence
                        ],
                        evidence=parent_evidence,
                        profile=profile,
                        follow_ups=follow_ups,
                        failed_follow_up_id=result.question_id,
                        revision_instruction=result.revision_instruction,
                        context=bundle.context,
                        **({"memory": memory} if memory is not None else {}),
                    )
                    old = follow_by_id[result.question_id]
                    follow_by_id[result.question_id] = follow_up
                    index = next(
                        index for index, item in enumerate(follow_ups)
                        if item.follow_up_id == result.question_id
                    )
                    follow_ups[index] = follow_up
                    questions[index] = AnalysisQuestion(
                        follow_up.follow_up_id,
                        follow_up.question,
                        follow_up.analysis_granularity,
                    )
                    question_revisions.append({
                        "follow_up_id": result.question_id,
                        "previous": old.to_dict(),
                        "replacement": follow_up.to_dict(),
                        "issues": result.issues,
                        "revision_instruction": result.revision_instruction,
                    })
                try:
                    repaired.append(self.analysis.analyze_follow_up(
                        bundle=bundle,
                        profile=profile,
                        follow_up=follow_up,
                        output_dir=run_dir / "follow_up" / result.question_id / f"relevance_repair_{round_index + 1}",
                        parent_evidence=[
                        all_evidence_by_id[evidence_id]
                        for evidence_id in follow_up.related_evidence_ids
                        if evidence_id in all_evidence_by_id
                        ],
                        revision_instruction=(
                            "" if result.revision_target == "QUESTION"
                            else result.revision_instruction
                        ),
                        coding_errors=(memory["Errors appear in coding"] if memory is not None else None),
                    ))
                except Exception as exc:
                    LOGGER.exception(
                        "Follow-up branch %s failed during A3 repair; dropping only this branch",
                        result.question_id,
                    )
                    branch_failures.append(self._failure_record(
                        stage="follow_up_repair",
                        branch_id=result.question_id,
                        question=follow_up.question,
                        exc=exc,
                    ))
                    evidence_by_id.pop(result.question_id, None)
            if question_revisions:
                write_json(
                    run_dir / f"05_follow_up_question_revisions_round_{round_index + 1}.json",
                    question_revisions,
                )
                write_json(run_dir / "04_follow_up_questions.json", follow_ups)
            evidence_by_id.update({item.question_id: item for item in repaired})
            follow_ups[:] = [
                item for item in follow_ups if item.follow_up_id in evidence_by_id
            ]
            questions = [
                item for item in questions if item.question_id in evidence_by_id
            ]
        raise AssertionError("unreachable")

    def _verify_with_audit(self, *, audit_path: Path, **kwargs):
        try:
            return self.verifier.verify(**kwargs)
        finally:
            # Keep component errors and completion text even when no relevance
            # decision could be parsed. Never save provider reasoning or secrets.
            write_json(audit_path, self.verifier.review_audit)

    def _parallel_map(self, items: Iterable[T], function: Callable[[T], R]) -> list[R]:
        items = list(items)
        if len(items) == 1 or self.config.parallel_workers == 1:
            return [function(item) for item in items]
        indexed: dict[int, R] = {}
        with ThreadPoolExecutor(max_workers=min(self.config.parallel_workers, len(items))) as pool:
            futures = {pool.submit(function, item): index for index, item in enumerate(items)}
            for future in as_completed(futures):
                indexed[futures[future]] = future.result()
        return [indexed[index] for index in range(len(items))]

    def _parallel_map_best_effort(
        self,
        items: Iterable[T],
        function: Callable[[T], R],
        *,
        stage: str,
        id_getter: Callable[[T], str],
        question_getter: Callable[[T], str],
    ) -> tuple[list[R], list[dict[str, object]]]:
        """Run independent branches without allowing one failure to erase the others."""
        items = list(items)
        successes: dict[int, R] = {}
        failures: dict[int, dict[str, object]] = {}
        if not items:
            return [], []

        def record_failure(index: int, item: T, exc: Exception) -> None:
            branch_id = id_getter(item)
            LOGGER.exception(
                "%s branch %s failed; continuing with the remaining branches",
                stage,
                branch_id,
                exc_info=exc,
            )
            failures[index] = self._failure_record(
                stage=stage,
                branch_id=branch_id,
                question=question_getter(item),
                exc=exc,
            )

        if len(items) == 1 or self.config.parallel_workers == 1:
            for index, item in enumerate(items):
                try:
                    successes[index] = function(item)
                except Exception as exc:
                    record_failure(index, item, exc)
        else:
            with ThreadPoolExecutor(max_workers=min(self.config.parallel_workers, len(items))) as pool:
                futures = {pool.submit(function, item): (index, item)
                           for index, item in enumerate(items)}
                for future in as_completed(futures):
                    index, item = futures[future]
                    try:
                        successes[index] = future.result()
                    except Exception as exc:
                        record_failure(index, item, exc)
        return (
            [successes[index] for index in sorted(successes)],
            [failures[index] for index in sorted(failures)],
        )

    @staticmethod
    def _failure_record(
        *, stage: str, branch_id: str, question: str, exc: Exception
    ) -> dict[str, object]:
        error = str(exc)
        return {
            "stage": stage,
            "branch_id": branch_id,
            "question": question,
            "error_type": type(exc).__name__,
            "error": error[-12_000:],
            "case_continued": True,
        }

    @staticmethod
    def _write_branch_failures(
        run_dir: Path, failures: list[dict[str, object]]
    ) -> None:
        write_json(run_dir / "branch_failures.json", failures)
