from __future__ import annotations

import json
import re
from typing import Any, Callable, TypeVar

from trajstoryagent_v5.json_utils import extract_json_object
from trajstoryagent_v5.agreement import REVIEW_INSTRUCTION, evidence_context, validate_topics
from trajstoryagent_v5.llm import LLMClient
from trajstoryagent_v5.models import AnalysisGranularity, AnalysisQuestion, DataBundle, EvidencePackage, FollowUpQuestion, TrajectoryDataProfile, TrajStoryReport, VerificationResult

T = TypeVar("T")


def _structured_call(
    llm: LLMClient,
    *,
    system: str,
    user: str,
    validator: Callable[[dict[str, Any]], T],
    attempts: int = 3,
) -> T:
    previous = ""
    error = ""
    for _ in range(attempts):
        repair = ""
        if error:
            repair = f"\nYour previous response was invalid. Error: {error}\nPrevious response:\n{previous}\n"
        previous = llm.complete(system=system, user=user + repair)
        try:
            return validator(extract_json_object(previous))
        except (KeyError, TypeError, ValueError) as exc:
            error = str(exc)
    raise RuntimeError(f"model did not return a valid structured response: {error}")


def _normalized_words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def _similar(left: str, right: str) -> bool:
    a, b = _normalized_words(left), _normalized_words(right)
    return bool(a and b and len(a & b) / len(a | b) >= 0.78)


def _question_key(text: str) -> str:
    """Normalize a question only far enough to catch literal repetitions."""
    return " ".join(re.findall(r"[a-z0-9_]+", text.lower()))


def _parse_analysis_granularity(item: dict[str, Any]) -> AnalysisGranularity:
    raw = item.get("analysis_granularity")
    if not isinstance(raw, dict):
        raise ValueError("analysis_granularity must be an object")
    base = raw.get("base")
    aggregation = raw.get("aggregation")
    allowed = {"point", "trajectory/trip", "user/entity", "group"}
    if base not in allowed:
        raise ValueError(f"analysis_granularity.base must be one of {sorted(allowed)}")
    if aggregation is not None and (not isinstance(aggregation, str) or not aggregation.strip()):
        raise ValueError("analysis_granularity.aggregation must be a non-empty string or null")
    return AnalysisGranularity(base, aggregation.strip() if aggregation else None)


class GoalDecomposer:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def decompose(self, bundle: DataBundle, profile: TrajectoryDataProfile) -> list[AnalysisQuestion]:
        def validate(payload: dict[str, Any]) -> list[AnalysisQuestion]:
            raw = payload.get("sub_questions")
            if not isinstance(raw, list) or len(raw) != 3:
                raise ValueError("sub_questions must contain exactly three items")
            questions = []
            for index, item in enumerate(raw, start=1):
                if not isinstance(item, dict):
                    raise ValueError("every sub-question must include an analysis_granularity")
                text = item.get("question")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("every sub-question must be non-empty")
                questions.append(
                    AnalysisQuestion(
                        f"Q{index}", text.strip(), _parse_analysis_granularity(item)
                    )
                )
            if any(_similar(a.question, b.question) for i, a in enumerate(questions) for b in questions[i + 1 :]):
                raise ValueError("the three sub-questions are not complementary")
            return questions

        return _structured_call(
            self.llm,
            system=(
                "You are TrajStory A1. Plan initial questions only; do not analyze data or propose follow-ups. "
                "Return JSON only."
            ),
            user=f"""
HIGH-LEVEL GOAL:
{bundle.goal}

DATA CONTEXT:
{bundle.context}

TRAJECTORY DATA PROFILE:
{json.dumps(profile.to_dict(), ensure_ascii=False)[:45000]}

Generate exactly three complementary sub-questions that directly serve the goal.
Do not map them mechanically to fixed temporal/spatial/mode categories. Do not ask a
question merely because a field exists. Each sub-question must pursue one clear analytical
objective, not necessarily one metric. Multiple related metrics and joint grouping conditions
are allowed when they serve that same comparison, pattern, or relationship. For example,
comparing vessel size across route classes may jointly examine length, width, and tonnage;
comparing movement by transport mode within each trip purpose is also one objective.
Do not split one coherent comparison into separate questions merely because it uses several
metrics. Select relevant metrics rather than listing every available field. Write one
interrogative sentence with a single question mark. Grouping and time-window conditions
may define the analysis but must not introduce an independent analytical objective. Do not combine requests such as describing
a pattern AND locating anomalies, comparing groups AND explaining causes, or measuring
an association AND testing a separate outcome. Distribute distinct analysis goals across
the three sub-questions; leave three additional goal-relevant directions, connections, or
refinements for adaptive follow-ups. Do not
perform statistical inference. For each question, plan its base analysis unit and optional
aggregation unit. This plan will be fixed before the separate code generator runs.
When a question needs fields from multiple tables, use the profiled table
relationship to identify a plausible analysis direction. The code stage must
check any chosen linkage key and cardinality against the data.

Return:
{{"sub_questions":[
  {{"question":"...","analysis_granularity":{{"base":"point|trajectory/trip|user/entity|group","aggregation":"field/grouping or null"}}}},
  {{"question":"...","analysis_granularity":{{"base":"...","aggregation":null}}}},
  {{"question":"...","analysis_granularity":{{"base":"...","aggregation":"..."}}}}
]}}
""",
            validator=validate,
        )

    def revise_question(
        self,
        *,
        bundle: DataBundle,
        profile: TrajectoryDataProfile,
        questions: list[AnalysisQuestion],
        failed_question_id: str,
        revision_instruction: str,
    ) -> AnalysisQuestion:
        by_id = {item.question_id: item for item in questions}
        if failed_question_id not in by_id:
            raise ValueError(f"unknown initial question: {failed_question_id}")
        retained = [item for item in questions if item.question_id != failed_question_id]

        def validate(payload: dict[str, Any]) -> AnalysisQuestion:
            raw = payload.get("replacement_question")
            if not isinstance(raw, dict):
                raise ValueError("replacement_question must be an object")
            text = raw.get("question")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("replacement question must be non-empty")
            if _similar(text, by_id[failed_question_id].question):
                raise ValueError("replacement question merely restates the rejected question")
            if any(_similar(text, item.question) for item in retained):
                raise ValueError("replacement question duplicates a retained question")
            return AnalysisQuestion(
                failed_question_id, text.strip(), _parse_analysis_granularity(raw)
            )

        return _structured_call(
            self.llm,
            system=(
                "You are TrajStory A1 revising a rejected analysis question. Replace the question; do not answer "
                "it or write code. Return JSON only."
            ),
            user=f"""
HIGH-LEVEL GOAL:
{bundle.goal}

DATA CONTEXT:
{bundle.context}

TRAJECTORY DATA PROFILE:
{json.dumps(profile.to_dict(), ensure_ascii=False)[:45000]}

REJECTED QUESTION:
{json.dumps(by_id[failed_question_id].to_dict(), ensure_ascii=False)}

A3 REJECTION REASON AND REVISION INSTRUCTION:
{revision_instruction}

RETAINED COMPLEMENTARY QUESTIONS:
{json.dumps([item.to_dict() for item in retained], ensure_ascii=False)}

Replace the rejected question with one clear, goal-relevant question that is fully
answerable from fields and verified relations in the supplied profile. Do not merely
paraphrase the rejected question or duplicate a retained objective. Preserve the same
question ID; the controller supplies it. Plan the fixed analysis granularity.

Return:
{{"replacement_question":{{"question":"...","analysis_granularity":
{{"base":"point|trajectory/trip|user/entity|group","aggregation":null}}}}}}
""",
            validator=validate,
        )


class EvidenceVerifier:
    """A3: independently validated relevance and optional cross-evidence review."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self.agreement_topics = []
        self.review_audit: dict[str, Any] = {}

    def verify(
        self,
        *,
        goal: str,
        questions: list[AnalysisQuestion],
        evidence: list[EvidencePackage],
        collect_agreement: bool = False,
        profile: TrajectoryDataProfile | None = None,
    ) -> list[VerificationResult]:
        self.agreement_topics = []
        self.review_audit = {
            "relevance": {"status": "pending", "errors": []},
            "agreement": {
                "status": "pending" if collect_agreement else "not_requested",
                "errors": [],
            },
            "attempts": [],
        }
        question_by_id = {item.question_id: item.question for item in questions}
        evidence_by_question = {item.question_id: item for item in evidence}
        if (len(question_by_id) != len(questions)
                or len(evidence_by_question) != len(evidence)
                or len({item.evidence_id for item in evidence}) != len(evidence)
                or set(question_by_id) != set(evidence_by_question)):
            raise ValueError("A3 requires exactly one evidence package per question")

        def validate(payload: dict[str, Any]) -> list[VerificationResult]:
            raw = payload.get("results")
            if not isinstance(raw, list) or len(raw) != len(questions):
                raise ValueError("results must cover every question")
            output = []
            seen = set()
            for item in raw:
                if not isinstance(item, dict):
                    raise ValueError("every relevance result must be an object")
                question_id = item.get("question_id")
                if isinstance(question_id, str):
                    question_id = question_id.strip()
                status = item.get("status")
                if isinstance(status, str):
                    status = status.strip().upper()
                if (not isinstance(question_id, str)
                        or question_id not in question_by_id or question_id in seen):
                    raise ValueError("invalid or duplicate question_id")
                if not isinstance(status, str) or status not in {"VALID", "REVISE"}:
                    raise ValueError("status must be VALID or REVISE")
                issues = item.get("issues")
                if issues is None:
                    issues = []  # No issue text supplied; not an extra validity judgment.
                if isinstance(issues, str):
                    issues = [issues] if issues.strip() else []
                if not isinstance(issues, list) or any(not isinstance(v, str) for v in issues):
                    raise ValueError("issues must be a list of strings or a single string")
                issues = [v.strip() for v in issues if v.strip()]
                # VALID never takes a repair route. Ignore even malformed unused
                # repair metadata instead of rejecting already-usable relevance results.
                instruction, target = "", "EVIDENCE"
                if status == "REVISE":
                    target = item.get("revision_target")
                    if isinstance(target, str):
                        target = target.strip().upper()
                    if not isinstance(target, str) or target not in {"EVIDENCE", "QUESTION"}:
                        raise ValueError("REVISE revision_target must be EVIDENCE or QUESTION")
                    instruction = item.get("revision_instruction")
                    if not isinstance(instruction, str) or not instruction.strip():
                        raise ValueError("REVISE requires a non-empty revision_instruction")
                    instruction = instruction.strip()
                output.append(
                    VerificationResult(question_id, status, issues, instruction, target)
                )
                seen.add(question_id)
            return output

        return self._review(
            system=(
                "You are TrajStory A3. Check question relevance and high-level goal relevance. "
                + ("Also summarize material cross-evidence disagreements using the supplied schema. "
                   "Do not settle disputed facts or change computed results. " if collect_agreement else
                   "Do not judge statistical correctness, evidence sufficiency, chart consistency, or next steps. ") +
                "Return JSON only."
            ),
            user=f"""
HIGH-LEVEL GOAL:
{goal}

DATA PROFILE:
{json.dumps(profile.to_dict(), ensure_ascii=False)[:45000] if profile else "Not supplied."}

QUESTION/EVIDENCE PACKAGES:
{json.dumps([{"question_id": q.question_id, "question": q.question,
              "evidence": evidence_by_question[q.question_id].to_dict()} for q in questions], ensure_ascii=False)}

For each branch return VALID if the evidence directly answers its question and still
serves the high-level goal. Otherwise return REVISE with concrete issues, a revision
target, and an instruction. Use revision_target QUESTION when the question itself is
off-goal, duplicates an answered objective, requests unavailable fields/relations, or
cannot be fully answered from the supplied profile. Use EVIDENCE only when the question
is appropriate and answerable but the generated evidence fails to answer it. Do not ask
for evidence regeneration when changing the question is necessary.
For VALID, revision_target and revision_instruction are unused: omit them or use null.
For REVISE, both an explicit EVIDENCE/QUESTION target and a nonempty instruction are required.

Return:
{{"results":[{{"question_id":"Q1","status":"VALID|REVISE",
"issues":[],"revision_target":"EVIDENCE|QUESTION","revision_instruction":""}}]}}
""" + ((REVIEW_INSTRUCTION + "\nEXECUTED EVIDENCE AND CODE:\n" +
          json.dumps(evidence_context(evidence), ensure_ascii=False)) if collect_agreement else ""),
            validator=validate,
            collect_agreement=collect_agreement,
            evidence_ids={e.evidence_id for e in evidence},
        )

    def _review(
        self, *, system: str, user: str,
        validator: Callable[[dict[str, Any]], list[VerificationResult]],
        collect_agreement: bool, evidence_ids: set[str],
    ) -> list[VerificationResult]:
        """Keep valid sections; retry only bad metadata, never analysis code.

        Normal path remains one combined call. Three total format attempts are
        allowed, independently of the controller's semantic REVISE budget.
        """
        results = None
        agreement_ready = not collect_agreement
        previous, errors = "", {}
        for attempt in range(1, 4):
            pending = (["results"] if results is None else []) + (
                ["agreement_topics"] if not agreement_ready else [])
            repair = ""
            if attempt > 1:
                repair = (
                    "\nA3 FORMAT REPAIR ONLY. Return ONLY these top-level sections: "
                    + ", ".join(pending)
                    + ". Other sections have already been validated and are retained; "
                    "do not reconsider them. Do not regenerate questions, code, or evidence.\n"
                    f"Section errors: {json.dumps(errors, ensure_ascii=False)}\n"
                    f"Previous response:\n{previous}\n"
                )
            previous = self.llm.complete(system=system, user=user + repair)
            record = {"attempt": attempt, "requested_sections": pending,
                      "response": previous, "errors": {}}
            self.review_audit["attempts"].append(record)
            errors = record["errors"]
            try:
                payload = extract_json_object(previous)
            except (KeyError, TypeError, ValueError) as exc:
                errors.update({section: str(exc) for section in pending})
            else:
                if results is None:
                    try:
                        results = validator(payload)
                        self.review_audit["relevance"]["status"] = "completed"
                        self.review_audit["relevance"]["results"] = [r.to_dict() for r in results]
                    except (KeyError, TypeError, ValueError) as exc:
                        errors["results"] = str(exc)
                if not agreement_ready:
                    try:
                        self.agreement_topics = validate_topics(
                            payload.get("agreement_topics"), evidence_ids)
                        agreement_ready = True
                        self.review_audit["agreement"].update(
                            status="completed", topic_count=len(self.agreement_topics))
                    except (KeyError, TypeError, ValueError) as exc:
                        errors["agreement_topics"] = str(exc)
            for section, error in errors.items():
                state = "relevance" if section == "results" else "agreement"
                self.review_audit[state]["errors"].append({"attempt": attempt, "error": error})
            if results is not None and agreement_ready:
                return results

        if not agreement_ready:
            # Empty usable topics is NOT a finding of no disagreement. All initial
            # branches are conservatively covered when malformed scope is unknown.
            self.review_audit["agreement"].update(
                status="unreviewed", affected_evidence_ids=sorted(evidence_ids),
                reason="Agreement metadata remained invalid after three format attempts.",
                use_rules=(
                    "Cross-evidence disagreement review is incomplete, not a finding of no conflict. "
                    "Use source evidence only as individually attributed results within its own scope. "
                    "Do not infer consensus or treat combinations of these results as established premises. "
                    "Follow-ups may independently test relationships with new computations; "
                    "disclose this review limitation in synthesis."
                ))
        if results is None:
            self.review_audit["relevance"]["status"] = "failed"
            error = self.review_audit["relevance"]["errors"][-1]["error"]
            raise A3RelevanceFormatError(
                f"A3 relevance response remained invalid after three format attempts: {error}")
        return results


class A3RelevanceFormatError(RuntimeError):
    """No usable relevance decision; not an evidence/question rejection."""


class CrossEvidenceExplorer:
    """A4: propose three follow-ups, with one bounded question review."""

    def __init__(self, llm: LLMClient, *, coverage_review_enabled: bool = True) -> None:
        self.llm = llm
        self.coverage_review_enabled = coverage_review_enabled
        self.last_candidates: list[FollowUpQuestion] = []
        self.last_duplicate_reviews: list[dict[str, Any]] = []

    @staticmethod
    def _planning_memory(memory: dict[str, list[str]]) -> dict[str, list[str]]:
        if set(memory) != {"Question", "Finding", "Errors appear in coding"}:
            raise ValueError("A4 memory must have exactly the three Manager fields")
        # Coding failures belong to the follow-up code writer, not question planning.
        return {key: memory[key] for key in ("Question", "Finding")}

    def explore(
        self,
        *,
        goal: str,
        initial_questions: list[AnalysisQuestion],
        evidence: list[EvidencePackage],
        profile: TrajectoryDataProfile,
        context: str = "",
        review_initial_questions: list[AnalysisQuestion] | None = None,
        memory: dict[str, list[str]] | None = None,
    ) -> list[FollowUpQuestion]:
        self.last_candidates = []
        self.last_duplicate_reviews = []
        evidence_ids = {item.evidence_id for item in evidence}
        relation_ids = profile.relation_ids()
        planning_memory = self._planning_memory(memory) if memory is not None else None

        def validate_candidates(payload: dict[str, Any]) -> list[FollowUpQuestion]:
            raw = payload.get("candidates")
            if not isinstance(raw, list) or len(raw) != 3:
                raise ValueError("candidates must contain exactly three items")
            output = []
            for index, item in enumerate(raw, start=1):
                if not isinstance(item, dict):
                    raise ValueError("every candidate must be an object")
                question = str(item.get("question", "")).strip()
                expected_new_information = str(item.get("expected_new_information", "")).strip()
                related = tuple(str(value) for value in item.get("related_evidence_ids", []))
                relations = tuple(str(value) for value in item.get("relation_ids", []))
                if not question:
                    raise ValueError("candidate question must not be empty")
                if memory is not None and not expected_new_information:
                    raise ValueError("memory-based A4 candidate needs expected_new_information")
                if not set(related) <= evidence_ids:
                    raise ValueError("invalid related evidence IDs")
                unknown_relations = set(relations) - relation_ids
                if unknown_relations:
                    raise ValueError(
                        f"unknown relation IDs {sorted(unknown_relations)}; "
                        f"valid relation IDs are {sorted(relation_ids)}. "
                        "Use [] when no listed relation supports the question"
                    )
                output.append(
                    FollowUpQuestion(
                        follow_up_id=f"F{index}",
                        question=question,
                        primary_evidence_id=related[0] if related else None,
                        analysis_granularity=_parse_analysis_granularity(item),
                        supporting_evidence_ids=related[1:],
                        relation_ids=relations,
                        purpose=str(item.get("purpose", "goal_relevant_exploration")),
                        expected_new_information=expected_new_information,
                    )
                )
            return output

        if memory is None:
            generator_prompt = f"""
HIGH-LEVEL GOAL:
{goal}

DATA CONTEXT:
{context}

EXECUTED INITIAL QUESTIONS:
{json.dumps([item.to_dict() for item in initial_questions], ensure_ascii=False)}

VALIDATED INITIAL EVIDENCE:
{json.dumps([item.to_dict() for item in evidence], ensure_ascii=False)}

DATA PROFILE:
{json.dumps(profile.to_dict(), ensure_ascii=False)}

PROFILED RELATIONS:
{json.dumps({"intra": [r.to_dict() for p in profile.files for r in p.relations],
             "cross": [r.to_dict() for r in profile.cross_file_relations]}, ensure_ascii=False)}

Generate exactly three goal-relevant next questions; they will be executed directly and
in parallel after a separate duplicate review. Use the three initial questions and their
validated evidence to seek useful new information. Prefer simple, answerable description
or comparison of a population, condition, measure, time pattern, or spatial pattern.
The questions may also extend or connect existing findings. Do not default
to refinement, explanation, interaction effects, controls, exception checking, robustness
testing, or a more complex method merely because prior evidence exists.

Questions are allowed to be related to earlier questions and to one another when the
underlying insights form a useful analytical story. They must not merely repeat an
already answered objective: each candidate must seek genuinely new information. Explore
diverse possibilities across the available fields, tables, relationships, populations,
times, and spatial structure, but include only directions that directly serve the goal.
There is no quota requiring every evidence item to parent the same number of candidates.
Evidence links are optional context, not parents. Use related_evidence_ids only when the
new calculation genuinely relies on an earlier result; otherwise return an empty list.

Constraints:
- Return exactly 3 candidates; fewer or more are invalid.
- Each candidate must pursue one clear analytical objective and use a single question
  mark at the end. One objective may jointly analyze multiple related metrics.
- Joint grouping and time-window conditions are allowed when they define that same
  relationship, pattern, or comparison. For example, comparing
  length, width, and tonnage across route classes is one vessel-size comparison.
- Do not split a coherent question into separate candidates solely by metric, and
  do not add unrelated metrics or independent analytical objectives.
- Do not combine requests such as profiling a pattern AND testing a peak, comparing
  groups AND explaining causes, or measuring an association AND testing another outcome.
- related_evidence_ids is optional and may be empty.
- Connections among validated insights are welcome but not mandatory.
- A negative or null result is acceptable; do not presuppose a positive relationship.
- This is the only follow-up hop.
- Follow agreement_context.use_rules when present. Shared, conditional and deferred
  arguments have different permissions for use. A deferred claim may be investigated,
  never assumed true. If investigating a cross-branch dispute, include the other
  participating evidence as supporting parents. Avoid repeating the same dispute in
  multiple follow-ups. Method-conditioned evidence must remain conditioned.
- For each candidate, plan the fixed base analysis unit and optional aggregation unit.

Return exactly one object in this shape, replacing example IDs with supplied IDs when needed.
Use a valid base (point, trajectory/trip, user/entity, group) and a grouping string or JSON null:
{{"candidates":[
{{"question":"...?","related_evidence_ids":[],"relation_ids":[],
"analysis_granularity":{{"base":"trajectory/trip","aggregation":null}}}},
{{"question":"...?","related_evidence_ids":[],"relation_ids":[],
"analysis_granularity":{{"base":"group","aggregation":"..."}}}},
{{"question":"...?","related_evidence_ids":[],"relation_ids":[],
"analysis_granularity":{{"base":"trajectory/trip","aggregation":null}}}}
]}}
"""
        else:
            generator_prompt = f"""
HIGH-LEVEL GOAL:
{goal}

COMPRESSED FIRST-ROUND MEMORY:
{json.dumps(planning_memory, ensure_ascii=False)}

DATA PROFILE:
{json.dumps(profile.to_dict(), ensure_ascii=False)}

Generate exactly three goal-relevant next questions. Use memory.Question to see
what was attempted and memory.Finding to see what was actually established.
First identify which parts of the high-level goal remain unanswered or only
partly answered by the Findings. Prefer questions that fill those gaps and can
produce a new main insight for the goal. A Question alone does not mean its
objective was answered. If a Finding is partial, ask for the specific missing
comparison or pattern; do not rerun the same attempted analysis unchanged.
Avoid near-rephrasings and superficial variants that change only wording, a
minor subgroup, time window, or metric without a meaningful new analytical angle.
Do not spend a follow-up on a peripheral data property while a substantive
part of the goal remains unanswered. If the Findings already answer the goal,
explore useful new goal-relevant directions. Findings are verified context,
not a requirement that every next question refine them.
Use the data profile to ensure the questions are answerable.
Prefer useful, distinct descriptive or comparative directions. A question may connect
findings or share a topic when it seeks a clearly different outcome, comparison,
or pattern that serves the goal. The three candidates should also differ in their
analytical objectives, not merely their phrasing or small slices of the same analysis.
Do not default to explanation, interaction effects, controls, robustness checks,
or more complex methods merely because findings
exist. Do not assume a hypothesized relationship is true or an unverified join works.
Respect the conditions and conflict-use restrictions retained in Finding.

Each question must pursue one clear analytical objective with a single question mark.
Related metrics may be combined within one coherent comparison; unrelated objectives
must not be bundled. Plan its base analysis unit and optional aggregation unit.
For each candidate, state in expected_new_information the specific unanswered
or partial aspect it would resolve and the new conclusion it could support.
Do not presume the result or copy an answer already present in Finding.
Return exactly three candidates. related_evidence_ids may contain only evidence IDs
listed in Finding, and only when the new analysis genuinely relies on those findings;
otherwise use []. relation_ids may contain only IDs in the data profile.

Return JSON only:
{{"candidates":[
{{"question":"...?","expected_new_information":"...","related_evidence_ids":[],"relation_ids":[],
"analysis_granularity":{{"base":"trajectory/trip","aggregation":null}}}},
{{"question":"...?","expected_new_information":"...","related_evidence_ids":[],"relation_ids":[],
"analysis_granularity":{{"base":"group","aggregation":"..."}}}},
{{"question":"...?","expected_new_information":"...","related_evidence_ids":[],"relation_ids":[],
"analysis_granularity":{{"base":"trajectory/trip","aggregation":null}}}}
]}}
"""

        def generate(
            review_feedback: dict[str, Any] | None = None,
            previous_candidates: list[FollowUpQuestion] | None = None,
        ) -> list[FollowUpQuestion]:
            feedback = ""
            if review_feedback is not None:
                feedback = (
                    "\nYOUR PREVIOUS CANDIDATES:\n"
                    + json.dumps(
                        [{"follow_up_id": item.follow_up_id, "question": item.question}
                         for item in previous_candidates or []],
                        ensure_ascii=False,
                    )
                    + "\nDUPLICATE REVIEW AND GOAL-GAP CHECK OF THOSE CANDIDATES:\n"
                    + json.dumps(review_feedback, ensure_ascii=False)
                    + "\nRegenerate all three questions. Replace repeated objectives and "
                      "questions that miss a substantive unanswered part of the goal; "
                      "keep useful directions where possible.\n"
                )
            return _structured_call(
                self.llm,
                system=(
                    "Propose three next analysis questions using the supplied goal, memory, "
                    "and data profile. Prioritize gaps in what Finding actually established. "
                    "Do not write code, answer questions, or infer new field relations. Return JSON only."
                    if memory is not None else
                    "You are TrajStory A4 Question Generator. Propose three next analysis "
                    "questions using the supplied context. Do not write code, answer "
                    "questions, or infer new field relations. Return JSON only."
                ),
                user=generator_prompt + feedback,
                validator=validate_candidates,
            )

        review_questions = review_initial_questions or initial_questions

        def review(candidates: list[FollowUpQuestion]) -> dict[str, Any]:
            candidate_ids = {item.follow_up_id for item in candidates}
            initial_ids = {item.question_id for item in review_questions}
            review_evidence = []
            if memory is None:
                review_evidence = [
                    {
                        key: raw[key]
                        for key in ("evidence_id", "question_id", "claim", "statistics",
                                    "visualizations", "justification")
                        if key in raw
                    }
                    for item in evidence
                    for raw in [item.to_dict()]
                ]

            def validate_review(payload: dict[str, Any]) -> dict[str, Any]:
                raw = payload.get("duplicates")
                if not isinstance(raw, list):
                    raise ValueError("duplicates must be a list")
                duplicates = []
                seen = set()
                for item in raw:
                    if not isinstance(item, dict):
                        raise ValueError("every duplicate must be an object")
                    follow_up_id = str(item.get("follow_up_id", "")).strip()
                    initial_id = str(item.get("initial_question_id", "")).strip()
                    if follow_up_id not in candidate_ids or initial_id not in initial_ids:
                        raise ValueError("duplicate review refers to an unknown question")
                    if follow_up_id in seen:
                        raise ValueError("duplicate review repeats a follow-up ID")
                    seen.add(follow_up_id)
                    duplicates.append({
                        "follow_up_id": follow_up_id,
                        "initial_question_id": initial_id,
                        "reason": str(item.get("reason", "")).strip(),
                    })
                if memory is not None and self.coverage_review_enabled and "coverage_issues" not in payload:
                    raise ValueError("memory-based A4 review needs coverage_issues list")
                raw_coverage_issues = (
                    payload.get("coverage_issues", []) if self.coverage_review_enabled else []
                )
                if memory is not None and self.coverage_review_enabled and not isinstance(raw_coverage_issues, list):
                    raise ValueError("memory-based A4 review needs coverage_issues list")
                if not isinstance(raw_coverage_issues, list):
                    raise ValueError("coverage_issues must be a list")
                coverage_issues = []
                for item in raw_coverage_issues:
                    if not isinstance(item, dict):
                        raise ValueError("every coverage issue must be an object")
                    follow_up_id = str(item.get("follow_up_id", "")).strip()
                    reason = str(item.get("reason", "")).strip()
                    if follow_up_id not in candidate_ids or not reason:
                        raise ValueError("coverage issue requires a known follow-up ID and reason")
                    if follow_up_id in {issue["follow_up_id"] for issue in coverage_issues}:
                        raise ValueError("coverage issues repeat a follow-up ID")
                    coverage_issues.append({"follow_up_id": follow_up_id, "reason": reason})
                return {"duplicates": duplicates, "coverage_issues": coverage_issues}

            if memory is None:
                review_prompt = f"""
PROPOSED FOLLOW-UP QUESTIONS:
{json.dumps([{"follow_up_id": item.follow_up_id, "question": item.question} for item in candidates], ensure_ascii=False)}

INITIAL QUESTIONS:
{json.dumps([{"question_id": item.question_id, "question": item.question} for item in review_questions], ensure_ascii=False)}

INITIAL EVIDENCE:
{json.dumps(review_evidence, ensure_ascii=False)}

Mark repetition of an initial question's analytical objective even if its evidence is
missing because that branch failed; a near-restatement is still a duplicate. Shared
fields, entities, or a connected topic alone do not make a question duplicate. Do not judge
answerability, importance, novelty relative to other follow-ups, or goal coverage.
Return {{"duplicates":[{{"follow_up_id":"F1","initial_question_id":"Q1","reason":"..."}}]}};
return an empty list when none repeat an answered initial objective.
"""
            else:
                coverage_instructions = (
                    """Also flag a coverage_issue when a candidate's expected new information is
already established by a Finding, does not directly advance the high-level
goal, or examines a peripheral data property while an obvious substantive
part of the goal remains unanswered. Do not require a predetermined list of
relations or a match to benchmark answers. If the Findings already address
the goal, allow useful new goal-relevant exploration. Do not judge statistical
correctness or answerability here.
Return {"duplicates":[{"follow_up_id":"F1","initial_question_id":"Q1","reason":"..."}],
"coverage_issues":[{"follow_up_id":"F2","reason":"..."}]};
use empty lists when none apply."""
                    if self.coverage_review_enabled else
                    """Do not judge how well the questions cover the goal or identify missing
goal aspects. Report only repetition of an initial analytical objective.
Return {"duplicates":[{"follow_up_id":"F1","initial_question_id":"Q1","reason":"..."}]};
use an empty list when none repeat."""
                )
                review_prompt = f"""
HIGH-LEVEL GOAL:
{goal}

COMPRESSED FIRST-ROUND MEMORY:
{json.dumps(planning_memory, ensure_ascii=False)}

DATA PROFILE:
{json.dumps(profile.to_dict(), ensure_ascii=False)}

PROPOSED FOLLOW-UP QUESTIONS:
{json.dumps([{"follow_up_id": item.follow_up_id, "question": item.question,
             "expected_new_information": item.expected_new_information} for item in candidates], ensure_ascii=False)}

Decide whether a proposed question substantially repeats an objective in
memory.Question or asks for an answer already contained in memory.Finding.
Flag near-rephrasings and superficial changes of subgroup, time window, or
metric. For a match to a Finding, report its originating Q-id as
initial_question_id. A shared topic or field alone is not enough: allow a
specific missing comparison or pattern when the original Finding is partial.

{coverage_instructions}
"""
            return _structured_call(
                self.llm,
                system=(
                    ("Review proposed questions against the goal and compressed memory. "
                     "Flag repetitions and clear failures to address unanswered goal content. "
                     "Return JSON only." if self.coverage_review_enabled else
                     "Review proposed questions against compressed memory for repetition only. "
                     "Do not judge goal coverage or missing goal content. Return JSON only.")
                    if memory is not None else
                    "You are TrajStory A4 Duplicate Reviewer. Decide only whether each "
                    "proposed follow-up repeats an initial question's analytical objective, "
                    "including a question whose execution failed. Return JSON only."
                ),
                user=review_prompt,
                validator=validate_review,
            )

        candidates = generate()
        for attempt in range(2):
            result = review(candidates)
            self.last_duplicate_reviews.append({
                "attempt": attempt + 1,
                "candidates": [item.to_dict() for item in candidates],
                **result,
            })
            if (not result["duplicates"] and not result["coverage_issues"]) or attempt == 1:
                break
            candidates = generate(result, candidates)
        self.last_candidates = candidates
        return candidates

    def revise_follow_up(
        self,
        *,
        goal: str,
        initial_questions: list[AnalysisQuestion],
        evidence: list[EvidencePackage],
        profile: TrajectoryDataProfile,
        follow_ups: list[FollowUpQuestion],
        failed_follow_up_id: str,
        revision_instruction: str,
        context: str = "",
        memory: dict[str, list[str]] | None = None,
    ) -> FollowUpQuestion:
        by_id = {item.follow_up_id: item for item in follow_ups}
        if failed_follow_up_id not in by_id:
            raise ValueError(f"unknown follow-up question: {failed_follow_up_id}")
        evidence_ids = {item.evidence_id for item in evidence}
        relation_ids = profile.relation_ids()
        planning_memory = self._planning_memory(memory) if memory is not None else None
        retained = [item for item in follow_ups if item.follow_up_id != failed_follow_up_id]
        seen = {
            _question_key(item.question)
            for item in [*initial_questions, *retained]
        }

        def validate(payload: dict[str, Any]) -> FollowUpQuestion:
            raw = payload.get("replacement_question")
            if not isinstance(raw, dict):
                raise ValueError("replacement_question must be an object")
            question = str(raw.get("question", "")).strip()
            related = tuple(str(value) for value in raw.get("related_evidence_ids", []))
            relations = tuple(str(value) for value in raw.get("relation_ids", []))
            if not question or _question_key(question) in seen:
                raise ValueError("replacement question must be new and non-empty")
            if _similar(question, by_id[failed_follow_up_id].question):
                raise ValueError("replacement question merely restates the rejected question")
            if not set(related) <= evidence_ids:
                raise ValueError("invalid related evidence IDs")
            unknown_relations = set(relations) - relation_ids
            if unknown_relations:
                raise ValueError(
                    f"unknown relation IDs {sorted(unknown_relations)}; "
                    f"valid relation IDs are {sorted(relation_ids)}. "
                    "Use [] when no listed relation supports the question"
                )
            return FollowUpQuestion(
                follow_up_id=failed_follow_up_id,
                question=question,
                primary_evidence_id=related[0] if related else None,
                analysis_granularity=_parse_analysis_granularity(raw),
                supporting_evidence_ids=related[1:],
                relation_ids=relations,
                purpose=str(raw.get("purpose", "goal_relevant_replacement")),
                expected_new_information=str(raw.get("expected_new_information", "")).strip(),
            )

        if memory is None:
            revision_prompt = f"""
HIGH-LEVEL GOAL:
{goal}

DATA CONTEXT:
{context}

VALIDATED INITIAL EVIDENCE:
{json.dumps([item.to_dict() for item in evidence], ensure_ascii=False)}

DATA PROFILE:
{json.dumps(profile.to_dict(), ensure_ascii=False)}

REJECTED FOLLOW-UP:
{json.dumps(by_id[failed_follow_up_id].to_dict(), ensure_ascii=False)}

A3 REJECTION REASON AND REVISION INSTRUCTION:
{revision_instruction}

QUESTIONS THAT MUST NOT BE DUPLICATED:
{json.dumps([item.to_dict() for item in [*initial_questions, *retained]], ensure_ascii=False)}

Generate one replacement next question that directly serves the high-level goal, seeks
new information, and is fully answerable from fields and verified relations in the data
profile. It may connect to prior insights, but must not merely paraphrase an answered or
rejected objective. Evidence links are optional; use related_evidence_ids only when the
replacement genuinely relies on prior evidence. Preserve the failed follow-up ID; the
controller supplies it.

Return:
{{"replacement_question":{{"question":"...","related_evidence_ids":[],
"relation_ids":[],"analysis_granularity":
{{"base":"point|trajectory/trip|user/entity|group","aggregation":null}}}}}}
"""
        else:
            revision_prompt = f"""
HIGH-LEVEL GOAL:
{goal}

COMPRESSED FIRST-ROUND MEMORY:
{json.dumps(planning_memory, ensure_ascii=False)}

DATA PROFILE:
{json.dumps(profile.to_dict(), ensure_ascii=False)}

REJECTED FOLLOW-UP:
{json.dumps(by_id[failed_follow_up_id].to_dict(), ensure_ascii=False)}

WHY THE QUESTION WAS REJECTED AND HOW TO REVISE IT:
{revision_instruction}

OTHER FOLLOW-UP QUESTIONS:
{json.dumps([item.to_dict() for item in retained], ensure_ascii=False)}

Replace the rejected question with one new, goal-relevant and answerable objective.
Compare it with every memory.Question, memory.Finding, and other follow-up.
Prefer a substantive part of the goal that the Findings have not answered or
have answered only partly. A Question alone does not mean the goal was answered;
ask for a distinct, answerable missing comparison rather than repeating a
failed analysis unchanged. Do not ask for an answer already in a Finding.
Avoid superficial variants that only change wording or make a minor subgroup,
time-window, or metric change without a meaningful new analytical angle.
Choose a question requiring genuinely new analysis, while allowing a related
topic when the outcome, comparison, or pattern is substantively different.
State the missing aspect and possible new conclusion in expected_new_information.
Use only fields and relations in the data profile. Evidence links may name only IDs in memory.Finding and only when
the replacement calculation genuinely relies on them. Do not infer new data facts.
Return JSON only:
{{"replacement_question":{{"question":"...?","expected_new_information":"...","related_evidence_ids":[],
"relation_ids":[],"analysis_granularity":
{{"base":"point|trajectory/trip|user/entity|group","aggregation":null}}}}}}
"""
        return _structured_call(
            self.llm,
            system=(
                "Replace the rejected next question with a semantically novel objective "
                "relative to Question and Finding; do not answer it, write code, "
                "or infer new relations. Return JSON only."
                if memory is not None else
                "You are TrajStory A4 revising a rejected next question. Replace it; do not answer it, write code, "
                "or infer new relations. Return JSON only."
            ),
            user=revision_prompt,
            validator=validate,
        )


