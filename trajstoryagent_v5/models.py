from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


def _clean(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


class Serializable:
    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))


@dataclass(frozen=True)
class DataFile(Serializable):
    file_id: str
    path: Path
    role: str = "trajectory data"

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path).expanduser().resolve())
        if not self.file_id.strip():
            raise ValueError("file_id must not be empty")
        if not self.path.is_file():
            raise FileNotFoundError(self.path)


@dataclass(frozen=True)
class DataBundle(Serializable):
    goal: str
    files: tuple[DataFile, ...]
    context: str = ""
    role: str = "trajectory analyst"

    def __post_init__(self) -> None:
        if not self.goal.strip():
            raise ValueError("goal must not be empty")
        if not self.files:
            raise ValueError("at least one data file is required")
        ids = [item.file_id for item in self.files]
        if len(ids) != len(set(ids)):
            raise ValueError("file_id values must be unique")


@dataclass
class ColumnProfile(Serializable):
    name: str
    physical_type: str
    semantic_role: str
    unique_count: int
    statistics: dict[str, Any] = field(default_factory=dict)
    sample_values: list[Any] = field(default_factory=list)
    raw_dtype: str = ""
    semantic_description: str = ""
    unit: str | None = None


@dataclass
class FileProfile(Serializable):
    file_id: str
    path: str
    role: str
    row_count: int
    columns: list[ColumnProfile]
    coverage: dict[str, Any] = field(default_factory=dict)
    relations: list["DataRelation"] = field(default_factory=list)


@dataclass
class DataRelation(Serializable):
    relation_id: str
    files: list[str]
    relation_type: str
    fields: list[str]
    description: str
    confidence: str = "inferred"
    verification: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryDataProfile(Serializable):
    files: list[FileProfile]
    cross_file_relations: list[DataRelation]

    def file_ids(self) -> set[str]:
        return {item.file_id for item in self.files}

    def relation_ids(self) -> set[str]:
        return {
            relation.relation_id
            for profile in self.files
            for relation in profile.relations
        } | {relation.relation_id for relation in self.cross_file_relations}


@dataclass(frozen=True)
class MethodDefinition(Serializable):
    """A stable analysis method that a plan may select by ID."""

    method_id: str
    operator: str
    description: str
    deterministic: bool = True


@dataclass(frozen=True)
class SemanticDefinition(Serializable):
    """A run-scoped lock for a derived concept shared across branches."""

    definition_id: str
    concept: str
    method_id: str
    source_file_ids: tuple[str, ...]
    fields: tuple[str, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticRegistry(Serializable):
    """Validated fields, methods, relations, and frozen run-scoped definitions."""

    field_ids: list[str]
    methods: list[MethodDefinition]
    relation_ids: list[str]
    definitions: list[SemanticDefinition] = field(default_factory=list)

    def method_ids(self) -> set[str]:
        return {item.method_id for item in self.methods}

    def definition_ids(self) -> set[str]:
        return {item.definition_id for item in self.definitions}

    def method(self, method_id: str) -> MethodDefinition:
        for item in self.methods:
            if item.method_id == method_id:
                return item
        raise KeyError(method_id)

    def definition(self, definition_id: str) -> SemanticDefinition:
        for item in self.definitions:
            if item.definition_id == definition_id:
                return item
        raise KeyError(definition_id)


@dataclass(frozen=True)
class AnalysisFilter(Serializable):
    field: str
    operator: Literal[
        "eq", "ne", "gt", "ge", "lt", "le", "in", "not_in",
        "between", "not_null", "date_eq", "contains",
    ]
    value: Any = None


@dataclass(frozen=True)
class AnalysisTransformation(Serializable):
    """Deterministic data preparation applied before a terminal recipe."""

    transformation_id: str
    method_id: str
    source_file_ids: tuple[str, ...]
    output_fields: tuple[str, ...]
    parameters: dict[str, Any] = field(default_factory=dict)
    definition_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnalysisPlan(Serializable):
    """Executable, validated contract between question planning and execution."""

    plan_id: str
    operator: Literal[
        "count", "distribution", "group_summary", "time_profile", "cross_tab",
        "correlation", "outlier_iqr", "trajectory_features", "custom",
    ]
    analysis_unit: str
    method_id: str
    source_file_ids: tuple[str, ...]
    transformations: tuple[AnalysisTransformation, ...] = ()
    value_fields: tuple[str, ...] = ()
    group_by: tuple[str, ...] = ()
    filters: tuple[AnalysisFilter, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)
    definition_ids: tuple[str, ...] = ()
    inherited_definition_ids: tuple[str, ...] = ()

    @property
    def is_recipe(self) -> bool:
        return self.operator != "custom"


@dataclass(frozen=True)
class AnalysisGranularity(Serializable):
    base: str
    aggregation: str | None = None


@dataclass(frozen=True)
class AnalysisQuestion(Serializable):
    question_id: str
    question: str
    analysis_granularity: AnalysisGranularity = field(
        default_factory=lambda: AnalysisGranularity("trajectory/trip")
    )




@dataclass
class EvidencePackage(Serializable):
    evidence_id: str
    question_id: str
    question: str
    analysis_granularity: AnalysisGranularity
    insight: str
    statistics: list[dict[str, Any]]
    visualizations: list[dict[str, Any]]
    supporting_scope: dict[str, Any]
    tables: list[dict[str, Any]] = field(default_factory=list)
    justification: str = ''
    limitations: list[str] = field(default_factory=list)
    evidence_strength: str = 'moderate'
    execution_notes: str = ''
    code_path: str | None = None
    raw_result_path: str | None = None
    interpretation_path: str | None = None
    parent_evidence_ids: list[str] = field(default_factory=list)
    analysis_plan: Any = None
    agreement_context: dict[str, Any] = field(default_factory=dict)
    comparison_records: list[dict[str, Any]] = field(default_factory=list)

    @property
    def claim(self):
        """Internal evidence-review alias; serialization always uses insight."""
        return self.insight

    @claim.setter
    def claim(self, value):
        self.insight = value



@dataclass
class VerificationResult(Serializable):
    question_id: str
    status: Literal["VALID", "REVISE"]
    issues: list[str] = field(default_factory=list)
    revision_instruction: str = ""
    revision_target: Literal["EVIDENCE", "QUESTION"] = "EVIDENCE"


@dataclass(frozen=True)
class FollowUpQuestion(Serializable):
    follow_up_id: str
    question: str
    primary_evidence_id: str | None = None
    analysis_granularity: AnalysisGranularity = field(
        default_factory=lambda: AnalysisGranularity("trajectory/trip")
    )
    supporting_evidence_ids: tuple[str, ...] = ()
    relation_ids: tuple[str, ...] = ()
    purpose: str = "goal_relevant_exploration"
    expected_new_information: str = ""
    @property
    def related_evidence_ids(self) -> tuple[str, ...]:
        """Optional context links; a follow-up need not have a parent."""
        return tuple(
            dict.fromkeys(
                [
                    *([self.primary_evidence_id] if self.primary_evidence_id else []),
                    *self.supporting_evidence_ids,
                ]
            )
        )






@dataclass(frozen=True)
class FrameworkConfig(Serializable):
    agreement_enabled: bool = True
    max_code_repairs: int = 2
    max_interpretation_repairs: int = 2
    max_relevance_repairs: int = 2
    code_timeout_seconds: int = 180
    parallel_workers: int = 3
    sample_rows_per_file: int = 3
    max_profile_categories: int = 12
    # Multi-table cases can legitimately contain around 90k characters once
    # complete sample rows and deterministic join candidates are included.
    # Keep a hard guard, but do not reject those valid, still modest prompts.
    max_profile_text_chars: int = 120_000
    memory_manager_enabled: bool = True
    relations_enabled: bool = True
    coverage_review_enabled: bool = True
    seed: int | None = 0
    min_support: int = 1

    def __post_init__(self) -> None:
        if type(self.min_support) is not int or self.min_support < 1:
            raise ValueError("min_support must be a positive integer")
        if not self.memory_manager_enabled:
            raise ValueError("Formal v5 requires first-round refined-insight memory")
        if not self.coverage_review_enabled:
            raise ValueError("Formal v5 requires question coverage review")
        if self.seed is not None and (type(self.seed) is not int or not 0 <= self.seed < 2**31):
            raise ValueError("seed must be an integer in [0, 2**31 - 1]")
        for name in (
            "max_code_repairs",
            "max_interpretation_repairs",
            "max_relevance_repairs",
            "code_timeout_seconds",
            "parallel_workers",
            "sample_rows_per_file",
            "max_profile_categories",
            "max_profile_text_chars",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class BranchInsight(Serializable):
    insight_id: str
    question_id: str
    insight: str
    evidence_ids: tuple[str, ...]


@dataclass
class TrajStoryReport(Serializable):
    insights: list[BranchInsight]
    summary: str
    evidence: list[EvidencePackage]
    questions: list[Any]
    follow_ups: list[Any]
