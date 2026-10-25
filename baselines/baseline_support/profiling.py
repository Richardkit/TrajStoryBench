from __future__ import annotations

import json
import math
import re
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .json_utils import extract_json_object
from .llm import LLMClient
from .models import (
    ColumnProfile,
    DataBundle,
    DataRelation,
    FileProfile,
    FrameworkConfig,
    TrajectoryDataProfile,
)


IDENTIFIER_RE = re.compile(r"(^id$|_id$|^id_|identifier|uuid|mmsi|order_no|trip_no)", re.I)
LATITUDE_RE = re.compile(r"(^lat$|latitude|_lat$|lat_)", re.I)
LONGITUDE_RE = re.compile(r"(^lon$|^lng$|longitude|_lon$|_lng$|lon_|lng_)", re.I)
TIME_RE = re.compile(r"(time|timestamp|datetime|date|hour)", re.I)
GEOMETRY_RE = re.compile(r"(geometry|geom|wkt|polyline|coordinates)", re.I)

SEMANTIC_ROLES = {
    "identifier",
    "timestamp",
    "latitude",
    "longitude",
    "geometry",
    "speed",
    "distance",
    "duration",
    "direction",
    "sequence",
    "category",
    "boolean",
    "measure",
    "attribute",
}
def _native(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    return value


def _semantic_role(name: str) -> str:
    if LATITUDE_RE.search(name):
        return "latitude"
    if LONGITUDE_RE.search(name):
        return "longitude"
    if GEOMETRY_RE.search(name):
        return "geometry"
    if IDENTIFIER_RE.search(name):
        return "identifier"
    if TIME_RE.search(name):
        return "timestamp"
    lower = name.lower()
    if any(token in lower for token in ("speed", "sog", "velocity")):
        return "speed"
    if any(token in lower for token in ("distance", "length")):
        return "distance"
    if any(token in lower for token in ("duration", "elapsed")):
        return "duration"
    if any(token in lower for token in ("course", "heading", "cog", "bearing")):
        return "direction"
    return "attribute"


def _physical_type(series: pd.Series, semantic_role: str) -> str:
    if semantic_role == "timestamp":
        return "timestamp"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "continuous_numeric" if series.nunique(dropna=True) > 20 else "discrete_numeric"
    return "categorical" if series.nunique(dropna=True) <= 50 else "text"


def _timestamp_values(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series) and "hour" in str(series.name).lower():
        return pd.Series(dtype="datetime64[ns]")
    return pd.to_datetime(series, errors="coerce", format="mixed", utc=True)


class StructuredTrajectoryProfiler:
    def __init__(self, llm: LLMClient, config: FrameworkConfig) -> None:
        self.llm = llm
        self.config = config
        self._frames: dict[str, pd.DataFrame] = {}

    def build(self, bundle: DataBundle) -> TrajectoryDataProfile:
        # A profiler can be reused across runs; never expose rows retained from an
        # earlier bundle to the current schema-reasoning call.
        self._frames = {}
        profiles = [self._profile_file(item) for item in bundle.files]
        join_candidates = (
            self._build_join_candidates(profiles)
            if self.config.relations_enabled else []
        )
        reasoning_payload = self._reason_about_schema(bundle, profiles, join_candidates)
        self._apply_column_semantics(profiles, reasoning_payload)
        for profile in profiles:
            profile.coverage = self._coverage(self._frames[profile.file_id], profile.columns)
        if self.config.relations_enabled:
            self._attach_intra_relations(profiles, reasoning_payload)
        # Join candidates inform A0's table reasoning but are not part of the
        # profile shared with question, code, or follow-up stages.
        cross = (
            self._build_cross_relations(profiles, reasoning_payload)
            if self.config.relations_enabled else []
        )
        return TrajectoryDataProfile(
            files=profiles,
            cross_file_relations=cross,
        )

    def _profile_file(self, data_file) -> FileProfile:
        suffix = data_file.path.suffix.lower()
        if suffix == ".csv":
            frame = pd.read_csv(data_file.path)
        elif suffix in {".parquet", ".pq"}:
            frame = pd.read_parquet(data_file.path)
        else:
            raise ValueError(f"unsupported input format: {data_file.path}")
        self._frames[data_file.file_id] = frame
        columns = [self._profile_column(frame[name]) for name in frame.columns]
        coverage = self._coverage(frame, columns)
        return FileProfile(
            file_id=data_file.file_id,
            path=str(data_file.path),
            role=data_file.role,
            row_count=len(frame),
            columns=columns,
            coverage=coverage,
        )

    def _profile_column(
        self, series: pd.Series, *, semantic_role: str | None = None
    ) -> ColumnProfile:
        name = str(series.name)
        semantic = semantic_role or _semantic_role(name)
        physical = _physical_type(series, semantic)
        clean = series.dropna()
        statistics: dict[str, Any] = {}
        if physical in {"continuous_numeric", "discrete_numeric"} and semantic != "identifier":
            numeric = pd.to_numeric(clean, errors="coerce").dropna()
            if not numeric.empty:
                quantiles = numeric.quantile([0, 0.25, 0.5, 0.75, 1]).to_dict()
                statistics = {
                    "min": _native(numeric.min()),
                    "max": _native(numeric.max()),
                    "mean": _native(numeric.mean()),
                    "std": _native(numeric.std(ddof=0)),
                    "quantiles": {str(k): _native(v) for k, v in quantiles.items()},
                }
        elif physical == "timestamp":
            if pd.api.types.is_numeric_dtype(clean):
                numeric = pd.to_numeric(clean, errors="coerce").dropna()
                if not numeric.empty:
                    statistics = {"min": _native(numeric.min()), "max": _native(numeric.max())}
            else:
                parsed = _timestamp_values(clean)
                parsed = parsed.dropna()
                if not parsed.empty:
                    statistics = {"min": str(parsed.min()), "max": str(parsed.max())}
        elif physical in {"categorical", "boolean", "discrete_numeric"}:
            counts = clean.astype(str).value_counts().head(self.config.max_profile_categories)
            statistics = {"top_values": {str(k): int(v) for k, v in counts.items()}}
        samples = [_native(value) for value in clean.head(self.config.sample_rows_per_file).tolist()]
        return ColumnProfile(
            name=name,
            physical_type=physical,
            semantic_role=semantic,
            unique_count=int(series.nunique(dropna=True)),
            statistics=statistics,
            sample_values=samples,
            raw_dtype=str(series.dtype),
        )

    def _coverage(self, frame: pd.DataFrame, columns: list[ColumnProfile]) -> dict[str, Any]:
        coverage: dict[str, Any] = {}
        temporal = [column for column in columns if column.semantic_role == "timestamp" and column.statistics]
        if temporal:
            coverage["temporal"] = {
                column.name: {
                    "min": column.statistics.get("min"),
                    "max": column.statistics.get("max"),
                }
                for column in temporal
            }
        latitudes = [column.name for column in columns if column.semantic_role == "latitude"]
        longitudes = [column.name for column in columns if column.semantic_role == "longitude"]
        if latitudes and longitudes:
            lat = pd.to_numeric(frame[latitudes[0]], errors="coerce")
            lon = pd.to_numeric(frame[longitudes[0]], errors="coerce")
            valid = lat.notna() & lon.notna()
            if valid.any():
                coverage["spatial"] = {
                    "latitude_field": latitudes[0],
                    "longitude_field": longitudes[0],
                    "min_lat": _native(lat[valid].min()),
                    "max_lat": _native(lat[valid].max()),
                    "min_lon": _native(lon[valid].min()),
                    "max_lon": _native(lon[valid].max()),
                }
        return coverage

    @staticmethod
    def _normalized_join_name(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.lower())

    @staticmethod
    def _join_value(value: Any) -> str | None:
        native = _native(value)
        if native is None:
            return None
        if isinstance(native, float):
            return format(native, ".12g")
        return str(native).strip().casefold()

    @staticmethod
    def _types_compatible(left: pd.Series, right: pd.Series) -> bool:
        left_numeric = pd.api.types.is_numeric_dtype(left)
        right_numeric = pd.api.types.is_numeric_dtype(right)
        left_datetime = pd.api.types.is_datetime64_any_dtype(left)
        right_datetime = pd.api.types.is_datetime64_any_dtype(right)
        if left_datetime or right_datetime:
            return left_datetime and right_datetime
        return left_numeric == right_numeric

    @staticmethod
    def _join_cardinality(left: pd.Series, right: pd.Series) -> str:
        left_non_null = left.dropna()
        right_non_null = right.dropna()
        left_unique = len(left_non_null) == left_non_null.nunique(dropna=True)
        right_unique = len(right_non_null) == right_non_null.nunique(dropna=True)
        if left_unique and right_unique:
            return "one_to_one"
        if left_unique:
            return "one_to_many"
        if right_unique:
            return "many_to_one"
        return "many_to_many"

    def _build_join_candidates(
        self, profiles: list[FileProfile], *, per_pair_limit: int = 8
    ) -> list[dict[str, Any]]:
        """Propose bounded, auditable field pairs before the LLM reasons about tables.

        Candidates are deliberately permissive enough to retain semantic aliases, while
        the A0 model may only accept candidate IDs and can never invent a field pair.
        """

        output: list[dict[str, Any]] = []
        candidate_index = 0
        for left_profile, right_profile in combinations(profiles, 2):
            left_frame = self._frames[left_profile.file_id]
            right_frame = self._frames[right_profile.file_id]
            ranked: list[tuple[float, dict[str, Any]]] = []
            left_columns = {item.name: item for item in left_profile.columns}
            right_columns = {item.name: item for item in right_profile.columns}
            for left_name in left_frame.columns:
                for right_name in right_frame.columns:
                    left_series = left_frame[left_name]
                    right_series = right_frame[right_name]
                    if not self._types_compatible(left_series, right_series):
                        continue

                    left_key = self._normalized_join_name(str(left_name))
                    right_key = self._normalized_join_name(str(right_name))
                    exact_name = bool(left_key and left_key == right_key)
                    name_similarity = SequenceMatcher(None, left_key, right_key).ratio()

                    left_clean = left_series.dropna()
                    right_clean = right_series.dropna()
                    left_sample = left_clean.sample(
                        n=min(1000, len(left_clean)), random_state=0
                    )
                    right_sample = right_clean.sample(
                        n=min(1000, len(right_clean)), random_state=0
                    )
                    left_values = {
                        normalized
                        for value in left_sample
                        if (normalized := self._join_value(value)) is not None
                    }
                    right_values = {
                        normalized
                        for value in right_sample
                        if (normalized := self._join_value(value)) is not None
                    }
                    intersection = left_values & right_values
                    union = left_values | right_values
                    minimum = min(len(left_values), len(right_values))
                    containment = len(intersection) / minimum if minimum else 0.0
                    jaccard = len(intersection) / len(union) if union else 0.0

                    left_column = left_columns[str(left_name)]
                    right_column = right_columns[str(right_name)]
                    same_role = left_column.semantic_role == right_column.semantic_role
                    identifier_pair = (
                        left_column.semantic_role == right_column.semantic_role == "identifier"
                    )
                    left_non_null = max(int(left_series.notna().sum()), 1)
                    right_non_null = max(int(right_series.notna().sum()), 1)
                    left_uniqueness = left_column.unique_count / left_non_null
                    right_uniqueness = right_column.unique_count / right_non_null
                    uniqueness_signal = max(left_uniqueness, right_uniqueness)

                    eligible = (
                        exact_name
                        or (len(intersection) >= 2 and containment >= 0.20)
                        or (identifier_pair and name_similarity >= 0.45)
                        or (name_similarity >= 0.72 and same_role)
                    )
                    if not eligible:
                        continue
                    name_signal = 1.0 if exact_name else 0.7 * name_similarity
                    cardinality = self._join_cardinality(left_series, right_series)
                    safe_cardinality = float(cardinality != "many_to_many")
                    score = (
                        0.35 * name_signal
                        + 0.25 * containment
                        + 0.10 * jaccard
                        + 0.10 * float(same_role)
                        + 0.15 * uniqueness_signal
                        + 0.05 * safe_cardinality
                    )
                    candidate = {
                        "files": [left_profile.file_id, right_profile.file_id],
                        "fields": [str(left_name), str(right_name)],
                        "name_exact": exact_name,
                        "name_similarity": round(name_similarity, 4),
                        "value_overlap": {
                            "intersection_count": len(intersection),
                            "jaccard": round(jaccard, 4),
                            "left_coverage": round(
                                len(intersection) / len(left_values) if left_values else 0.0, 4
                            ),
                            "right_coverage": round(
                                len(intersection) / len(right_values) if right_values else 0.0, 4
                            ),
                            "containment": round(containment, 4),
                        },
                        "semantic_role_match_from_names": same_role,
                        "uniqueness": {
                            "left": round(left_uniqueness, 4),
                            "right": round(right_uniqueness, 4),
                        },
                        "cardinality": cardinality,
                        "score": round(score, 4),
                    }
                    ranked.append((score, candidate))

            ranked.sort(
                key=lambda item: (
                    -item[0], item[1]["fields"][0], item[1]["fields"][1]
                )
            )
            for _, candidate in ranked[:per_pair_limit]:
                candidate_index += 1
                candidate["candidate_id"] = f"J{candidate_index}"
                output.append(candidate)
        return output

    @staticmethod
    def _rows(frame: pd.DataFrame, count: int, *, random: bool) -> list[dict[str, Any]]:
        if frame.empty:
            return []
        selected = (
            frame.sample(n=min(count, len(frame)), random_state=0)
            if random
            else frame.head(count)
        )
        return [
            {str(key): _native(value) for key, value in row.items()}
            for row in selected.to_dict(orient="records")
        ]

    def _reason_about_schema(
        self,
        bundle: DataBundle,
        profiles: list[FileProfile],
        join_candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        compact = {
            "files": [
                {
                    "file_id": profile.file_id,
                    "role": profile.role,
                    "row_count": profile.row_count,
                    "columns": [
                        {
                            "name": column.name,
                            "raw_dtype": column.raw_dtype,
                            "physical_type": column.physical_type,
                            "unique_count": column.unique_count,
                            "statistics": column.statistics,
                            "sample_values": column.sample_values,
                        }
                        for column in profile.columns
                    ],
                }
                for profile in profiles
            ],
            "complete_head_rows": {
                file_id: self._rows(frame, self.config.sample_rows_per_file, random=False)
                for file_id, frame in self._frames.items()
            },
            "complete_random_rows": {
                file_id: self._rows(frame, self.config.sample_rows_per_file, random=True)
                for file_id, frame in self._frames.items()
            },
        }
        if self.config.relations_enabled:
            compact["deterministic_cross_table_join_candidates"] = join_candidates
        profile_text = json.dumps(compact, ensure_ascii=False, default=str)
        if len(profile_text) > self.config.max_profile_text_chars:
            raise ValueError(
                f"A0 schema prompt has {len(profile_text)} characters, exceeding "
                f"max_profile_text_chars={self.config.max_profile_text_chars}; refusing "
                "to truncate the JSON or silently omit complete rows"
            )

        intra_instruction = (
            "Also propose only useful structural relationships inside each individual table. The\n"
            "relationship label is free-form: do not force it into identifier/temporal/spatial/semantic."
            if self.config.relations_enabled else
            "Do not infer relationships among fields within a table. Return intra_file_relations as []."
        )
        cross_instruction = (
            "For cross-table analysis, describe the relationship between each relevant pair of tables.\n"
            "Relationship labels remain free-form. State what the tables mean to each other, not\n"
            "how to join them: omit join keys, candidate IDs, cardinalities, overlap scores, and\n"
            "code instructions from relationship labels and descriptions. Inspect\n"
            "deterministic_cross_table_join_candidates and select candidate IDs only\n"
            "when the fields are a semantically valid row-linkage key without incorrect row\n"
            "multiplication. Reject coincidental overlaps in categories, timestamps, measures,\n"
            "or coordinates. Candidate IDs are internal A0 clues and are not shared with\n"
            "later stages. An empty list is correct when the tables are related conceptually\n"
            "but have no safe join candidate."
            if self.config.relations_enabled else
            "Do not infer relationships between tables or candidate joins. Return cross_file_relations as []."
        )
        prompt = f"""
DATA CONTEXT:
{bundle.context}

Jointly classify every column using its name, raw dtype, statistics, per-column values,
and the supplied complete rows. Complete rows preserve relationships among values in the
same record. The random rows are a deterministic random sample (seed 0), not additional
evidence for answering the analytical goal.

For semantic_role use exactly one of:
{sorted(SEMANTIC_ROLES)}

{intra_instruction}

{cross_instruction}

Do not answer the analytical goal and do not propose insights. Use no columns or files
absent from the supplied profile.

Return exactly:
{{
  "column_semantics": [
    {{"file_id":"F1","columns":[
      {{"name":"column_name","semantic_role":"attribute","description":"what the field represents",
        "unit":null}}
    ]}}
  ],
  "intra_file_relations": [
    {{"file_id":"F1","relationship":"free-form relationship label",
      "fields":["field1","field2"],"description":"..."}}
  ],
  "cross_file_relations": [
    {{"files":["F1","F2"],"relationship":"free-form table relationship label",
      "description":"how the two tables relate as tables",
      "join_candidate_ids":["J1"]}}
  ]
}}

PROFILE AND COMPLETE ROWS:
{profile_text}
"""
        previous = ""
        error = ""
        for _ in range(3):
            repair = ""
            if error:
                repair = (
                    f"\nYour previous response was invalid: {error}\n"
                    f"Previous response:\n{previous}\nReturn a corrected complete JSON object.\n"
                )
            previous = self.llm.complete(
                system=(
                    "Jointly infer column semantics and concise structural relationships "
                    "from deterministic table profiles and complete sample rows. Return JSON only."
                    if self.config.relations_enabled else
                    "Infer column semantics from deterministic table profiles and complete "
                    "sample rows. Do not infer field or table relationships. Return JSON only."
                ),
                user=prompt + repair,
            )
            try:
                payload = extract_json_object(previous)
                self._validate_schema_payload(payload, profiles, join_candidates)
                return payload
            except (KeyError, TypeError, ValueError) as exc:
                error = str(exc)
        raise RuntimeError(f"A0 did not return a valid complete schema response: {error}")

    def _validate_schema_payload(
        self,
        payload: dict[str, Any],
        profiles: list[FileProfile],
        join_candidates: list[dict[str, Any]],
    ) -> None:
        expected = {
            profile.file_id: {column.name for column in profile.columns}
            for profile in profiles
        }
        raw_files = payload.get("column_semantics")
        if not isinstance(raw_files, list):
            raise ValueError("column_semantics must be a list")
        seen_files: set[str] = set()
        for raw_file in raw_files:
            if not isinstance(raw_file, dict):
                raise ValueError("each column_semantics item must be an object")
            file_id = str(raw_file.get("file_id", ""))
            if file_id not in expected or file_id in seen_files:
                raise ValueError(f"unknown or duplicate semantic file_id: {file_id!r}")
            raw_columns = raw_file.get("columns")
            if not isinstance(raw_columns, list):
                raise ValueError(f"column_semantics[{file_id}].columns must be a list")
            seen_columns: set[str] = set()
            for raw_column in raw_columns:
                if not isinstance(raw_column, dict):
                    raise ValueError("each semantic column must be an object")
                name = str(raw_column.get("name", ""))
                if name not in expected[file_id] or name in seen_columns:
                    raise ValueError(f"unknown or duplicate column {file_id}.{name}")
                if raw_column.get("semantic_role") not in SEMANTIC_ROLES:
                    raise ValueError(f"invalid semantic_role for {file_id}.{name}")
                description = raw_column.get("description")
                if not isinstance(description, str) or not description.strip():
                    raise ValueError(f"missing semantic description for {file_id}.{name}")
                unit = raw_column.get("unit")
                if unit is not None and not isinstance(unit, str):
                    raise ValueError(f"unit must be a string or null for {file_id}.{name}")
                seen_columns.add(name)
            if seen_columns != expected[file_id]:
                missing = sorted(expected[file_id] - seen_columns)
                raise ValueError(f"column semantics omitted {file_id} columns: {missing}")
            seen_files.add(file_id)
        if seen_files != set(expected):
            raise ValueError(f"column semantics omitted files: {sorted(set(expected) - seen_files)}")

        for key in ("intra_file_relations", "cross_file_relations"):
            if not isinstance(payload.get(key), list):
                raise ValueError(f"{key} must be a list")
        if not self.config.relations_enabled and payload["intra_file_relations"]:
            raise ValueError("intra_file_relations must be empty in the no-relations ablation")
        if not self.config.relations_enabled and payload["cross_file_relations"]:
            raise ValueError("cross_file_relations must be empty in the no-relations ablation")

        candidate_by_id = {item["candidate_id"]: item for item in join_candidates}
        for relation in payload["cross_file_relations"]:
            if not isinstance(relation, dict):
                raise ValueError("each cross_file_relations item must be an object")
            files = [str(value) for value in relation.get("files", [])]
            if len(files) != 2 or len(set(files)) != 2 or not set(files) <= set(expected):
                raise ValueError("cross_file_relations contains invalid files")
            candidate_ids = relation.get("join_candidate_ids", [])
            if not isinstance(candidate_ids, list) or not all(
                isinstance(value, str) for value in candidate_ids
            ):
                raise ValueError("join_candidate_ids must be a list of candidate IDs")
            if len(candidate_ids) != len(set(candidate_ids)):
                raise ValueError("join_candidate_ids must not contain duplicates")
            for candidate_id in candidate_ids:
                candidate = candidate_by_id.get(candidate_id)
                if candidate is None:
                    raise ValueError(f"unknown join candidate ID: {candidate_id}")
                if set(candidate["files"]) != set(files):
                    raise ValueError(
                        f"join candidate {candidate_id} does not belong to relation files"
                    )

    def _apply_column_semantics(
        self, profiles: list[FileProfile], payload: dict[str, Any]
    ) -> None:
        semantic_by_file = {
            str(item["file_id"]): {str(column["name"]): column for column in item["columns"]}
            for item in payload["column_semantics"]
        }
        for profile in profiles:
            frame = self._frames[profile.file_id]
            updated = []
            for name in frame.columns:
                semantic = semantic_by_file[profile.file_id][str(name)]
                column = self._profile_column(
                    frame[name], semantic_role=str(semantic["semantic_role"])
                )
                column.semantic_description = str(semantic["description"]).strip()
                unit = semantic.get("unit")
                column.unit = str(unit).strip() if isinstance(unit, str) and unit.strip() else None
                updated.append(column)
            profile.columns = updated

    def _attach_intra_relations(self, profiles: list[FileProfile], payload: dict[str, Any]) -> None:
        profile_by_id = {item.file_id: item for item in profiles}
        counters: dict[str, int] = {}
        for raw in payload.get("intra_file_relations", []):
            file_id = str(raw.get("file_id", ""))
            if file_id not in profile_by_id:
                continue
            valid_fields = {column.name for column in profile_by_id[file_id].columns}
            fields = [str(field) for field in raw.get("fields", [])]
            if not fields or not set(fields) <= valid_fields:
                continue
            counters[file_id] = counters.get(file_id, 0) + 1
            relation_type = str(raw.get("relationship", "")).strip()
            if not relation_type:
                continue
            profile_by_id[file_id].relations.append(
                DataRelation(
                    relation_id=f"R_{file_id}_{counters[file_id]}",
                    files=[file_id],
                    relation_type=relation_type,
                    fields=fields,
                    description=str(raw.get("description", "")),
                    confidence="inferred",
                    verification={"method": "schema_presence_only", "valid": None},
                )
            )

    def _build_cross_relations(
        self,
        profiles: list[FileProfile],
        payload: dict[str, Any],
    ) -> list[DataRelation]:
        profile_by_id = {item.file_id: item for item in profiles}
        output: list[DataRelation] = []
        for index, raw in enumerate(payload.get("cross_file_relations", []), start=1):
            files = [str(value) for value in raw.get("files", [])]
            if len(files) != 2 or len(set(files)) != 2 or not set(files) <= set(profile_by_id):
                continue
            relation_type = str(raw.get("relationship", "")).strip()
            description = str(raw.get("description", "")).strip()
            if not relation_type or not description:
                continue
            relation_id = f"R_cross_{index}"
            output.append(
                DataRelation(
                    relation_id=relation_id,
                    files=files,
                    relation_type=relation_type,
                    fields=[],
                    description=description,
                    confidence="inferred",
                )
            )
        return output
