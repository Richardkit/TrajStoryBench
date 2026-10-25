"""Stable, composable primitives for LLM-generated trajectory analyses.

The functions in this module deliberately compute data products rather than
domain conclusions.  Generated programs remain responsible for selecting an
analysis, while this layer owns repetitive parsing, provenance, plotting, and
artifact serialization details.
"""

from __future__ import annotations

import json
import hashlib
import copy
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_ID_FIELD_RE = re.compile(r"^(trajectory|trip|user|taxi|vessel|vehicle|order)_?id$", re.I)
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
TOOL_LAYER_VERSION = "trajstory.analysis_tools.v2"


class PlotResult:
    """Figure-compatible result that also supports ``figure, axis = ...``."""

    def __init__(self, figure: Any, axis: Any) -> None:
        self.figure = figure
        self.axis = axis

    def __iter__(self):
        yield self.figure
        yield self.axis

    def __getattr__(self, name: str) -> Any:
        return getattr(self.figure, name)


def _native(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, Mapping):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        return [_native(item) for item in list(value)]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _normalize_chart_axis(axis: Mapping[str, Any], axis_name: str) -> dict[str, Any]:
    """Validate a bounded, JSON-native observation of one plotted axis.

    ``value`` may be a flat sequence or a mapping of series names to sequences.
    The latter represents grouped lines/bars and heat-map rows without requiring a
    vision model to recover the values from image pixels.
    """

    if not isinstance(axis, Mapping):
        raise ValueError(f"{axis_name} must be an object with description and value")
    description = str(axis.get("description", "")).strip()
    if not description:
        raise ValueError(f"{axis_name}.description must be non-empty")
    if "value" not in axis:
        raise ValueError(f"{axis_name}.value is required")
    value = _native(axis["value"])

    def validate_bounded(item: Any, *, depth: int = 0) -> None:
        if depth > 3:
            raise ValueError(f"{axis_name}.value is nested too deeply")
        if isinstance(item, list):
            if not item:
                raise ValueError(f"{axis_name}.value must not be empty")
            if len(item) > 50:
                raise ValueError(f"{axis_name}.value sequences are limited to 50 values")
            for child in item:
                validate_bounded(child, depth=depth + 1)
        elif isinstance(item, dict):
            if not item:
                raise ValueError(f"{axis_name}.value must not be empty")
            if len(item) > 50:
                raise ValueError(f"{axis_name}.value is limited to 50 named series")
            for child in item.values():
                validate_bounded(child, depth=depth + 1)
        elif item is not None and not isinstance(item, (str, int, float, bool)):
            raise ValueError(f"{axis_name}.value contains a non-JSON scalar")

    validate_bounded(value)
    if len(json.dumps(value, ensure_ascii=False)) > 16_000:
        raise ValueError(f"{axis_name}.value exceeds the structured observation budget")
    return {"description": description, "value": value}


def parse_timestamps(values: pd.Series | Iterable[Any], *, utc: bool = True) -> pd.Series:
    """Parse heterogeneous timestamps without relying on first-row inference.

    Pandas' ``format='mixed'`` handles ISO strings with and without fractional
    seconds.  A deterministic anchor year is added for month-day-only values.
    Invalid values become ``NaT`` rather than aborting an analysis.
    """

    series = values.copy() if isinstance(values, pd.Series) else pd.Series(values)
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce", utc=utc)
    text = series.astype("string").str.strip()
    month_day = text.str.match(r"^\d{1,2}-\d{1,2}(?:\s|T)", na=False)
    anchored = text.where(~month_day, "2000-" + text)
    return pd.to_datetime(anchored, errors="coerce", format="mixed", utc=utc)


def haversine_km(
    lat1: Any, lon1: Any, lat2: Any, lon2: Any
) -> np.ndarray:
    """Vectorized great-circle distance in kilometres."""

    lat1_arr = np.radians(np.asarray(lat1, dtype=float))
    lon1_arr = np.radians(np.asarray(lon1, dtype=float))
    lat2_arr = np.radians(np.asarray(lat2, dtype=float))
    lon2_arr = np.radians(np.asarray(lon2, dtype=float))
    dlat = lat2_arr - lat1_arr
    dlon = lon2_arr - lon1_arr
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_arr) * np.cos(lat2_arr) * np.sin(dlon / 2.0) ** 2
    return 6371.0088 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def point_to_polyline_haversine_km(
    point_latitude: Any,
    point_longitude: Any,
    route_latitude: Any,
    route_longitude: Any,
) -> np.ndarray:
    """Distance from each point to the nearest polyline segment in kilometres.

    Segment projection is performed in the point-centred local tangent plane;
    the final point-to-projection distance is always evaluated by Haversine.
    This precise algorithm is the implementation behind
    ``route_assignment_mean_point_to_polyline_v1``.
    """

    plat = np.asarray(point_latitude, dtype=float).reshape(-1)
    plon = np.asarray(point_longitude, dtype=float).reshape(-1)
    rlat = np.asarray(route_latitude, dtype=float).reshape(-1)
    rlon = np.asarray(route_longitude, dtype=float).reshape(-1)
    if len(rlat) < 2 or len(rlat) != len(rlon):
        raise ValueError("a route polyline requires at least two paired coordinates")
    if len(plat) != len(plon):
        raise ValueError("point latitude and longitude arrays must have equal length")
    if not np.isfinite(np.r_[rlat, rlon]).all():
        raise ValueError("route coordinates must be finite")
    if not len(plat):
        return np.asarray([], dtype=float)

    earth_radius = 6371.0088
    plat_rad = np.radians(plat)[:, None]
    plon_rad = np.radians(plon)[:, None]
    alat = np.radians(rlat[:-1])[None, :]
    alon = np.radians(rlon[:-1])[None, :]
    blat = np.radians(rlat[1:])[None, :]
    blon = np.radians(rlon[1:])[None, :]

    def wrapped_delta(value: np.ndarray) -> np.ndarray:
        return (value + np.pi) % (2.0 * np.pi) - np.pi

    ax = earth_radius * wrapped_delta(alon - plon_rad) * np.cos(plat_rad)
    ay = earth_radius * (alat - plat_rad)
    bx = earth_radius * wrapped_delta(blon - plon_rad) * np.cos(plat_rad)
    by = earth_radius * (blat - plat_rad)
    dx, dy = bx - ax, by - ay
    denominator = dx * dx + dy * dy
    safe_denominator = np.where(denominator > 0.0, denominator, 1.0)
    fraction = np.clip(-(ax * dx + ay * dy) / safe_denominator, 0.0, 1.0)
    fraction = np.where(denominator > 0.0, fraction, 0.0)
    closest_lat = alat + fraction * (blat - alat)
    segment_delta_lon = wrapped_delta(blon - alon)
    closest_lon = plon_rad + wrapped_delta(alon - plon_rad) + fraction * segment_delta_lon
    distances = haversine_km(
        plat_rad * 180.0 / np.pi,
        plon_rad * 180.0 / np.pi,
        closest_lat * 180.0 / np.pi,
        closest_lon * 180.0 / np.pi,
    )
    return np.nanmin(distances, axis=1)


def assign_trajectories_to_routes(
    points: pd.DataFrame,
    routes: pd.DataFrame,
    *,
    trajectory_id: str,
    point_latitude: str,
    point_longitude: str,
    route_id: str,
    route_latitude: str,
    route_longitude: str,
    point_order: str | None = None,
    route_order: str | None = None,
    route_name: str | None = None,
    route_class: str | None = None,
    carry_columns: Sequence[str] = (),
    output_route_id: str = "assigned_route_id",
    output_route_name: str = "assigned_route_name",
    output_route_class: str = "assigned_route_class",
    output_distance: str = "assignment_distance_km",
) -> pd.DataFrame:
    """Assign each trajectory to the route with minimum mean point distance.

    The six ID/coordinate column names after ``routes`` are required keyword-only
    arguments. Coordinates are degrees; distances are kilometres. Supply
    ``route_order`` if route rows are not already in polyline order.
    Output columns default to the supplied trajectory ID, ``assigned_route_id``
    and ``assignment_distance_km``. Optional route name/class inputs add
    ``assigned_route_name``/``assigned_route_class``; ``carry_columns`` copy the
    first non-null point value per trajectory. No distance threshold is applied.
    """

    point_columns = [trajectory_id, point_latitude, point_longitude, *carry_columns]
    route_columns = [route_id, route_latitude, route_longitude]
    if point_order:
        point_columns.append(point_order)
    if route_order:
        route_columns.append(route_order)
    if route_name:
        route_columns.append(route_name)
    if route_class:
        route_columns.append(route_class)
    _require_columns(points, point_columns)
    _require_columns(routes, route_columns)
    point_work = points.loc[:, list(dict.fromkeys(point_columns))].copy()
    route_work = routes.loc[:, list(dict.fromkeys(route_columns))].copy()
    point_work[point_latitude] = pd.to_numeric(point_work[point_latitude], errors="coerce")
    point_work[point_longitude] = pd.to_numeric(point_work[point_longitude], errors="coerce")
    route_work[route_latitude] = pd.to_numeric(route_work[route_latitude], errors="coerce")
    route_work[route_longitude] = pd.to_numeric(route_work[route_longitude], errors="coerce")
    point_work = point_work.dropna(subset=[trajectory_id, point_latitude, point_longitude])
    route_work = route_work.dropna(subset=[route_id, route_latitude, route_longitude])
    if point_order:
        point_work = point_work.sort_values([trajectory_id, point_order], kind="stable")
    if route_order:
        route_work = route_work.sort_values([route_id, route_order], kind="stable")
    route_groups = sorted(route_work.groupby(route_id, sort=False), key=lambda item: str(item[0]))
    if not route_groups:
        raise ValueError("no valid reference routes are available")

    rows: list[dict[str, Any]] = []
    for identifier, group in point_work.groupby(trajectory_id, sort=False, dropna=False):
        distances: list[tuple[str, Any, float, pd.DataFrame]] = []
        for candidate_id, route in route_groups:
            if len(route) < 2:
                continue
            per_point = point_to_polyline_haversine_km(
                group[point_latitude], group[point_longitude],
                route[route_latitude], route[route_longitude],
            )
            distances.append((str(candidate_id), candidate_id, float(np.nanmean(per_point)), route))
        if not distances:
            raise ValueError("every reference route has fewer than two valid coordinates")
        _, selected_id, selected_distance, selected_route = min(
            distances, key=lambda item: (item[2], item[0])
        )
        row: dict[str, Any] = {
            trajectory_id: identifier,
            output_route_id: selected_id,
            output_distance: selected_distance,
        }
        if route_name:
            row[output_route_name] = selected_route[route_name].iloc[0]
        if route_class:
            row[output_route_class] = selected_route[route_class].iloc[0]
        for column in carry_columns:
            values = group[column].dropna()
            row[column] = values.iloc[0] if not values.empty else None
        rows.append(row)
    return pd.DataFrame(rows)


def match_events_to_trajectory_points(
    events: pd.DataFrame,
    points: pd.DataFrame,
    *,
    event_latitude: str,
    event_longitude: str,
    point_latitude: str,
    point_longitude: str,
    trajectory_id: str,
    output_trajectory_id: str = "matched_trajectory_id",
    output_distance: str = "match_distance_km",
    carry_point_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Match each event spatially against all supplied trajectory points.

    Return the event rows with ``matched_trajectory_id``, ``match_distance_km``
    and any requested ``carry_point_columns``. Coordinates are degrees and the
    distance is kilometres. Carried columns overwrite same-named event columns.
    Restrict points to the appropriate entity/day before calling: this helper
    applies no entity, time-window or maximum-distance constraint. Equidistant
    matches use the first point in input order; invalid event coordinates produce
    null matched values.
    """

    _require_columns(events, [event_latitude, event_longitude])
    _require_columns(
        points,
        [trajectory_id, point_latitude, point_longitude, *carry_point_columns],
    )
    output = events.copy()
    point_work = points.reset_index(drop=True).copy()
    point_work[point_latitude] = pd.to_numeric(point_work[point_latitude], errors="coerce")
    point_work[point_longitude] = pd.to_numeric(point_work[point_longitude], errors="coerce")
    point_work = point_work.dropna(subset=[trajectory_id, point_latitude, point_longitude])
    if point_work.empty:
        raise ValueError("no valid trajectory points are available for event matching")
    event_lat = pd.to_numeric(output[event_latitude], errors="coerce")
    event_lon = pd.to_numeric(output[event_longitude], errors="coerce")
    matched_indices: list[int | None] = []
    matched_distances: list[float | None] = []
    for latitude, longitude in zip(event_lat, event_lon):
        if not np.isfinite(latitude) or not np.isfinite(longitude):
            matched_indices.append(None)
            matched_distances.append(None)
            continue
        distances = haversine_km(
            latitude, longitude, point_work[point_latitude], point_work[point_longitude]
        )
        position = int(np.nanargmin(distances))
        matched_indices.append(position)
        matched_distances.append(float(distances[position]))
    output[output_trajectory_id] = [
        point_work.iloc[index][trajectory_id] if index is not None else None
        for index in matched_indices
    ]
    output[output_distance] = matched_distances
    for column in carry_point_columns:
        output[column] = [
            point_work.iloc[index][column] if index is not None else None
            for index in matched_indices
        ]
    return output


def map_categories(
    frame: pd.DataFrame,
    *,
    source_field: str,
    output_field: str,
    mapping: Mapping[str, Any],
    default: Any = None,
) -> pd.DataFrame:
    """Create one deterministic categorical grouping without mutating the input."""

    _require_columns(frame, [source_field])
    output = frame.copy()
    normalized = {str(key): value for key, value in mapping.items()}
    output[output_field] = output[source_field].map(
        lambda value: normalized.get(str(value), default) if pd.notna(value) else default
    )
    return output


def add_normalized_progress(
    frame: pd.DataFrame,
    *,
    trajectory_id: str,
    order_by: str,
    bins: int = 10,
    progress_column: str = "normalized_progress",
    bin_column: str = "progress_bin",
) -> pd.DataFrame:
    """Return point rows ordered within trajectory with progress in [0, 1]."""

    _require_columns(frame, [trajectory_id, order_by])
    if bins < 2:
        raise ValueError("bins must be at least 2")
    output = frame.copy()
    if not pd.api.types.is_numeric_dtype(output[order_by]):
        parsed = parse_timestamps(output[order_by])
        output = output.assign(_trajstory_order=parsed)
        sort_column = "_trajstory_order"
    else:
        sort_column = order_by
    output = output.sort_values([trajectory_id, sort_column], kind="stable")
    position = output.groupby(trajectory_id, dropna=False).cumcount()
    count = output.groupby(trajectory_id, dropna=False)[trajectory_id].transform("size")
    denominator = (count - 1).where(count > 1, 1)
    output[progress_column] = position / denominator
    output[bin_column] = np.minimum(
        np.floor(output[progress_column] * bins).astype(int), bins - 1
    )
    return output.drop(columns=["_trajstory_order"], errors="ignore")


def trajectory_features(
    frame: pd.DataFrame,
    *,
    trajectory_id: str,
    timestamp: str,
    latitude: str,
    longitude: str,
    speed: str | None = None,
    carry_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Build a robust one-row-per-trajectory feature table.

    Distances are kilometres and duration is minutes. Without ``speed``,
    ``mean_speed`` is total path distance divided by elapsed hours; median/max
    speed remain null. With ``speed``, all three speed summaries retain the input
    column's units. Original columns are retained only via ``carry_columns``,
    which copy the first non-null value and overwrite same-named output fields.
    See TOOL_REFERENCE for the complete returned-column contract.
    """

    required = [trajectory_id, timestamp, latitude, longitude, *carry_columns]
    if speed:
        required.append(speed)
    _require_columns(frame, required)
    work = frame.loc[:, list(dict.fromkeys(required))].copy()
    work["_timestamp"] = parse_timestamps(work[timestamp])
    work["_latitude"] = pd.to_numeric(work[latitude], errors="coerce")
    work["_longitude"] = pd.to_numeric(work[longitude], errors="coerce")
    if speed:
        work["_speed"] = pd.to_numeric(work[speed], errors="coerce")
    work = work.sort_values([trajectory_id, "_timestamp"], kind="stable")

    rows: list[dict[str, Any]] = []
    for identifier, group in work.groupby(trajectory_id, dropna=False, sort=False):
        group = group.reset_index(drop=True)
        valid_geo = group[["_latitude", "_longitude"]].notna().all(axis=1)
        geo = group.loc[valid_geo]
        if len(geo) >= 2:
            segment_distance = haversine_km(
                geo["_latitude"].iloc[:-1], geo["_longitude"].iloc[:-1],
                geo["_latitude"].iloc[1:], geo["_longitude"].iloc[1:],
            )
            path_distance = float(np.nansum(segment_distance))
            displacement = float(haversine_km(
                [geo["_latitude"].iloc[0]], [geo["_longitude"].iloc[0]],
                [geo["_latitude"].iloc[-1]], [geo["_longitude"].iloc[-1]],
            )[0])
        else:
            segment_distance = np.asarray([], dtype=float)
            path_distance = 0.0
            displacement = 0.0

        valid_time = group["_timestamp"].dropna()
        duration_minutes = (
            float((valid_time.iloc[-1] - valid_time.iloc[0]).total_seconds() / 60.0)
            if len(valid_time) >= 2 else 0.0
        )
        if speed:
            speeds = group["_speed"].dropna()
            mean_speed = float(speeds.mean()) if not speeds.empty else None
            median_speed = float(speeds.median()) if not speeds.empty else None
            max_speed = float(speeds.max()) if not speeds.empty else None
        else:
            mean_speed = (path_distance / (duration_minutes / 60.0)) if duration_minutes > 0 else None
            median_speed = None
            max_speed = None

        first = group.iloc[0]
        last = group.iloc[-1]
        start_time = valid_time.iloc[0] if not valid_time.empty else pd.NaT
        end_time = valid_time.iloc[-1] if not valid_time.empty else pd.NaT
        row: dict[str, Any] = {
            trajectory_id: identifier,
            "point_count": int(len(group)),
            "start_time": start_time,
            "end_time": end_time,
            "start_date": str(start_time.date()) if pd.notna(start_time) else None,
            "start_hour": int(start_time.hour) if pd.notna(start_time) else None,
            "duration_minutes": duration_minutes,
            "distance_km": path_distance,
            "displacement_km": displacement,
            "mean_speed": mean_speed,
            "median_speed": median_speed,
            "max_speed": max_speed,
            "start_latitude": _native(first["_latitude"]),
            "start_longitude": _native(first["_longitude"]),
            "end_latitude": _native(last["_latitude"]),
            "end_longitude": _native(last["_longitude"]),
        }
        for column in carry_columns:
            non_null = group[column].dropna()
            row[column] = _native(non_null.iloc[0]) if not non_null.empty else None
        rows.append(row)
    return pd.DataFrame(rows)


def grouped_summary(
    frame: pd.DataFrame | None = None,
    *,
    df: pd.DataFrame | None = None,
    group_by: Sequence[str],
    metrics: Sequence[str],
    aggregations: Sequence[str] = ("count", "mean", "median", "std"),
) -> pd.DataFrame:
    """Compute a flat, JSON-friendly grouped summary table."""

    frame = _resolve_frame_argument(frame, df)
    _require_columns(frame, [*group_by, *metrics])
    if not group_by or not metrics:
        raise ValueError("group_by and metrics must be non-empty")
    result = frame.groupby(list(group_by), dropna=False)[list(metrics)].agg(list(aggregations))
    result.columns = [f"{metric}_{aggregation}" for metric, aggregation in result.columns]
    return result.reset_index()


def safe_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    on: str | Sequence[str],
    how: str = "inner",
    validate: str | None = None,
) -> pd.DataFrame:
    """Validate key presence, then call pandas merge with the given cardinality.

    Both tables must use the same key names; rename first for differently named
    keys. Overlapping non-key columns retain pandas' ``_x``/``_y`` suffixes:
    this helper does not resolve duplicates or preserve their original names.
    Project or rename those columns explicitly before merging.
    """

    keys = [on] if isinstance(on, str) else list(on)
    _require_columns(left, keys)
    _require_columns(right, keys)
    return left.merge(right, on=keys, how=how, validate=validate)


def correlation_table(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
    method: str = "pearson",
) -> pd.DataFrame:
    """Return a tidy pairwise correlation table with finite pair counts."""

    _require_columns(frame, columns)
    if method not in {"pearson", "spearman", "kendall"}:
        raise ValueError("method must be pearson, spearman, or kendall")
    numeric = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    correlations = numeric.corr(method=method)
    rows = []
    for left_index, left in enumerate(columns):
        for right in columns[left_index + 1:]:
            pair_count = int(numeric[[left, right]].dropna().shape[0])
            rows.append({
                "left": left,
                "right": right,
                "correlation": _native(correlations.loc[left, right]),
                "pair_count": pair_count,
                "method": method,
            })
    return pd.DataFrame(rows)


def plot_group_comparison(
    frame: pd.DataFrame | None = None,
    *,
    df: pd.DataFrame | None = None,
    group: str,
    value: str,
    kind: str = "box",
    title: str = "",
    ylabel: str | None = None,
) -> Any:
    """Create a version-stable box or mean-bar comparison figure."""

    frame = _resolve_frame_argument(frame, df)
    _require_columns(frame, [group, value])
    clean = frame[[group, value]].copy()
    clean[value] = pd.to_numeric(clean[value], errors="coerce")
    clean = clean.dropna(subset=[group, value])
    if clean.empty:
        raise ValueError("no finite observations are available for the group comparison")
    labels = [str(item) for item in clean[group].drop_duplicates().tolist()]
    figure, axis = plt.subplots(figsize=(max(6.0, 0.8 * len(labels)), 4.5))
    if kind == "box":
        values = [clean.loc[clean[group].astype(str) == label, value].to_numpy() for label in labels]
        axis.boxplot(values, tick_labels=labels, patch_artist=True)
    elif kind == "bar":
        means = clean.assign(_group=clean[group].astype(str)).groupby("_group", sort=False)[value].mean()
        means = means.reindex(labels)
        axis.bar(labels, means.to_numpy())
    else:
        raise ValueError("kind must be 'box' or 'bar'")
    axis.set_xlabel(group)
    axis.set_ylabel(ylabel or value)
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=35)
    figure.tight_layout()
    return PlotResult(figure, axis)


def plot_profile(
    frame: pd.DataFrame | None = None,
    *,
    df: pd.DataFrame | None = None,
    x: str,
    y: str,
    group: str | None = None,
    title: str = "",
) -> Any:
    """Plot mean y over x for each group after numeric and empty-data checks."""

    frame = _resolve_frame_argument(frame, df)
    if group is None:
        frame = frame.copy()
        group = "_trajstory_series"
        frame[group] = "all"
    _require_columns(frame, [x, y, group])
    clean = frame[[x, y, group]].copy()
    clean[x] = pd.to_numeric(clean[x], errors="coerce")
    clean[y] = pd.to_numeric(clean[y], errors="coerce")
    clean = clean.dropna(subset=[x, y, group])
    if clean.empty:
        raise ValueError("no finite observations are available for the profile plot")
    summarized = clean.groupby([group, x], dropna=False)[y].mean().reset_index()
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    for label, values in summarized.groupby(group, sort=False):
        values = values.sort_values(x)
        axis.plot(values[x], values[y], marker="o", linewidth=1.5, label=str(label))
    axis.set_xlabel(x)
    axis.set_ylabel(y)
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    return PlotResult(figure, axis)


def plot_heatmap(
    frame: pd.DataFrame | None = None,
    *,
    df: pd.DataFrame | None = None,
    row: str,
    column: str,
    value: str,
    aggregation: str = "mean",
    title: str = "",
) -> Any:
    """Create an annotated pivot heatmap using matplotlib only."""

    frame = _resolve_frame_argument(frame, df)
    numeric = frame.copy()
    if {row, column, value} <= set(numeric.columns):
        numeric[value] = pd.to_numeric(numeric[value], errors="coerce")
        matrix = numeric.pivot_table(index=row, columns=column, values=value, aggfunc=aggregation)
    else:
        # Generated code sometimes passes an already-pivoted matrix. Accept a
        # categorical first column as its index and plot the remaining numeric cells.
        if numeric.index.name is None and len(numeric.columns) > 1:
            first = numeric.columns[0]
            if not pd.api.types.is_numeric_dtype(numeric[first]):
                numeric = numeric.set_index(first)
        matrix = numeric.apply(pd.to_numeric, errors="coerce")
        matrix = matrix.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if matrix.empty:
        raise ValueError("heatmap pivot is empty")
    figure, axis = plt.subplots(
        figsize=(max(6.0, 0.8 * len(matrix.columns)), max(4.0, 0.5 * len(matrix.index)))
    )
    image = axis.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(matrix.columns)), labels=[str(item) for item in matrix.columns], rotation=35)
    axis.set_yticks(range(len(matrix.index)), labels=[str(item) for item in matrix.index])
    axis.set_xlabel(column)
    axis.set_ylabel(row)
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label=f"{aggregation} {value}")
    figure.tight_layout()
    return PlotResult(figure, axis)


def plot_trajectories(
    frame: pd.DataFrame | None = None,
    *,
    df: pd.DataFrame | None = None,
    trajectory_id: str,
    latitude: str,
    longitude: str,
    order_by: str | None = None,
    max_trajectories: int = 50,
    title: str = "",
) -> Any:
    """Plot a bounded set of trajectories with coordinate validation."""

    frame = _resolve_frame_argument(frame, df)
    columns = [trajectory_id, latitude, longitude] + ([order_by] if order_by else [])
    _require_columns(frame, columns)
    clean = frame.loc[:, columns].copy()
    clean[latitude] = pd.to_numeric(clean[latitude], errors="coerce")
    clean[longitude] = pd.to_numeric(clean[longitude], errors="coerce")
    clean = clean.dropna(subset=[trajectory_id, latitude, longitude])
    selected = clean[trajectory_id].drop_duplicates().head(max_trajectories)
    clean = clean[clean[trajectory_id].isin(selected)]
    if clean.empty:
        raise ValueError("no valid coordinates are available for the trajectory plot")
    figure, axis = plt.subplots(figsize=(7.0, 6.0))
    for _, values in clean.groupby(trajectory_id, sort=False):
        if order_by:
            values = values.sort_values(order_by, kind="stable")
        axis.plot(values[longitude], values[latitude], alpha=0.55, linewidth=1.0)
    axis.set_xlabel(longitude)
    axis.set_ylabel(latitude)
    axis.set_title(title)
    figure.tight_layout()
    return PlotResult(figure, axis)


def statistic(name: str, value: Any, description: str) -> dict[str, Any]:
    """Create one raw statistic entry without interpretation."""

    return {"name": str(name), "value": _native(value), "description": str(description)}


def table_from_frame(
    name: str,
    frame: pd.DataFrame,
    description: str,
    *,
    max_rows: int = 100,
) -> dict[str, Any]:
    """Convert a DataFrame to the bounded raw-table artifact schema."""

    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    bounded = frame.head(max_rows)
    return {
        "name": str(name),
        "columns": [str(column) for column in bounded.columns],
        "rows": [[_native(value) for value in row] for row in bounded.itertuples(index=False, name=None)],
        "description": str(description),
    }


class AnalysisSession:
    """Own approved inputs, raw-field provenance, plots, and result emission."""

    def __init__(
        self,
        input_files: Mapping[str, str | Path],
        output_dir: str | Path,
        *,
        require_visualization: bool = False,
        metadata_only_fallback: bool = False,
    ) -> None:
        self.input_files = {str(key): Path(value).expanduser().resolve() for key, value in input_files.items()}
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._loaded_fields: dict[str, set[str]] = {}
        self._scope_ids: list[str] = []
        self._scope_id_set: set[str] = set()
        self._measurements: dict[str, dict[str, Any]] = {}
        self._saved_visualizations: list[dict[str, Any]] = []
        self._visualization_errors: list[str] = []
        self._require_visualization = require_visualization
        self._metadata_only_fallback = metadata_only_fallback

    def measure(
        self, file_id: str, *, name: str, operation: str,
        column: str | None = None, filters: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        """Compute a bounded raw-column reduction and capture its actual context.

        This deliberately does not accept model-written semantic IDs, units, scope
        summaries or precomputed values. Custom/derived computations remain supported
        through statistic(), but are not eligible for automatic conflict admission.
        """
        if operation not in {"row_count", "mean", "min", "max", "nunique"}:
            raise ValueError("unsupported measurement operation")
        if operation == "row_count" and column is not None:
            raise ValueError("row_count does not take a column")
        if operation != "row_count" and not isinstance(column, str):
            raise ValueError("column is required for this measurement")
        frame = self.load_table(file_id)
        normalized_filters = []
        for spec in filters:
            if set(spec) != {"column", "op", "value"}:
                raise ValueError("filter requires exactly column, op and value")
            key, op, value = spec["column"], spec["op"], spec["value"]
            _require_columns(frame, [key])
            if op not in {"eq", "ne", "gt", "ge", "lt", "le", "in"}:
                raise ValueError("unsupported measurement filter")
            series = frame[key]
            if op == "in":
                if not isinstance(value, (list, tuple)):
                    raise ValueError("in filter requires a list")
                mask = series.isin(value)
            else:
                mask = getattr(series, op)(value)
            frame = frame.loc[mask.fillna(False)]
            normalized_filters.append(_native(dict(spec)))
        if operation == "row_count":
            value = int(len(frame))
        else:
            _require_columns(frame, [column])
            series = frame[column]
            if operation != "nunique" and not pd.api.types.is_numeric_dtype(series):
                raise ValueError("numeric reduction requires a numeric raw column")
            value = _native(getattr(series, operation)())
        # Exact row selection, including source row indices, without exporting raw rows.
        selected_hash = hashlib.sha256(
            pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()
        with self.input_files[file_id].open("rb") as source:
            source_hash = hashlib.file_digest(source, "sha256").hexdigest()
        context = {
            "version": "raw_reduction.v1", "file_id": file_id, "source_sha256": source_hash,
            "selection_sha256": selected_hash, "row_count": len(frame),
            "filters": sorted(normalized_filters, key=lambda x: json.dumps(x, sort_keys=True)),
            "grain": "source_row", "column": column, "operation": operation,
            "unit": "count" if operation in {"row_count", "nunique"} else "native_column_unit",
            "missing_policy": "pandas_skip_null" if operation != "row_count" else "include_rows",
        }
        mid = f"M{len(self._measurements) + 1}"
        result = statistic(name, value, f"{operation} of raw {file_id}.{column or '*'}")
        result["measurement_id"] = mid
        self._measurements[mid] = {"result": copy.deepcopy(result), "context": context}
        return result

    @classmethod
    def from_environment(cls) -> "AnalysisSession":
        try:
            inputs = json.loads(os.environ["TRAJSTORY_INPUT_FILES"])
            output_dir = os.environ["TRAJSTORY_OUTPUT_DIR"]
        except KeyError as exc:
            raise RuntimeError(f"missing executor environment variable: {exc.args[0]}") from exc
        if not isinstance(inputs, dict) or not inputs:
            raise RuntimeError("TRAJSTORY_INPUT_FILES must contain a non-empty JSON object")
        metadata_only = os.environ.get("TRAJSTORY_METADATA_ONLY_FALLBACK") == "1"
        return cls(
            inputs, output_dir, require_visualization=not metadata_only,
            metadata_only_fallback=metadata_only,
        )

    def load_table(
        self,
        file_id: str,
        *,
        columns: Sequence[str] | None = None,
        track_ids: bool = True,
    ) -> pd.DataFrame:
        """Load an approved CSV/Parquet table and record raw-field lineage."""

        if file_id not in self.input_files:
            raise KeyError(f"unknown input file_id {file_id!r}; available: {sorted(self.input_files)}")
        path = self.input_files[file_id]
        requested = list(columns) if columns is not None else None
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path, usecols=requested)
        elif path.suffix.lower() in {".parquet", ".pq"}:
            frame = pd.read_parquet(path, columns=requested)
        else:
            raise ValueError(f"unsupported input format: {path.suffix}")
        fields = {str(column) for column in frame.columns}
        self._loaded_fields.setdefault(file_id, set()).update(fields)
        if track_ids:
            inferred = [column for column in frame.columns if _ID_FIELD_RE.match(str(column))]
            self.track_ids(frame, inferred)
        return frame

    def record_fields(self, file_id: str, fields: Sequence[str]) -> None:
        """Record raw fields used by a custom loader after validating the file ID."""

        if file_id not in self.input_files:
            raise KeyError(f"unknown input file_id {file_id!r}")
        self._loaded_fields.setdefault(file_id, set()).update(str(field) for field in fields)

    def track_ids(self, frame: pd.DataFrame, columns: Sequence[str], *, limit: int = 500) -> None:
        """Add bounded trajectory/user/entity identifiers to the evidence scope."""

        present = [column for column in columns if column in frame.columns]
        for column in present:
            for value in frame[column].dropna().astype(str).unique():
                if value not in self._scope_id_set:
                    self._scope_id_set.add(value)
                    self._scope_ids.append(value)
                if len(self._scope_ids) >= limit:
                    return

    def save_figure(
        self,
        figure: Any,
        filename: str,
        description: str,
        *,
        chart_type: str,
        x_axis: Mapping[str, Any],
        y_axis: Mapping[str, Any],
        statistic_names: Sequence[str] = (),
        dpi: int = 140,
    ) -> dict[str, Any] | None:
        """Save a figure together with the bounded data observation it encodes."""
        raw_figure = figure.figure if isinstance(figure, PlotResult) else figure
        try:
            safe_name = _SAFE_FILENAME_RE.sub("_", Path(filename).name)
            if Path(safe_name).suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                safe_name += ".png"
            normalized_type = str(chart_type).strip()
            if not normalized_type:
                raise ValueError("chart_type must be non-empty")
            normalized_names = [str(value).strip() for value in statistic_names]
            if any(not value for value in normalized_names):
                raise ValueError("statistic_names must contain non-empty names")
            if len(normalized_names) != len(set(normalized_names)):
                raise ValueError("statistic_names must not contain duplicates")
            destination = self.output_dir / safe_name
            if any(item["path"] == safe_name for item in self._saved_visualizations):
                raise ValueError("saved visualizations must use distinct filenames")
            raw_figure.savefig(destination, dpi=dpi, bbox_inches="tight")
            visualization = {
                "path": safe_name,
                "description": str(description),
                "observation": {
                    "chart_type": normalized_type,
                    "x_axis": _normalize_chart_axis(x_axis, "x_axis"),
                    "y_axis": _normalize_chart_axis(y_axis, "y_axis"),
                    "statistic_names": normalized_names,
                },
            }
        except Exception as exc:
            if not self._metadata_only_fallback:
                raise
            self._visualization_errors.append(f"{type(exc).__name__}: {exc}")
            return None
        finally:
            plt.close(raw_figure)
        self._saved_visualizations.append(copy.deepcopy(visualization))
        return visualization

    def emit_result(
        self,
        *,
        statistics: Sequence[Mapping[str, Any]],
        tables: Sequence[Mapping[str, Any]] = (),
        visualizations: Sequence[Mapping[str, Any]] = (),
        execution_notes: str = "",
    ) -> Path:
        """Write the raw result with provenance derived from actual table loads."""

        if self._metadata_only_fallback:
            visualizations = [item for item in visualizations if item is not None]
        if not self._loaded_fields:
            raise ValueError("no approved input table was loaded through AnalysisSession")
        if len(tables) > 1:
            raise ValueError("an atomic A2 branch may emit at most one result table")
        if len(self._saved_visualizations) > 2:
            raise ValueError("an atomic A2 branch may save at most two visualizations")
        if len(visualizations) > 2:
            raise ValueError("an atomic A2 branch may reference at most two visualizations")
        if self._saved_visualizations:
            saved_paths = {
                (self.output_dir / item["path"]).resolve()
                for item in self._saved_visualizations
            }
            supplied_paths = []
            for item in visualizations:
                supplied_path = Path(str(item.get("path", "")))
                supplied_path = (
                    supplied_path.resolve() if supplied_path.is_absolute()
                    else (self.output_dir / supplied_path).resolve()
                )
                if supplied_path not in saved_paths or supplied_path in supplied_paths:
                    raise ValueError("emitted visualization does not match the saved figure")
                supplied_paths.append(supplied_path)
            # save_figure already captured the plotted data. Do not depend on
            # generated code remembering to forward its return value.
            visualizations = list(self._saved_visualizations)
        if self._require_visualization and not 1 <= len(visualizations) <= 2:
            raise ValueError("every analysis branch must save one or two structured visualizations")
        if self._metadata_only_fallback and not visualizations:
            has_statistic = any(
                isinstance(item, Mapping) and item.get("value") is not None
                for item in statistics
            )
            has_table = any(
                isinstance(item, Mapping) and bool(item.get("rows"))
                for item in tables
            )
            if not (has_statistic or has_table):
                raise ValueError(
                    "metadata-only fallback requires a computed value or nonempty result table"
                )
        normalized_visualizations = []
        available_statistic_names = {
            str(item.get("name", "")).strip() for item in statistics
            if isinstance(item, Mapping) and str(item.get("name", "")).strip()
        }
        for item in visualizations:
            path = Path(str(item.get("path", "")))
            if not path.name:
                raise ValueError("visualization path must not be empty")
            candidate = path.resolve() if path.is_absolute() else (self.output_dir / path).resolve()
            try:
                relative_path = candidate.relative_to(self.output_dir)
            except ValueError as exc:
                raise ValueError("visualization must be inside the branch output directory") from exc
            if not candidate.is_file():
                raise ValueError(f"visualization does not exist: {candidate}")
            observation = item.get("observation")
            if not isinstance(observation, Mapping):
                raise ValueError("every visualization requires a structured observation")
            chart_type = str(observation.get("chart_type", "")).strip()
            if not chart_type:
                raise ValueError("visualization observation requires chart_type")
            statistic_names = [str(value).strip() for value in observation.get("statistic_names", [])]
            unknown_names = set(statistic_names) - available_statistic_names
            if unknown_names:
                raise ValueError(
                    f"visualization references unknown statistic names: {sorted(unknown_names)}"
                )
            normalized_visualizations.append({
                "path": str(relative_path),
                "description": str(item.get("description", "")),
                "observation": {
                    "chart_type": chart_type,
                    "x_axis": _normalize_chart_axis(observation.get("x_axis", {}), "x_axis"),
                    "y_axis": _normalize_chart_axis(observation.get("y_axis", {}), "y_axis"),
                    "statistic_names": statistic_names,
                },
            })
        comparison_records = []
        seen_measurements = set()
        for index, item in enumerate(statistics):
            mid = item.get("measurement_id")
            if mid is None:
                continue
            if (mid not in self._measurements or mid in seen_measurements
                    or dict(item) != self._measurements[mid]["result"]):
                raise ValueError("measurement must be emitted once and unchanged")
            seen_measurements.add(mid)
            comparison_records.append({
                "origin": "session.measure.v1", "statistic_index": index,
                "value": item["value"], "context": self._measurements[mid]["context"],
            })
        payload = {
            "producer": TOOL_LAYER_VERSION,
            "statistics": [_native(dict(item)) for item in statistics],
            "comparison_records": comparison_records,
            "tables": [_native(dict(item)) for item in tables],
            "visualizations": normalized_visualizations,
            "supporting_scope": {
                "files": sorted(self._loaded_fields),
                "fields": sorted({field for fields in self._loaded_fields.values() for field in fields}),
                "trajectory_or_user_ids": self._scope_ids,
            },
            "execution_notes": str(execution_notes),
        }
        if self._metadata_only_fallback and not normalized_visualizations:
            payload["analysis_mode"] = "metadata_only"
            if self._visualization_errors:
                payload["visualization_errors"] = self._visualization_errors
        destination = self.output_dir / "raw_analysis_result.json"
        destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return destination


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"missing required columns: {missing}; available: {list(frame.columns)}")


def _resolve_frame_argument(
    frame: pd.DataFrame | None, df: pd.DataFrame | None
) -> pd.DataFrame:
    if frame is not None and df is not None:
        raise ValueError("pass either frame or df, not both")
    resolved = frame if frame is not None else df
    if resolved is None:
        raise ValueError("a DataFrame must be supplied as frame or df")
    return resolved


_TOOL_IMPORT_REFERENCE = """
STABLE ANALYSIS TOOL CONTRACT
Use the exact signatures and returned column names below; do not guess aliases.
In signatures, parameters after * MUST be passed by keyword. Column parameters
are strings naming existing columns, not Series. Import every helper you call:

from trajstoryagent_v5.v5_framework.analysis_tools import (
    add_normalized_progress, assign_trajectories_to_routes,
    correlation_table, grouped_summary, haversine_km, map_categories,
    match_events_to_trajectory_points, parse_timestamps,
    point_to_polyline_haversine_km, plot_group_comparison, plot_heatmap,
    plot_profile, plot_trajectories, safe_merge, statistic, table_from_frame,
    trajectory_features,
)
""".strip()

_SESSION_REFERENCE = """
SESSION AND OUTPUT
from trajstoryagent_v5.v5_framework.analysis_tools import AnalysisSession

Required pattern:
1. session = AnalysisSession.from_environment()
2. df = session.load_table("file_id", columns=[...])
3. Compute with helpers; pandas/numpy/scipy are available for custom analyses.
4. Save one or two distinct figures with their actual plotted values and emit the raw result.
Target 60--140 lines, at most one table (<=100 rows) and two visualizations.
Do not write claims or manually construct supporting_scope/comparison_records.

- AnalysisSession.from_environment() -> session configured by the executor.
- session.load_table(file_id, *, columns=None, track_ids=True) -> full CSV/Parquet
  DataFrame, optionally restricted to listed raw columns. file_id must be in the
  input catalog. Derived/helper output columns cannot be requested here.
- session.measure(file_id, *, name, operation, column=None, filters=()) -> statistic
  dict. Operations: row_count, mean, min, max, nunique. row_count counts raw rows
  (including nulls) and forbids column; others require a raw column and skip nulls.
  mean/min/max require numeric dtype. Filters are ANDed objects with exactly
  {"column": "raw_column", "op": "eq|ne|gt|ge|lt|le|in", "value": ...}; in takes a list.
  This reloads the source, not a transformed DataFrame. Emit the returned dict once,
  unchanged: it records source-row context for conflict admission. Use statistic()
  for custom/derived metrics; no comparison metadata is required for those.
""".strip()

_ARTIFACT_HELPER_REFERENCE = """
IN-MEMORY ARTIFACT HELPERS (these do not write files)
- statistic(name, value, description) -> {name, value, description}; converts numpy
  and pandas values to JSON-compatible values. Non-finite numbers become null.
- table_from_frame(name, frame, description, *, max_rows=100) -> table dict with
  name, columns, rows, description. Keeps the first max_rows; it does not aggregate.
""".strip()

_SESSION_OUTPUT_REFERENCE = """
NATIVE SESSION ARTIFACT WRITING
- session.save_figure(figure, filename, description, *, chart_type, x_axis, y_axis,
  statistic_names=(), dpi=140) -> visualization dict with path, description,
  observation. Saves inside the branch output directory, closes the figure, and
  automatically registers its structured observation for emit_result().
  x_axis/y_axis: {"description": "...", "value": [...]} or value={"series": [...]}.
  Every sequence must be nonempty and <=50 items; use actual plotted values with
  aligned x/y ordering (evenly spaced samples if needed). Multiple series are allowed.
  statistic_names must exactly match names emitted in statistics, not table columns
  or labels. Emit decisive plotted values in statistics or the bounded table too.
- session.emit_result(*, statistics, tables=(), visualizations=(), execution_notes="")
  -> Path to raw_analysis_result.json. Pass statistics and optional tables;
  saved figures are included automatically even if visualizations is omitted or [].
  Provenance is collected from actual loads. At most one table is accepted.
  The execution validator requires one or two saved visualizations, even for
  null/negative findings. Use distinct filenames and structured observations.
  Only the orchestrator's final metadata-only fallback after repeated chart
  failures may omit the figure; it must emit computed statistics or a result table.
""".strip()

_ANALYSIS_HELPER_REFERENCE = """
TRAJECTORY FEATURES AND GEOMETRY
- parse_timestamps(values, *, utc=True) -> datetime Series. Handles mixed datetime
  strings; month-day-only dates use year 2000; invalid values become NaT. For numeric
  Unix epochs, first convert with pd.to_datetime(values, unit=the_known_unit, utc=True).
- haversine_km(lat1, lon1, lat2, lon2) -> elementwise distance(s) in km; inputs are
  numeric scalars/arrays in degrees, using numpy broadcasting, not a pairwise matrix.
- point_to_polyline_haversine_km(point_latitude, point_longitude, route_latitude,
  route_longitude) -> 1D distance array in km, one per point. Inputs are coordinate
  arrays, not field names; route arrays require >=2 finite paired points in path order.
- trajectory_features(frame, *, trajectory_id, timestamp, latitude, longitude,
  speed=None, carry_columns=()) -> one row per trajectory, ordered internally by time.
  Exact output columns: the supplied trajectory_id, point_count, start_time, end_time,
  start_date, start_hour, duration_minutes, distance_km, displacement_km, mean_speed,
  median_speed, max_speed, start_latitude, start_longitude, end_latitude, end_longitude.
  Distances use coordinates in degrees and return km; duration is minutes.
  With speed: speed summaries retain the input speed units (no automatic conversion).
  Without speed: mean_speed is distance_km / elapsed hours (null for zero duration);
  median/max speed are null.
  carry_columns adds each field's first non-null value, NOT an aggregate. Avoid names
  colliding with computed outputs: carried values overwrite them. Other input columns
  are not retained. Do not assume outputs such as avg_speed_kmh, points, or n_trips.
- add_normalized_progress(frame, *, trajectory_id, order_by, bins=10,
  progress_column="normalized_progress", bin_column="progress_bin") -> original
  point rows plus those two columns, sorted per trajectory. Progress is point rank
  divided by (point count - 1), NOT elapsed time/distance; singleton progress is 0.
  bins>=2; progress_bin ranges 0..bins-1. This does NOT aggregate points into bins.

ROUTE ASSIGNMENT AND EVENT MATCHING
- assign_trajectories_to_routes(points, routes, *, trajectory_id, point_latitude,
  point_longitude, route_id, route_latitude, route_longitude, point_order=None,
  route_order=None, route_name=None, route_class=None, carry_columns=(),
  output_route_id="assigned_route_id", output_route_name="assigned_route_name",
  output_route_class="assigned_route_class", output_distance="assignment_distance_km")
  -> one row per trajectory with valid coordinates. All SIX ID/coordinate keyword
  arguments are required. There are NO latitude/longitude/order_by aliases here.
  Coordinates are degrees. Choose the route minimizing the mean point-to-polyline
  distance over trajectory points; routes need >=2 valid points. There is no distance
  rejection threshold; equal-distance ties use lexicographic string route IDs.
  Use point_order/route_order for ordering columns, or supply already ordered rows.
  Returns: trajectory_id under its supplied name, assigned_route_id,
  assignment_distance_km; adds assigned_route_name/class only when route_name/class
  is supplied (copies the selected route's first row). output_* options rename these.
  carry_columns copies first non-null point values; other input fields are not kept.
  In particular, default output does NOT include route_id, route_class or vessel_id.
  Example (replace column names with those in the supplied schema):
  assigned = assign_trajectories_to_routes(
      points, routes, trajectory_id="trajectory_id", point_latitude="lat",
      point_longitude="lon", route_id="route_id", route_latitude="lat",
      route_longitude="lon", point_order="point_index", route_order="point_index",
      route_class="route_class", carry_columns=["vessel_id"])
  # Use assigned["assigned_route_class"] directly; no route metadata merge is needed.
  # To join by route ID instead, rename assigned_route_id to route_id first,
  # or request output_route_id="route_id" in the call above.
- match_events_to_trajectory_points(events, points, *, event_latitude, event_longitude,
  point_latitude, point_longitude, trajectory_id,
  output_trajectory_id="matched_trajectory_id", output_distance="match_distance_km",
  carry_point_columns=()) -> event rows plus matched_trajectory_id, match_distance_km,
  and carried point columns. Coordinates are degrees. Matches nearest point across
  ALL supplied points; no entity/day/time-window/distance restriction is automatic.
  Filter compatible point populations before calling. Ties use first input point;
  invalid event coordinates yield null matches. At least one valid point is required.
  carry_point_columns overwrites same-named event columns; rename first if needed.

GROUPING AND JOINS
- map_categories(frame, *, source_field, output_field, mapping, default=None)
  -> copy of frame plus output_field. Mapping keys and non-null source values are
  compared as strings; null/unmapped values use default. Existing output_field is replaced.
- grouped_summary(frame=None, *, df=None, group_by, metrics,
  aggregations=("count", "mean", "median", "std")) -> grouping columns plus flat
  <metric>_<aggregation> columns, e.g. speed_kmh_mean. Pass nonempty lists for
  group_by/metrics. count is non-null count per metric, NOT distinct trips/entities;
  choose nunique on an ID for that. For raw row counts use groupby(...).size().
- correlation_table(frame, *, columns, method="pearson") -> columns
  [left, right, correlation, pair_count, method], one row per unordered pair.
  method supports pearson/spearman/kendall. Nonnumeric values become null;
  pair_count counts rows non-null in both columns; unavailable correlations are null/NaN.
- safe_merge(left, right, *, on, how="inner", validate=None) -> pandas merged frame.
  on is a column name or list present on BOTH sides. validate supports one_to_one,
  one_to_many, many_to_one, many_to_many; choose the actual intended cardinality.
  No left_on/right_on/suffixes arguments: explicitly rename/project columns first.
  This checks keys/cardinality, NOT semantic validity or overlapping non-key columns.
  Overlaps become <column>_x and <column>_y, so the bare column no longer exists.
  When adding trajectory labels to points, merge only keys and NEW columns:
  labeled = safe_merge(points, labels[["trajectory_id", "anomaly_status"]],
                       on="trajectory_id", how="left", validate="many_to_one")
  # Do not re-merge start_region/end_region when already present in points.

PLOTTING
All four plotting helpers return PlotResult: use fig = helper(...) or fig, ax =
helper(...); save fig through the framework's figure writer or fig.savefig(...).
They accept frame positionally or
df=, never both. They return figures, NOT the aggregated DataFrame.
- plot_group_comparison(frame=None, *, df=None, group, value, kind="box", title="",
  ylabel=None): kind is box or bar; bar plots MEAN value per group, not sums/counts.
- plot_profile(frame=None, *, df=None, x, y, group=None, title=""):
  plots mean y per (group, x), sorted by numeric x. For time axes derive a numeric
  time index first; group=None makes one series. Non-numeric x/y values are dropped.
- plot_heatmap(frame=None, *, df=None, row, column, value, aggregation="mean", title=""):
  pivots long-form data with the specified aggregation; also accepts a pre-pivoted
  numeric matrix. row/column/value are still required as labels for the matrix case.
- plot_trajectories(frame=None, *, df=None, trajectory_id, latitude, longitude,
  order_by=None, max_trajectories=50, title=""):
  plots longitude (x) versus latitude (y) for the first max_trajectories unique IDs;
  this display limit does not change the population used for analysis.

REPAIR CHECKLIST
- NameError for a helper: add its explicit import; preserve unrelated working logic.
- Unexpected/missing keyword: use its full signature above, including required kwargs.
- KeyError after a merge: inspect overlapping columns/_x/_y; do not assume missing
  raw data. Before pd.to_numeric, select one Series, not a duplicate-named DataFrame.
- Series.fillna does not accept a numpy ndarray. Use an index-aligned Series or
  series.where(series.notna(), replacement_array) for row-wise replacement.
- Check actual helper output names and units before accessing or joining them.
- For derived features, import and call the helper first; never load derived columns
  as if they were raw fields. Use the smallest repair that addresses the traceback.
""".strip()

# Both frameworks receive these exact computational contracts and import the
# same implementations. Artifact writers remain specific to each framework.
SHARED_TOOL_REFERENCE = "\n\n".join((
    _TOOL_IMPORT_REFERENCE, _ARTIFACT_HELPER_REFERENCE, _ANALYSIS_HELPER_REFERENCE,
))
TOOL_REFERENCE = "\n\n".join((
    _TOOL_IMPORT_REFERENCE, _SESSION_REFERENCE, _ARTIFACT_HELPER_REFERENCE,
    _SESSION_OUTPUT_REFERENCE, _ANALYSIS_HELPER_REFERENCE,
))


def analysis_tool_metadata() -> dict[str, Any]:
    """Identify the exact shared implementation and documentation for a run."""
    return {
        "enabled": True,
        "module": "trajstoryagent_v5.v5_framework.analysis_tools",
        "version": TOOL_LAYER_VERSION,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(SHARED_TOOL_REFERENCE.encode("utf-8")).hexdigest(),
    }
