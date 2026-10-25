"""Experimental, GT-free conditional modelling and exception-aware compression.

The numerical kernels are deterministic. Metric/condition binding is proposed by
the backbone and is NOT certified by these kernels. No population inference is
made from correlated trajectory observations.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


class EvidenceCollector:
    """Export sufficient statistics from complete, explicitly keyed observations."""

    def __init__(self):
        self.measures = []

    def add_measure(self, frame, *, value, conditions, observation_id, unit,
                    observation_unit, description, scope, kind="continuous"):
        if len(self.measures) >= 3:
            raise ValueError("at most three question-relevant measures")
        if not 1 <= len(conditions) <= 2 or len(set(conditions)) != len(conditions):
            raise ValueError("one or two distinct categorical condition columns required")
        if not observation_id or frame.duplicated(observation_id).any():
            raise ValueError("observation_id must identify exactly one row per observation")
        if not all(isinstance(x, str) and x.strip() for x in
                   (unit, observation_unit, description, scope)):
            raise ValueError("unit, observation_unit, description and scope required")
        if kind not in ("continuous", "indicator"):
            raise ValueError("kind must be continuous or indicator")
        y = pd.to_numeric(frame[value], errors="coerce")
        valid = np.isfinite(y) & frame[conditions].notna().all(axis=1)
        data = frame.loc[valid, conditions].copy()
        data["__value"] = y[valid].astype(float)
        if len(data) < 2:
            raise ValueError("at least two finite observations required")
        if kind == "indicator" and not data["__value"].isin([0., 1.]).all():
            raise ValueError("indicator observations must be exactly zero or one")
        for col in conditions:
            data[col] = data[col].astype(str)
            if data[col].nunique() > 12:
                raise ValueError("condition cardinality exceeds 12; use meaningful complete bins, never top-k filtering")
        cells = []
        for key, group in data.groupby(conditions, sort=True, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            a = group["__value"].to_numpy()
            mean = float(a.mean())
            cells.append({"condition": dict(zip(conditions, key)), "n": len(a),
                          "mean": mean, "sse": float(np.square(a - mean).sum())})
        self.measures.append({"description": description, "unit": unit,
            "observation_unit": observation_unit, "scope": scope, "kind": kind,
            "conditions": conditions, "source_rows": len(frame),
            "excluded_missing_rows": int((~valid).sum()), "cells": cells})

    def finish(self, reason=""):
        path = Path(os.environ["TRAJSTORY_OUTPUT_DIR"]) / "observations.json"
        path.write_text(json.dumps({"version": 1, "measures": self.measures,
            "unsupported_reason": reason}, ensure_ascii=False, indent=2), encoding="utf-8")


def aggregate(cells):
    n = sum(c["n"] for c in cells)
    if n <= 0:
        raise ValueError("empty support")
    mean = sum(c["n"] * c["mean"] for c in cells) / n
    sse = sum(c["sse"] + c["n"] * (c["mean"] - mean)**2 for c in cells)
    return {"n": n, "mean": mean, "sse": sse,
            "sd": math.sqrt(max(0., sse / n))}


def conditional_partition(measure, *, penalty=1., min_support=1):
    """Exact recursion over complete categorical splits (<=2 dimensions).

    Objective = normalized within-SSE + log2(n+1) per leaf + split code cost.
    This is descriptive regularization, not a likelihood/BIC significance test.
    Exactness is only over this bounded tree grammar, not all partitions.
    """
    cells = measure["cells"]
    total = aggregate(cells)
    scale = max(total["sse"] / total["n"], 1e-12)
    leaf_cost = penalty * math.log2(total["n"] + 1)
    evaluations = 0

    def visit(subset, remaining, condition):
        nonlocal evaluations
        evaluations += 1
        stats = aggregate(subset)
        leaf = dict(condition=condition, **stats)
        best = {"objective": stats["sse"] / scale + leaf_cost,
                "leaves": [leaf], "split": None}
        for dim in remaining:
            values = sorted({c["condition"][dim] for c in subset})
            groups = [[c for c in subset if c["condition"][dim] == v] for v in values]
            if len(values) < 2 or any(sum(c["n"] for c in g) < min_support for g in groups):
                continue
            children = [visit(g, [d for d in remaining if d != dim], {**condition, dim: v})
                        for v, g in zip(values, groups)]
            score = sum(c["objective"] for c in children) + penalty * math.log2(1 + len(remaining))
            if score < best["objective"] - 1e-10:
                best = {"objective": score, "split": dim,
                        "leaves": [l for c in children for l in c["leaves"]]}
        return best

    best = visit(cells, measure["conditions"], {})
    within = sum(l["sse"] for l in best["leaves"])
    between = sum(l["n"] * (l["mean"] - total["mean"])**2 for l in best["leaves"])
    if not math.isclose(total["sse"], within + between, abs_tol=1e-7, rel_tol=1e-9):
        raise ArithmeticError("variance decomposition failed")
    return {**{k: v for k, v in measure.items() if k != "cells"}, **best,
            "total": total, "within_sse": within, "between_sse": between,
            "explained_fraction": between / total["sse"] if total["sse"] > 0 else 0.,
            "scale": scale, "states_evaluated": evaluations,
            "penalty": penalty, "min_support": min_support}




def condition_text(condition):
    return ", ".join(f"{k}={v}" for k, v in condition.items()) or "all observations"


def render_facts(partition, compression=None):
    """Deterministic verbalization, including support and within-group spread."""
    desc, unit = partition["description"], partition["unit"]
    total = partition["total"]
    prefix = (f"{desc} ({unit}; {partition['observation_unit']}; {partition['scope']}): "
              f"overall mean {total['mean']:.4g}, SD {total['sd']:.4g}, n={total['n']}.")
    leaves = partition["leaves"]
    if len(leaves) == 1:
        return [prefix, "The penalized conditional model retained a pooled description; this does not establish absence of group differences."]
    facts = [prefix]
    if compression is None:
        for leaf in leaves:
            facts.append(f"{condition_text(leaf['condition'])}: mean {leaf['mean']:.4g}, "
                         f"within-group SD {leaf['sd']:.4g}, n={leaf['n']}.")
    else:
        defaults = compression["default_indices"]
        if defaults:
            scopes = "; ".join(condition_text(leaves[i]["condition"]) for i in defaults)
            actual = [leaves[i]["mean"] for i in defaults]
            facts.append(f"For {scopes}, means are represented by {compression['default_mean']:.4g}; "
                         f"actual means range {min(actual):.4g}–{max(actual):.4g}.")
            facts.append("Within-group SD and n respectively: " + "; ".join(
                f"{condition_text(leaves[i]['condition'])}: {leaves[i]['sd']:.4g}, {leaves[i]['n']}"
                for i in defaults) + ".")
        for i in compression["exception_indices"]:
            leaf = leaves[i]
            facts.append(f"Explicit exception {condition_text(leaf['condition'])}: mean {leaf['mean']:.4g}, "
                         f"within-group SD {leaf['sd']:.4g}, n={leaf['n']}.")
    low, high = min(leaves, key=lambda l: l['mean']), max(leaves, key=lambda l: l['mean'])
    facts.append(f"Highest mean: {condition_text(high['condition'])} ({high['mean']:.4g}); "
                 f"lowest: {condition_text(low['condition'])} ({low['mean']:.4g}); "
                 f"difference {high['mean'] - low['mean']:.4g} {unit}.")
    return facts
