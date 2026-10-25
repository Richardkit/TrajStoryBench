"""Conservative admission of numerical conflicts using executed result references.

Free-text contradiction, aliases, inferred units and missing legacy metadata are
not decidable here. They remain unconfirmed instead of becoming attack edges.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from trajstoryagent_v5.v5_framework.analysis_tools import TOOL_LAYER_VERSION


CONTEXT_KEYS = {
    "version", "file_id", "source_sha256", "selection_sha256", "row_count",
    "filters", "grain", "column", "operation", "unit", "missing_policy",
}


def _resolve(argument, evidence_by_id, artifacts):
    ref = argument.get("result_ref")
    if not isinstance(ref, dict):
        return None, "missing_result_reference"
    eid, index = ref.get("evidence_id"), ref.get("statistic_index")
    if not isinstance(eid, str) or eid not in evidence_by_id or eid not in argument["evidence_ids"]:
        return None, "unknown_result_evidence"
    if type(index) is not int or index < 0:
        return None, "invalid_statistic_index"
    if argument["requires"]:
        return None, "compound_argument_not_supported"
    e = evidence_by_id[eid]
    if eid not in artifacts:
        try:
            artifacts[eid] = json.loads(Path(e.raw_result_path).read_text())
        except (OSError, TypeError, ValueError):
            artifacts[eid] = {}
    raw = artifacts[eid]
    if (not isinstance(raw, dict)
            or raw.get("producer") not in {"trajstory.analysis_tools.v1", TOOL_LAYER_VERSION}):
        return None, "missing_execution_artifact"
    try:
        stat = raw["statistics"][index]
        if not isinstance(stat, dict) or stat != e.statistics[index]:
            return None, "evidence_artifact_mismatch"
    except (KeyError, IndexError, TypeError):
        return None, "invalid_statistic_reference"
    ledger = raw.get("comparison_records", [])
    if not isinstance(ledger, list):
        return None, "malformed_execution_context"
    records = [r for r in ledger if isinstance(r, dict)
               and r.get("statistic_index") == index]
    if len(records) != 1:
        return None, "missing_or_ambiguous_execution_context"
    r = records[0]
    c = r.get("context")
    if (r.get("origin") != "session.measure.v1" or not isinstance(c, dict)
            or set(c) != CONTEXT_KEYS or c["version"] != "raw_reduction.v1"
            or c["grain"] != "source_row"
            or c["operation"] not in {"row_count", "mean", "min", "max", "nunique"}):
        return None, "unsupported_execution_context"
    if (not all(isinstance(c[k], str) and len(c[k]) == 64 for k in ["source_sha256", "selection_sha256"])
            or type(c["row_count"]) is not int or c["row_count"] < 0
            or not isinstance(c["filters"], list)):
        return None, "incomplete_execution_context"
    value = stat.get("value")
    if type(value) not in {int, float} or not math.isfinite(value) or value != r.get("value"):
        return None, "unsupported_or_changed_numeric_result"
    return {"evidence_id": eid, "statistic_index": index, "value": value, "context": c}, None


def admit_topics(topics, evidence):
    """Return only canonical scalar conflicts plus an audit of every proposed edge."""
    evidence_by_id = {e.evidence_id: e for e in evidence}
    artifacts = {}
    admitted, decisions = [], []
    for topic in topics:
        arguments = {a["argument_id"]: a for a in topic["arguments"]}
        resolved = {aid: _resolve(a, evidence_by_id, artifacts) for aid, a in arguments.items()}
        edges = []
        used = set()
        for left, right in topic["attacks"]:
            a, ae = resolved[left]
            b, be = resolved[right]
            decision = {"topic_id": topic["topic_id"], "attack": [left, right]}
            if ae or be:
                decision.update(status="comparability_unconfirmed", reasons={left: ae, right: be})
            elif a["evidence_id"] == b["evidence_id"]:
                decision.update(status="comparability_unconfirmed", reason="not_cross_branch")
            elif a["context"] != b["context"]:
                # Different contexts can be compatible; do not assert semantic equivalence
                # or opposition. Report exactly which execution fields differ.
                decision.update(status="different_context", different_fields=sorted(
                    k for k in CONTEXT_KEYS if a["context"][k] != b["context"][k]))
            else:
                op = a["context"]["operation"]
                equal = (a["value"] == b["value"] if op in {"row_count", "nunique"} else
                         math.isclose(a["value"], b["value"], rel_tol=1e-9, abs_tol=1e-12))
                if equal:
                    decision.update(status="compatible_numeric_results")
                else:
                    decision.update(status="admitted_numeric_conflict")
                    edges.append([left, right])
                    used.update([left, right])
            decisions.append(decision)
        if not topic["attacks"]:
            decisions.append({"topic_id": topic["topic_id"], "status": "no_attack_candidate"})
        if edges:
            canonical = []
            for aid in sorted(used):
                r = resolved[aid][0]
                c = r["context"]
                # A3 cannot smuggle a broader natural-language inference into a
                # numerically checked edge. Negotiate only this executed scalar.
                canonical.append({
                    "argument_id": aid, "evidence_ids": [r["evidence_id"]], "requires": [],
                    "statement": (f"Executed {c['operation']}({c['column'] or '*'}) on "
                                  f"{c['file_id']} with scope {c['selection_sha256']} "
                                  f"equals {r['value']}; this states only the recorded scalar."),
                    "basis": f"{r['evidence_id']}.statistics[{r['statistic_index']}]",
                    "result_ref": {"evidence_id": r["evidence_id"], "statistic_index": r["statistic_index"]},
                    "execution_context": c,
                })
            admitted.append({"topic_id": topic["topic_id"],
                             "description": "Different executed scalar values under identical recorded contexts",
                             "arguments": canonical, "attacks": edges,
                             "evidence_ids": sorted({e for a in canonical for e in a["evidence_ids"]})})
    return admitted, {
        "version": "execution_context_gate.v1", "proposed_topic_count": len(topics),
        "admitted_topic_count": len(admitted), "decisions": decisions,
        "rules": "Only exact recorded raw-reduction contexts with incompatible scalar values are admitted. "
                 "Unknown context is not proof of compatibility; unadmitted prose remains unadjudicated.",
    }
