"""Bounded, evidence-grounded adaptation of partial argumentation agreements.

LLMs expose topic arguments and explicit branch positions; Python enumerates
dependency-closed, conflict-free candidates and maximizes minimum Hamming
satisfaction. Unknown positions are excluded, not treated as rejection. This is
an engineering adaptation, not an implementation of formal AF semantics.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from itertools import combinations
from pathlib import Path

from trajstoryagent_v5.v5_framework.json_utils import extract_json_object, write_json


REVIEW_INSTRUCTION = """
In the SAME response, also return "agreement_topics": [] if no material cross-branch
disagreement exists. Do not invent conflict between different questions, populations,
units, time ranges, or explicitly conditional methods. Missing discussion is not rejection.
Inspect the executed code as well as evidence. Identify only disagreements affecting
subsequent use of the evidence. Each topic has 2 or 3 participating evidence IDs and
2--8 atomic arguments (bounded claims or method-use proposals already grounded in evidence).
Include unaffected parts only when needed to express partial agreement within that topic.
Record exact scope/method conditions in statements. Do not declare a method correct simply
because it ran, was used first, or has more supporters. Code problems require concrete citations.
Each argument has source evidence IDs, a nonempty basis pointing to supplied statistics or code,
and IDs of arguments it requires. Attacks are pairs of incompatible arguments in this topic,
not differences of scope. Arguments and dependencies are local to each topic, with no cycles.
No new calculations or fabricated numbers. Do not request code repair merely for disagreement;
REVISE remains reserved for question/goal irrelevance.
Schema in addition to results:
"agreement_topics": [{"topic_id":"T1","description":"...",
"evidence_ids":["E1","E2"],"arguments":[
{"argument_id":"a","statement":"...","evidence_ids":["E1"],
"basis":"statistic name or code excerpt and its relevance","requires":[]},
{"argument_id":"b","statement":"...","evidence_ids":["E2"],
"basis":"...","requires":[]}],"attacks":[["a","b"]]}]
At most 3 topics. Return all required keys, including empty lists.
For each argument add result_ref={"evidence_id":"E1","statistic_index":0}
when it is an exact assertion about a supplied executed scalar. Otherwise use null.
statistic_index is zero-based. Do not invent references, comparison_records, scope,
units, or method metadata. The program admits only incompatible scalar results
with matching execution-recorded contexts; qualitative/custom/legacy assertions
remain unconfirmed when this context is absent. Candidate prose is never proof of conflict.
"""


def validate_topics(topics, evidence_ids):
    if not isinstance(topics, list) or len(topics) > 3:
        raise ValueError("agreement_topics must be a list of at most 3 topics")
    seen = set()
    for t in topics:
        if not isinstance(t, dict):
            raise ValueError("every agreement topic must be an object")
        tid = t["topic_id"]
        if not isinstance(tid, str) or not tid or tid in seen:
            raise ValueError("invalid/duplicate topic_id")
        seen.add(tid)
        participants = t["evidence_ids"]
        if (not isinstance(participants, list) or not 2 <= len(participants) <= 3
                or any(not isinstance(eid, str) for eid in participants)
                or len(set(participants)) != len(participants)
                or not set(participants) <= evidence_ids):
            raise ValueError("topic requires 2--3 distinct known participants")
        if not isinstance(t["description"], str) or not t["description"].strip():
            raise ValueError("topic description required")
        args = t["arguments"]
        if not isinstance(args, list) or not 2 <= len(args) <= 8:
            raise ValueError("topic requires 2--8 arguments")
        if any(not isinstance(a, dict) for a in args):
            raise ValueError("every topic argument must be an object")
        ids = [a["argument_id"] for a in args]
        if any(not isinstance(a, str) or not a for a in ids) or len(set(ids)) != len(ids):
            raise ValueError("invalid/duplicate argument IDs")
        by_id = {a["argument_id"]: a for a in args}
        for a in args:
            for key in ("statement", "basis"):
                if not isinstance(a[key], str) or not a[key].strip():
                    raise ValueError(f"argument {key} required")
            if (not isinstance(a["evidence_ids"], list) or not a["evidence_ids"]
                    or any(not isinstance(eid, str) for eid in a["evidence_ids"])
                    or not set(a["evidence_ids"]) <= set(participants)):
                raise ValueError("argument sources must be topic participants")
            if (not isinstance(a["requires"], list)
                    or any(not isinstance(aid, str) for aid in a["requires"])
                    or not set(a["requires"]) <= set(ids)):
                raise ValueError("unknown dependency")
        def visit(aid, stack):
            if aid in stack:
                raise ValueError("cyclic argument dependency")
            for dep in by_id[aid]["requires"]:
                visit(dep, stack | {aid})
        for aid in ids:
            visit(aid, set())
        if not isinstance(t["attacks"], list):
            raise ValueError("attacks must be a list")
        for pair in t["attacks"]:
            if (not isinstance(pair, list) or len(pair) != 2
                    or any(not isinstance(aid, str) for aid in pair)
                    or not set(pair) <= set(ids) or pair[0] == pair[1]):
                raise ValueError("invalid attack pair")
    return topics


def evidence_context(evidence):
    return [dict(item.to_dict(), executed_code=(
        Path(item.code_path).read_text() if item.code_path else "")) for item in evidence]


def update_position(llm, evidence_id, topics, evidence, goal, output_dir):
    """Exactly one LLM completion per involved branch, across all its topics."""
    selected = [t for t in topics if evidence_id in t["evidence_ids"]]
    participants = {eid for t in selected for eid in t["evidence_ids"]}
    prompt = {
        "goal": goal, "your_evidence_id": evidence_id,
        "topics": selected,
        "shared_evidence": evidence_context([e for e in evidence if e.evidence_id in participants]),
    }
    write_json(output_dir / f"{evidence_id}_request.json", prompt)
    system = (
        "You are the original TrajStory analysis branch updating your position once after "
        "reading shared evidence. Do not defend your original answer automatically. "
        "For EVERY argument in EVERY supplied topic, return accept, reject, or unknown, "
        "with a concrete evidence/code reason. Unknown means not established, not rejection. "
        "Accept only the exact bounded statement, including method/scope conditions. "
        "Do not compute, change measured values, rewrite code, or invent evidence. "
        "An alleged error without sufficient grounds must remain unknown. Return JSON: "
        '{"positions":[{"topic_id":"T1","arguments":[{"argument_id":"a",'
        '"stance":"accept|reject|unknown","reason":"..."}]}]}.'
    )
    try:
        response = llm.complete(system=system, user=json.dumps(prompt, ensure_ascii=False))
        write_json(output_dir / f"{evidence_id}_response.json", {"response": response})
        payload = extract_json_object(response)
        positions = payload["positions"]
        if not isinstance(positions, list) or len(positions) != len(selected):
            raise ValueError("positions must cover exactly the supplied topics")
        expected = {t["topic_id"]: {a["argument_id"] for a in t["arguments"]} for t in selected}
        seen = set()
        for p in positions:
            tid = p["topic_id"]
            if tid not in expected or tid in seen:
                raise ValueError("invalid/duplicate position topic")
            seen.add(tid)
            args = p["arguments"]
            if len(args) != len(expected[tid]) or {a["argument_id"] for a in args} != expected[tid]:
                raise ValueError("position arguments must cover topic exactly")
            for a in args:
                if a["stance"] not in {"accept", "reject", "unknown"} or not str(a["reason"]).strip():
                    raise ValueError("invalid stance/reason")
        result = {"evidence_id": evidence_id, "status": "completed", "positions": positions}
    except Exception as exc:
        # A malformed update must not create a second reasoning round or fake consensus.
        result = {"evidence_id": evidence_id, "status": "unresolved", "error": str(exc),
                  "positions": [{"topic_id": t["topic_id"], "arguments": [
                      {"argument_id": a["argument_id"], "stance": "unknown",
                       "reason": "Branch update unavailable; no position inferred."}
                      for a in t["arguments"]]} for t in selected]}
    write_json(output_dir / f"{evidence_id}_position.json", result)
    return result


def commit_topic(topic, updates):
    ids = [a["argument_id"] for a in topic["arguments"]]
    positions = {}
    for update in updates:
        if update["evidence_id"] in topic["evidence_ids"]:
            p = next(p for p in update["positions"] if p["topic_id"] == topic["topic_id"])
            positions[update["evidence_id"]] = {a["argument_id"]: a["stance"] for a in p["arguments"]}
    if set(positions) != set(topic["evidence_ids"]):
        raise ValueError("missing participant position")
    candidates = []
    for size in range(len(ids) + 1):
        for selected in combinations(ids, size):
            s = set(selected)
            if any(x in s and y in s for x, y in topic["attacks"]):
                continue
            if any(a["argument_id"] in s and not set(a["requires"]) <= s for a in topic["arguments"]):
                continue
            # Nothing can be adopted solely through shared abstention.
            if any(not any(p[aid] == "accept" for p in positions.values()) for aid in s):
                continue
            scores = []
            for p in positions.values():
                known = {aid for aid in ids if p[aid] != "unknown"}
                if not known:
                    scores = []
                    break
                matches = sum((aid in s) == (p[aid] == "accept") for aid in known)
                scores.append(Fraction(matches, len(known)))
            if scores:
                candidates.append((min(scores), list(selected), [float(x) for x in scores]))
    maximum = max((x[0] for x in candidates), default=None)
    best = [x for x in candidates if x[0] == maximum] if maximum is not None else []
    # Never resolve tied alternatives by arbitrary enumeration order.
    common = set.intersection(*(set(x[1]) for x in best)) if best else set()
    shared = {aid for aid in common if all(p[aid] == "accept" for p in positions.values())}
    # Shared downstream claims cannot silently rely on disputed prerequisites.
    while True:
        reduced = {aid for aid in shared if all(
            set(a["requires"]) <= shared for a in topic["arguments"] if a["argument_id"] == aid)}
        if reduced == shared:
            break
        shared = reduced
    return {
        **topic, "minimum_satisfaction": float(maximum) if maximum is not None else None,
        "candidate_count": len(candidates),
        "best_candidates": [{"argument_ids": x[1], "participant_satisfaction": dict(zip(positions, x[2]))} for x in best],
        "shared_argument_ids": sorted(shared),
        "conditional_argument_ids": sorted(common - shared),
        "deferred_argument_ids": sorted(set(ids) - common),
        "status": "partial_commit" if best else "unresolved",
    }


def build_agreement(llm, topics, evidence, goal, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    participants = sorted({eid for t in topics for eid in t["evidence_ids"]})
    if len(participants) > 3:
        raise ValueError("at most three branch updates allowed")
    if participants:
        with ThreadPoolExecutor(max_workers=len(participants)) as pool:
            updates = list(pool.map(lambda eid: update_position(
                llm, eid, topics, evidence, goal, output_dir), participants))
    else:
        updates = []
    result = {
        "protocol": "bounded_partial_agreement_v1",
        "branch_update_calls": len(participants),
        "updates": updates,
        "topics": [commit_topic(t, updates) for t in topics],
        "use_rules": (
            "Shared arguments are usable only within their stated scope; agreement is not factual verification. "
            "Conditional arguments require explicit method/scope attribution and disclosure of dissent. "
            "Deferred arguments are not established premises: investigate them or omit dependent claims. "
            "Do not combine incompatible assignments/denominators. Original computed evidence is immutable. "
            "Unmentioned evidence remains individually supported, not unanimously accepted. "
            "Follow-up computations may supersede an argument only with explicit new evidence and matching scope; "
            "report that change with evidence IDs. No additional debate round is authorized."
        ),
    }
    write_json(output_dir / "commitment.json", result)
    return result


def downstream_context(agreement):
    """Keep audit responses on disk; downstream needs arguments, decisions and reasons."""
    return {k: v for k, v in agreement.items() if k != "updates"} | {
        "positions": [{"evidence_id": u["evidence_id"], "status": u["status"],
                       "positions": u["positions"]} for u in agreement["updates"]]
    }
