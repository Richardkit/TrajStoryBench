import json
import sys
from pathlib import Path

import pytest

FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
TRAJSTORY_ROOT = FRAMEWORK_ROOT.parent
MANIFEST = TRAJSTORY_ROOT / "trajstory_examples" / "insight" / "manifest.json"
sys.path.insert(0, str(FRAMEWORK_ROOT))

from src.trajstory import build_case_spec, load_manifest_cases, parse_trajstory_result


def test_case_spec_is_sanitised_and_multitable():
    entries = load_manifest_cases(
        MANIFEST,
        trajstory_root=TRAJSTORY_ROOT,
        case_ids=["danish_ais_01_shared_corridor_day"],
    )
    spec = build_case_spec(entries[0], trajstory_root=TRAJSTORY_ROOT, sample_rows=5)

    assert spec["format"] == "trajstory-agent-input-v1"
    assert [table["name"] for table in spec["tables"]] == ["main", "vessels"]
    assert all(Path(table["path"]).is_file() for table in spec["tables"])
    serialised = json.dumps(spec)
    assert '"insights"' not in serialised
    assert '"insight_list"' not in serialised
    assert '"summary"' not in serialised


def test_unknown_case_is_rejected():
    with pytest.raises(ValueError, match="Unknown TrajStory case IDs"):
        load_manifest_cases(
            MANIFEST,
            trajstory_root=TRAJSTORY_ROOT,
            case_ids=["not_a_real_case"],
        )


def test_result_parser_accepts_fenced_structured_json():
    result = parse_trajstory_result(
        """Result follows:\n```json
        {"insights": [{"statement": "A differs from B.", "evidence": ["n=20"],
        "dimensions": ["group"], "confidence": 0.8}], "summary": "A concise summary."}
        ```"""
    )
    assert result["pred_insights"] == ["A differs from B."]
    assert result["pred_summary"] == "A concise summary."
    assert result["structured_insights"][0]["confidence"] == 0.8


def test_result_parser_rejects_empty_outputs():
    with pytest.raises(ValueError, match="non-empty insights"):
        parse_trajstory_result('{"insights": [], "summary": "none"}')


@pytest.mark.parametrize("summary_fields", [{}, {"summary": ""}, {"summary": "Summary"}])
def test_result_parser_preserves_partial_insights(summary_fields):
    payload = {"insights": ["One supported finding."], **summary_fields}
    result = parse_trajstory_result(json.dumps(payload))
    assert result["pred_insights"] == payload["insights"]
    assert result["pred_summary"] == summary_fields.get("summary", "")
