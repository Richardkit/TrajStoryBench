from pathlib import Path
import pytest
from trajstoryagent_v5.v5_framework.checkpoint_paths import relocate_path


def test_frozen_paths_resolve_without_changing_unrelated_values(tmp_path):
    assert Path(relocate_path("B:/TrajStory/results/case/00_input.json", tmp_path)) == tmp_path / "results/case/00_input.json"
    assert Path(relocate_path("B:" + chr(92) + "TrajStory" + chr(92) + "input.csv", tmp_path)) == tmp_path / "input.csv"
    for value in ("A scientific insight", "B:/TrajStoryElse/input.csv", 42, None):
        assert relocate_path(value, tmp_path) == value


def test_parent_traversal_rejected(tmp_path):
    with pytest.raises(ValueError):
        relocate_path("B:/TrajStory/../secret", tmp_path)
