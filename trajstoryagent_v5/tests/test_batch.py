import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from trajstoryagent_v5.v5_framework import batch
from trajstoryagent_v5.v5_framework import case_adapter
from trajstoryagent_v5.v5_framework import providers


def setup_batch(tmp_path, monkeypatch):
    case = tmp_path/'case.json'
    case.write_text('{}')
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'cases':[{'case_id':'case01','json_path':'case.json'}]}))
    monkeypatch.setattr(case_adapter, 'load_trajstory_case', lambda *a, **k: SimpleNamespace(files=[]))
    monkeypatch.setattr(providers, 'credentials', lambda *a: None)
    return ['--model','gpt','--manifest',str(manifest),'--repo-root',str(tmp_path),'--output',str(tmp_path/'run')]


def test_dry_run_and_bad_selection(tmp_path, monkeypatch):
    args = setup_batch(tmp_path, monkeypatch)
    assert batch.main(args+['--dry-run']) == 0
    assert not (tmp_path/'run').exists()
    with pytest.raises(ValueError, match='Unknown case'):
        batch.main(args+['--case-ids','missing'])


def test_success_uses_frozen_independent_package_and_refuses_overwrite(tmp_path, monkeypatch):
    args = setup_batch(tmp_path, monkeypatch)
    def execute(command, cwd, **kwargs):
        assert command[command.index('-m')+1] == 'trajstoryagent_v5'
        assert (cwd/'trajstoryagent_v5/cli.py').is_file()
        assert not (cwd/'scripts').exists()
        dest = Path(command[command.index('--output')+1])
        dest.mkdir(parents=True)
        (dest/'result.json').write_text('{}')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(batch.subprocess, 'run', execute)
    assert batch.main(args) == 0
    with pytest.raises(FileExistsError):
        batch.main(args)


def test_failed_case_recorded_without_retry(tmp_path, monkeypatch):
    args = setup_batch(tmp_path, monkeypatch)
    calls=[]
    def fail(*a, **k):
        calls.append(a)
        return SimpleNamespace(returncode=2)
    monkeypatch.setattr(batch.subprocess, 'run', fail)
    assert batch.main(args) == 1
    assert len(calls) == 1
    rows=json.loads((tmp_path/'run/batch_results.json').read_text())
    assert rows[0]['state']=='failed'
