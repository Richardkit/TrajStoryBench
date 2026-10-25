import json
from pathlib import Path

import pytest

from trajstoryagent_v5.input_paths import resolve_input_csv
from trajstoryagent_v5.case_adapter import load_trajstory_case


def make_case(tmp_path):
    (tmp_path/'input_trips.csv').write_text('trip_id\nt1\n',encoding='utf-8')
    (tmp_path/'input_context.csv').write_text('name\nreference\n',encoding='utf-8')
    case=tmp_path/'case.json'
    case.write_text(json.dumps(dict(metadata=dict(goal='Analyze the supplied input'),
        dataset_csv_path='trips.csv',input_references=[dict(path='context.csv'),dict(path='trips.csv')])),encoding='utf-8')
    return case


def test_v5_loads_projected_files_without_requiring_originals(tmp_path):
    case=make_case(tmp_path)
    bundle=load_trajstory_case(case,tmp_path)
    assert [(f.file_id,f.path.name) for f in bundle.files]==[
        ('main','input_trips.csv'),('context','input_context.csv')]


def test_missing_projection_never_falls_back_to_original(tmp_path):
    old=tmp_path/'trips.csv';old.write_text('secret_derived_value\n100\n',encoding='utf-8')
    with pytest.raises(FileNotFoundError,match='no original CSV fallback'):
        resolve_input_csv(old)


def test_existing_input_prefix_is_not_applied_twice(tmp_path):
    path=tmp_path/'input_trips.csv';path.write_text('id\n1\n',encoding='utf-8')
    assert resolve_input_csv(path)==path.resolve()


def test_only_declared_fully_derived_table_is_omitted(tmp_path):
    directory=tmp_path/'trajstory_examples/insight/LaDe/15_delivery_acceptance_bursts'
    assert resolve_input_csv(directory/'acceptance_bursts.csv') is None
    assert resolve_input_csv(directory/'input_acceptance_bursts.csv') is None
    with pytest.raises(FileNotFoundError):
        resolve_input_csv(directory/'other.csv')


