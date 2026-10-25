import json
from types import SimpleNamespace

import pytest
from trajstoryagent_v5 import evaluate, hyperparameter_replay as hp, ablation_replay as ab
from trajstoryagent_v5.ablation_memory_profile import NoMemoryConfig
from trajstoryagent_v5.models import FrameworkConfig


def test_two_independent_rounds_threshold_before_average(tmp_path):
    calls=[]
    def create(**kwargs):
        calls.append(kwargs)
        rating=5 if len(calls)==1 else 7
        payload={'choices':[{'finish_reason':'stop','message':{'content':f'<rating>{rating}</rating>'}}]}
        return SimpleNamespace(model_dump=lambda:payload)
    client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result=evaluate.score(['pred'],['ref'],tmp_path/'eval',client)
    assert len(calls)==2 and calls[0]==calls[1]
    assert result['g_eval']==.6 and result['coverage_at_0_6']==.5
    with pytest.raises(FileExistsError):evaluate.score(['pred'],['ref'],tmp_path/'eval',client)
    assert len(calls)==2


def test_memory_guard_only_relaxed_in_ablation():
    with pytest.raises(ValueError):FrameworkConfig(memory_manager_enabled=False)
    assert not NoMemoryConfig(memory_manager_enabled=False).memory_manager_enabled
    with pytest.raises(ValueError):NoMemoryConfig(memory_manager_enabled=False,min_support=0)


def test_frozen_unsupported_observations_never_reexport(tmp_path,monkeypatch):
    frozen=tmp_path/'old';frozen.mkdir()
    (frozen/'branch.json').write_text(json.dumps({'status':'failed'}))
    monkeypatch.setattr(hp.refinement,'observations',lambda *a:pytest.fail('Unexpected export'))
    result=hp.branch('gpt',{'question_id':'Q1','question':'q','insight':'original'},{},{},tmp_path/'new',frozen=frozen,partition_penalty=3)
    assert result['status']=='failed'
    assert result['insights']['conditional_compressed']=='original'


def test_hyperparameter_math_and_frozen_observations(tmp_path,monkeypatch):
    measure=dict(description='duration',unit='min',observation_unit='trip',scope='all',conditions=['group'],
                 cells=[dict(condition={'group':g},n=3,mean=v,sse=1.) for g,v in [('a',0.),('b',10.)]])
    frozen=tmp_path/'old';frozen.mkdir()
    (frozen/'observations.json').write_text(json.dumps({'measures':[measure]}))
    monkeypatch.setattr(hp.refinement,'observations',lambda *a:pytest.fail('Unexpected export'))
    def interpret(slug,evidence,bundle,facts,target):
        hp.write(target/'interpretation.json',{'use_refinement':True,'insight':'refined'})
        return 'refined'
    monkeypatch.setattr(hp,'interpret',interpret)
    args=('gpt',{'question_id':'Q1','question':'q','insight':'original'},{},{},tmp_path/'new')
    result=hp.branch(*args,frozen=frozen,partition_penalty=3,min_support=2,tolerance_sd=.1,distortion_weight=5)
    assert result['status']=='computed'
    cert=hp.read(tmp_path/'new/mathematical_certificate.json')
    expected=hp.conditional_partition(measure,penalty=3,min_support=2)
    assert cert['partitions']==[expected]
    assert cert['compressions']==[hp.compress_with_exceptions(expected,distortion_weight=5,tolerance_sd=.1)]
    with pytest.raises(AssertionError):hp.branch(*args,frozen=frozen,partition_penalty=1)


def test_no_compression_reuses_original_uncompressed_text(tmp_path):
    old=tmp_path/'old';old.mkdir()
    evidence={'question_id':'Q1','question':'q','insight':'original'}
    ab.write(old/'original_evidence.json',evidence)
    ab.write(old/'branch.json',{'status':'computed','insights':{'conditional_only':'plain','conditional_compressed':'compressed'},
                              'arm_status':{'conditional_only':'applied'}})
    ab.write(old/'conditional_only/facts.json',[{'value':1}])
    result=ab.conditional_branch('qwen',evidence,{},{},tmp_path/'new',cached_first=old)
    assert result['insights']=={'conditional_only':'plain'}
    assert not (tmp_path/'new/conditional_compressed').exists()


def test_batch_scoring_weights_references_and_records_failure(tmp_path,monkeypatch):
    from pathlib import Path
    from trajstoryagent_v5 import evaluate_batch as eb
    cases=[]
    for cid in ('a','b','c'):
        case=tmp_path/f'{cid}.json';case.write_text(json.dumps({'insights':['ref']}))
        gen=tmp_path/'generation'/cid;gen.mkdir(parents=True)
        (gen/'result.json').write_text(json.dumps({'insights':['pred']}))
        cases.append({'case_id':cid,'json_path':case.name})
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'cases':cases}))
    def fake(args):
        output=Path(args[args.index('--output')+1])
        if output.name=='c':raise ValueError('Failed judge')
        count,value=(1,.2) if output.name=='a' else (3,.6)
        eb.write_json(output/'comparison.json',dict(reference_count=count,g_eval=value,coverage_at_0_6=value))
    monkeypatch.setattr(eb,'score_case',fake)
    assert eb.main(['--generation-root',str(tmp_path/'generation'),'--manifest',str(manifest),
                    '--repo-root',str(tmp_path),'--output',str(tmp_path/'scores')])==1
    result=json.loads((tmp_path/'scores/comparison.json').read_text())
    assert result['means']['g_eval']==pytest.approx(.5)
    assert not result['complete'] and result['failures'][0]['case_id']=='c'
