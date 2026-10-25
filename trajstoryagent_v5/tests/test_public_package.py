import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import trajstoryagent_v5
from trajstoryagent_v5.v5_framework import cli
from trajstoryagent_v5.v5_framework import export_support
from trajstoryagent_v5.v5_framework.execution import PythonCodeExecutor
from trajstoryagent_v5.v5_framework.identity import FRAMEWORK_ID, IMPLEMENTATION_ID
from trajstoryagent_v5.v5_framework import refinement


def test_only_paper_compression_kernel_and_default_support_are_exposed():
    from trajstoryagent_v5.v5_framework import conditional_math
    from trajstoryagent_v5.v5_framework import compression_math
    import inspect
    assert not hasattr(conditional_math,'compress_with_exceptions')
    assert refinement.compress_with_exceptions is compression_math.compress_with_exceptions
    assert inspect.signature(conditional_math.conditional_partition).parameters['min_support'].default==1


@pytest.mark.parametrize('model', ['gpt', 'qwen'])
def test_public_cli_uses_formal_identity_and_snapshots_only_this_package(tmp_path, monkeypatch, model):
    (tmp_path/'input_data.csv').write_text('id,value\na,3\n', encoding='utf8')
    case=tmp_path/'case.json'
    case.write_text(json.dumps({'metadata':{'goal':'Compare'},'dataset_csv_path':'data.csv'}),encoding='utf8')
    monkeypatch.setattr('trajstoryagent_v5.v5_framework.providers.credentials',lambda generator:None)
    monkeypatch.setenv('OPENAI_API_KEY','offline-placeholder')
    monkeypatch.setenv('DASHSCOPE_API_KEY','offline-placeholder')
    clients=[]
    monkeypatch.setattr('trajstoryagent_v5.v5_framework.transport.ResilientClient',lambda **kw:clients.append(kw) or object())
    class Agent:
        def __init__(self,client,*,generator,config):
            assert generator==model and config.memory_manager_enabled and config.min_support==1
        def run(self,bundle,output):
            assert bundle.files[0].path.name=='input_data.csv'
            (output/'predictions.json').write_text(json.dumps({'framework':FRAMEWORK_ID,
                'insights':{'conditional_compressed':['Retained insight.']}}),encoding='utf8')
            return SimpleNamespace(insights=[SimpleNamespace(insight='Retained insight.')])
    monkeypatch.setattr(trajstoryagent_v5,'TrajStoryAgent',Agent)
    before=export_support.OpenAICompatibleClient
    output=tmp_path/'output'
    cli.main(['--case',str(case),'--repo-root',str(tmp_path),'--model',model,'--output',str(output)])
    assert export_support.OpenAICompatibleClient is before
    assert clients[0]['temperature']==0 and clients[0]['reasoning_effort']=='none'
    result=json.loads((output/'result.json').read_text(encoding='utf8'))
    manifest=json.loads((output/'run_manifest.json').read_text(encoding='utf8'))
    assert result['framework']==manifest['framework']==FRAMEWORK_ID
    assert result['implementation']==manifest['implementation']==IMPLEMENTATION_ID
    assert result['insights']==['Retained insight.']
    for relative,digest in manifest['source_sha256'].items():
        assert Path(relative).parts[0]=='trajstoryagent_v5'
        assert hashlib.sha256((output/'source_snapshot'/relative).read_bytes()).hexdigest()==digest


def test_executor_loads_v5_tools_and_supplied_data(tmp_path):
    data=tmp_path/'input_values.csv';data.write_text('value\n2\n4\n',encoding='utf8')
    code="from trajstoryagent_v5.v5_framework.analysis_tools import AnalysisSession\ns=AnalysisSession.from_environment()\nprint(float(s.load_table('main').value.mean()))\n"
    result=PythonCodeExecutor().run(code,tmp_path/'run',input_files={'main':str(data)})
    assert result.success,result.diagnostic
    assert result.stdout.strip()=='3.0'


def test_package_imports_without_other_project_directories(tmp_path):
    package=Path(trajstoryagent_v5.__file__).parent
    isolated=tmp_path/'isolated';isolated.mkdir()
    shutil.copytree(package,isolated/'trajstoryagent_v5',ignore=shutil.ignore_patterns('__pycache__','tests','build','*.egg-info'))
    code='''import importlib,importlib.abc,pkgutil,sys
sys.path.insert(0,sys.argv[1])
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'trajstoryagent_v1','trajstoryagent_v2','trajstoryagent_v3','trajstoryagent_v4','scripts','insightbench'}:
   raise AssertionError('Forbidden dependency: '+fullname)
sys.meta_path.insert(0,Block())
import trajstoryagent_v5 as pkg
for item in pkgutil.iter_modules(pkg.__path__):
 if item.name != '__main__':importlib.import_module('trajstoryagent_v5.'+item.name)
agent=pkg.TrajStoryAgent(None,generator='gpt')
assert agent.config.memory_manager_enabled
assert agent.__class__.__module__=='trajstoryagent_v5.v5_framework.controller'
print(pkg.FRAMEWORK_ID)
'''
    run=subprocess.run([sys.executable,'-I','-c',code,str(isolated)],cwd=isolated,
        capture_output=True,text=True,timeout=60)
    assert run.returncode==0,run.stderr
    assert run.stdout.strip()==FRAMEWORK_ID


@pytest.mark.parametrize('mode',['unsupported','export_failed','declined','interpret_failed'])
def test_refinement_fallback_retains_initial_insight(tmp_path,monkeypatch,mode):
    measure=dict(description='duration',unit='min',observation_unit='trip',scope='all',
        conditions=['group'],cells=[dict(condition={'group':g},n=3,mean=v,sse=1.) for g,v in [('a',0.),('b',10.)]])
    def observations(*args):
        if mode=='export_failed':raise ValueError('offline export failure')
        return {'measures':[],'unsupported_reason':'Unsupported relation'} if mode=='unsupported' else {'measures':[measure]}
    monkeypatch.setattr(refinement,'observations',observations)
    def interpret(slug,evidence,bundle,facts,folder):
        if mode=='interpret_failed':raise ValueError('offline interpretation failure')
        refinement.write(folder/'interpretation.json',{'use_refinement':False,'insight':evidence['insight']})
        return evidence['insight']
    monkeypatch.setattr(refinement,'interpret',interpret)
    result=refinement.run_branch('gpt',dict(question_id='Q1',question='Compare?',insight='Original measured insight.'),{}, {},tmp_path/'branch')
    assert result['insights']['conditional_compressed']=='Original measured insight.'
    assert result['arm_status']['conditional_compressed']!='applied'
