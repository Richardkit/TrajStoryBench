"""Fresh ablations and user-produced checkpoint reuse, without external models."""
import copy
import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from trajstoryagent_v5.v5_framework import controller
from trajstoryagent_v5.v5_framework.models import (
    FrameworkConfig, DataBundle, DataFile, AnalysisQuestion, EvidencePackage,
    TrajectoryDataProfile, BranchInsight, TrajStoryReport,
)
from trajstoryagent_v5.ablation import experiment, ablation_replay as replay
from trajstoryagent_v5.ablation.ablation_memory_profile import MemoryAblation, DataProfilingAblation


def wire(agent, counts, monkeypatch, *, checkpoint=False):
    agent.config=replace(agent.config, agreement_enabled=False)
    if not checkpoint:
        if not isinstance(agent,DataProfilingAblation):
            agent.profiler=SimpleNamespace(build=lambda b:TrajectoryDataProfile([],[]))
        agent.decomposer=SimpleNamespace(decompose=lambda b,p:[AnalysisQuestion('Q1','Count input rows')])
        def verify(**kw):
            replay.write(kw['run_dir']/'03_initial_review_audit_round_0.json',{})
            replay.write(kw['run_dir']/'03_initial_agreement_topics_round_0.json',[])
            return kw['evidence']
        agent._verify_and_repair_initial=verify
    def initial(**kw):
        counts.append('initial')
        q=kw['question']
        return EvidencePackage('EQ1',q.question_id,q.question,q.analysis_granularity,'raw',[],[],{})
    agent.analysis.engine=SimpleNamespace(analyze_initial=initial)
    explorer=SimpleNamespace(explore=lambda **kw:[],last_duplicate_reviews=[],last_candidates=[])
    if isinstance(agent,MemoryAblation):agent.explorer.delegate=explorer
    else:agent.explorer=explorer
    def refine(slug,evidence,bundle,profile,folder):
        counts.append('full_refinement')
        result=dict(question_id=evidence['question_id'],status='computed',
            insights={'conditional_only':'plain','conditional_compressed':'compressed'},
            arm_status={'conditional_only':'declined','conditional_compressed':'declined'})
        replay.write(folder/'branch.json',result)
        replay.write(folder/'original_evidence.json',evidence)
        return result
    agent.refiner=refine
    def conditional(slug,evidence,bundle,profile,folder,**kw):
        counts.append('conditional_refinement')
        return dict(question_id=evidence['question_id'],insights={'conditional_only':'plain'},arm_status={'conditional_only':'declined'})
    monkeypatch.setattr(replay,'conditional_branch',conditional)
    def memory(*a,**kw):
        counts.append('memory')
        return {'Question':[],'Finding':[],'Errors appear in coding':[]}
    monkeypatch.setattr(controller.MemoryManagerAgent,'compress',memory)
    def summary(**kw):
        return TrajStoryReport([BranchInsight('I1',e.question_id,e.insight,(e.evidence_id,)) for e in kw['evidence']],
                              'summary',kw['evidence'],kw['questions'],kw['follow_ups'])
    agent.synthesizer=SimpleNamespace(synthesize=summary)
    return agent

@pytest.fixture
def bundle(tmp_path):
    path=tmp_path/'input_data.csv';path.write_text('x\n1\n2\n')
    return DataBundle('Count rows',(DataFile('main',path),))

@pytest.mark.parametrize('variant', experiment.VARIANTS[:4])
def test_fresh_runs_initial_analysis_and_disables_only_requested_stage(tmp_path,monkeypatch,bundle,variant):
    counts=[]
    if variant=='without_memory':agent=MemoryAblation(None,'gpt',None)
    elif variant=='without_data_profiling':agent=DataProfilingAblation(None,'gpt',FrameworkConfig().to_dict())
    else:agent=replay.ReplayAblation(None,variant,None,generator='gpt')
    wire(agent,counts,monkeypatch).run(bundle,tmp_path/'out')
    assert counts.count('initial')==1
    assert ('memory' in counts)==(variant!='without_memory')
    assert ('conditional_refinement' in counts)==(variant=='without_compression')
    assert ('full_refinement' in counts)==(variant in ('without_memory','without_data_profiling'))
    if variant=='without_data_profiling':
        profile=replay.read(tmp_path/'out/01_profile.json')
        assert profile['profile_kind']=='raw_schema_and_first_rows_only'
        assert 'row_count' not in profile['files'][0]
    if variant=='without_memory':assert not (tmp_path/'out/03_memory.json').exists()

@pytest.mark.parametrize('variant',['without_conditional_refinement','without_compression','without_memory'])
def test_user_generated_full_run_can_be_replayed_without_initial_analysis(tmp_path,monkeypatch,bundle,variant):
    full=tmp_path/'user_full';counts=[]
    wire(controller.TrajStoryAgent(None,generator='gpt'),counts,monkeypatch).run(bundle,full)
    replay.write(full/'run_manifest.json',{'model':'gpt-6-luna','input_sha256':{str(bundle.files[0].path):hashlib.sha256(bundle.files[0].path.read_bytes()).hexdigest()}})
    before={str(p):p.read_bytes() for p in full.rglob('*') if p.is_file()}
    experiment.prepare(full,'gpt',variant)
    counts=[]
    agent=MemoryAblation(None,'gpt',full) if variant=='without_memory' else replay.ReplayAblation(None,variant,full,generator='gpt')
    wire(agent,counts,monkeypatch,checkpoint=True).run(bundle,tmp_path/'replay')
    assert 'initial' not in counts
    assert before=={str(p):p.read_bytes() for p in full.rglob('*') if p.is_file()}


def test_checkpoint_agreement_is_copied_without_model_calls(tmp_path):
    source=tmp_path/'full';dest=tmp_path/'replay'
    replay.write(source/'03_agreement/commitment.json',{'topics':[{'id':'conflict'}]})
    assert replay.replay_agreement(source,dest)=={'topics':[{'id':'conflict'}]}
    assert replay.read(dest/'commitment.json')['topics']

@pytest.mark.parametrize('variant',experiment.VARIANTS[:4])
def test_fresh_preflight_needs_no_historical_directory_or_credentials(tmp_path,monkeypatch,variant):
    (tmp_path/'input_data.csv').write_text('x\n1\n')
    case=tmp_path/'case.json';case.write_text(json.dumps({'metadata':{'goal':'Count'},'dataset_csv_path':'data.csv'}))
    monkeypatch.setattr(experiment,'credentials',lambda *a:pytest.fail('preflight accessed credentials'))
    assert experiment.main(['--mode','fresh','--case',str(case),'--repo-root',str(tmp_path),
        '--model','gpt','--variant',variant,'--output',str(tmp_path/'new'),'--dry-run'])==0
    assert not (tmp_path/'new').exists()


def test_invalid_mode_combinations_rejected(tmp_path):
    base=['--model','gpt','--variant','without_memory','--output',str(tmp_path/'new')]
    for args in [['--mode','fresh','--source-run','x'],['--mode','checkpoint','--case','x']]:
        with pytest.raises(SystemExit):experiment.main(base+args)
    with pytest.raises(SystemExit):experiment.main(['--model','gpt','--variant','without_data_profiling',
        '--mode','checkpoint','--source-run','x','--output',str(tmp_path/'new')])


def test_batch_fresh_dispatch_has_no_checkpoint_dependency(tmp_path,monkeypatch):
    from pathlib import Path
    from trajstoryagent_v5.v5_framework import batch,providers
    csv=tmp_path/'input_data.csv';csv.write_text('x\n1\n')
    case=tmp_path/'case.json';case.write_text(json.dumps({'metadata':{'goal':'Count'},'dataset_csv_path':'data.csv'}))
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'cases':[{'case_id':'sample','json_path':'case.json'}]}))
    monkeypatch.setattr(providers,'credentials',lambda *a:None)
    calls=[]
    def execute(command,cwd,**kw):
        calls.append(command)
        assert '--source-run' not in command and '--checkpoint-manifest' not in command
        assert command[command.index('--mode')+1]=='fresh'
        assert Path(command[command.index('--case')+1])==case
        assert (cwd/'trajstoryagent_v5/ablation/experiment.py').exists()
        dest=Path(command[command.index('--output')+1]);dest.mkdir(parents=True)
        replay.write(dest/'result.json',{'insights':['mock']})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(batch.subprocess,'run',execute)
    assert batch.main(['--model','gpt','--experiment','without_memory','--mode','fresh',
        '--repo-root',str(tmp_path),'--manifest',str(manifest),'--output',str(tmp_path/'batch')])==0
    assert len(calls)==1
    assert replay.read(tmp_path/'batch/batch_manifest.json')['experiment']['mode']=='fresh'


def test_existing_checkpoint_path_not_rebased_into_source_snapshot(tmp_path,monkeypatch):
    from pathlib import Path
    from trajstoryagent_v5.v5_framework import checkpoint_paths as paths
    monkeypatch.setattr(paths,'ORIGINAL_ROOT',str(tmp_path).replace('\\','/'))
    monkeypatch.setattr(paths,'ROOT',tmp_path/'source_snapshot')
    data=tmp_path/'input.csv';data.write_text('x\n1\n')
    assert Path(paths.relocate_path(str(data)))==data
    assert Path(paths.relocate_path(str(data),tmp_path/'override'))==tmp_path/'override/input.csv'
