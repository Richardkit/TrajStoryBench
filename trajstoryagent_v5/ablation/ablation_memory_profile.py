"""Isolated A0/Memory ablation adapters for the accepted A0-A5 v5 lineage."""
from __future__ import annotations
import copy
from dataclasses import fields
import json
from pathlib import Path
import shutil

from trajstoryagent_v5.ablation import ablation_replay as common
from trajstoryagent_v5.v5_framework.profiling import _native

read,write=common.read,common.write
legacy=common.legacy
ARMS=('without_memory','without_data_profiling')

def bundle_from(data):
    return legacy.DataBundle(**{**data,'files':tuple(legacy.DataFile(**f) for f in data['files'])})

class NoMemoryConfig(common.FrameworkConfig):
    """Relax only the base class's mandatory-Memory guard, in this adapter."""
    def __post_init__(self):
        values={f.name:getattr(self,f.name) for f in fields(self)}
        values['memory_manager_enabled']=True
        common.FrameworkConfig(**values)
        if self.memory_manager_enabled is not False:
            raise ValueError('Memory ablation must disable the Memory Manager')

class RawHandoffLabelClient:
    def __init__(self,client):self.client=client
    def __getattr__(self,name):return getattr(self.client,name)
    def complete(self,*,system,user,**kwargs):
        # Rename the representation only; retain the existing planning/review task.
        user=user.replace('\nCOMPRESSED FIRST-ROUND MEMORY:\n','\nRAW FIRST-ROUND RECORDS (NO MEMORY MANAGER):\n')
        system=system.replace('compressed memory','raw first-round records')
        return self.client.complete(system=system,user=user,**kwargs)

class RawHandoffExplorer:
    """Use the same goal-gap reviewer while bypassing learned memory and errors."""
    def __init__(self,delegate,owner):self.delegate,self.owner=delegate,owner;self.handoff=None
    def __getattr__(self,name):return getattr(self.delegate,name)
    def explore(self,**kwargs):
        if kwargs.get('memory') is not None:raise RuntimeError('Learned memory leaked into the ablation')
        questions=kwargs.get('review_initial_questions') or kwargs['initial_questions']
        self.handoff=dict(Question=[json.dumps(q.to_dict(),ensure_ascii=False) for q in questions],
                          Finding=[json.dumps(e.to_dict(),ensure_ascii=False) for e in self.owner.refined_first],
                          **{'Errors appear in coding':[]})
        write(self.owner.run_dir/'03_raw_first_round_handoff.json',dict(
            mode='uncompressed_original_records',memory_manager_called=False,coverage_review_enabled=True,handoff=self.handoff))
        return self.delegate.explore(**{**kwargs,'memory':copy.deepcopy(self.handoff)})
    def revise_follow_up(self,**kwargs):
        if self.handoff is None:raise RuntimeError('Missing original-record handoff')
        if kwargs.get('memory') is not None:raise RuntimeError('Learned memory leaked into follow-up revision')
        return self.delegate.revise_follow_up(**{**kwargs,'memory':copy.deepcopy(self.handoff)})

class MemoryAblation(common.TrajStoryAgent):
    def __init__(self,llm,model,source):
        self.source=Path(source) if source is not None else None
        values=read(self.source/'00_config.json') if self.source is not None else common.FrameworkConfig().to_dict();values['memory_manager_enabled']=False
        super().__init__(RawHandoffLabelClient(llm),generator=model,config=NoMemoryConfig(**values))
        self.explorer=RawHandoffExplorer(self.explorer,self)
        self.refined_first=[]
        if self.source is None:
            return
        self.saved_profile=common.load_profile(read(self.source/'01_profile.json'))
        self.planned=[common.load_question(q) for q in read(self.source/'02_initial_questions.json')]
        self.validated=[common.load_evidence(e) for e in read(self.source/'03_validated_initial_evidence.json')]
        self.final_questions=[common.load_question(q) for q in read(self.source/'06_report.json')['questions']]
        self.first_failures=[f for f in read(self.source/'branch_failures.json') if str(f.get('stage','')).startswith('initial_')]
        self.profiler.build=lambda bundle:copy.deepcopy(self.saved_profile)
        self.decomposer.decompose=lambda bundle,profile:copy.deepcopy(self.planned)
    def build_initial_agreement(self, topics, initial, goal, output):
        if self.source is None:
            return super().build_initial_agreement(topics, initial, goal, output)
        return common.replay_agreement(self.source, output)
    def _parallel_map_best_effort(self,items,function,**kwargs):
        if self.source is not None and kwargs.get('stage')=='initial_analysis':
            for q in self.planned:self.history.add(q.question)
            return copy.deepcopy(self.validated),copy.deepcopy(self.first_failures)
        return super()._parallel_map_best_effort(items,function,**kwargs)
    def _verify_and_repair_initial(self,*,questions,**kwargs):
        if self.source is None:
            return super()._verify_and_repair_initial(questions=questions, **kwargs)
        questions[:]=copy.deepcopy(self.final_questions)
        latest=max(self.source.glob('03_initial_review_audit_round_*.json'),key=lambda p:int(p.stem.rsplit('_',1)[1]))
        self.verifier.review_audit=read(latest)
        self.verifier.agreement_topics=read(self.source/latest.name.replace('review_audit','agreement_topics'))
        return copy.deepcopy(self.validated)
    def refine_initial(self,bundle,profile,initial,run_dir):
        if self.source is None:
            saved, enriched = super().refine_initial(bundle,profile,initial,run_dir)
            self.refined_first=copy.deepcopy(enriched)
            return saved,enriched
        # Reuse all first-round refinement, including the compressed realization.
        records=read(self.source/'early/predictions.json')['branches']
        self.initial_refinements={r['question_id']:copy.deepcopy(r) for r in records}
        enriched=copy.deepcopy(initial)
        for e in enriched:e.insight=self.initial_refinements[e.question_id]['insights']['conditional_compressed']
        saved=bundle_from(read(self.source/'03_actual_planning_bundle.json'))
        assert saved.goal==bundle.goal and saved.files==bundle.files
        shutil.copytree(self.source/'early',run_dir/'early')
        self.refined_first=copy.deepcopy(enriched)
        return saved,enriched
    def run(self,bundle,output_dir):
        self.run_dir=Path(output_dir)
        report=super().run(bundle,output_dir)
        assert not (self.run_dir/'03_memory.json').exists() and not (self.run_dir/'03_memory_fallback.json').exists()
        framework=read(self.run_dir/'00_framework.json')
        framework.update(framework='v5_ablation',variant='without_memory',memory_manager_enabled=False,
                         memory_representation='full uncompressed first-round records',coding_error_bank_enabled=False,
                         first_round_replayed_from=str(self.source) if self.source is not None else None,first_round_refinement_reused=self.source is not None)
        write(self.run_dir/'00_framework.json',framework)
        return report

class SchemaOnlyProfile(legacy.TrajectoryDataProfile):
    def to_dict(self):
        # Required internal dataclass defaults never become fabricated prompt facts.
        return dict(profile_kind='raw_schema_and_first_rows_only',files=[dict(
            file_id=f.file_id,path=f.path,role=f.role,columns=[dict(name=c.name,raw_dtype=c.raw_dtype,sample_values=c.sample_values) for c in f.columns]) for f in self.files],
            cross_file_relations=[])

class SchemaOnlyProfiler:
    def __init__(self,sample_rows=3):self.sample_rows=sample_rows
    def build(self,bundle):
        import pandas as pd
        files=[]
        for item in bundle.files:
            if item.path.suffix.lower()!='.csv':raise ValueError('This frozen cohort requires projected CSV inputs')
            frame=pd.read_csv(item.path,nrows=self.sample_rows)
            columns=[legacy.ColumnProfile(name=str(name),physical_type='unprofiled',semantic_role='unprofiled',unique_count=-1,
                statistics={},sample_values=[_native(v) for v in frame[name].tolist()],raw_dtype=str(frame[name].dtype)) for name in frame.columns]
            files.append(legacy.FileProfile(file_id=item.file_id,path=str(item.path),role=item.role,row_count=-1,columns=columns,coverage={},relations=[]))
        return SchemaOnlyProfile(files,[])

class DataProfilingAblation(common.TrajStoryAgent):
    def __init__(self,llm,model,config):
        super().__init__(llm,generator=model,config=common.FrameworkConfig(**config))
        self.profiler=SchemaOnlyProfiler(self.config.sample_rows_per_file)
    def run(self,bundle,output_dir):
        report=super().run(bundle,output_dir)
        run=Path(output_dir);framework=read(run/'00_framework.json')
        framework.update(framework='v5_ablation',variant='without_data_profiling',a0_enabled=False,
                         schema_view='names, raw dtypes and first three original rows only',first_round_replayed=False,
                         profiled_semantics_statistics_coverage_relations=False)
        write(run/'00_framework.json',framework)
        return report
