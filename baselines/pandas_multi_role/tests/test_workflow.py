import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from pandas_multi_role.workflow import Config, PandasRoleWorkflow, validate_citations
from pandas_multi_role.role_runner import offline_factory


class FakeTools(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class WorkflowTests(unittest.TestCase):
    def test_global_budget_includes_all_roles(self):
        self.assertEqual(Config().analysis_calls_per_branch,11)
        self.assertLessEqual(6*Config().analysis_calls_per_branch+15,83)
        for value in [0,20,True,1.5]:
            with self.assertRaises(ValueError):Config(value)

    def test_exact_output_citations(self):
        steps=[dict(observation='count = 3')]
        valid=dict(insight='Three rows.',citations=[dict(step=1,quote='count = 3')],limitations=[])
        validate_citations(valid,steps)
        for citation in [dict(step=2,quote='3'),dict(step=1,quote='count = 4'),dict(step=True,quote='3')]:
            with self.assertRaises(ValueError):
                validate_citations({**valid,'citations':[citation]},steps)

    def test_complete_separate_role_flow_and_original_frames_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);frames=[pd.DataFrame({'x':[1,2,3]})]
            base=offline_factory(3,'df')
            def factory(role,recorder):
                if role.startswith('analyst'):
                    return FakeTools(callbacks=[recorder],responses=[AIMessage(content='',tool_calls=[
                        dict(name='python_repl_ast',args={'query':"df['branch_marker']=1; print(len(df))"},id='act')]),
                        AIMessage(content='unverified analyst note should not leak')])
                return base(role,recorder)
            wf=PandasRoleWorkflow(factory,offline=True)
            state=wf.run(SimpleNamespace(goal='Count rows',context='Test only'),frames,
                         [dict(dataframe='df',file_id='main',role='data')],out)
            self.assertEqual(state['insights_count'],6)
            self.assertEqual(state['requests_used'],27)
            self.assertEqual(state['tool_calls'],6)
            self.assertFalse(state['eligible_for_scoring'])
            self.assertNotIn('branch_marker',frames[0].columns)
            interp=json.loads((out/'roles/interpreter_Q1/input.json').read_text())
            self.assertNotIn('unverified analyst note',str(interp))
            self.assertNotIn('tables',interp['payload'])
            self.assertIn('steps',interp['payload'])
            planner=json.loads((out/'roles/planner_2/input.json').read_text())
            self.assertEqual(len(planner['payload']['prior_evidence']),3)
            self.assertNotIn('steps',planner['payload']['prior_evidence'][0])
            synth=json.loads((out/'roles/synthesizer/input.json').read_text())
            self.assertEqual(set(synth['payload']),{'goal','approved_insights'})
            self.assertEqual(len(wf.recorders),21)

    def test_branch_limit_uses_executed_evidence_without_extra_analyst_summary_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=offline_factory(3,'df')
            def factory(role,recorder):
                if role.startswith('analyst'):
                    return FakeTools(callbacks=[recorder],responses=[AIMessage(content='',tool_calls=[
                        dict(name='python_repl_ast',args={'query':'print(len(df))'},id='act')])]*7)
                return base(role,recorder)
            wf=PandasRoleWorkflow(factory,config=Config(60),offline=True)
            state=wf.run(SimpleNamespace(goal='Count',context=''),[pd.DataFrame({'x':[1,2,3]})],
                         [dict(dataframe='df',file_id='main',role='data')],Path(tmp))
            self.assertEqual(state['requests_used'],57)
            self.assertEqual(state['tool_calls'],42)
            self.assertEqual(state['insights_count'],6)

    def test_unknown_parent_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            def factory(role,recorder):
                return FakeTools(callbacks=[recorder],responses=[AIMessage(content=json.dumps(dict(questions=[
                    dict(question=f'q{i}',parent_evidence_ids=['invented']) for i in range(3)])))])
            wf=PandasRoleWorkflow(factory,offline=True)
            with self.assertRaises(ValueError):
                wf._plan(1,SimpleNamespace(goal='Count',context=''),[],[],Path(tmp))

    def test_synthesis_cannot_add_unverified_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=offline_factory(3,'df')
            def factory(role,recorder):
                if role=='synthesizer':
                    return FakeTools(callbacks=[recorder],responses=[AIMessage(content=json.dumps(dict(
                        selected_evidence_ids=['invented'],summary='unsupported')))])
                return base(role,recorder)
            wf=PandasRoleWorkflow(factory,offline=True)
            with self.assertRaises(ValueError):
                wf.run(SimpleNamespace(goal='Count',context=''),[pd.DataFrame({'x':[1,2,3]})],
                       [dict(dataframe='df',file_id='main',role='data')],Path(tmp))


if __name__=='__main__':unittest.main()
