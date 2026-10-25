"""Offline evidence handoff, context isolation, and whole-case budget checks."""
import copy
import json
import tempfile
import types
import unittest
from unittest.mock import patch
from pathlib import Path
from trajstory_multi_role import RoleWorkflow,load_input_cases,ConsecutiveNetworkFailures,DEFAULT_MODEL_BUDGET


class Sandbox:
    instances=[]
    def __init__(self,timeout,kernel_name):
        self.last_execution_error=False
        self.instances.append(self)
    def __enter__(self): return self
    def __exit__(self,*args): return False
    def execute_code(self,code): return "2 measured rows\n",[]


class Factory:
    def __init__(self, *, exhaust=False, bad_interpreter=None, summary=True, fail_synthesis=False):
        self.requests={};self.clients=[];self.exhaust=exhaust;self.bad=bad_interpreter
        self.summary=summary;self.fail_synthesis=fail_synthesis
    def __call__(self,role):
        self.clients.append(role)
        def create(**request):
            self.requests.setdefault(role,[]).append(copy.deepcopy(request))
            n=len(self.requests[role])
            payload=json.loads(request['messages'][-1]['content']) if not role.startswith('analyst') else {}
            if role.startswith('analyst') and (n==1 or self.exhaust):
                message=types.SimpleNamespace(role='assistant',content=None,tool_calls=[
                    types.SimpleNamespace(id=f'call{n}',type='function',function=types.SimpleNamespace(
                        name='jupyter_code_interpreter',arguments=json.dumps({'code':"print('2 measured rows')"})))])
            else:
                if role.startswith('planner'):
                    parents=[x['evidence_id'] for x in payload['approved_prior_evidence']]
                    result=dict(questions=[dict(question=f'Question {i}',parent_evidence_ids=parents[:1]) for i in range(3)])
                elif role.startswith('analyst'):result=None
                elif role.startswith('interpreter') or role.startswith('verifier'):
                    result=dict(insight='There are 2 measured rows.',citations=[dict(step=1,quote='2 measured rows')],limitations=[])
                    if role==self.bad:result['citations'][0]['quote']='UNEXECUTED NUMBER 100'
                    if role.startswith('verifier'):result.update(status='supported',reason='output confirms count')
                else:
                    if self.fail_synthesis:raise RuntimeError('offline non-network failure')
                    result=dict(selected_evidence_ids=[x['evidence_id'] for x in payload['approved_evidence']],
                        report_markdown='## Final Output\n1. Task Understanding\n2. Approach Summary\n3. Key Implementation\n4. Results & Explanation')
                    if self.summary:result['summary']='Six coherent findings about the same measured count.'
                message=types.SimpleNamespace(role='assistant',content=json.dumps(result) if result is not None else 'Measured computation complete.',tool_calls=None)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)],
                usage=types.SimpleNamespace(prompt_tokens=10,completion_tokens=2,total_tokens=12))
        return types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))


class Tests(unittest.TestCase):
    def test_projected_inputs_required_without_raw_csv_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);raw=root/'points.csv';raw.write_text('raw_only\n99\n')
            cp=root/'case.json';cp.write_text(json.dumps(dict(dataset_csv_path=str(raw),metadata=dict(goal='Check data'))))
            mp=root/'manifest.json';mp.write_text(json.dumps(dict(cases=[dict(case_id='demo',json_path=str(cp))])))
            with self.assertRaises(FileNotFoundError):load_input_cases(mp,['demo'])
            projected=root/'input_points.csv';projected.write_text('x\n1\n')
            self.assertEqual(load_input_cases(mp,['demo'])[0]['sources'][0]['source'],str(projected))

    def run_workflow(self,factory,budget=60):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);csv=root/'data.csv';csv.write_text('x\n1\n2\n',encoding='utf8')
            args=types.SimpleNamespace(max_iterations=budget,model='OFFLINE',generation_kwargs={},
                is_multimodal=False,code_timeout=30,kernel_name='dsa_eval',verbose=False)
            Sandbox.instances.clear()
            workflow=RoleWorkflow(factory,args,sandbox_factory=Sandbox)
            row=workflow.run(dict(case_id='demo',goal='Check row count',sources=[dict(source=str(csv),role='data')]),root/'out')
            self.assertEqual(len(Sandbox.instances),6)
            self.assertEqual(len(list((root/'out/branches').glob('*/analysis.ipynb'))),6)
            return row,factory.requests

    def test_isolated_roles_explicit_handoff_and_coherent_insights(self):
        factory=Factory();row,requests=self.run_workflow(factory)
        self.assertEqual(row['status'],'completed')
        self.assertEqual(row['token_usage']['llm_calls'],27)
        self.assertEqual(len(factory.clients),21)
        self.assertEqual(len(row['pred_insights']),6)
        # Coherent/repeated insight text is permitted; never collapse by theme.
        self.assertEqual(len(set(row['pred_insights'])),1)
        for role,calllist in requests.items():
            if not role.startswith('analyst'):
                self.assertEqual(len(calllist),1)
                self.assertEqual([m['role'] for m in calllist[0]['messages']],['system','user'])
                self.assertNotIn('tools',calllist[0])
            else:
                self.assertEqual(len(calllist[0]['messages']),2)
                self.assertIn('tools',calllist[0])
                self.assertNotIn('question',calllist[0]['tools'][0]['function']['parameters']['properties'])
        second=json.loads(requests['planner_2'][0]['messages'][1]['content'])
        self.assertEqual(len(second['approved_prior_evidence']),3)
        first=json.loads(requests['planner_1'][0]['messages'][1]['content'])
        self.assertEqual(first['approved_prior_evidence'],[])
        self.assertNotIn('head',first);self.assertNotIn('columns',first)
        interpreted=json.loads(requests['interpreter_Q1'][0]['messages'][1]['content'])
        self.assertNotIn('parent_evidence',interpreted)
        self.assertEqual(interpreted['steps'][0]['observation'].strip(),'2 measured rows')

    def test_citation_failure_retains_other_branches(self):
        row,requests=self.run_workflow(Factory(bad_interpreter='interpreter_Q1'))
        self.assertEqual(row['status'],'partial');self.assertTrue(row['eligible_for_scoring'])
        self.assertEqual(len(row['pred_insights']),5)
        self.assertNotIn('verifier_Q1',requests)

    def test_missing_summary_is_eligible(self):
        row,_=self.run_workflow(Factory(summary=False))
        self.assertEqual(row['status'],'partial');self.assertEqual(row['pred_summary'],'')
        self.assertTrue(row['eligible_for_scoring']);self.assertEqual(len(row['pred_insights']),6)

    def test_synthesis_failure_retains_approved_insights(self):
        row,_=self.run_workflow(Factory(fail_synthesis=True))
        self.assertEqual(row['status'],'partial');self.assertEqual(len(row['pred_insights']),6)
        self.assertEqual(row['token_usage']['llm_calls'],27)
        self.assertEqual(row['token_usage']['missing_usage_calls'],1)

    def test_exhausted_analysts_do_not_exceed_case_budget(self):
        row,requests=self.run_workflow(Factory(exhaust=True))
        self.assertEqual(row['token_usage']['llm_calls'],57)
        self.assertTrue(all(len(req)==7 for role,req in requests.items() if role.startswith('analyst')))

    def test_global_guard_counts_failed_requests_and_stops_before_call_61(self):
        with tempfile.TemporaryDirectory() as temp:
            args=types.SimpleNamespace(max_iterations=60,model='OFFLINE',generation_kwargs={})
            f=Factory();w=RoleWorkflow(f,args,sandbox_factory=Sandbox);w.out=Path(temp);w.calls=60
            with self.assertRaisesRegex(RuntimeError,'WholeCaseModelBudgetExceeded'):
                w._client('planner_1').chat.completions.create()
            self.assertEqual(f.requests,{})

    def test_83_budget_allocation(self):
        self.assertEqual(DEFAULT_MODEL_BUDGET,83)
        row,requests=self.run_workflow(Factory(exhaust=True),budget=83)
        self.assertEqual(row['max_model_calls_per_case'],83)
        self.assertEqual(row['token_usage']['llm_calls'],81)
        self.assertTrue(all(len(req)==11 for role,req in requests.items() if role.startswith('analyst')))

    def test_network_retries_free_and_success_resets_streak_at_budget_edge(self):
        with tempfile.TemporaryDirectory() as temp,patch('trajstory_multi_role.time.sleep'):
            outcomes=iter([ConnectionError()]*4+[None]+[ConnectionError()]*4+[None])
            attempted=[]
            def create(**request):
                attempted.append(copy.deepcopy(request));error=next(outcomes)
                if error is not None:raise error
                return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content='ok',tool_calls=None))],usage=None)
            factory=lambda role:types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
            w=RoleWorkflow(factory,types.SimpleNamespace(max_iterations=83));w.out=Path(temp);w.calls=81
            call=w._client('planner_1').chat.completions.create
            call(messages=['same input']);self.assertEqual(w.calls,82);self.assertEqual(w.consecutive_network_failures,0)
            call(messages=['next input']);self.assertEqual(w.calls,83);self.assertEqual(w.consecutive_network_failures,0)
            self.assertEqual(len(w.records),10)
            self.assertEqual(sum(not r['counts_toward_budget'] for r in w.records),8)
            self.assertEqual(attempted[:5],[{'messages':['same input']}]*5)
            self.assertEqual(len(list(Path(temp).glob('roles/planner_1/call_*/request.json'))),10)
            with self.assertRaisesRegex(RuntimeError,'WholeCaseModelBudgetExceeded'):call()
            self.assertEqual(len(attempted),10)

    def test_five_network_failures_interrupt_case_and_skip_remaining_roles(self):
        with tempfile.TemporaryDirectory() as temp,patch('trajstory_multi_role.time.sleep'):
            roles=[]
            def factory(role):
                roles.append(role)
                def create(**request):raise ConnectionError('offline network failure')
                return types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
            args=types.SimpleNamespace(max_iterations=83,model='OFFLINE',generation_kwargs={})
            w=RoleWorkflow(factory,args,sandbox_factory=Sandbox)
            with self.assertRaises(ConsecutiveNetworkFailures):
                w.run(dict(case_id='demo',goal='Check',sources=[]),Path(temp)/'out')
            row=json.loads((Path(temp)/'out/result.json').read_text(encoding='utf8'))
            self.assertEqual(roles,['planner_1'])
            self.assertEqual(row['status'],'interrupted')
            self.assertEqual(row['token_usage']['llm_calls'],0)
            self.assertEqual(row['token_usage']['model_calls_started'],5)
            self.assertEqual(row['token_usage']['network_failures'],5)


if __name__=='__main__':unittest.main()
