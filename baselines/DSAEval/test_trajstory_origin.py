"""Offline fidelity, context, input, transport and output checks for the origin adapter."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from trajstory_origin import (ROOT,UPSTREAM,Transport,NetworkInterrupted,load_cases,
    load_upstream,parse_report,run_case,verify_upstream)

def response(code=None,insights=None,summary='A summary'):
    if code is not None:
        message=NS(role='assistant',content=None,tool_calls=[NS(id='call_'+str(len(code)),type='function',function=NS(name='jupyter_code_interpreter',arguments=json.dumps(dict(code=code))))])
    else:message=NS(role='assistant',content='## Final Output\n**4. Results & Explanation**\n```json\n'+json.dumps(dict(insights=insights or ['Measured finding'],summary=summary))+'\n```',tool_calls=None)
    return NS(choices=[NS(message=message)],usage=NS(prompt_tokens=10,completion_tokens=5,total_tokens=15))

class Sandbox:
    instances=[]
    def __init__(self,timeout):self.executions=[];self.instances.append(self)
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def execute_code(self,code):self.executions.append(code);return 'Measured output\n',[]

class OriginTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.Agent,_,cls.prompts=load_upstream()
    def execute(self,responses,budget=83):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);csv=folder/'input_data.csv';csv.write_text('x\n1\n')
            requests=[];answers=iter(responses)
            def create(**request):requests.append(copy.deepcopy(request));return next(answers)
            client=NS(chat=NS(completions=NS(create=create)))
            args=NS(model='OFFLINE',max_model_calls=budget,code_timeout=180,is_multimodal=False,verbose=False)
            Sandbox.instances.clear()
            row=run_case(dict(case_id='demo',goal='Analyze rows',sources=[dict(source=str(csv),role='table')]),folder/'out',args,client,{},(self.Agent,Sandbox,self.prompts))
            self.assertEqual(len(Sandbox.instances),1)
            self.assertTrue((folder/'out/analysis.ipynb').exists())
            self.assertFalse((folder/'out/branches').exists())
            return row,requests,list(Sandbox.instances[0].executions)
    def test_upstream_files_match_pinned_git_objects(self):
        origin=verify_upstream()
        for name in origin['sha256']:
            expected=subprocess.check_output(['git','-C',str(ROOT/'DSAEval'),'show',origin['commit']+':'+name])
            self.assertEqual((UPSTREAM/name).read_bytes(),expected)
    def test_one_conversation_kernel_and_unrestricted_tool_iterations(self):
        row,requests,executions=self.execute([response('x=1'),response('print(x)'),response(insights=['One','Two'])])
        self.assertEqual(row['agent_invocations'],1)
        self.assertEqual(row['status'],'partial')
        self.assertEqual(row['pred_insights'],['One','Two'])
        self.assertEqual(row['token_usage']['llm_calls'],3)
        self.assertIn('x=1',executions);self.assertIn('print(x)',executions)
        self.assertEqual([m['role'] for m in requests[0]['messages']],['system','user'])
        self.assertTrue(any(m['role']=='tool' for m in requests[1]['messages']))
        self.assertTrue(all('tools' in request for request in requests))
        self.assertTrue(all('question' not in request['tools'][0]['function']['parameters']['properties'] for request in requests))
    def test_no_forced_report_or_extraction_request_at_budget_limit(self):
        row,requests,_=self.execute([response('x=1'),response('print(x)')],budget=2)
        self.assertEqual(len(requests),2);self.assertFalse(row['native_finished']);self.assertEqual(row['pred_insights'],[])
        self.assertTrue(all(request['tool_choice']=='auto' for request in requests))
    def test_missing_summary_and_less_than_six_accepted(self):
        row,requests,_=self.execute([response(insights=['One'],summary='')])
        self.assertEqual(row['pred_insights'],['One']);self.assertEqual(row['pred_summary'],'')
        self.assertTrue(row['eligible_for_scoring']);self.assertEqual(len(requests),1)
    def test_projected_inputs_required_gold_excluded(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'raw.csv').write_text('x\n1\n')
            cp=root/'case.json';cp.write_text(json.dumps(dict(dataset_csv_path=str(root/'raw.csv'),metadata=dict(goal='Analyze'),insights=['SECRET'],summary='SECRET_SUMMARY')))
            mp=root/'manifest.json';mp.write_text(json.dumps(dict(cases=[dict(case_id='demo',json_path=str(cp))])))
            with self.assertRaises(FileNotFoundError):load_cases(mp,['demo'])
            (root/'input_raw.csv').write_text('x\n1\n');cases=load_cases(mp,['demo'])
            self.assertNotIn('SECRET',json.dumps(cases));self.assertTrue(cases[0]['sources'][0]['source'].endswith('input_raw.csv'))
    def test_network_failures_free_and_stop_after_five(self):
        with tempfile.TemporaryDirectory() as temp,patch('trajstory_origin.time.sleep'):
            calls=[]
            def create(**request):calls.append(request);raise ConnectionError('offline')
            t=Transport(NS(chat=NS(completions=NS(create=create))),temp,83,{})
            with self.assertRaises(NetworkInterrupted):t.create(model='OFFLINE')
            self.assertEqual(len(calls),5);self.assertEqual(t.charged,0);self.assertEqual(t.usage()['network_failures'],5)
            with self.assertRaises(NetworkInterrupted):t.create(model='OFFLINE')
            self.assertEqual(len(calls),5)
    def test_success_resets_network_streak_and_retries_same_input(self):
        with tempfile.TemporaryDirectory() as temp,patch('trajstory_origin.time.sleep'):
            outcomes=iter([ConnectionError()]*4+[response()]+[ConnectionError()]*4+[response()]);calls=[]
            def create(**request):
                calls.append(copy.deepcopy(request));value=next(outcomes)
                if isinstance(value,Exception):raise value
                return value
            t=Transport(NS(chat=NS(completions=NS(create=create))),temp,2,{})
            t.create(messages=['same']);self.assertEqual(t.streak,0)
            t.create(messages=['next']);self.assertEqual(t.charged,2)
            self.assertEqual(calls[:5],[dict(messages=['same'])]*5)
            with self.assertRaisesRegex(RuntimeError,'BudgetExceeded'):t.create()

if __name__=='__main__':unittest.main()
