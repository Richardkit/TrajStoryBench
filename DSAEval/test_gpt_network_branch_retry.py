"""Offline checks that branch recovery resumes saved evidence without rerunning analysis."""
import json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_dsaeval_gpt_network_branch_retry import recovered_text

class RecoveryTests(unittest.TestCase):
 def recover(self,role):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);old=root/'original';out=root/'retry'
   steps=[dict(code="print('2 rows')",observation='2 rows\n',execution_error=False)]
   source=dict(goal='Count rows',question=dict(question='How many?',question_id='Q1'),steps=steps)
   candidate=dict(insight='There are 2 rows.',citations=[dict(step=1,quote='2 rows')],limitations=[])
   review=dict(candidate,status='supported')
   def write(p,obj):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj),encoding='utf8')
   write(old/'branches/Q1/execution.json',dict(steps=steps,analyst_note='Existing analysis',log={}))
   write(old/'roles/interpreter_Q1/input.json',dict(system='original interpreter',payload=source))
   write(old/'roles/interpreter_Q1/output.json',candidate)
   before=(old/'branches/Q1/execution.json').read_bytes()
   call=Mock(side_effect=[candidate,review] if role=='interpreter_Q1' else [review])
   w=SimpleNamespace(out=out,spec=dict(details=[dict(branch='Q1',last_network_role=role)]),_call=call)
   item=recovered_text(w,old,'Q1',dict(goal='Count rows'))
   self.assertEqual(item['insight'],candidate['insight'])
   self.assertEqual(before,(old/'branches/Q1/execution.json').read_bytes())
   self.assertTrue((out/'branches/Q1/result.json').exists())
   return [args.args[0] for args in call.call_args_list]
 def test_resume_interpreter_and_verifier_only(self):
  self.assertEqual(self.recover('interpreter_Q1'),['interpreter_Q1','verifier_Q1'])
 def test_resume_verifier_only(self):
  self.assertEqual(self.recover('verifier_Q1'),['verifier_Q1'])
if __name__=='__main__':unittest.main()
