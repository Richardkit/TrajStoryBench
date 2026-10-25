"""Small live interface probe on a synthetic table, never benchmark data."""

import sys as _baseline_sys
from pathlib import Path as _BaselinePath
_baseline_repo = next(p for p in _BaselinePath(__file__).resolve().parents if (p / "baselines" / "__init__.py").is_file())
_baseline_sys.path.insert(0, str(_baseline_repo))
import baselines  # Register the shared baseline module location.

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from pandas_multi_role.react_runner import Worker,validate_code,write
from trajstoryagent_v1.llm import OpenAICompatibleClient
from trajstoryagent_v1.json_utils import extract_json_object
from langchain_core.language_models.llms import LLM
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_experimental.tools.python.tool import PythonAstREPLTool
import pandas as pd

class ProbeLLM(LLM):
    client:Any
    folder:Any
    calls:int=0
    @property
    def _llm_type(self):return 'mimo_probe'
    def _call(self,prompt,stop=None,run_manager=None,**kwargs):
        self.calls+=1
        result=self.client.complete(system='Follow the requested ReAct format.',user=prompt,json_mode=False)
        write(self.folder/f'llm_{self.calls}.json',dict(prompt=prompt,response=result,metadata=self.client.last_response_metadata))
        for s in stop or []:
            if s in result:result=result.split(s,1)[0]
        return result

class ProbeTool(PythonAstREPLTool):
    worker_handle:Any
    def _run(self,query,run_manager=None):return self.worker_handle.call(query)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=ROOT/'results/pandas/interface_probe');args=p.parse_args();folder=args.output_dir;folder.mkdir(parents=True,exist_ok=True)
    import winreg
    if not os.getenv('XIAOMI_API_KEY'):
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,'Environment') as k:os.environ['XIAOMI_API_KEY']=winreg.QueryValueEx(k,'XIAOMI_API_KEY')[0]
    df=pd.DataFrame({'x':[1,2,3]});df.to_csv(folder/'sample.csv',index=False)
    write(folder/'catalog.json',[dict(file_id='main',path=str((folder/'sample.csv').resolve()),role='synthetic probe')])
    worker=Worker(folder/'catalog.json',folder)
    try:
        # Check persistence and rejection before the live call; these are adapter tests.
        assert worker.call('probe_value = 7; print(probe_value)').strip()=='7'
        assert worker.call('probe_value + 1').strip()=='8'
        assert 'ValueError' in worker.call('import os')
        native=OpenAICompatibleClient(model='mimo-v2.6-flash',reasoning_effort='none',temperature=0,max_tokens=1000)
        llm=ProbeLLM(client=native,folder=folder)
        agent=create_pandas_dataframe_agent(llm,df,agent_type='zero-shot-react-description',allow_dangerous_code=True,
            max_iterations=5,return_intermediate_steps=True,agent_executor_kwargs={'handle_parsing_errors':True})
        agent.tools=[ProbeTool(worker_handle=worker)]
        result=agent.invoke({'input':'Use the Python tool to compute the sum of column x in df. Then give Final Answer as a JSON object with key sum and the computed number.'})
        output=extract_json_object(result['output'])
        assert output.get('sum')==6 and len(result['intermediate_steps'])>=1
        assert not native.last_response_metadata.get('reasoning_present')
        write(folder/'result.json',dict(passed=True,answer=output,llm_calls=llm.calls,steps=[dict(tool=a.tool,input=a.tool_input,observation=b) for a,b in result['intermediate_steps']],worker_persistence='passed',guard_rejection='passed'))
        print(json.dumps(dict(passed=True,sum=6,llm_calls=llm.calls,tool_steps=len(result['intermediate_steps']))))
    finally:worker.close()

if __name__=='__main__':main()
