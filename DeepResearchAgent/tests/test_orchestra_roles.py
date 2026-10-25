"""Check real context isolation and completion auditing without model requests."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src.agent.orchestra_roles import (
    OrchestraInterpreterAgent, OrchestraAnalystAgent, ROLE_NAMES, role_audit,
)
from src.agent.tool_calling_agent import ToolCallingAgent
from src.agent.types import AgentResponse, AgentExtra
from src.session import SessionContext


def test_text_role_has_private_inputs_and_no_tools(tmp_path):
    from src.agent.orchestra_handoffs import store_for, execution_artifact
    agent = OrchestraInterpreterAgent(workdir=str(tmp_path/'agent'/'orchestra_interpreter'),
                                     model_name='test', use_memory=False)
    model = AsyncMock(return_value=SimpleNamespace(success=True, message='{"candidates":[]}', extra=None))
    parent = SessionContext()
    identifier='E_'+'a'*32
    execution_artifact(store_for(agent,parent),identifier,[dict(step=1,args={'code':'print(3)'},observation='Stdout: 3')])
    async def run():
        with patch('src.agent.orchestra_roles.model_manager', new=model):
            await agent(task=json.dumps(dict(instruction='first branch evidence',evidence_ids=[identifier])), ctx=parent)
            await agent(task=json.dumps(dict(instruction='second branch evidence',evidence_ids=[identifier])), ctx=parent)
    asyncio.run(run())
    assert model.await_count == 2
    assert set(model.call_args.kwargs) == {'model', 'messages', 'max_completion_tokens'}
    assert model.call_args.kwargs['max_completion_tokens']==1800
    assert 'first branch' not in str(model.call_args.kwargs['messages'])
    audit = role_audit(tmp_path, parent.id)
    assert len(audit['dispatches']) == 2
    assert audit['separate_role_contexts']
    assert 'orchestra_analyst' in audit['missing_roles']


def test_concurrent_analyst_dispatches_use_private_instances(tmp_path):
    agent = OrchestraAnalystAgent(workdir=str(tmp_path/'agent'/'orchestra_analyst'),
                                 model_name='test', use_memory=False)
    seen = []
    instances = []  # Keep fake executors alive; Python may otherwise reuse id().
    async def fake_call(self, task, files=None, ctx=None, **kwargs):
        instances.append(self)
        seen.append((id(self), self.workdir, ctx.id, task))
        from pathlib import Path
        Path(self.workdir, 'tracer.json').write_text(json.dumps({'sessions': {ctx.id: [
            {'tool': {'actions': [{'name': 'trajstory_analysis', 'args': '{"code":"print(3)"}',
                                  'output': 'Stdout: 3'}]}}]}}),encoding='utf8')
        return AgentResponse(success=True, message='evidence',
                             extra=AgentExtra(data={'done': True, 'result': 'evidence'}))
    async def run():
        with patch.object(ToolCallingAgent, '__call__', new=fake_call):
            await asyncio.gather(agent(task='A', ctx=SessionContext()),
                                 agent(task='B', ctx=SessionContext()))
    asyncio.run(run())
    assert len({x[0] for x in seen}) == 2
    assert len({x[1] for x in seen}) == 2
    assert len({x[2] for x in seen}) == 2
    assert all('Do not write the whole-case' in x[3] for x in seen)


def test_role_audit_cannot_reuse_another_case(tmp_path):
    for role in ROLE_NAMES:
        folder=tmp_path/'agent'/role/'dispatches'/'a'
        folder.mkdir(parents=True)
        (folder/'input.json').write_text(json.dumps(dict(case_session_id='other',
            dispatch_session_id=role)),encoding='utf8')
        (folder/'output.json').write_text('{"success":true}',encoding='utf8')
    assert set(role_audit(tmp_path, 'current')['missing_roles']) == set(ROLE_NAMES)


def test_gpt6_structured_calls_keep_zero_temperature_and_omit_penalty():
    from src.model.openai.chat import ChatOpenAI
    from src.agent.planning_agent import PlanDecision
    from src.message.types import HumanMessage
    client=ChatOpenAI(model='gpt-6-luna',temperature=0,reasoning={'reasoning_effort':'none'})
    request=asyncio.run(client._build_params(messages=[HumanMessage(content='probe')],
                                           response_format=PlanDecision))
    assert request['params']['temperature']==0
    assert request['params']['reasoning_effort']=='none'
    assert 'frequency_penalty' not in request['params']
    assert request['params']['response_format']['type']=='json_schema'


def test_analyst_latest_feedback_does_not_repeat_old_stdout(tmp_path):
    record=SimpleNamespace(tool=None,observation={})
    tracer=SimpleNamespace(add_record=AsyncMock(),save_to_json=AsyncMock())
    tasks=[];calls=[]
    async def messages(self,task,**kwargs):
        tasks.append(task)
        return []
    async def think(self,messages,task_id,step_number,ctx=None,record=None,**kwargs):
        calls.append(step_number)
        if step_number<2:
            record.tool={'actions':[{'name':'trajstory_analysis','output':f'UNIQUE_STDOUT_{step_number}'}]}
            return {'done':False,'result':None,'reasoning':None}
        record.tool={'actions':[{'name':'done','output':'completed'}]}
        return {'done':True,'result':'completed','reasoning':''}
    async def run():
        agent=ToolCallingAgent(workdir=str(tmp_path),model_name='test',use_memory=False,use_todo=False)
        await agent.initialize()
        with patch.object(ToolCallingAgent,'_get_tracer_and_record',new=AsyncMock(return_value=(tracer,record))), \
             patch.object(ToolCallingAgent,'_get_messages',new=messages), \
             patch.object(ToolCallingAgent,'_think_and_tool',new=think):
            await agent(task='focused task',ctx=SessionContext(),tool_feedback_mode='latest')
    asyncio.run(run())
    assert 'UNIQUE_STDOUT_0' in tasks[1]
    assert 'UNIQUE_STDOUT_1' in tasks[2] and 'UNIQUE_STDOUT_0' not in tasks[2]
    assert tracer.add_record.await_count==3


def test_repeated_unable_to_execute_done_stops_after_two_calls(tmp_path):
    record=SimpleNamespace(tool=None,observation={})
    tracer=SimpleNamespace(add_record=AsyncMock(),save_to_json=AsyncMock())
    async def think(self,messages,task_id,step_number,ctx=None,record=None,**kwargs):
        record.tool={'actions':[{'name':'done','output':'Missing path'}]}
        return dict(done=True,result='Missing path',reasoning='No input')
    async def run():
        agent=ToolCallingAgent(workdir=str(tmp_path),model_name='test',use_memory=False,use_todo=False,max_steps=50)
        await agent.initialize()
        with patch.object(ToolCallingAgent,'_get_tracer_and_record',new=AsyncMock(return_value=(tracer,record))), \
             patch.object(ToolCallingAgent,'_get_messages',new=AsyncMock(return_value=[])), \
             patch.object(ToolCallingAgent,'_think_and_tool',new=think):
            response=await agent(task='focused',ctx=SessionContext(),min_analysis_calls=1,
                                 premature_done_repeat_limit=2,tool_feedback_mode='latest')
            assert not response.success
    asyncio.run(run())
    assert tracer.add_record.await_count==2


def test_host_bound_path_is_supplied_when_planner_omits_it(tmp_path):
    from pathlib import Path
    from src.agent.orchestra_handoffs import store_for, write
    agent=OrchestraAnalystAgent(workdir=str(tmp_path/'agent/orchestra_analyst'),model_name='test')
    parent=SessionContext();path='B:/case/input.json';seen=[]
    write(store_for(agent,parent)/'case_context.json',dict(spec_path=path))
    async def execute(self,task,files=None,ctx=None,**kwargs):
        seen.append(task)
        Path(self.workdir,'tracer.json').write_text(json.dumps({'sessions':{ctx.id:[
            {'tool':{'actions':[dict(name='trajstory_analysis',args={'code':'print(3)'},output='Stdout: 3')]}}]}}),encoding='utf8')
        return AgentResponse(success=True,message='Complete',extra=AgentExtra(data=dict(done=True,result='Complete')))
    async def run():
        with patch.object(ToolCallingAgent,'__call__',new=execute):
            await agent(task='Calculate counts; use the exact spec_path from the task.',ctx=parent)
            await agent(task='Calculate counts using '+path,ctx=parent)
    asyncio.run(run())
    assert len(seen)==2 and all(text.count(path)==1 for text in seen)
