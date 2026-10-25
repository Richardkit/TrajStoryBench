"""Six-round insight checkpoints, with a final summary on round 37."""

import sys as _baseline_sys
from pathlib import Path as _BaselinePath
_baseline_repo = next(p for p in _BaselinePath(__file__).resolve().parents if (p / "baselines" / "__init__.py").is_file())
_baseline_sys.path.insert(0, str(_baseline_repo))
import baselines  # Register the shared baseline module location.

import json
from typing import Any
from pydantic.v1 import Field
from langchain.agents import AgentExecutor
from langchain_core.agents import AgentFinish, AgentAction, AgentStep
from langchain.agents.format_scratchpad.tools import format_to_tool_messages
from langchain_core.prompts import ChatPromptTemplate
from baseline_support.json_utils import extract_json_object

MAX_ROUNDS = 37
CHECKPOINTS = {6: 1, 12: 2, 18: 3, 24: 4, 30: 5, 36: 6}
ROUND_POLICY = (
    'INSIGHT CHECKPOINT PROTOCOL: rounds 6,12,18,24,30,36 require at least '
    '1,2,3,4,5,6 cumulative distinct evidence-backed insights respectively. '
    'Checkpoint submissions count as Agent rounds. At each checkpoint you '
    'must return JSON containing the cumulative insight strings and a summary '
    'of findings so far. If the quota is missed, the program immediately ends '
    'with the existing insights and summary; no extra analysis or repair call '
    'is allowed. If all checkpoints pass, round 37 produces the final summary '
    'using the existing insights. Never fabricate findings to satisfy a quota. '
    'Choose analyses autonomously; use computed evidence.\n\n'
)

class CheckpointPandasExecutor(AgentExecutor):
    rounds_used: int = 0
    completed_normally: bool = False
    insights: list[str] = Field(default_factory=list)
    checkpoints: list[dict] = Field(default_factory=list)
    summary: str = ''
    termination_reason: str = ''
    checkpoint_llm: Any = Field(default=None, exclude=True, repr=False)
    checkpoint_prompt: Any = Field(default=None, exclude=True, repr=False)
    progress_callback: Any = Field(default=None, exclude=True, repr=False)

    def _save(self):
        if self.progress_callback:
            self.progress_callback(dict(rounds_used=self.rounds_used, round_limit=37,
                insights=self.insights, summary=self.summary, checkpoints=self.checkpoints,
                termination_reason=self.termination_reason))

    def _parse(self, text):
        try:
            obj = extract_json_object(text)
            values = obj.get('insights')
            if not isinstance(values, list) or not all(isinstance(x,str) and x.strip() for x in values):
                raise ValueError('insights must be a list of nonempty strings')
            # Cumulative list is model-authored; exact duplicates never count twice.
            values = list(dict.fromkeys(x.strip() for x in values))
            if len(values) > 6:
                raise ValueError('At most six insight strings are permitted')
            summary = obj.get('summary')
            if not isinstance(summary,str) or not summary.strip():
                raise ValueError('summary must be a nonempty string')
            return values, summary.strip(), None
        except (ValueError, TypeError) as exc:
            return None, None, str(exc)

    def _finish(self, reason):
        self.completed_normally = True
        self.termination_reason = reason
        if not self.summary:
            self.summary = ('Existing findings: ' + ' '.join(self.insights)) if self.insights else 'No evidence-backed insight was completed.'
        self._save()
        return AgentFinish(return_values={'output':json.dumps(
            {'insights':self.insights,'summary':self.summary},ensure_ascii=False)}, log=reason)

    def _iter_next_step(self, name_to_tool_map, color_mapping, inputs,
                        intermediate_steps, run_manager=None):
        if self.rounds_used >= MAX_ROUNDS:
            raise RuntimeError('No round 38 is allowed')
        self.rounds_used += 1
        round_id = self.rounds_used
        counter = f'CURRENT ROUND: {round_id}/37. Registered insights: {json.dumps(self.insights,ensure_ascii=False)}.\n\n'
        if round_id in CHECKPOINTS or round_id == 37:
            quota = CHECKPOINTS.get(round_id,6)
            if round_id == 37:
                instruction = ('FINAL SUMMARY ROUND. Return JSON with the existing insights unchanged and '
                    'a coherent summary supported by them. Do not call tools or add new insights.\n')
            else:
                instruction = (f'CHECKPOINT ROUND {round_id}: submit at least {quota} cumulative insights now. '
                    'Return ONLY JSON {"insights":[cumulative insight strings],"summary":"summary so far"}. '
                    'No tools are available in this checkpoint round. If fewer insights are supported, '
                    'return only those supported; the program will end with this partial result.\n')
            # Some compatible providers keep issuing tools despite tool_choice=none
            # when the context contains assistant tool calls. Submit progress from
            # a text-only evidence transcript, without tool schemas or tool roles.
            evidence = json.dumps([{'tool':a.tool,'input':a.tool_input,
                                   'observation':str(b)} for a,b in intermediate_steps],
                                  ensure_ascii=False,default=str)
            prompt = self.checkpoint_prompt.invoke({**inputs,
                'input':counter+instruction+inputs['input']+
                    '\n\nExecuted evidence and saved progress:\n'+evidence,
                'agent_scratchpad':[]})
            callbacks = run_manager.get_child() if run_manager else None
            response = self.checkpoint_llm.invoke(prompt, config={'callbacks':callbacks})
            text = response.content if isinstance(response.content,str) else json.dumps(response.content)
            values, summary, error = self._parse(text)
            if values is not None:
                if round_id != 37:
                    self.insights = values
                self.summary = summary
            passed = error is None and len(self.insights) >= quota
            self.checkpoints.append(dict(round=round_id,required_insights=quota,
                insight_count=len(self.insights),passed=passed,parse_error=error,raw_output=text,
                returned_tool_calls=getattr(response,'tool_calls',[])))
            self._save()
            if round_id == 37:
                yield self._finish('final_summary_round_37' if error is None else 'summary_format_error')
            elif not passed:
                yield self._finish(f'checkpoint_quota_missed_round_{round_id}')
            else:
                message = json.dumps({'checkpoint':round_id,'insights':self.insights,
                    'summary_so_far':self.summary},ensure_ascii=False)
                yield AgentStep(action=AgentAction(tool='checkpoint_progress',tool_input=message,
                    log='Checkpoint progress: '+message),observation=message)
            return
        counted_inputs = {**inputs,'input':counter+inputs['input']}
        for step in super()._iter_next_step(name_to_tool_map,color_mapping,counted_inputs,
                                           intermediate_steps,run_manager):
            if isinstance(step,AgentFinish):
                # Before round 37 a model finish is only a progress proposal.
                values, summary, error = self._parse(step.return_values.get('output',''))
                if values is not None:
                    self.insights,self.summary = values,summary
                message = json.dumps({'round':round_id,'insights':self.insights,
                    'note':'Progress saved. Continue until the next checkpoint; final summary is round 37.'},ensure_ascii=False)
                self._save()
                yield AgentStep(action=AgentAction(tool='checkpoint_progress',tool_input=message,
                    log=message),observation=message)
            else:
                yield step
        self._save()

def apply_round_budget(executor, llm, progress_callback=None):
    prompts = [x for x in executor.agent.runnable.steps if isinstance(x,ChatPromptTemplate)]
    if len(prompts) != 1:
        raise RuntimeError('Cannot identify the official Pandas prompt')
    return CheckpointPandasExecutor(agent=executor.agent,tools=executor.tools,
        return_intermediate_steps=True,max_iterations=37,
        max_execution_time=executor.max_execution_time,early_stopping_method='force',
        handle_parsing_errors=executor.handle_parsing_errors,
        checkpoint_llm=llm,
        checkpoint_prompt=prompts[0],progress_callback=progress_callback)
