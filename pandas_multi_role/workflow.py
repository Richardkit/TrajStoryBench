import json
from dataclasses import dataclass
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_experimental.agents import create_pandas_dataframe_agent
from baseline_support.json_utils import extract_json_object, write_json
from scripts.pandas_repl_compat import apply_repl_compat
from scripts.pandas_native_budget import validate_answer
from scripts.pandas_usage_logging import ProviderUsageRecorder
from . import prompts
from .network_requests import ConsecutiveNetworkFailures

PROTOCOL = 'pandas_separate_context_roles_two_rounds_v1'


@dataclass(frozen=True)
class Config:
    max_model_calls: int = 83

    def __post_init__(self):
        if type(self.max_model_calls) is not int or self.max_model_calls < 21:
            raise ValueError('total model-call budget must be an integer >=21')

    @property
    def analysis_calls_per_branch(self):
        # 2 planner + 6 interpreter + 6 verifier + 1 synthesis calls.
        return (self.max_model_calls - 15) // 6


def validate_citations(payload, steps):
    text = payload.get('insight')
    if not isinstance(text, str) or len(text.split()) > 180:
        raise ValueError('insight must be a string of at most 180 words')
    citations = payload.get('citations')
    if not isinstance(citations, list) or (text.strip() and not citations):
        raise ValueError('nonempty insights require execution citations')
    for citation in citations:
        if not isinstance(citation, dict):
            raise ValueError('invalid citation')
        n, quote = citation.get('step'), citation.get('quote')
        if type(n) is not int or not 1 <= n <= len(steps):
            raise ValueError('unknown execution step')
        if not isinstance(quote, str) or not quote.strip() or quote not in str(steps[n-1]['observation']):
            raise ValueError('citation excerpt is absent from tool output')
    if not isinstance(payload.get('limitations', []), list) or not all(
            isinstance(x, str) for x in payload.get('limitations', [])):
        raise ValueError('limitations must be strings')
    return payload


class PandasRoleWorkflow:
    def __init__(self, model_factory, *, config=None, offline=False):
        self.factory = model_factory
        self.config = config or Config()
        self.offline = offline
        self.recorders = {}
        self.tool_calls = 0

    def _model(self, role, directory):
        recorder = ProviderUsageRecorder(directory)
        self.recorders[role] = recorder
        return self.factory(role, recorder)

    def _call(self, role, system, payload, out):
        # A fresh model wrapper and fresh messages: no implicit shared role history.
        directory = out/'roles'/role
        write_json(directory/'input.json', dict(role=role, system=system, payload=payload))
        model = self._model(role, directory)
        response = model.invoke([SystemMessage(content=system), HumanMessage(
            content=json.dumps(payload, ensure_ascii=False, default=str))])
        if getattr(response, 'tool_calls', None):
            raise ValueError('non-computation roles cannot request tools')
        result = extract_json_object(response.content)
        write_json(directory/'output.json', result)
        return result

    def _plan(self, number, bundle, catalog, prior, out):
        payload = dict(round=number, goal=bundle.goal, context=bundle.context, tables=catalog,
                       prior_evidence=prior)
        result = self._call(f'planner_{number}', prompts.PLANNER, payload, out)
        questions = result.get('questions')
        if not isinstance(questions, list) or len(questions) != 3:
            raise ValueError('planner must return exactly three questions')
        seen, allowed = set(), {e['evidence_id'] for e in prior}
        for i, q in enumerate(questions, 1):
            text = q.get('question')
            if not isinstance(text, str) or not text.strip() or text.strip().casefold() in seen:
                raise ValueError('planner questions must be nonempty and distinct')
            seen.add(text.strip().casefold())
            parents = q.get('parent_evidence_ids', [])
            if not isinstance(parents, list) or not all(isinstance(x, str) for x in parents) or not set(parents) <= allowed:
                raise ValueError('unknown parent evidence')
            q.update(question_id=('Q' if number == 1 else 'F')+str(i), parent_evidence_ids=parents)
        write_json(out/f'round_{number}_questions.json', questions)
        return questions

    def _branch(self, question, bundle, catalog, frames, prior, out):
        qid = question['question_id']; directory = out/'branches'/qid
        parents = [e for e in prior if e['evidence_id'] in question['parent_evidence_ids']]
        payload = dict(goal=bundle.goal, context=bundle.context, question=question, tables=catalog,
                       parent_evidence=parents)
        write_json(directory/'analyst_input.json', payload)
        task = prompts.ANALYST.format(limit=self.config.analysis_calls_per_branch) + '\n' + json.dumps(payload, ensure_ascii=False, default=str)
        write_json(out/'roles'/('analyst_'+qid)/'input.json',dict(role='analyst_'+qid,task=task))
        model = self._model('analyst_'+qid, out/'roles'/('analyst_'+qid))
        # Private copies and private REPL namespaces for every branch/round.
        local = [df.copy(deep=True) for df in frames]
        agent = apply_repl_compat(create_pandas_dataframe_agent(model,
            local if len(local)>1 else local[0], agent_type='tool-calling',
            allow_dangerous_code=True, return_intermediate_steps=True,
            max_iterations=self.config.analysis_calls_per_branch, max_execution_time=None))
        result = agent.invoke({'input':task})
        steps = [dict(tool=a.tool, input=a.tool_input, observation=str(b), log=a.log)
                 for a,b in result['intermediate_steps']]
        self.tool_calls += len(steps)
        write_json(directory/'execution.json', dict(steps=steps, analyst_note=result['output']))
        write_json(out/'roles'/('analyst_'+qid)/'output.json',dict(execution_path=str(directory/'execution.json'),
                  analyst_note=result['output']))
        if not steps:
            raise ValueError('no executed evidence')
        source = dict(goal=bundle.goal, question=question, steps=steps)
        candidate = self._call('interpreter_'+qid, prompts.INTERPRETER, source, out)
        validate_citations(candidate, steps)
        if not candidate['insight'].strip():
            write_json(directory/'result.json', dict(state='no_supported_insight', candidate=candidate))
            return None
        review = self._call('verifier_'+qid, prompts.VERIFIER, dict(**source,candidate=candidate), out)
        status = review.get('status')
        if status not in ('supported','revised','rejected'):
            raise ValueError('unknown verifier status')
        validate_citations(review, steps)
        if status == 'supported' and review['insight'] != candidate['insight']:
            raise ValueError('supported insight must be unchanged')
        if status == 'rejected':
            if review['insight'].strip():
                raise ValueError('rejected insight must be empty')
            write_json(directory/'result.json', dict(state='rejected', candidate=candidate, review=review))
            return None
        if not review['insight'].strip():
            raise ValueError('approved insight must be nonempty')
        record = dict(evidence_id='E_'+qid, question_id=qid, question=question['question'],
                      insight=review['insight'], citations=review['citations'],
                      limitations=list(dict.fromkeys(candidate.get('limitations',[])+review.get('limitations',[]))),
                      execution_path=str(directory/'execution.json'), verifier_status=status)
        write_json(directory/'result.json', dict(state='approved', evidence=record))
        return record

    def run(self, bundle, frames, tables, out):
        out = Path(out)
        catalog = [{**table, 'rows':len(df), 'columns':[
            dict(name=str(c),dtype=str(df[c].dtype)) for c in df.columns],
            'head':df.head(5).to_dict('records')} for table,df in zip(tables,frames)]
        evidence, failures = [], []
        self.recorders.clear()
        self.tool_calls = 0
        for number in (1,2):
            write_json(out/'status.json', dict(state='planning',round=number,protocol=PROTOCOL))
            prior = list(evidence)
            questions = self._plan(number,bundle,catalog,prior,out)
            for q in questions:
                write_json(out/'status.json', dict(state='analyzing',round=number,branch=q['question_id'],protocol=PROTOCOL))
                try:
                    item = self._branch(q,bundle,catalog,frames,prior,out)
                    if item:evidence.append(item)
                except ConsecutiveNetworkFailures:
                    raise
                except Exception as exc:
                    # Do not persist provider exception details or trigger retries.
                    failure = dict(question_id=q['question_id'],error_type=type(exc).__name__)
                    failures.append(failure)
                    write_json(out/'branches'/q['question_id']/'failure.json',failure)
        answer = self._call('synthesizer',prompts.SYNTHESIZER,dict(goal=bundle.goal,
                            approved_insights=evidence),out)
        ids = answer.get('selected_evidence_ids');by_id={e['evidence_id']:e for e in evidence}
        if not isinstance(ids,list) or not all(isinstance(x,str) for x in ids) or len(ids)>6 or len(set(ids))!=len(ids) or not set(ids)<=set(by_id):
            raise ValueError('synthesizer selected invalid evidence IDs')
        selected = [by_id[x] for x in ids]
        parsed = dict(insights=[e['insight'] for e in selected],summary=answer.get('summary'))
        kind = validate_answer(parsed)
        summaries = {r:c.summary() for r,c in self.recorders.items()}
        calls = sum(x['requests_used'] for x in summaries.values())
        if calls > self.config.max_model_calls:
            raise RuntimeError('global call budget exceeded')
        usage = dict(source='provider_reported_usage',model_calls_started=calls,
            calls_with_usage=sum(x['calls_with_usage'] for x in summaries.values()),
            missing_usage_calls=sum(x['missing_usage_calls'] for x in summaries.values()),
            complete=all(x['complete'] for x in summaries.values()),
            **{k:sum(x[k] for x in summaries.values()) for k in ('prompt_tokens','completion_tokens','total_tokens')})
        write_json(out/'model_usage.json',dict(summary=usage,roles=summaries))
        write_json(out/'insights.json',dict(insights=selected,summary=parsed['summary']))
        write_json(out/'parsed_output.json',dict(answer=parsed,output_kind=kind))
        write_json(out/'branch_failures.json',failures)
        state = dict(state='returned',protocol=PROTOCOL,requests_used=calls,request_limit=self.config.max_model_calls,
                     tool_calls=self.tool_calls,
                     output_kind=kind,insights_count=len(selected),eligible_for_scoring=kind!='empty' and not self.offline,
                     token_usage=usage,branch_failures=len(failures))
        write_json(out/'status.json',state)
        return state
