"""Case-scoped references and bounded, deduplicated role inputs."""
import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path

CHUNK_CHARS = 1600
ROLE_INPUT_CHARS = 12000
TASK_CHARS = 1800
ROLE_OUTPUT_CHARS = 6000
REF = re.compile(r'([ER]_[0-9a-f]{32})(?:/S([1-9][0-9]*)(?:/([CK])([1-9][0-9]*))?)?')

PLANNER_INDEXED_GUIDANCE = (
    '\nPlanner context is a bounded current-state index, not a conversation log. '
    'Reuse ready artifact IDs instead of requesting the same result again. '
    'E steps are [step, stdout chunk count, code chunk count]. '
    'Prefer preferred_steps (latest successful outputs), not initial schema previews. '
    'A chunk selector must include its step: E_<id>/S2/C1, never E_<id>/C1. '
    'Never repeat evidence text, citations or old plans in thinking, analysis or plan_update. '
    'Keep thinking and analysis under 160 characters each and plan_update under 240. '
    'Errors show occurrence counts, not repeated messages; correct the handoff instead of '
    'retrying unchanged tasks. Identical dispatches reuse saved results; repeated '
    'dispatch-only rounds stop without rerunning agents. If latest_synthesis meets '
    'the goal and required roles are complete, finish with exactly that JSON. '
    'Otherwise name the missing evidence and reuse existing IDs before computing more. '
    'New computation must pass through interpretation, verification and a new synthesis; '
    'never submit a stale report or repeat analysis because a handoff failed.'
)


def dispatch_key(role, task, files=()):
    """Exact task equivalence only; no semantic merging of different questions."""
    try:
        value = json.loads(task)
    except (ValueError, TypeError):
        value = str(task).strip()
    if isinstance(value, dict) and set(value) <= {'instruction', 'evidence_ids'}:
        refs = value.get('evidence_ids')
        if isinstance(refs, list) and all(isinstance(ref, str) for ref in refs):
            value = dict(value, evidence_ids=sorted(set(refs)))
    return json.dumps([role, value, list(files or [])], ensure_ascii=False, sort_keys=True)


def reusable_dispatch(payload):
    if payload.get('success'):
        return True
    try:
        error = json.loads(payload.get('result') or payload.get('error') or '')
    except (ValueError, TypeError):
        return False
    # These fail before a model call. Transient/model-output errors remain retryable.
    detail = error.get('detail', '') if isinstance(error, dict) else ''
    return detail.startswith(('Evidence batch exceeds', 'Invalid evidence ID:',
                              'Unknown evidence ID', 'Wrong upstream role',
                              'Task must be JSON', 'Use a short instruction',
                              'Supply 1 to 24 evidence_ids', 'Instruction exceeds'))


def task_hint(task):
    # The case path is already host-bound and carries no information about the question.
    text = re.sub(r'(?:[A-Za-z]:[\\/]|/)[^\s,]*case_input\.json', '[case]', str(task))
    text = re.sub(r'^Using trajstory_analysis with spec_path\s*\[case\]\s*,?\s*', '', text,
                  flags=re.IGNORECASE)
    return REF.sub('[reference]', text)[:120]


class PlannerEvidenceState:
    """One entry per artifact and error, with full history left in local logs."""

    def __init__(self):
        self.artifacts = {}
        self.errors = {}
        self.completed_roles = set()
        self.latest_synthesis = None
        self.synthesis_stale = False
        self.reused_dispatches = 0
        self.events = 0

    def observe(self, role, task, success, result, synthesis_basis=None):
        self.events += 1
        try:
            payload = json.loads(result)
        except (ValueError, TypeError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        try:
            request = json.loads(task)
        except (ValueError, TypeError):
            request = {}
        hint = request.get('instruction', task) if isinstance(request, dict) else task
        hint = task_hint(hint)
        if success:
            self.completed_roles.add(role)
            identifier = payload.get('evidence_id') or payload.get('result_id')
            if isinstance(identifier, str) and REF.fullmatch(identifier):
                entry = dict(id=identifier, role=role.removeprefix('orchestra_'), topic=hint)
                if identifier.startswith('E_'):
                    entry['steps'] = [[s['step'], s['stdout_chunks'], s['code_chunks']]
                                      for s in payload.get('index', [])]
                    entry['preferred_steps'] = payload.get('successful_steps', [])[-2:]
                else:
                    entry['count'] = payload.get('count', 0)
                if self.artifacts.get(identifier) != entry and self.latest_synthesis is not None:
                    self.synthesis_stale = True
                self.artifacts[identifier] = entry
            elif role == 'orchestra_synthesizer' and isinstance(payload.get('insights'), list):
                # Keep the exact latest approved report once, not every round's copy.
                self.latest_synthesis = payload
                self.synthesis_stale = (synthesis_basis is not None and
                                        set(synthesis_basis) != set(self.artifacts))
            return
        detail = str(payload.get('detail') or result).split('Available:', 1)[0].strip()[:180]
        refs = tuple(sorted(set(m.group(0) for m in REF.finditer(str(task)))))
        key = (role, detail, refs)
        prior = self.errors.pop(key, None)
        self.errors[key] = dict(role=role.removeprefix('orchestra_'), error=detail,
                                attempts=1 + (prior['attempts'] if prior else 0), topic=hint)
        if self.latest_synthesis is not None:
            self.synthesis_stale = True

    def render(self, max_chars):
        value = dict(completed_roles=sorted(self.completed_roles),
                     artifacts=list(self.artifacts.values()),
                     recent_errors=list(self.errors.values())[-6:])
        if self.latest_synthesis is not None and not self.synthesis_stale:
            # All review evidence is already behind the completed synthesis.
            # Final submission needs the exact report once, not its old inputs again.
            value = dict(completed_roles=sorted(self.completed_roles),
                         ready_artifact_count=len(self.artifacts),
                         latest_synthesis=self.latest_synthesis)
        elif self.synthesis_stale:
            value['synthesis_status'] = 'stale: process new evidence through roles and synthesize again'
        if self.reused_dispatches:
            value['reused_dispatches'] = self.reused_dispatches
            value['reuse_notice'] = 'Unchanged tasks reused saved results; use ready IDs or correct the task.'
        encode = lambda: json.dumps(value, ensure_ascii=False, separators=(',', ':'))
        text = encode()
        if len(text) > max_chars:
            # Drop only descriptive hints; preserve every artifact ID and selector.
            if 'artifacts' in value:
                value['artifacts'] = [{k: v for k, v in entry.items() if k != 'topic'}
                                      for entry in value['artifacts']]
            if 'recent_errors' in value:
                value['recent_errors'] = [{k: v for k, v in entry.items() if k != 'topic'}
                                          for entry in value['recent_errors']]
            text = encode()
        if len(text) > max_chars:
            # Older errors remain on disk. One latest correction is enough in context.
            if 'recent_errors' in value:
                value['recent_errors'] = value['recent_errors'][-1:]
            text = encode()
        if len(text) > max_chars:
            raise ValueError('Planner artifact index exceeds its context budget; '
                             'reuse existing evidence rather than create more artifacts')
        return text


def model_limits(agent, messages):
    limit=getattr(agent,'model_input_char_limit',None)
    size=sum(len(str(message.content)) for message in messages)
    if limit is not None and size>limit:
        raise ValueError(f'Model input exceeds {limit} characters; split the task/evidence instead of copying history')
    output=getattr(agent,'model_output_token_limit',None)
    return {'max_completion_tokens':output} if output is not None else {}


def unique_content(values):
    canonical={};aliases={};seen={}
    for identifier,value in values.items():
        key=json.dumps(value,ensure_ascii=False,sort_keys=True)
        if key in seen:aliases[identifier]=seen[key]
        else:canonical[identifier]=value;seen[key]=identifier
    return canonical,aliases


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf8')


def store_for(agent, ctx):
    if ctx is None:
        raise ValueError('A case context is required')
    # 128-bit case keys keep evidence paths below Windows MAX_PATH.
    key = hashlib.sha256(ctx.id.encode()).hexdigest()[:32]
    return Path(agent.workdir).resolve().parent / 'handoffs' / key


def chunks(text):
    # Exact consecutive slices, including all whitespace; no content is lost.
    return [text[i:i+CHUNK_CHARS] for i in range(0, len(text), CHUNK_CHARS)]


def execution_artifact(store, identifier, steps):
    indexed=[]
    for step in steps:
        args=step['args']
        if isinstance(args, str):
            try: args=json.loads(args)
            except json.JSONDecodeError: args={}
        code=args.get('code', '') if isinstance(args, dict) else ''
        observation=str(step['observation'])
        indexed.append(dict(step=step['step'], success=bool(observation) and not
            observation.startswith('TrajStory analysis error:'),
            stdout=chunks(observation), code=chunks(code)))
    write(store / (identifier+'.json'), dict(kind='execution', steps=indexed))
    successful=[s for s in indexed if s['success']]
    # The planner sees a small index, never the underlying code/history/stdout.
    return dict(status='ready' if successful else 'no_executed_evidence', evidence_id=identifier,
        successful_steps=[s['step'] for s in successful],
        index=[dict(step=s['step'], stdout_chunks=len(s['stdout']), code_chunks=len(s['code']),
                    preview=s['stdout'][0][:100]) for s in successful[-6:]],
        selector='evidence_id/S<step>/C<chunk> for stdout; /K<chunk> for code',
        omitted_index_entries=max(0,len(successful)-6))


def task_payload(task):
    if len(task)>3000:
        raise ValueError('Use a short instruction and evidence_ids; do not paste code, logs or prior responses')
    value=json.loads(task)
    if not isinstance(value,dict) or set(value)-{'instruction','evidence_ids'}:
        raise ValueError('Task must be JSON with only instruction and evidence_ids')
    instruction=value.get('instruction','')
    refs=value.get('evidence_ids')
    if not isinstance(instruction,str) or len(instruction)>TASK_CHARS:
        raise ValueError('Instruction exceeds the compact task limit')
    if not isinstance(refs,list) or not refs or len(refs)>24 or not all(isinstance(x,str) for x in refs):
        raise ValueError('Supply 1 to 24 evidence_ids')
    return instruction,list(dict.fromkeys(refs))


def resolve(store, refs, role):
    sources={};results={};pending=list(refs);visited=set()
    while pending:
        ref=pending.pop(0)
        if ref in visited:continue
        visited.add(ref)
        match=REF.fullmatch(ref)
        if not match:raise ValueError('Invalid evidence ID: '+ref)
        identifier,step_no,kind,chunk_no=match.groups()
        path=store/(identifier+'.json')
        if not path.is_file():raise ValueError('Unknown evidence ID in this case: '+identifier)
        item=json.loads(path.read_text(encoding='utf8'))
        if identifier.startswith('R_'):
            if role=='orchestra_interpreter':raise ValueError('Interpretation requires executed E_ evidence IDs')
            if step_no:raise ValueError('Role results do not have step selectors')
            expected='orchestra_interpreter' if role=='orchestra_verifier' else 'orchestra_verifier'
            if item.get('role')!=expected:
                raise ValueError('Wrong upstream role for '+role)
            results[identifier]=item['payload']
            # Verification receives the same exact evidence once; synthesis
            # receives reviewed statements only, never the execution history.
            if role=='orchestra_verifier':pending.extend(item['source_ids'])
            continue
        if role=='orchestra_synthesizer':raise ValueError('Synthesis accepts only verifier result IDs')
        for step in item['steps']:
            if not step['success'] or (step_no and step['step']!=int(step_no)):continue
            pairs=[('C',step['stdout'])] if not kind else [(kind,step['stdout'] if kind=='C' else step['code'])]
            for letter,parts in pairs:
                for number,text in enumerate(parts,1):
                    if chunk_no and number!=int(chunk_no):continue
                    sources[f"{identifier}/S{step['step']}/{letter}{number}"]=text
    if not sources and not results:raise ValueError('No matching executed evidence')
    return sources,results


def role_input(store, task, role, system_chars=0):
    instruction,refs=task_payload(task)
    sources,results=resolve(store,refs,role)
    if role=='orchestra_verifier' and not results:
        raise ValueError('Verification requires an interpreter R_ result ID')
    if role=='orchestra_verifier':
        # Exact quoted text already occurs in the source chunk. Send its location
        # rather than copying the excerpt a second time in every candidate.
        results=deepcopy(results)
        for value in results.values():
            for item in value['candidates']:
                for citation in item.get('citations', []):
                    quote=citation.pop('quote')
                    start=sources[citation['evidence_id']].find(quote)
                    if start < 0:
                        raise ValueError('Stored citation does not match execution evidence')
                    citation['quote_span']=[start, start+len(quote)]
    if role=='orchestra_synthesizer':
        results={key:dict(reviews=[dict(status=item['status'],insight=item['insight'],
            limitations=item.get('limitations',[]),
            evidence_ids=list(dict.fromkeys(c['evidence_id'] for c in item.get('citations',[]))))
            for item in value['reviews'] if item['status'] in ('supported','revised')])
            for key,value in results.items()}
    evidence,aliases=unique_content(sources)
    upstream,result_aliases=unique_content(results)
    value=dict(instruction=instruction,evidence=evidence,upstream_results=upstream)
    if aliases:value['evidence_aliases']=aliases
    if result_aliases:value['result_aliases']=result_aliases
    text=json.dumps(value,ensure_ascii=False)
    if len(text)+system_chars>ROLE_INPUT_CHARS:
        ids=list(sources)
        raise ValueError('Evidence batch exceeds 12000 characters; select fewer /S#/C# IDs. '
                         'Available: '+', '.join(ids[:12]))
    return text,sources


def validate_result(role, payload, sources):
    if not isinstance(payload,dict):raise ValueError('Role result must be JSON object')
    if role=='orchestra_synthesizer':
        from src.trajstory.output import parse_trajstory_result
        parsed=parse_trajstory_result(json.dumps(payload,ensure_ascii=False))
        if len(parsed['pred_insights'])>6:raise ValueError('Synthesis must return at most six insights')
        payload['insights']=list(dict.fromkeys(parsed['pred_insights']))
        return
    key='candidates' if role=='orchestra_interpreter' else 'reviews'
    items=payload.get(key)
    if not isinstance(items,list) or len(items)>6:raise ValueError('Return at most six '+key)
    # Remove only byte-equivalent structured repeats, preserving all distinct
    # evidence IDs, scopes, qualifications and numerical statements.
    items=list({json.dumps(item,ensure_ascii=False,sort_keys=True):item for item in items}.values())
    payload[key]=items
    for item in items:
        if not isinstance(item,dict) or not isinstance(item.get('insight'),str):raise ValueError('Invalid insight')
        if len(item['insight'])>900:raise ValueError('Keep each insight under 900 characters')
        if role=='orchestra_verifier' and item.get('status') not in ('supported','revised','rejected'):
            raise ValueError('Invalid review status')
        if role=='orchestra_verifier' and item['status']=='rejected' and item['insight'].strip():
            raise ValueError('Rejected findings must have empty insight text')
        citations=item.get('citations',[])
        if not isinstance(citations,list) or len(citations)>4:raise ValueError('Use at most four concise citations')
        citations=list({json.dumps(c,sort_keys=True):c for c in citations}.values())
        item['citations']=citations
        if item['insight'].strip() and not citations:raise ValueError('Nonempty insights require citations')
        for citation in citations:
            if not isinstance(citation,dict):raise ValueError('Invalid citation')
            identifier=citation.get('evidence_id');quote=citation.get('quote')
            if isinstance(quote,str) and len(quote)>240:
                raise ValueError('Citation quote exceeds 240 characters; copy a shorter exact excerpt')
            if identifier not in sources or not isinstance(quote,str) or not quote.strip() or len(quote)>240 or quote not in sources[identifier]:
                raise ValueError('Citation must exactly match supplied execution evidence')
