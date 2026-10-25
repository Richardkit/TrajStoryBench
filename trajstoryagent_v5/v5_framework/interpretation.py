"""Formal v5 interpretation implementation."""
import json
from trajstoryagent_v5.v5_framework.json_utils import extract_json_object
from trajstoryagent_v5.v5_framework.export_support import read, write, client

def interpret(slug, evidence, bundle, facts, directory):
    """Language realization only; no judge/reference input. No formal NL guarantee."""
    cached = directory/'interpretation.json'
    if cached.exists():
        return read(cached)['insight']
    rows = [{'id': f'C{i+1}', 'text': text} for i, text in enumerate(facts)]
    original = {k: evidence[k] for k in ('statistics','tables','supporting_scope','limitations')}
    prompt = ('Answer the fixed analytical question using the original computed evidence and the new deterministic certificate facts. '
              'Stay within the original question. Preserve important conditions, comparisons, numerical support, within-group variation and material exceptions. '
              'Do not describe algorithms or certificate IDs in the insight. A pooled fit does not prove groups are identical. '
              'Indicator means are empirical proportions. Do not infer statistical significance, causality or population-level generalizations. '
              'A compression default is approximate; use supplied exact means/ranges when precision matters. '
              'If the new measure is irrelevant or has a wrong analytical unit, set use_refinement=false and explain. '
              'Return JSON {"use_refinement":true,"insight":"one coherent factual insight, at most 180 words",'
              '"used_fact_ids":["C1"],"limitations":["..."],"binding_concerns":[]}. '
              'List only actually used facts; do not invent evidence.\n')
    prompt += json.dumps({'goal':bundle['goal'],'question':evidence['question'],
                         'granularity':evidence['analysis_granularity'],
                         'computed_original':original, 'certificate_facts':rows}, ensure_ascii=False)
    llm = client(slug)
    error = ''
    for attempt in range(2):
        request = prompt + ('\nRepair output format: ' + error if error else '')
        write(directory/f'request_{attempt+1}.json', {'system':'Interpret computed evidence only. JSON only.', 'user':request})
        response = llm.complete(system='Interpret computed evidence only. JSON only.', user=request)
        (directory/f'response_{attempt+1}.txt').write_text(response, encoding='utf-8')
        try:
            result = extract_json_object(response)
            if type(result.get('use_refinement')) is not bool:
                raise ValueError('use_refinement must be boolean')
            if result['use_refinement']:
                if not isinstance(result.get('insight'), str) or not result['insight'].strip():
                    raise ValueError('insight missing')
                if len(result['insight'].split()) > 180:
                    raise ValueError('insight exceeds 180-word cap')
                ids = result.get('used_fact_ids', [])
                if not ids or not set(ids) <= {r['id'] for r in rows}:
                    raise ValueError('use at least one existing certificate fact ID')
                if result.get('binding_concerns'):
                    result['use_refinement'] = False
            if not result['use_refinement']:
                result['insight'] = evidence['insight']
            result['formal_guarantee'] = 'none for free prose; structured certificate only'
            write(cached, result)
            return result['insight']
        except (ValueError, TypeError) as exc:
            error = str(exc)
    raise RuntimeError('Interpretation failed: ' + error)

