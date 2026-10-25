"""Independent two-round GPT insight evaluation; never generates or selects predictions."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re

from trajstoryagent_v5.v5_framework.json_utils import write_json


def metrics(matrix):
    if not matrix or any(not row for row in matrix):
        raise ValueError('References and predictions must be nonempty')
    best = [max(row) for row in matrix]
    return {'g_eval': sum(best)/len(best),
            'coverage_at_0_6': sum(x >= .6 for x in best)/len(best), 'maxima': best}


def parse_rating(response):
    choice = response['choices'][0]
    message = choice['message']
    reasoning = ((response.get('usage') or {}).get('completion_tokens_details') or {}).get('reasoning_tokens')
    match = re.fullmatch(r'\s*<rating>(\d+)</rating>\s*', message.get('content') or '')
    if choice['finish_reason'] != 'stop' or message.get('reasoning_content') or reasoning not in (None, 0) or not match or not 0 <= int(match[1]) <= 10:
        raise ValueError('Invalid judge rating or unexpected reasoning')
    return int(match[1])/10


def score(predictions, references, output, client, workers=4):
    if not predictions or not references or workers < 1:
        raise ValueError('Nonempty predictions/references and positive workers required')
    output = Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('Use a new evaluation directory')
    prompts = json.loads((Path(__file__).parent.parent/'roles/judge_prompts.json').read_text(encoding='utf-8'))
    jobs = {}; ids = []
    for ref in references:
        row=[]
        for pred in predictions:
            req=dict(model='gpt-6-luna',messages=[dict(role='system',content=prompts['G_EVAL_BASIC_SYSTEM_MESSAGE']),
                dict(role='user',content=prompts['G_EVAL_BASIC_TEMPLATE'].format(answer=pred,gt_answer=ref))],
                temperature=0,top_p=1,reasoning_effort='none',max_completion_tokens=50,logprobs=True,top_logprobs=5)
            key=hashlib.sha256(json.dumps(req,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
            jobs[key]=req;row.append(key)
        ids.append(row)
    output.mkdir(parents=True,exist_ok=True)
    with (output/'manifest.json').open('x',encoding='utf-8') as f:
        json.dump(dict(predictions=predictions,references=references,jobs=jobs,job_ids=ids,
                       rounds=2,rating_policy='discrete rating / 10; per-reference maximum; coverage separately per round'),f,ensure_ascii=False,indent=2)
    rounds=[]
    for rnd in (1,2):
        def one(key):
            dest=output/f'round_{rnd}'/key
            write_json(dest/'request.json',jobs[key])
            for attempt in range(1,4):
                try:
                    response=client.chat.completions.create(**jobs[key]).model_dump()
                    write_json(dest/f'response_{attempt}.json',response)
                    value=parse_rating(response)
                    write_json(dest/'result.json',dict(score=value,rating=round(value*10),request_hash=key))
                    return key,value
                except Exception as exc:
                    write_json(dest/f'error_{attempt}.json',dict(type=type(exc).__name__))
                    if attempt==3:raise
        with ThreadPoolExecutor(max_workers=workers) as pool:
            values=dict(pool.map(one,jobs))
        matrix=[[values[k] for k in row] for row in ids]
        write_json(output/f'round_{rnd}/matrix.json',matrix)
        rounds.append(metrics(matrix))
    result=dict(rounds=rounds,reference_count=len(references),prediction_count=len(predictions),
                **{m:sum(r[m] for r in rounds)/2 for m in ('g_eval','coverage_at_0_6')})
    write_json(output/'comparison.json',result)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',type=Path,required=True)
    p.add_argument('--result',type=Path,required=True,help='Generation result.json with insights list')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args(argv)
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise FileExistsError('Use a new evaluation directory')
    references=json.loads(args.case.read_text(encoding='utf-8'))['insights']
    predictions=json.loads(args.result.read_text(encoding='utf-8'))['insights']
    if not isinstance(predictions,list) or not all(isinstance(x,str) and x.strip() for x in predictions):
        raise ValueError('Expected a nonempty list of prediction strings')
    from trajstoryagent_v5.v5_framework.providers import credentials
    credentials('gpt')
    from openai import OpenAI
    client=OpenAI(api_key=os.environ['OPENAI_API_KEY'],base_url='https://api.openai.com/v1',timeout=180,max_retries=2)
    result=score(predictions,references,args.output,client,args.workers)
    write_json(args.output/'input_hashes.json',{str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.case,args.result)})
    print(json.dumps(result))


if __name__=='__main__':
    main()
