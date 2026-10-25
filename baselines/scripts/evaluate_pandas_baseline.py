"""Two-round discrete GPT scoring of Pandas Agent and matched cached baselines."""

import sys as _baseline_sys
from pathlib import Path as _BaselinePath
_baseline_repo = next(p for p in _BaselinePath(__file__).resolve().parents if (p / "baselines" / "__init__.py").is_file())
_baseline_sys.path.insert(0, str(_baseline_repo))
import baselines  # Register the shared baseline module location.

import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from baseline_support.runtime import credentials
from baseline_support.runtime import read,write,sha
from baseline_support.evaluation import original_prompts
from baseline_support.evaluation import digest,metrics
from baseline_support.evaluation import paired
SOURCE=ROOT/'results/v3_result/mimo26_matched15_gpt_judge_20261012'
V1MATH=ROOT/'results/v3_result/v1math_mimo15_20261012'
ARMS=('pandas_agent','v1','v1_compressed','insightbench')


def freeze(folder):
    cfg=read(folder/'manifest.json')
    if cfg.get('validation_mode')!='benchmark':raise ValueError('Offline fake smoke outputs cannot be evaluated as benchmark results')
    if cfg['model']!='mimo-v2.6-flash':raise ValueError('Cached baselines require MiMo generation')
    for rel,h in cfg['input_sha256'].items():
        if sha(ROOT/rel)!=h:raise RuntimeError('Input changed: '+rel)
    summary=read(folder/'trajstory_case_results.json');lookup={r['case_id']:r for r in summary}
    if set(lookup)!={c['case_id'] for c in cfg['cases']} or any(r['status']!='completed' for r in summary):raise RuntimeError('Incomplete generation; do not silently skip cases')
    vm=read(V1MATH/'manifest.json');template,system=original_prompts();frozen=dict(judge='gpt-6-luna',jobs={},cases=[])
    for c in cfg['cases']:
        cid=c['case_id'];pred=dict(pandas_agent=lookup[cid]['pred_insights'],
            v1=read(SOURCE/'mimo/v1'/cid/'predictions.json')['predictions'],
            v1_compressed=read(V1MATH/'refine'/vm['case_directory'][cid]/'predictions.json')['predictions']['v1_compressed'],
            insightbench=read(SOURCE/'mimo/insightbench'/cid/'predictions.json')['predictions'])
        refs=read(ROOT/c['json_path'])['insights']
        for arm,values in pred.items():
            if not 0<len(values)<=6:raise ValueError('Invalid prediction count')
            ids=[]
            for ref in refs:
                row=[]
                for value in values:
                    req=dict(model='gpt-6-luna',messages=[dict(role='system',content=system),dict(role='user',content=template.format(answer=value,gt_answer=ref))],temperature=0,top_p=1,max_completion_tokens=50,reasoning_effort='none',logprobs=True,top_logprobs=5)
                    k=digest(req);frozen['jobs'][k]=req;row.append(k)
                ids.append(row)
            frozen['cases'].append(dict(case_id=cid,arm=arm,predictions=values,references=refs,job_ids=ids))
    frozen['input_hash']=digest(frozen);p=folder/'judge/manifest.json'
    if p.exists():assert read(p)==frozen
    else:write(p,frozen)
    return frozen


def evaluate(folder,frozen):
    from openai import OpenAI
    credentials(['OPENAI_API_KEY']);client=OpenAI(api_key=os.environ['OPENAI_API_KEY'],timeout=180,max_retries=2)
    caches=[(SOURCE/'mimo/judges/gpt',read(SOURCE/'mimo/judges/gpt/manifest.json')),(V1MATH/'judge',read(V1MATH/'judge/manifest.json'))]
    for rnd in (1,2):
        def score(k):
            dest=folder/'judge'/f'round_{rnd}'/k
            if (dest/'result.json').exists():return
            write(dest/'request.json',frozen['jobs'][k])
            for root,manifest in caches:
                cached=root/f'round_{rnd}'/k/'result.json'
                if k in manifest['jobs'] and cached.exists():
                    assert manifest['jobs'][k]==frozen['jobs'][k]
                    write(dest/'result.json',dict(read(cached),reused_from=str(cached),source_sha256=sha(cached)));return
            for attempt in range(3):
                body=client.chat.completions.create(**frozen['jobs'][k]).model_dump();write(dest/f'response_{time.time_ns()}.json',body)
                choice=body['choices'][0];message=choice['message'];usage=body.get('usage') or {};rt=(usage.get('completion_tokens_details') or {}).get('reasoning_tokens')
                if message.get('reasoning_content') or rt not in (None,0):raise RuntimeError('Unexpected reasoning')
                match=re.fullmatch(r'\s*<rating>(\d+)</rating>\s*',message.get('content') or '')
                if choice['finish_reason']=='stop' and match and 0<=int(match[1])<=10:
                    rating=int(match[1]);write(dest/'result.json',dict(rating=rating,score=rating/10,request_hash=k,returned_model=body.get('model'),usage=usage));return
            raise RuntimeError('Invalid rating')
        pending=list(frozen['jobs'])
        for attempt in range(3):
            failed=[]
            with ThreadPoolExecutor(max_workers=8) as pool:
                jobs={pool.submit(score,k):k for k in pending}
                for i,f in enumerate(as_completed(jobs),1):
                    try:f.result()
                    except Exception:
                        k=jobs[f];failed.append(k);write(folder/'judge'/f'round_{rnd}'/k/f'error_{time.time_ns()}.json',dict(error=traceback.format_exc()))
                    if i%100==0 or i==len(pending):write(folder/'evaluation_status.json',dict(state='evaluating',round=rnd,processed=i,total=len(pending),errors=len(failed)))
            if not failed:break
            pending=failed
        if failed:raise RuntimeError('Unresolved scores')
    rounds={n:{k:read(folder/'judge'/f'round_{n}'/k/'result.json')['score'] for k in frozen['jobs']} for n in (1,2)};rows=[]
    for c in frozen['cases']:
        ms=[metrics([[rounds[n][k] for k in row] for row in c['job_ids']]) for n in (1,2)]
        rows.append(dict(case_id=c['case_id'],arm=c['arm'],rounds=ms,**{m:statistics.mean(x[m] for x in ms) for m in ('g_eval','coverage_at_0_6')}))
    lookup={(c['case_id'],c['arm']):c for c in rows};ids=sorted({c['case_id'] for c in rows})
    write(folder/'comparison.json',dict(judge='gpt-6-luna',rounds=2,cases=rows,
        means={a:{m:statistics.mean(x[m] for x in rows if x['arm']==a) for m in ('g_eval','coverage_at_0_6')} for a in ARMS},
        paired={'pandas_minus_'+a:{m:paired([lookup[c,'pandas_agent'][m]-lookup[c,a][m] for c in ids]) for m in ('g_eval','coverage_at_0_6')} for a in ARMS if a!='pandas_agent'}))
    write(folder/'evaluation_status.json',dict(state='completed'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results-dir',type=Path,default=ROOT/'results/pandas/mimo15_20261012');p.add_argument('--run',action='store_true');args=p.parse_args()
    frozen=freeze(args.results_dir);print(json.dumps(dict(frozen_requests=len(frozen['jobs']),cases=len(frozen['cases'])//4)))
    if args.run:evaluate(args.results_dir,frozen)


if __name__=='__main__':main()
