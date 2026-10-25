"""Frozen two-round GPT evaluation for one batch of the minimal Pandas baseline."""

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
from baseline_support.evaluation import original_prompts
from baseline_support.evaluation import digest,metrics
from baseline_support.evaluation import paired

GENERATOR = 'gpt'

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8');tmp.replace(p)

def freeze(batch):
    generation=read(batch/'generation.json');template,system=original_prompts();frozen={'jobs':{},'cases':[]}
    for row in generation:
        refs=read(ROOT/row['json_path'])['insights'];ids=[];pred=[]
        if row['eligible_for_scoring']:
            pred=read(Path(row['directory'])/'parsed_output.json')['answer']['insights']
            for ref in refs:
                keys=[]
                for value in pred:
                    req=dict(model='gpt-6-luna',messages=[dict(role='system',content=system),dict(role='user',content=template.format(answer=value,gt_answer=ref))],temperature=0,top_p=1,max_completion_tokens=50,reasoning_effort='none',logprobs=True,top_logprobs=5)
                    key=digest(req);frozen['jobs'][key]=req;keys.append(key)
                ids.append(keys)
        frozen['cases'].append(dict(case_id=row['case_id'],eligible=row['eligible_for_scoring'],predictions=pred,references=refs,job_ids=ids,generation=row))
    frozen['input_hash']=digest(frozen);p=batch/'judge/manifest.json'
    if p.exists():assert read(p)==frozen,'Frozen judge inputs changed'
    else:write(p,frozen)
    return frozen

def evaluate(batch,frozen,report_profile='current'):
    if report_profile not in ('current','legacy-full','legacy-qwen','legacy-budget'):
        raise ValueError('Unknown report profile')
    from openai import OpenAI
    import winreg
    key=os.environ.get('OPENAI_API_KEY')
    if not key:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,'Environment') as k:key=winreg.QueryValueEx(k,'OPENAI_API_KEY')[0]
    client=OpenAI(api_key=key,base_url='https://api.openai.com/v1',timeout=180,max_retries=2)
    for rnd in (1,2):
        def score(k):
            dest=batch/'judge'/f'round_{rnd}'/k
            if (dest/'result.json').exists():return
            write(dest/'request.json',frozen['jobs'][k])
            for attempt in range(3):
                body=client.chat.completions.create(**frozen['jobs'][k]).model_dump();write(dest/f'response_{time.time_ns()}.json',body)
                choice=body['choices'][0];msg=choice['message'];usage=body.get('usage') or {};rt=(usage.get('completion_tokens_details') or {}).get('reasoning_tokens')
                if msg.get('reasoning_content') or rt not in (None,0):raise RuntimeError('Unexpected judge reasoning')
                match=re.fullmatch(r'\s*<rating>(\d+)</rating>\s*',msg.get('content') or '')
                if choice['finish_reason']=='stop' and match and 0<=int(match[1])<=10:
                    rating=int(match[1]);write(dest/'result.json',dict(rating=rating,score=rating/10,request_hash=k,returned_model=body.get('model'),usage=usage));return
            raise RuntimeError('Invalid rating')
        pending=list(frozen['jobs']);failed=[]
        for retry in range(3):
            failed=[]
            with ThreadPoolExecutor(max_workers=8) as pool:
                jobs={pool.submit(score,k):k for k in pending}
                for i,f in enumerate(as_completed(jobs),1):
                    try:f.result()
                    except Exception:
                        k=jobs[f];failed.append(k);write(batch/'judge'/f'round_{rnd}'/k/f'error_{time.time_ns()}.json',dict(error=traceback.format_exc()))
                    if i%50==0 or i==len(pending):write(batch/'evaluation_status.json',dict(state='evaluating',round=rnd,processed=i,total=len(pending),failed=len(failed)))
            if not failed:break
            pending=failed
        if failed:raise RuntimeError('Unresolved judging failures; cached successful scores retained')
    values={n:{k:read(batch/'judge'/f'round_{n}'/k/'result.json')['score'] for k in frozen['jobs']} for n in (1,2)}
    rows=[]
    for c in frozen['cases']:
        scores=[metrics([[values[n][k] for k in r] for r in c['job_ids']]) for n in (1,2)] if c['eligible'] else [dict(g_eval=0.,coverage_at_0_6=0.) for _ in (1,2)]
        rows.append(dict(case_id=c['case_id'],eligible=c['eligible'],rounds=scores,**{m:statistics.mean(x[m] for x in scores) for m in ('g_eval','coverage_at_0_6')},generation=c['generation']))
    ib=ROOT/'results/insightbench/full125_20261011';baseline=[]
    if (ib/'comparison.json').exists() and (ib/'evaluation/gpt/manifest.json').exists():
        refs={c['case_id']:c['references'] for c in read(ib/'evaluation/gpt/manifest.json')['cases'] if c['generator']==GENERATOR}
        saved={c['case_id']:c for c in read(ib/'comparison.json')['judges']['gpt']['cases'] if c['generator']==GENERATOR}
        if all(c['case_id'] in refs and c['references']==refs[c['case_id']] and c['case_id'] in saved for c in frozen['cases']):baseline=[saved[c['case_id']] for c in frozen['cases']]
    selected=[r for r in rows if r['eligible']]
    complete=sum(len(c['predictions'])==6 and c['eligible'] for c in frozen['cases'])
    partial=sum(0<len(c['predictions'])<6 and c['eligible'] for c in frozen['cases'])
    report=dict(generator=GENERATOR,judge='gpt-6-luna',judge_rounds=2,total=len(rows),valid_outputs=len(selected),complete_outputs=complete,partial_outputs=partial,failed_or_empty_outputs=len(rows)-len(selected),success_rate=len(selected)/len(rows),
        score_protocol='Discrete rating/10; reference maxima and coverage>=0.6 separately in each GPT round, average rounds, equal-case macro. Score valid partial outputs using actual insight strings; technical failure/empty/invalid=0.',
        means={m:statistics.mean(r[m] for r in rows) for m in ('g_eval','coverage_at_0_6')},
        successful_only_means={m:statistics.mean(r[m] for r in selected) if selected else None for m in ('g_eval','coverage_at_0_6')},cases=rows,
        summary_evaluation='Presence/format only; G-eval and coverage evaluate insight sentences.')
    # Preserve historical report schemas without maintaining separate scorers.
    if report_profile != 'current':
        for field in ('complete_outputs','partial_outputs','failed_or_empty_outputs'):
            report.pop(field)
        report['score_protocol']='Discrete rating/10; per-round reference maxima and coverage>=0.6, average rounds, then equal-case macro. Failed/invalid generation=0.'
        if report_profile == 'legacy-full':
            report.pop('generator')
        elif report_profile == 'legacy-qwen':
            report['generator']='qwen3.8-flash'
    if baseline:
        if report_profile == 'legacy-qwen':
            report['matched_insightbench_generator']='qwen'
        report['matched_insightbench_means']={m:statistics.mean(r[m] for r in baseline) for m in ('g_eval','coverage_at_0_6')}
        report['paired_vs_insightbench']={m:paired([a[m]-b[m] for a,b in zip(rows,baseline)]) for m in ('g_eval','coverage_at_0_6')}
    write(batch/'report.json',report);write(batch/'evaluation_status.json',dict(state='completed',total_requests=len(frozen['jobs']),rounds=2))
    return report

def main():
    global GENERATOR
    p=argparse.ArgumentParser();p.add_argument('--batch-dir',type=Path,required=True);p.add_argument('--run',action='store_true');p.add_argument('--generator',choices=['gpt','qwen'],required=True);p.add_argument('--report-profile',choices=['current','legacy-full','legacy-qwen','legacy-budget'],default='current');args=p.parse_args();GENERATOR=args.generator;frozen=freeze(args.batch_dir)
    if args.run:evaluate(args.batch_dir,frozen,args.report_profile)
    print(json.dumps(dict(cases=len(frozen['cases']),requests_per_round=len(frozen['jobs']),run=args.run)))

if __name__=='__main__':main()
