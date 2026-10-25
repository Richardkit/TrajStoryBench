"""Score a selected generation batch without selecting or replacing experiments."""
import argparse
import hashlib
import json
from pathlib import Path
from trajstoryagent_v5.evaluation.evaluate import main as score_case
from trajstoryagent_v5.v5_framework.json_utils import write_json


def main(argv=None):
    root=Path(__file__).resolve().parents[2]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--generation-root',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=root/'trajstory_examples/insight/manifest.json')
    p.add_argument('--repo-root',type=Path,default=root)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case-ids',nargs='+')
    p.add_argument('--workers',type=int,default=4,help='Concurrent pair requests within each case')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args(argv)
    cases=json.loads(args.manifest.read_text(encoding='utf-8'))['cases']
    ids=[c['case_id'] for c in cases]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate case IDs')
    wanted=set(args.case_ids or ids)
    if wanted-set(ids):raise ValueError('Unknown case IDs')
    cases=[c for c in cases if c['case_id'] in wanted]
    if not cases:raise ValueError('No cases selected')
    output=args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):raise FileExistsError('Use a fresh output directory')
    plan=[]
    for c in cases:
        cid=c['case_id']
        if not cid or cid in ('.','..') or any(x in cid for x in '/\\:'):raise ValueError('Unsafe case ID')
        case=(args.repo_root/c['json_path']).resolve()
        result=(args.generation_root/cid/'result.json').resolve()
        for path in (case,result):
            if not path.is_file():raise FileNotFoundError(path)
        predictions=json.loads(result.read_text(encoding='utf-8')).get('insights')
        if not isinstance(predictions,list) or not predictions:raise ValueError('Empty/invalid predictions: '+cid)
        plan.append(dict(case_id=cid,case=str(case),result=str(result),
                         input_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (case,result)}))
    if args.dry_run:
        print(json.dumps(dict(ready=True,cases=len(plan),judge_rounds=2)));return 0
    output.mkdir(parents=True,exist_ok=True)
    with (output/'batch_manifest.json').open('x',encoding='utf-8') as f:json.dump(plan,f,ensure_ascii=False,indent=2)
    rows=[];failures=[]
    for item in plan:
        cid=item['case_id']
        try:
            for path,digest in item['input_sha256'].items():
                if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise RuntimeError('Input changed')
            score_case(['--case',item['case'],'--result',item['result'],'--output',str(output/cid),'--workers',str(args.workers)])
            rows.append(dict(case_id=cid,**json.loads((output/cid/'comparison.json').read_text(encoding='utf-8'))))
        except Exception as exc:failures.append(dict(case_id=cid,error_type=type(exc).__name__))
        weight=sum(r['reference_count'] for r in rows)
        write_json(output/'comparison.json',dict(complete=not failures and len(rows)==len(plan),cases=rows,failures=failures,
            aggregation='reference-insight weighted over successfully evaluated cases; failures explicitly listed',
            means={m:sum(r[m]*r['reference_count'] for r in rows)/weight for m in ('g_eval','coverage_at_0_6')} if weight else {}))
    return 1 if failures else 0


if __name__=='__main__':raise SystemExit(main())
