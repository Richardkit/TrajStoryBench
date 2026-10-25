"""Generate one fresh formal multi-role v5 case; scoring is a separate operation."""
from __future__ import annotations

import argparse
import hashlib
import json
from .identity import FRAMEWORK_ID, IMPLEMENTATION_ID
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True, type=Path, help='TrajStory case JSON')
    parser.add_argument('--model', choices=['gpt', 'qwen'], required=True)
    parser.add_argument('--output', required=True, type=Path, help='New, empty output directory')
    parser.add_argument('--repo-root', type=Path, default=ROOT, help='Root for case-relative input paths')
    args = parser.parse_args(argv)

    # Check output before credentials, network calls, or any generation.
    output = args.output.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'Refusing to overwrite an existing run: {output}')

    from trajstoryagent_v5.case_adapter import load_trajstory_case
    from trajstoryagent_v5.json_utils import write_json
    from trajstoryagent_v5.transport import ResilientClient
    from trajstoryagent_v5 import FrameworkConfig, TrajStoryAgent
    from trajstoryagent_v5.refinement import COST_MODEL
    from trajstoryagent_v5.terminology import InsightClient
    from trajstoryagent_v5 import export_support as refinement
    from trajstoryagent_v5.providers import credentials

    case = args.case.expanduser().resolve()
    bundle = load_trajstory_case(case, args.repo_root, input_csvs=True)
    credentials(args.model)
    model, url = refinement.MODELS[args.model]
    key_name = 'OPENAI_API_KEY' if args.model == 'gpt' else 'DASHSCOPE_API_KEY'
    client = ResilientClient(model=model, base_url=url, api_key=os.environ[key_name],
                             temperature=0, reasoning_effort='none', max_tokens=16000)
    # Export/refinement calls receive the same terminology and projected-input
    # contract as profiling, analysis, verification, Memory and summary.
    exporter_client = refinement.OpenAICompatibleClient
    config = FrameworkConfig()
    output.mkdir(parents=True, exist_ok=True)
    os.environ['TRAJSTORY_LLM_AUDIT_PATH'] = str(output/'llm_calls.jsonl')
    package = Path(__file__).resolve().parent
    source_paths = [p for p in package.rglob('*') if p.is_file() and p.suffix in {'.py','.md','.json','.txt','.toml'} and 'tests' not in p.parts and '__pycache__' not in p.parts]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    write_json(output/'run_manifest.json', {
        'framework': FRAMEWORK_ID, 'implementation': IMPLEMENTATION_ID, 'generator': args.model,
        'schema_version': 2,
        'input_policy': 'required input_ CSV projections; no original CSV fallback',
        'compression_cost_model': COST_MODEL,
        'model': model, 'temperature': 0, 'reasoning_effort': 'none',
        'case': str(case), 'config': config.to_dict(),
        'input_sha256': {str(p): sha(p) for p in [case, *(f.path for f in bundle.files)]},
        'source_sha256': {str(Path(package.name)/p.relative_to(package)): sha(p) for p in source_paths},
        'prediction_source': 'predictions.json:insights.conditional_compressed',
        'summary_source': 'finalized branch insights, after both rounds of refinement',
    })
    # Save the exact source used by this new run; historical snapshots stay intact.
    for source in source_paths:
        target = output/'source_snapshot'/package.name/source.relative_to(package)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    refinement.OpenAICompatibleClient = lambda **kwargs: InsightClient(ResilientClient(**kwargs))
    try:
        report = TrajStoryAgent(client, generator=args.model, config=config).run(bundle, output)
    finally:
        refinement.OpenAICompatibleClient = exporter_client
    predictions = json.loads((output/'predictions.json').read_text(encoding='utf-8'))
    if predictions.get('framework') != FRAMEWORK_ID:
        raise RuntimeError('Prediction framework does not match the formal v5 implementation')
    if [item.insight for item in report.insights] != predictions['insights']['conditional_compressed']:
        raise RuntimeError('Report insights and evaluation predictions must be identical')
    write_json(output/'result.json', {
        'generator': args.model, 'case_json': str(case), 'source_run': str(output),
        'framework': FRAMEWORK_ID, 'implementation': IMPLEMENTATION_ID,
        'compression_cost_model': COST_MODEL,
        'schema_version': 2,
        'insights': predictions['insights']['conditional_compressed'],
        'source_sha256': sha(output/'predictions.json'),
    })
    print(json.dumps({'state': 'completed', 'output': str(output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
