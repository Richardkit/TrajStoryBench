"""Evidence exporter owned by v5; no historical experiment controller."""
import ast
import json
from pathlib import Path
from trajstoryagent_v5.code_generation import extract_python_source
from trajstoryagent_v5.execution import PythonCodeExecutor, validate_generated_code
from trajstoryagent_v5.llm import OpenAICompatibleClient
from trajstoryagent_v5.providers import MODELS

ARMS = ('conditional_only', 'conditional_compressed')

def read(path):
    from .checkpoint_paths import relocate_decoded, relocate_path
    return relocate_decoded(json.loads(Path(relocate_path(str(path))).read_text(encoding='utf-8')))

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)

def client(slug):
    model, url = MODELS[slug]
    return OpenAICompatibleClient(model=model, base_url=url, temperature=0,
        reasoning_effort='none', max_tokens=10000, timeout_seconds=180, max_retries=2)

def check_export_code(code):
    validate_generated_code(code)
    # Additional containment of this narrow, computation-only exporter. This
    # remains a guardrail, not an OS sandbox (same limitation as base A2).
    allowed = {'pandas', 'numpy', 'math', 'datetime', 'collections', 'itertools',
               'scipy', 'sklearn', 'trajstoryagent_v5'}
    forbidden_methods = {'read_csv','read_parquet','read_json','read_excel','to_csv','to_json',
                         'to_pickle','write_text','write_bytes','unlink','remove','rmdir',
                         'save','savefig','load','dump','dumps','system','popen'}
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Import):
            if any(a.name.split('.')[0] not in allowed for a in node.names):
                raise ValueError('exporter import outside computation allowlist')
        if isinstance(node, ast.ImportFrom) and (node.module or '').split('.')[0] not in allowed:
            raise ValueError('exporter import outside computation allowlist')
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in {'open','getattr','setattr','globals','locals'}:
                raise ValueError('direct filesystem/dynamic access is disallowed')
            if isinstance(node.func, ast.Attribute) and node.func.attr in forbidden_methods:
                raise ValueError('direct filesystem access is disallowed: ' + node.func.attr)
        if isinstance(node, ast.Attribute) and node.attr in {'input_files','output_dir','__dict__','__class__'}:
            raise ValueError('exporter may not access filesystem handles or dynamic internals')

EXPORT_CONTRACT = '''
Create a small supplemental computation for the FIXED QUESTION using full named inputs.
This is a research prototype: export at most THREE relevant numerical measures with
one or two categorical conditioning dimensions. Do not explore adjacent questions.
Use the old code as a field/metric guide, but correct a wrong or insufficient statistic
when the question requires a different relation. Never reuse a conclusion as data.
Useful measures: per-entity/trip/day arrival or duration; count per complete time bin;
distance/speed; or 0/1 indicators for a precisely computed event or sequence relation.
For nearest-next-destination questions compute actual next-stop vs available candidates;
sequence similarity alone cannot answer that question. Do not use linear mean/SD of
angles, clock times crossing midnight, or raw lat/lon as a surrogate for route shape.
If this numeric conditional grammar does not fit, export no measures and explain why.

API (these exact classes already exist):
from trajstoryagent_v5.analysis_tools import AnalysisSession
from trajstoryagent_v5.conditional_math import EvidenceCollector
session = AnalysisSession.from_environment()
df = session.load_table('main')  # use ONLY named file IDs from input catalogue
collector = EvidenceCollector()
collector.add_measure(frame, value='numeric_column', conditions=['group_column'],
    observation_id=['entity_id', 'date'], unit='minutes', observation_unit='entity-day',
    description='arrival time', scope='observed weekdays, all specified users',
    kind='continuous')  # or kind='indicator' for EXACT 0/1 values
collector.finish()  # required; or finish(reason='unsupported relation ...') if no measures

The frame contains one record per stated observational unit, not duplicated GPS points
masquerading as trips/users. observation_id columns must uniquely identify every row.
Condition columns are complete, meaningful categorical partitions (<=12 values each).
Use existing meaningful bins; if needed state all bin boundaries explicitly. Never
select top groups, arbitrary IDs, or most favorable subsets to satisfy the size cap.
Preserve all observations within the stated question scope. Include zeros for complete
observed intervals where appropriate; do not invent zero activity outside observation.
Numerical values MUST be computed, not typed as answers. Null handling is explicit.
Validate join cardinality. Source tables must never be modified. No network, os, sys,
pathlib, direct file reads/writes, plots, session.emit_result(), prints, or narrative.
The only artifact writer is collector.finish(). pandas/numpy/math/scipy/sklearn and
the two imported project APIs are available. Return Python source only.
'''

def observations(slug, evidence, bundle, profile, directory):
    cached = directory/'observations.json'
    if cached.exists():
        return read(cached)
    old_code = Path(evidence['code_path']).read_text(encoding='utf-8')
    inputs = {'goal': bundle['goal'], 'context': bundle.get('context', ''),
              'files': bundle['files'], 'question': evidence['question'],
              'granularity': evidence['analysis_granularity'],
              'profile': profile}
    # Resume an interrupted export, or recover a program rejected by the old
    # overbroad pandas.rename guard. Original responses/executions remain intact.
    for index, saved in enumerate(sorted(directory.glob('export_attempt_*/response.txt'))):
        replay=directory/f'saved_program_replay_{index+1}'
        if (replay/'execution.json').exists():
            continue
        try:
            code=extract_python_source(saved.read_text(encoding='utf-8'))
            check_export_code(code)
            result=PythonCodeExecutor(timeout_seconds=180).run(code,replay,
                input_files={f['file_id']:f['path'] for f in bundle['files']})
            write(replay/'execution.json',{'source_response':str(saved),'success':result.success,'diagnostic':result.diagnostic})
            if result.success:
                payload=read(replay/'observations.json')
                if isinstance(payload.get('measures'),list) and len(payload['measures'])<=3:
                    write(cached,payload)
                    return payload
        except Exception as exc:
            write(replay/'error.json',{'error':str(exc)})
    prompt = EXPORT_CONTRACT + '\nMODEL-FACING INPUTS:\n' + json.dumps(inputs, ensure_ascii=False)[:45000]
    prompt += '\nPREVIOUS EXECUTED ANALYSIS (read-only context):\n' + old_code[:25000]
    previous = error = ''
    llm = client(slug)
    for attempt in range(2):
        folder = directory/f'export_attempt_{attempt+1}'
        if (folder/'response.txt').exists():
            folder = directory/f'export_recovery_attempt_{attempt+1}'
        folder.mkdir(parents=True, exist_ok=True)
        repair = ('\nPrevious program:\n' + previous + '\nRepair this failure:\n' + error[-6500:]) if error else ''
        request = prompt + repair
        write(folder/'request.json', {'system': 'Export computed observational evidence; Python only.', 'user': request})
        response = llm.complete(system='Export computed observational evidence; Python only.', user=request, json_mode=False)
        (folder/'response.txt').write_text(response, encoding='utf-8')
        try:
            previous = extract_python_source(response)
            check_export_code(previous)
            result = PythonCodeExecutor(timeout_seconds=180).run(previous, folder,
                input_files={f['file_id']: f['path'] for f in bundle['files']})
            write(folder/'execution.json', {'success': result.success, 'diagnostic': result.diagnostic})
            if not result.success:
                raise ValueError(result.diagnostic)
            payload = read(folder/'observations.json')
            if not isinstance(payload.get('measures'), list) or len(payload['measures']) > 3:
                raise ValueError('invalid collector output')
            write(cached, payload)
            return payload
        except Exception as exc:
            error = str(exc)
            write(folder/'error.json', {'error': error})
    raise RuntimeError('Evidence export failed: ' + error[-1200:])
