"""Offline runtime check: configuration, all framework registries and analysis tool."""

import sys as _baseline_sys
from pathlib import Path as _BaselinePath
_baseline_repo = next(p for p in _BaselinePath(__file__).resolve().parents if (p / "baselines" / "__init__.py").is_file())
_baseline_sys.path.insert(0, str(_baseline_repo))
import baselines  # Register the shared baseline module location.

import asyncio
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
FRAMEWORK = ROOT / 'baselines/DeepResearchAgent'
sys.path.insert(0, str(FRAMEWORK))
os.environ['TRAJSTORY_MINIMAL_IMPORTS'] = '1'
for name in ['src.config', 'src.agent', 'src.environment', 'src.interaction',
             'src.memory', 'src.model', 'src.prompt', 'src.skill', 'src.tool',
             'src.version', 'src.session', 'src.task']:
    importlib.import_module(name)
    print('IMPORT_OK', name)

from src.trajstory import build_case_spec, load_manifest_cases
from src.tool.trajstory_analysis import TrajstoryAnalysisTool

entries = load_manifest_cases(ROOT / 'trajstory_examples/insight/manifest.json',
                             trajstory_root=ROOT,
                             case_ids=['danish_ais_01_shared_corridor_day'])
case = build_case_spec(entries[0], trajstory_root=ROOT, sample_rows=3)
with tempfile.TemporaryDirectory(prefix='orchestra_check_', dir=ROOT / '.runtime') as folder:
    spec = Path(folder) / 'case_input.json'
    spec.write_text(json.dumps(case), encoding='utf8')
    tool = TrajstoryAnalysisTool(trajstory_root=str(ROOT), use_trajstory_helpers=False)
    result = asyncio.run(tool(spec_path=str(spec), code="print(len(df)); print(df.shape)"))
    assert result.success, result.message
    print('ANALYSIS_OK', result.message)
print('OFFLINE_CHECK_PASSED')
