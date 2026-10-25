"""Verify DSAEval's real persistent Jupyter session without an API call."""

import sys as _baseline_sys
from pathlib import Path as _BaselinePath
_baseline_repo = next(p for p in _BaselinePath(__file__).resolve().parents if (p / "baselines" / "__init__.py").is_file())
_baseline_sys.path.insert(0, str(_baseline_repo))
import baselines  # Register the shared baseline module location.

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'baselines/DSAEval'))
os.environ['JUPYTER_PATH'] = str(ROOT / '.runtime/dsaeval/share/jupyter')
from data_agent import DataScienceAgent
from jupyter_sandbox import JupyterSandbox

with JupyterSandbox(timeout=30, kernel_name='dsa_eval') as sandbox:
    assert not sandbox.last_execution_error, 'Default science imports failed'
    text, images = sandbox.execute_code("probe_frame = pd.DataFrame({'x': [1, 2, 3]}); print(probe_frame.x.sum())")
    assert '6' in text and not sandbox.last_execution_error, text
    text, images = sandbox.execute_code("print(len(probe_frame)); plt.plot(probe_frame.x); plt.show()")
    assert '3' in text and images and not sandbox.last_execution_error, text
    print('JUPYTER_EXECUTION_AND_PERSISTENCE_OK', len(images), 'plot(s)')
print('OFFLINE_CHECK_PASSED')
