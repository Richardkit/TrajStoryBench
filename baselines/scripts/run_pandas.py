"""One Pandas run entry: choose a workflow; existing model/run arguments pass through."""
from __future__ import annotations

import sys as _baseline_sys
from pathlib import Path as _BaselinePath
_baseline_repo = next(p for p in _BaselinePath(__file__).resolve().parents if (p / "baselines" / "__init__.py").is_file())
_baseline_sys.path.insert(0, str(_baseline_repo))
import baselines  # Register the shared baseline module location.

import argparse
import importlib
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
WORKFLOWS={
    'roles':('pandas_multi_role.role_runner',{}),
    'native':('pandas_multi_role.native_runner',{}),
    'checkpoint':('pandas_multi_role.checkpoint_runner',{'policy':'minimal'}),
    'checkpoint-bounded':('pandas_multi_role.checkpoint_runner',{'policy':'bounded'}),
    'legacy-react':('pandas_multi_role.react_runner',{}),
}


def main(argv=None):
    args=list(sys.argv[1:] if argv is None else argv)
    parser=argparse.ArgumentParser(description=__doc__,add_help=False)
    parser.add_argument('--workflow',choices=WORKFLOWS)
    selected,remaining=parser.parse_known_args(args)
    if selected.workflow is None:
        parser.print_help()
        print('Workflow help: --workflow NAME --help. No workflow is executed by default.')
        if args and args not in (['--help'],['-h']):parser.error('Choose --workflow')
        return 0
    module,kwargs=WORKFLOWS[selected.workflow]
    engine=importlib.import_module(module)
    previous=sys.argv
    try:
        sys.argv=[str(Path(__file__)),*remaining]
        result=engine.main(**kwargs)
    finally:
        sys.argv=previous
    return 0 if result is None else result


if __name__=='__main__':raise SystemExit(main())
