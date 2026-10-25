"""Run one baseline using its existing arguments; append --help for its options. Use the selected framework's Python environment."""
import argparse
import runpy
import sys
from . import REPO_ROOT, BASELINES_ROOT
ENTRIES = {
    "llm-only": ("module", "llm_only_baseline"),
    "evaluate": ("module", "baseline_support.evaluate"),
    "codegen": ("path", "baselines/scripts/run_codegen_baseline.py"),
    "pandas": ("path", "baselines/scripts/run_pandas.py"),
    "dsaeval": ("path", "baselines/DSAEval/trajstory_multi_role.py"),
    "agentorchestra": ("path", "baselines/DeepResearchAgent/examples/run_trajstory.py"),
    "insightbench": ("path", "baselines/scripts/run_insightbench_generation_no_rouge.py"),
}
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=ENTRIES)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    kind, target = ENTRIES[args.method]
    sys.argv = [target, *args.arguments]
    if kind == "module":
        runpy.run_module(target, run_name="__main__")
    else:
        path = REPO_ROOT / target
        sys.path.insert(0, str(path.parent))
        runpy.run_path(str(path), run_name="__main__")
if __name__ == "__main__":
    main()
