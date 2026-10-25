"""Prepare CodeGen; generation/execution requires --run or --smoke-offline."""

import sys as _baseline_sys
from pathlib import Path as _BaselinePath
_baseline_repo = next(p for p in _BaselinePath(__file__).resolve().parents if (p / "baselines" / "__init__.py").is_file())
_baseline_sys.path.insert(0, str(_baseline_repo))
import baselines  # Register the shared baseline module location.

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from codegen_baseline.workflow import (Config, MODELS, ModelSession, OfflineClient,
    make_client, prepare, read, run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--model", choices=MODELS, default="gpt")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--max-model-calls", type=int, default=83)
    parser.add_argument("--code-timeout", type=int, default=180)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--smoke-offline", action="store_true")
    args = parser.parse_args()
    config = Config(request_limit=args.max_model_calls, code_timeout=args.code_timeout)
    out = args.results_dir.resolve()
    manifest = prepare(ROOT, args.case_id, args.model, out, config, args.smoke_offline)
    if not args.run and not args.smoke_offline:
        print(json.dumps(dict(state="prepared", case_id=args.case_id,
                              planned_model_responses=2, code_execution_limit=1)))
        return
    client = (OfflineClient(read(out / "model_input.json")["tables"][0]["path"])
              if args.smoke_offline else make_client(args.model))
    session = ModelSession(client, out, config)
    state = run(ROOT, out, manifest, session)
    print(json.dumps(state))
    if state["state"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
