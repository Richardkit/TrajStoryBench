"""Adapt CodeGen outputs to the existing frozen, two-round GPT judge."""

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
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from codegen_baseline import PROTOCOL
from codegen_baseline.workflow import parse_answer, read, sha, write


def collect(directories, generator):
    rows, identifiers = [], set()
    for directory in directories:
        directory = Path(directory).resolve()
        manifest = read(directory / "manifest.json")
        state = read(directory / "status.json")
        if manifest["protocol"] != PROTOCOL or manifest["generator"] != generator:
            raise ValueError("Mixed CodeGen protocol or generator")
        if manifest["validation_mode"] != "benchmark":
            raise ValueError("Offline outputs must never enter benchmark scoring")
        if state["state"] not in {"returned", "empty", "failed"}:
            raise ValueError("Case generation has not finished")
        cid = manifest["case_id"]
        if cid in identifiers:
            raise ValueError("Duplicate case ID; select one predeclared result")
        identifiers.add(cid)
        if sha(ROOT / manifest["json_path"]) != manifest["case_json_sha256"]:
            raise ValueError("Reference case changed since generation")
        eligible = bool(state.get("eligible_for_scoring"))
        parsed = directory / "parsed_output.json"
        if eligible:
            answer = parse_answer(json.dumps(read(parsed)["answer"]))
            execution = read(directory / "execution_result.json")
            response = read(directory / "answer_response.json")
            raw_answer = parse_answer(response["choices"][0]["message"]["content"])
            if (not answer["insights"] or answer != raw_answer or state["execution_attempts"] != 1
                    or state["state"] != "returned" or not execution["success"]
                    or execution["execution_attempts"] != 1):
                raise ValueError("Invalid scored CodeGen output")
        rows.append(dict(case_id=cid, json_path=manifest["json_path"], directory=str(directory),
            eligible_for_scoring=eligible, source_sha256=sha(parsed) if parsed.exists() else None,
            status_sha256=sha(directory / "status.json"), manifest_sha256=sha(directory / "manifest.json")))
    if not rows:
        raise ValueError("No generation results supplied")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--generator", choices=["gpt", "qwen"], required=True)
    parser.add_argument("--run", action="store_true", help="Send two independent GPT judge rounds")
    args = parser.parse_args()
    rows = collect(args.generation_dirs, args.generator)
    target = args.batch_dir.resolve() / "generation.json"
    if target.exists() and read(target) != rows:
        raise ValueError("Frozen scoring inputs changed; use a fresh batch directory")
    write(target, rows)
    from baselines.scripts import evaluate_pandas_checkpoint_batch as judge
    judge.GENERATOR = args.generator
    frozen = judge.freeze(args.batch_dir.resolve())
    if args.run:
        report = judge.evaluate(args.batch_dir.resolve(), frozen)
        report.update(framework=PROTOCOL, fixed_insight_quota=False,
                      output_count_note="Actual supported insight count; six is not a requirement.")
        # The common evaluator's historical complete/partial labels assume six insights.
        for name in ("complete_outputs", "partial_outputs"):
            report.pop(name, None)
        write(args.batch_dir / "report.json", report)
    print(json.dumps(dict(cases=len(rows), requests_per_round=len(frozen["jobs"]),
                          judge="gpt-6-luna", rounds=2, run=args.run)))


if __name__ == "__main__":
    main()
