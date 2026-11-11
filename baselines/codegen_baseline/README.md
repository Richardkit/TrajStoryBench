# CodeGen (DiscoveryBench adaptation)

The method generates a complete Python program, executes it **once**, then asks
the **same model in the same conversation** to generate insights followed by a
summary from the executed evidence. Normal completion uses two model responses.
There is no iterative exploration, separate planner/verifier, code repair, format
repair, answer-quality retry or automatic whole-case retry.

References:
- Paper: https://arxiv.org/html/2407.01725#S5.SS1
- Original one-execution tool: https://github.com/allenai/discoverybench/blob/main/agents/coder_utils.py

## Input and adaptation

Inputs come from the current 125-case catalog via `load_trajstory_case(...,
input_csvs=True)`. Only goal, dataset description, analyst role, projected CSV
paths, table roles and CSV headers reach generation. No original CSV fallback,
reference insight, reference question, intermediate gold hypothesis, workflow tag,
notebook, reference summary or model-based preprocessing is supplied. When an
input manifest is available, CSV hashes are checked against it.

Only the selected CSVs are copied into execution/input/. The generated program
loads these relative paths itself. A separate process receives an allowlist
environment without provider credentials, user Python paths or proxies. Audit
guards restrict reads to its execution area and Python runtime, writes to
execution/artifacts/, and block network/process access. These are Python-level
guardrails, **not an OS sandbox for adversarial code**. Inputs in the main
repository are not executed against or modified.

This adaptation uses two JSON responses rather than the original LangChain
action parser; a complete Python script must explicitly print evidence. It has a
fixed independent synthetic code-format example, not a trajectory task example.
The final output is `{"insights":["..."],"summary":"..."}`. There is no fixed
insight count; each insight is at most 180 whitespace-separated words. Plain and
JSON-fenced objects are accepted; invalid content is retained without model repair.
Figures may be saved, but the final model gets text evidence, not image pixels.

The program gets 180 seconds by default. Full stdout/stderr are retained on disk;
the final model sees at most approximately 60,000 bytes, split 80%/20% between
stdout/stderr, with deterministic prefix/suffix truncation disclosed to the model.
Execution errors or timeouts fail the case without a final synthesis call.

## Budget and evaluation

The common 83-request ceiling includes final synthesis; the ordinary method uses
only two responses. It does not spend unused calls. SDK retries are disabled.
Transport failures retry the identical request without consuming this budget;
five consecutive transport failures stop the case. A response resets the streak.
HTTP errors, empty/truncated/invalid responses and bad answers do not trigger
regeneration. Actual attempts, counted requests, provider usage and timings are
recorded; missing usage is explicit.

Models: GPT `gpt-6-luna` and Qwen `qwen3.8-flash`, temperature 0 and reasoning
disabled. Evaluation is explicitly separate: two independent `gpt-6-luna` rounds,
temperature 0, reasoning disabled, using the InsightBench-style judge. Reference
maxima and Coverage@0.6 are computed within each round before
averaging. Full-batch means include failed/invalid/empty cases as zero;
successful-only means are separately reported. Summary is not semantically scored.
Offline fake output is explicitly excluded from scoring.

## Commands

Use a fresh output directory for each invocation. Preparation is the default;
`--run` starts actual generation and code execution. Preparing a directory does
not make it reusable for another invocation: use a new directory for the run.

Run from the repository root in the Python environment for this baseline.

```bash
python -m baselines codegen --case-id danish_ais_01_shared_corridor_day --model gpt --results-dir results/codegen/gpt/PREPARE_NEW

# Use --model qwen for Qwen. The default call ceiling is 83.
python -m baselines codegen --case-id danish_ais_01_shared_corridor_day --model gpt --results-dir results/codegen/gpt/RUN_NEW --run

# Explicit offline smoke run; never eligible for scoring.
python -m baselines codegen --case-id danish_ais_01_shared_corridor_day --model gpt --results-dir work/codegen/OFFLINE_NEW --smoke-offline

# Freeze scoring inputs only; adding --run sends both GPT judge rounds.
python baselines/scripts/evaluate_codegen_baseline.py --generation-dirs results/codegen/gpt/RUN_NEW --batch-dir results/codegen/gpt/EVAL_NEW --generator gpt
```

Artifacts: manifest and source snapshot; model input; exact code/answer requests
and raw responses; usage/transport attempts; submitted program; full stdout/stderr;
bounded model observation; parsed output; terminal status and judge-compatible
job_result. Started directories cannot automatically be restarted or overwritten.
