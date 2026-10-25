# Pandas multi-role workflow

This is an explicitly separate variant of the original single-agent Pandas baseline.
It reuses LangChain's official DataFrame factory, PythonAstREPLTool and the existing
full-stdout/shared-namespace compatibility fix. It does not modify the native baseline,
v4/v5, DSAEval, source data, goals or reference insights.

## Organization

1. Planner sees the goal, case context, named table schemas and first five rows.
2. Three private analysis branches use copied full DataFrames and separate Python
   namespaces. Each branch can inspect/compute/repair within its own tool loop.
3. A separate interpreter receives its assigned question and execution transcript.
   The analyst's unverified final note is not supplied as factual evidence.
4. A separate verifier receives the candidate, code and actual tool outputs. Rejected
   or failed branches yield no scored insight; no re-generation follows a rejection.
5. Round-two planner sees the original input metadata plus approved round-one insights,
   citations, limitations and execution artifact paths. It selects three follow-ups.
   Each follow-up analyst receives only the selected parent records, not every role's history.
6. Synthesizer selects up to six approved insights by ID and writes the summary. Evaluated
   insight sentences are copied exactly, not rewritten by the summarizer.

Every role call has its own model-facing input/output artifact. All roles use the same
backbone/model configuration. Only analyst branches get the DataFrame/Python tools.
Branches are currently executed sequentially to avoid shared stdout-capture races.
There are no mathematical conditional-decomposition/compression stages in this Pandas
variant; those remain a separate TrajStory method. It is not named as original AgentPoirot.

## Budgets and outputs

Default global budget: 83 **model requests**, covering all roles, not 83 per worker.
Output fields `request_limit` and `requests_used` refer to this request budget;
the two analysis rounds are separate workflow stages.
Reserve up to 15 calls for 2 planners, 6 interpreters, 6 verifiers and 1 synthesizer.
Each of six analysts can use floor((83-15)/6)=11 calls, so the maximum is 81 with default
settings. Unused branch calls are not redistributed. The interpreter may still use the
executed evidence if an analyst reaches its limit without a textual final note.
No automatic format re-tries or whole-case retries are added by this workflow.
For this role workflow, SDK retries are disabled. Transport connection failures and
timeouts retry the same model request without restarting the analysis or consuming
the request budget. Five consecutive network failures stop the entire case; a
successful response resets the streak. Other errors are not retried by this policy.
`requests_used` counts logical requests excluding terminal network failures;
`model_calls_started` retains attempted invocation telemetry. Network failure totals
and the current streak are recorded separately. Usage remains provider-reported.

`status.json`, `job_result.json`, `model_usage.json`, `parsed_output.json` and `insights.json`
record progress, failures, actual usage and output. `insight` is the branch text field;
parsed_output.answer.insights is the compatibility string list for existing evaluation.
Empty/failed outputs count as zero; partial supported outputs remain eligible.
Verification is an LLM generation check plus structural exact-output citation checking,
not an independent proof that every generated computation is correct.
Benchmark scores must be produced separately with two independent GPT-6-Luna rounds.

The original 25-case single-agent results are not comparable as a pure agent-count
ablation: this variant adds planning, verifier and output-selection changes and reallocates
the call budget. Keep both protocols separately labeled when testing them.

## Commands

Use the existing Pandas environment and a fresh output directory:

```powershell
& B:/TrajStory/.runtime/pandas_agent/Scripts/python.exe -X utf8 scripts/run_pandas_multi_role.py --case-id <case_id> --model gpt --results-dir results/pandas_multi_role/gpt/<fresh-case> --run
& B:/TrajStory/.runtime/pandas_agent/Scripts/python.exe -X utf8 scripts/run_pandas_multi_role.py --case-id <case_id> --model qwen --results-dir results/pandas_multi_role/qwen/<fresh-case> --run
```

Without `--run`, only prepare inputs/manifests. `--smoke-offline` exercises the actual
DataFrame/Python tool pipeline with scripted model outputs and cannot be scored.
