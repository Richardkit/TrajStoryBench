# TrajStory adapter entries

## Original-workflow entry (prepared 2026-10-18; no model test started)

`python baselines/DSAEval/trajstory_origin.py` runs `trajstory_origin.py`
from the repository root. It loads the unmodified upstream agent,
prompts, tools, utilities and Jupyter sandbox pinned in `origin_upstream/`.
One task uses one agent, one continuous conversation and one persistent kernel;
there are no fixed branches, independent verification roles or exact-citation
filters. Only input/output, provider settings, Windows runtime and the requested
83-call/five-network-failure policy are adapted. The original 20-call budget is
available explicitly. Generation requires `--run`; dry-run is the default.
Use `--help` for the available arguments.
The existing multi-role entry and results are retained separately.

## Existing multi-role entry: separate-context roles (2026-10-15)

The Python entry `python -m baselines dsaeval` targets `trajstory_multi_role.py`, protocol
`dsaeval_separate_context_roles_two_rounds_v1`. The default budget and transport
policy below apply to new runs. Completed GPT/Qwen experiments retain their
recorded configurations and frozen source snapshots.

- Match the latest Pandas role workflow: two planner rounds, three analyst
  branches per round; each branch hands executed evidence to a separate
  interpreter, then an independent verifier; one final synthesis role.
- Each role gets a separate provider client and private message context.
  Planning round two receives approved round-one evidence explicitly. Each
  analyst receives only its question, file paths and selected parent evidence;
  interpreters/verifiers receive the assigned question and execution transcript.
  There is no whole-case shared conversation.
- Retain DSAEval file-path inputs and native Jupyter code tool. No DataFrames,
  schemas or sample rows are preloaded for the planner. Each analyst loads files
  itself and has a private persistent Jupyter kernel and its own notebook.
  Kernels/variables are not shared across branches, matching Pandas private REPLs.
- Reuse the native science prompt for analysts, correcting runtime information
  and appending the role/output duties. Text-only planning, interpretation,
  verification and synthesis use explicit role prompts. Upstream `prompts.py`
  and `data_agent.py` are unchanged by this role conversion.
- Default whole-case model budget is 83. Successful responses and non-network
  API failures count; network/transport failures do not. Reserve at most 15
  text-role calls (2+6+6+1). Each analyst has `(83-15)//6=11` decisions, hence
  at most 81 counted calls on the full path; two calls remain unused.
- Retry the identical request after a network failure with short bounded backoff.
  Abort after five consecutive transport failures; a successful response resets
  the streak. Save the partial result as `interrupted` and propagate the error so
  callers stop rather than advance to more cases or scoring. Citation, JSON,
  code-execution and HTTP status errors are not transport errors.
  SDK retries are disabled in the current launcher so five failures means five
  visible attempts, not five groups of hidden SDK retries. Record total attempts,
  charged calls and network failures separately. No semantic/format repair turns
  or additional model calls are introduced by the transport retry policy.
- Exact quotation checks require interpreted/verified insight citations to exist
  in the tool output. Verification is model-based, not independent numerical
  reexecution. Code and observations remain auditable per role and branch.
- Related/coherent insights are allowed; no distinct-insight requirement.
  Partial approved insights remain eligible without a summary or six outputs.
  If synthesis fails, retain approved branch text without manufacturing a summary.
- Fresh results directory required. Old locked and native-single-agent entries
  and outputs remain available for historical reproduction and must not be mixed.

Preparation only, no API or Jupyter execution:

```bash
python -m baselines dsaeval --case-ids danish_ais_01_shared_corridor_day --results-dir results/dsaeval/gpt/runs/budget83_new --model gpt-6-luna --max-model-calls 83 --dry-run
```

Use the same Python entry with `--model qwen3.8-flash` and a fresh directory under
`results/dsaeval/qwen/runs/` for Qwen. Timestamped historical batch/retry scripts
are frozen experiment entries, not the default entry for the new policy.

## Previous native single-agent adapter (2026-10-14)

`python baselines/DSAEval/trajstory_native.py` is the separate single-agent entry. It retains the upstream `DataScienceAgent`
tool loop with one invocation and one persistent Jupyter kernel per case.

- Pass only staged CSV paths and the goal; the agent loads and inspects files.
  No DataFrames, computed schemas or gold answers are supplied.
- Reuse `DSAgent_PROMPT_LLM` / `DSAgent_PROMPT_VLM` from unchanged `prompts.py`.
  Only correct the paths, CPU environment, actual iteration limit and code timeout.
- Keep the native four-section Markdown report. The user request adds six
  factual insights and a summary, represented in one JSON block inside Results
  & Explanation for downstream scoring. Save the full report and notebook.
- Native freedom to perform repeated successful calculations, change direction,
  verify results and repair errors. No question registration/locking, six-stage
  schedule, successful-execution cap, checkpoints or forced answer-only turn.
- Default 83 total model decisions per case (including the report), configurable
  with `--max-iterations`. This is a ceiling; early completion is allowed.
  Exhaustion is failure with no extra model call. Transport retries are separate.
  Code calls default to 180 seconds, not the upstream server timeout.
- Preserve the upstream text-only versus multimodal observation switch.
- Use a fresh empty results directory. New protocol: `native_free_analysis_v1`.
  These results must not be combined with locked-question results.
- Local Jupyter is not an enforced filesystem isolation boundary; gold answers
  are excluded from staged inputs and prompts, not inaccessible to arbitrary code.

Example (no model call):

```bash
python baselines/DSAEval/trajstory_native.py --case-ids danish_ais_01_shared_corridor_day --results-dir results/dsaeval_native_new --model gpt-6-luna --max-iterations 83 --dry-run
```

Evaluation uses two independent GPT judge rounds; calculate
coverage@0.6 in each round before averaging. Historical DeepSeek guidance below
does not apply to new evaluations.

## Historical six-exploration adapter (preserved)

The remainder documents the legacy `trajstory_six_explorations.py` entry.
Its code, tests and old results remain unchanged. Its default worst-case model
decision budget is 49 (up to two question registrations per exploration),
superseding the earlier 43-call statement below.

This directory is the upstream DSAEval repository at commit
`160f3d577e8a1b6a7e4b6ae8c5db168ba9d188f1`, with a small adapter in
`trajstory_six_explorations.py` and two backward-compatible extensions to the
agent and sandbox. The original `runner.py` and DSAEval benchmark remain usable.

## What the adapter runs

For each TrajStory case, the adapter creates **one** DSAEval agent and **one**
Jupyter kernel. It gives the agent six requests in order:

1. Explorations 1–3 seek distinct initial directions.
2. Explorations 4–6 deepen, check, or challenge earlier results.
3. One final model call summarizes the six findings without new analysis.

The first three requests share conversation and kernel state. Thus the
execution is serial throughout; it is **not** TrajStoryAgent's three
independent first-round branches followed by three follow-ups. The adapter
does not add a separate planner agent or parallel workers to DSAEval.

Since 2026-09-21 the adapter uses `locked_single_question_v1`:

- First the same agent registers one atomic question in a tool-free call. It
  must not bundle multiple hypotheses or screen candidate directions.
- The execution phase locks that exact question. Tool arguments and the final
  JSON must repeat it. A changed question or a batch of multiple tool calls is
  rejected before execution.
- One self-contained code call must perform the relevant loading, calculation
  and optional plotting for that question. After the first error-free execution,
  tools are immediately removed and only the finding may be returned. An empty
  or uninteresting result is not permission to try a different question.
- Only failed executions allow repair of the same calculation. The execution
  phase still has at most six model calls (at most five execution attempts and
  an answer-only call), not six new questions. Rejected calls also consume turns.
- Planning and summary offer no tools; unsolicited tool calls on answer-only
  turns are never executed. Fixed kernel initialization is not question exploration.

A successful no-repair case uses six question registrations, six code executions,
six answers and one summary: **19 model calls, 6 analysis executions**. The
default worst-case logical model-call budget is 43 including planning and summary
(provider transport retries are separate). The six-question budget cannot be
increased by increasing `--max-iterations-per-exploration`.

Results record `exploration_protocol`, `code_execution_count`,
`successful_analysis_count`, and `repair_execution_count`, as well as the locked
question and blocked-call count in each stage log. Failed cases may complete
fewer than six questions; they are not silently replaced. Question identity
checks do **not** prove the semantic scope of arbitrary Python: bundling hidden
subquestions inside one code block is forbidden by the prompt but still needs
code audit. Do not describe this as a formal semantic verifier.

Use a fresh output directory. The adapter refuses old-protocol result directories,
and the Qwen125 launcher no longer accepts old-protocol rows as completed new runs.
Previously produced results are unchanged and must not be relabeled as this protocol.

The input folder contains copies of only the case's declared CSV files. The
gold insights, summary, and case JSON are not copied into the agent's input.
Each case gets a separate kernel and conversation. The output format is the
usual `trajstory_case_results.json` with `pred_insights` and `pred_summary`,
plus per-exploration logs and an executable notebook. Evaluation is separate
and can use the same DeepSeek judge as the other frameworks.

## Setup and dry run

Install `baselines/DSAEval/requirements.txt` in a suitable Python environment.
Use that environment's `python` in the commands below. The default kernel name is
`python3`; `--kernel-name dsa_eval` selects that named kernel if registered.
This adapter does not need the original DSAEval datasets or reference answers.

From the TrajStory repository root:

```bash
python baselines/DSAEval/trajstory_six_explorations.py \
  --case-ids danish_ais_01_shared_corridor_day \
  --results-dir results/dsaeval_six_dry_run \
  --model qwen3.8-flash --dry-run
```

The dry run validates all selected CSV paths and prints the exact input file
list without starting a kernel or using an API key.

## Actual generation

The adapter selects the provider-specific key environment variable from the
model name: `OPENAI_API_KEY` for GPT, `DASHSCOPE_API_KEY` for Qwen, and
`DEEPSEEK_API_KEY` for DeepSeek. It defaults to each provider's corresponding
API endpoint; `--base-url` and `--api-key-env` override those choices. Qwen
defaults to the Beijing DashScope endpoint used by the current comparison,
and disables thinking.

```bash
python baselines/DSAEval/trajstory_six_explorations.py \
  --case-ids danish_ais_01_shared_corridor_day \
  --results-dir results/dsaeval_six_qwen \
  --model qwen3.8-flash
```

Add `--is-multimodal` only when the selected model endpoint accepts image
inputs. In text mode, the agent is instructed to print numbers that support
its conclusions. `--code-timeout` defaults to 180 seconds per Jupyter call.
The adaptation does not invoke DSAEval's own evaluator or expose reference
answers to the generation agent.

## Fidelity and validation limits

The original `DataScienceAgent` tool loop, persistent conversation, Jupyter
execution, and notebook capture are retained. The TrajStory adapter replaces
the original task prompt and Markdown report format with six case-specific
requests and a JSON result contract. The single-question registration and
successful-execution limit are additional benchmark budget adaptations, not
features claimed to exist in upstream DSAEval. `--max-iterations-per-exploration`
controls repair/formatting turns, not the number of questions or successful analyses.

The adapter stages only declared CSV inputs and excludes reference answers
from the prompt. DSAEval's original Jupyter kernel is a local Python process,
so the prompt is not a filesystem access control boundary. For a benchmark
run that requires enforced input isolation, launch the runner in an isolated
container or filesystem view containing only the staged CSVs and output path.
The offline tests verify question locking, multi-call rejection, stopping after
success, bounded error repairs, and six successful analyses in a complete case.
The following Danish AIS 12 pilot results predate the locked-question protocol:
the pilot
then completed with both GPT-5.6 Luna and Qwen3.8-Flash under the final
answer-only request flow:

| Model | Explorations | Model calls | Input tokens | Output tokens |
| --- | ---: | ---: | ---: | ---: |
| GPT-5.6 Luna | 6/6 | 15 | 84,243 | 4,811 |
| Qwen3.8-Flash | 6/6 | 33 | 315,569 | 9,616 |

The pilot results and notebooks are in
`../results/dsaeval_pilot_gpt56_luna_danish12_answeronly_20260920/` and
`../results/dsaeval_pilot_qwen38_flash_danish12_answeronly_20260920/`.
These runs validate execution and output formatting for one case. Numeric
claims still need the same external evidence evaluation as other frameworks;
the Qwen pilot contains a vessel-type proportion statement that conflicts with
its own reported counts.
