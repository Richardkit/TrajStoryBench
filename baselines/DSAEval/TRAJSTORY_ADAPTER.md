# DSAEval TrajStory adapter

The Python entry `python -m baselines dsaeval` targets `trajstory_multi_role.py`, protocol
`dsaeval_separate_context_roles_two_rounds_v1`.

- The workflow has two planner rounds, with three analyst
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
  SDK retries are disabled so five failures means five
  visible attempts, not five groups of hidden SDK retries. Record total attempts,
  charged calls and network failures separately. No semantic/format repair turns
  or additional model calls are introduced by the transport retry policy.
- Exact quotation checks require interpreted/verified insight citations to exist
  in the tool output. Verification is model-based, not independent numerical
  reexecution. Code and observations remain auditable per role and branch.
- Related/coherent insights are allowed; no distinct-insight requirement.
  Partial approved insights remain eligible without a summary or six outputs.
  If synthesis fails, retain approved branch text without manufacturing a summary.
- Use a fresh results directory for each run.

## Commands

Run from the repository root in a Python environment with
`baselines/DSAEval/requirements.txt` installed. The default Jupyter kernel is
`python3`; use `--kernel-name NAME` to select a registered kernel.

Preparation only, no API or Jupyter execution:

```bash
python -m baselines dsaeval --case-ids danish_ais_01_shared_corridor_day --results-dir results/dsaeval/gpt/runs/budget83_new --model gpt-6-luna --max-model-calls 83 --dry-run
```

Use the same Python entry with `--model qwen3.8-flash` and a fresh directory under
`results/dsaeval/qwen/runs/` for Qwen. Remove `--dry-run` to generate.
The default configuration uses temperature 0, reasoning disabled and text-only
observations. Set `OPENAI_API_KEY` for GPT or `DASHSCOPE_API_KEY` for Qwen.

Evaluation uses two independent GPT judge rounds. Compute reference maxima and
Coverage@0.6 within each round before averaging the two round results.
