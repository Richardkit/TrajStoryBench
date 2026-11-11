# AgentOrchestra TrajStory adapter

Protocol: `native_planner_bounded_indexed_multi_role`.

- The native planner chooses questions, parallel dispatch and completion.
  The four task roles use separate contexts and the existing native limits.
- The runner defaults to `--max-model-requests 83` per case.
  This is one shared counter for Planner, all task roles, Analyst internal calls,
  corrections and final synthesis/completion. It is not 83 calls per agent.
  Reservations are atomic across parallel dispatches; different cases start fresh.
  The next invocation is rejected before calling the provider when all slots are
  used. Exhaustion fails explicitly as `ModelRequestBudgetExceeded`; no automatic
  partial report or uncounted finalization call is issued. The native 10-round
  and 50-Analyst-step guards also remain active and can stop a case earlier.
- Budgeted GPT/Qwen calls disable hidden OpenAI SDK retries. The case counter
  centrally retries connection failures, timeouts and retryable HTTP statuses
  (408/409/429/500/502/503/504); these do not consume the 83 model slots.
  Five consecutive network failures across the sample stop it explicitly as
  `SampleNetworkFailureLimitExceeded`. A non-network response resets the streak;
  once stopped, later in-flight responses cannot restart the sample. Already
  in-flight requests may finish, but no new request is sent after the stop.
  Model corrections still count. Tool executions, deterministic artifact reuse
  and external judges do not consume generation requests. Results include
  `request_budget.requests_used` (excluding network failures), total
  `requests_started` (including network attempts), network failures/streak and
  separate role totals. Parallel in-flight slots are released on network failure.
- The planner context defaults to `planner_context_max_chars=6000`. The planner sees one
  current entry per artifact ID, short task hints and selector counts. Repeated
  errors are merged into an occurrence count; raw quotes, stdout, prior plans
  and every previous response are not replayed. Full dispatch logs are preserved.
  If hints do not fit, only hints are removed; source IDs/selectors remain.
- Once a synthesis succeeds, the planner receives the exact latest synthesis
  once and role-completion status, rather than also receiving its old inputs.
  If a new upstream artifact or unresolved handoff appears afterward, the report
  becomes stale: its text leaves context and the artifact/selector index returns.
  A stale report cannot be submitted as completion. A fresh synthesis uses the
  evidence available before its dispatch, so parallel new computation is not hidden.
  The full planner request still has the separate 24,000-character guard.
- Interpreter/verifier prompts explicitly require JSON objects with candidates
  or reviews lists. Bare lists are wrapped without altering the findings or
  relaxing citation, count, size, evidence or role-isolation checks.
- Verifier inputs encode prior exact quotations as character spans into the
  source chunk. The source text is sent once; original quotes remain on disk.
  Reviewer outputs still require exact quotes and are checked normally.
- Indexed mode reuses successful dispatch results only within the same case when
  role, task and files match exactly. JSON key ordering and repeated/reordered
  evidence IDs are normalized; different instructions are never merged by meaning.
  Identical simultaneous dispatches share one execution. Reuse is logged as
  `dispatch_reused`; it does not manufacture a new role context or new evidence.
- Deterministic handoff failures are reused while the artifact catalog is unchanged;
  model/transport failures remain retryable. Two consecutive rounds containing only
  reused dispatches stop explicitly as a failure without rerunning agents. This is
  an adapter stopping rule, in addition to the native round/step limits.
- Compact task hints omit the shared spec path so questions remain distinguishable.
  Execution indexes identify the latest two successful steps as preferred evidence;
  the full step/chunk selectors remain available. No evidence is silently truncated
  or substituted, and citations still require exact source excerpts.

## Commands

Run from the repository root in the framework's Python environment.
`--input-csvs` selects the projected input tables. Set `OPENAI_API_KEY` for GPT
or `DASHSCOPE_API_KEY` for Qwen.

```bash
python -m baselines agentorchestra --case-ids danish_ais_01_shared_corridor_day --results-dir results/agentorchestra_new --model-name openai/gpt-6-luna --input-csvs --max-model-requests 83 --dry-run
```

Remove --dry-run to call the model. Use a fresh output directory.
Current output records are tagged native_planner_bounded_indexed_multi_role.
The default generator is openai/gpt-6-luna; use openai/qwen3.8-flash for Qwen.
