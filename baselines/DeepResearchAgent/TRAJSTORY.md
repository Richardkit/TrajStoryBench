# Active: bounded indexed multi-role adapter (2026-10-16)

Protocol: `native_planner_bounded_indexed_multi_role`. This is an
AgentOrchestra adapter revision, not a TrajStoryAgent version.

- The native planner still chooses questions, parallel dispatch and completion.
  The four task roles still use separate contexts and the existing native limits.
- The active runner additionally defaults to `--max-model-requests 83` per case.
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
  Existing outputs are unchanged; future runs record their own request budget.
- This runner opts into `planner_context_max_chars=6000`. The planner sees one
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
  an adapter stopping rule, in addition to the native round/step limits. Generic
  AgentBus callers that do not enable indexed context retain their prior behavior.
- Compact task hints omit the shared spec path so questions remain distinguishable.
  Execution indexes identify the latest two successful steps as preferred evidence;
  the full step/chunk selectors remain available. No evidence is silently truncated
  or substituted, and citations still require exact source excerpts.

Historical experiment outputs and frozen source snapshots are not rewritten.
The interrupted Qwen batch must not resume into this protocol as if it were
the same experiment; use a fresh run directory when retesting.

# Historical: indexed multi-role adapter (2026-10-15)

After the frozen five-case v4 pilot, host-bound case paths are supplied directly
to analysts even when the planner omits them. Two identical premature done
responses without execution stop that analyst instead of repeating up to 50
steps. Interpreter records retain only actually cited source IDs for subsequent
verification. Citation limits are explicit in role instructions and errors.
These post-pilot wiring fixes have offline validation only; the completed pilot
remains labeled v4 and its scores are unchanged. Current protocol is
native_planner_indexed_multi_role_v4_1.

Duplicate-input/output guards (2026-10-15):
- Identical source text under different IDs is included once with source-ID
  aliases. All original provenance remains available for exact citation checks.
- Byte-equivalent candidate/review objects and repeated citations are removed;
  different populations, scopes and evidence IDs are not semantically merged.
  Exact duplicate final insight strings are removed without padding the count.
- Planner request body: at most 24,000 characters; API completion at most
  2,500 tokens. Analyst request body: at most 18,000 characters; API completion
  at most 2,400 tokens. These optional limits apply to this adapter only.
- Interpreter/verifier system+user input remains capped at 12,000 characters,
  completion at 1,800 tokens and accepted output at 6,000 characters.
  Synthesis input is capped at 12,000 characters, completion at 1,400 tokens,
  and final JSON at 4,000 characters. These are distinct character/token units.
- Each candidate insight is at most 900 characters with up to four exact
  citations, each at most 240 characters. Full execution records remain on disk.
  Oversize inputs fail before model calls; evidence must be split explicitly.
  These caps do not represent a common case-level budget across frameworks.

Protocol: native_planner_indexed_multi_role_v4. Default results directory:
results/agentorchestra_indexed_multi_role_v4. The paused v3 pilot remains
paused and its records are unchanged. No generation is started by this update.

Analysts compute one focused question. They do not generate six candidates,
repeat code/stdout in their completion note or print full case metadata.
Full code, outputs and failed repairs remain in each private tracer directory.
The planner receives only completion status, evidence ID and a small index
of successful execution steps. Latest-step tool feedback replaces repeated
cumulative stdout in analyst inputs; this is an explicit adaptation change.

Interpreter/verifier/synthesizer tasks must be short JSON strings containing
only instruction and evidence_ids. Case-scoped deterministic lookup retrieves
only the selected records; it does not give roles arbitrary file access.
No raw logs or prior role response bodies are copied into the dispatch task.

Execution IDs use E_<dispatch id>/S<step>/C<chunk> for exact stdout slices
and /K<chunk> for code slices. Slices contain at most 1,600 characters, with
all original content preserved in the stored artifact. A bare E_ ID selects
all successful stdout steps. If the resulting system+user role input exceeds
12,000 characters, lookup rejects the batch before a model call and returns
selectors for narrowing it; it does not silently cut evidence or rerun code.
Repeated reference IDs and overlapping source chunks are deduplicated.
Failed-step details remain in logs, outside interpretation and verification.

Interpreter results are stored behind R_ IDs; the planner receives a short
count/preview index. Verification retrieves those candidates and the exact
source evidence once. Citations must name a supplied chunk ID and quote an
exact excerpt. Synthesizers retrieve only supported/revised verifier findings,
without code, raw stdout or repair history. Their complete final JSON must
fit 4,000 characters, below the retained 5,000-character planner history limit.
Dynamic scheduling, role isolation and final-synthesizer equality checks remain.

# Historical: dynamic multi-role adapter v3 (2026-10-15)

The user's multi-role requirement supersedes the earlier two-role minimal
configuration. The active entry remains examples/run_trajstory.py, using
configs/trajstory.py. Protocol: native_planner_dynamic_multi_role_v3.
Default results: results/agentorchestra_dynamic_multi_role_v3.

Five role types are active: the native planning agent, orchestra_analyst,
orchestra_interpreter, orchestra_verifier and orchestra_synthesizer. The
native Planner and AgentBus choose the number of subtasks, concurrency,
dependencies, follow-ups and stopping point. There is no fixed two-round
or three-question dispatch schedule. Each required role must complete at
least one dispatch before a case is accepted; no outer retry enforces that.
The Planner must relay a successful synthesizer's actual answer. An answer
that bypasses roles or changes the synthesized insight list/Summary fails
role validation, with the raw output retained.

Each analyst dispatch owns a fresh executor instance, SessionContext and
tracer directory, including simultaneous dispatches of the same role.
Interpreter/verifier/synthesizer calls use fresh system+user messages with
no shared analyst conversation and no tool access. All roles use the same
configured backbone. Task-specific inputs and outputs are saved per dispatch
under _framework/agent/<role>/dispatches/<id>, associated with the case session.
Evidence must be supplied explicitly in dependent dispatch tasks; automatic
shared-history injection is disabled. A verifier reviews supplied execution
evidence; it does not perform an independent numerical rerun. Citation checks
are currently model-based, not an exact-excerpt validator like the current
Pandas/DSAEval multi-role wrappers.

Current Pandas/DSAEval role adapters both use fixed two rounds of three
questions with separate analyst, interpreter, verifier and synthesis calls.
Pandas uses preloaded DataFrames/private persistent REPLs; DSAEval supplies
projected CSV paths/private persistent Jupyter kernels. This adapter retains
AgentOrchestra's dynamic orchestration and per-call Python state reset.
Its native 10-planner-round/50-analyst-step limits are not the 60-call whole-case
budget in those wrappers. This budget difference must be resolved explicitly
before claiming a matched-budget performance comparison.

The v3 role roster and role instructions are task-adaptation changes, not
unmodified upstream functionality. TrajStory numerical helpers remain disabled;
the requested 5,000-character planner result-history limit is retained.
Upstream planning and tool-calling prompt templates remain unchanged.

# Historical: minimal adapter v2 (2026-10-14)

The active runner reuses the native Planner and generic tool-calling agent.
The Planner independently chooses subtasks, concurrency, analyses and when to
finish. The task requests six factual insight strings supported by the data
and a coherent Summary. Insights may be coherent and related; there is no
distinctness requirement or additional instruction about analysis directions.

No fixed two-round protocol, six-dispatch schedule, required specialist roles,
independent verifier/synthesis stages or per-question analysis budget is enabled.
The upstream agent settings remain: 50 steps, 10 tools and 5 review steps.
The native AgentBus maximum is 10 planner rounds, a ceiling rather than a target.
Per-sub-agent result history is set to 5,000 characters by user request
(2026-10-13), to reduce truncation of the six-insight answer. The adapter does
not enable extra prior-result injection into subtasks. Override with
--history-result-max-chars if needed.

Only data access and output compatibility are adapted: sanitised case input,
multi-table trajstory_analysis tool, gold-answer isolation and parsing of
insights + summary. No outer correction or count-repair loop is applied.
The minimal configuration disables the TrajStoryAgent-specific trajectory
helpers. Agents write general-purpose Python using pandas/numpy and the
available scientific libraries. Historical configurations retain the helper
option for reproducibility.

Each analysis call starts a fresh Python environment and reloads the input
tables. This follows the upstream python_interpreter's per-call state reset;
it is not a persistent Jupyter/REPL session. The wrapper adds case metadata,
preloaded tables and input validation. Tool stdout is capped at 50,000 characters;
the execution agent receives up to 12,000 characters per action in local feedback,
with a 32,000-character cumulative feedback window when external memory is off.

structure_matches_request records whether six insights and a nonempty Summary
were returned. Nonempty valid insights are preserved when Summary is missing
or empty or the count differs. output_status is complete, partial or
count_mismatch (>6). Empty/invalid insight lists remain output validation failures.
The parser never generates a missing Summary, invents insights or truncates
the generated list to force a count. Framework failures remain failures; no
outer recovery of unfinished child answers is enabled.
External memory, unrelated application services and tools are disabled for the
case environment. This remains a TrajStory data adapter, not every original tool.

The 10 planner rounds and 50 execution steps are native limits. They do not
constitute a shared case-level model-call budget. A planner round can dispatch
multiple subtasks; each subtask may call the model repeatedly. Native exhaustion
also invokes the planner once more to mark its plan failed.

Compared with the original repository, planning/tool-calling prompt templates,
dynamic dispatch and native per-agent limits are retained. Data access, final
output format, disabled external services/memory, local tool-result feedback
and the explicitly requested 300-to-5,000 result-history expansion are adapter
differences. Keep those differences explicit in baseline reporting.

Windows entry:

```powershell
& 'B:\TrajStory\scripts\run_agentorchestra_windows.ps1' --case-ids danish_ais_01_shared_corridor_day --results-dir 'B:\TrajStory\results\agentorchestra_minimal_new' --model-name openai/gpt-6-luna --dry-run
```

Remove --dry-run to call the model. Default results directory is
results/agentorchestra_trajstory_minimal_v2. Current output records are tagged
native_planner_minimal_v2. The default generator is openai/gpt-6-luna.
Keep v1 minimal and previous controlled-budget results separate.
TRAJSTORY_CONTROLLED_LEGACY.md records the earlier 3+3 protocol; its specialist
classes and tests remain for historical reproducibility but are not dispatched
by the active minimal configuration.
