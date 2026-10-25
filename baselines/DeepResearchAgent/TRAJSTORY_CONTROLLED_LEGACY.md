# TrajStory compatibility

This adapter runs TrajStory InsightBench cases through DeepResearchAgent's
AgentBus while preserving the existing evaluation contract.

The pipeline uses the same three parallel discovery agents (temporal/spatial,
behavior/comparison, and anomaly) twice: first for breadth, then for distinct
questions targeted at the first verifier's coverage gaps. Each discovery round
is followed by an independent evidence verifier, then a synthesis agent builds
the final report. The runner passes an exact dispatch schedule to AgentBus, so
successful runs have exactly six exploration dispatches, with no extra or
missing specialist calls. Verifier and synthesis calls are not exploration
dispatches. A planner schedule violation fails the case rather than silently
changing the budget. Later stages receive prior results through an opt-in
AgentBus context channel. The default 300-character bus history remains
unchanged for other applications.

The controlled-budget version limits each of the six specialist dispatches to
one successful initial analysis and at most one successful deeper check of the
same question. Failed code may be repaired twice per phase without consuming
the successful-analysis budget. One model step may execute only one analysis,
and each specialist has a nine-step internal limit. The execution cap is
enforced in `src/agent/trajstory_exploration_budget.py` and the tool-calling
agent. No model-generated phase or goal metadata is required; the semantic
rule that a follow-up stay on the same question is stated in the specialist
and planner prompts and must be checked in traces. Verifier and synthesis
agents retain their 15-step limit and are not counted as exploration.

Previous pilot and Qwen outputs were generated with the older, unbounded
internal-analysis setting. Keep them separate from new controlled-budget runs.
`../scripts/rerun_deepresearchagent_step15.py` selects the 18 pilot cases with
unfinished exploration dispatches, runs two cases concurrently (at most six
exploration agents), and reuses the two cases whose explorations all completed.
Its separate `evaluate --wait` mode uses the same DeepSeek judge and prediction
selection as the previous pilot, then produces a merged 20-case comparison.

Gold-answer isolation is intentional: `insight.json` is read only by the host
adapter, which emits a sanitised `case_input.json` containing metadata, table
paths, and schema previews. Agents only receive that sanitised file.

Validate a case without loading the framework or calling a model:

```bash
python examples/run_trajstory.py \
  --case-ids danish_ais_01_shared_corridor_day \
  --dry-run
```

Run one case after installing the repository dependencies and setting the API
key required by the selected provider:

```bash
python examples/run_trajstory.py \
  --case-ids danish_ais_01_shared_corridor_day \
  --model-name openai/gpt-5.6-luna
```

Multiple explicitly named cases are allowed. Without `--case-ids`, the runner
uses only the first manifest case unless `--limit` is supplied, preventing an
accidental full-manifest run.

Results are written to
`../results/agentorchestra_trajstory/trajstory_case_results.json` and include
`pred_insights` and `pred_summary`, so the current TrajStory evaluation scripts
can consume them with a small path/configuration change.

The adapter has been exercised with GPT-5.6 Luna and Qwen3.8-Flash. The older
20-case GPT pilot produced 17 valid final reports; two failures came from
exhausted GPT credit during planning, and one case ended with an empty
validated insight list. Those older outputs predate the controlled-budget
setting and must not be mixed with new results.
