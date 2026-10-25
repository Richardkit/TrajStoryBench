# TrajStoryAgent v5 — formal two-round workflow

This package contains one implementation: the multi-role workflow underlying
the formal GPT/Qwen experiments and the supplied paper's framework section.
All framework code and the Memory Manager role prompt are contained here.
Only third-party Python libraries and the explicitly supplied case data are
external dependencies. There are no persistent-worker or single-agent variants.

## Source layout

- `v5_framework/`: complete v5 workflow, generation, batch execution, and shared utilities.
- `ablation/`: ablation implementations and the shared checkpoint experiment dispatcher.
- `hyperparameter_study/`: hyperparameter replay implementation and its entry point.
- `evaluation/`: single-case and batch scoring.
- `roles/`: prompt resources; `tests/`: offline verification.

The root command modules are compatibility entry points. Existing documented commands remain valid.

## Workflow

1. Load only projected `input_*.csv` tables, then profile fields and relations.
2. Plan three initial questions with analysis-unit/grouping configurations.
3. Generate and execute code; interpret initial insights; review evidence and
   repair or revise questions within the configured limits.
4. For each accepted first-round branch, export observations, perform
   conditional decomposition, compress with explicit exceptions, and interpret
   the resulting facts into a refined insight.
5. Summarize attempted questions, accepted insights and coding failures into
   Memory. Supply original/refined insight texts and accepted conditional facts
   separately as planning context. Original measured evidence is preserved.
6. Plan and analyze three follow-up questions using this context, Memory,
   original parent evidence and scope restrictions. Review the new evidence.
7. Refine accepted second-round evidence using the same decomposition and
   compression. Reuse first-round refinement records without rerunning them.
8. Collect the final branch insights and generate only their summary. Report
   insights and evaluation predictions must contain identical text.

Unsupported, failed or declined refinement retains its initial insight. A failed
analysis branch contributes no fabricated insight. No Memory update occurs after
the final round. Memory contains qualitative summaries; it does not replace the
original numerical evidence. This is not a persistent Python worker architecture.

## Run

Python 3.11 or later is required. From the parent directory:

```powershell
python -m pip install -r trajstoryagent_v5/requirements.txt
python -m trajstoryagent_v5 --help
python -m trajstoryagent_v5 --case path/to/insight.json --repo-root path/to/data-root --model gpt --output path/to/new-empty-run
```

Set `OPENAI_API_KEY` for GPT or `DASHSCOPE_API_KEY` for Qwen in the environment.
Keys are never included in manifests. On Windows, the loader also supports the
current Python environment's Conda state and the user's environment registry.
Model settings are `gpt-6-luna` / `qwen3.8-flash`, temperature 0, reasoning off.
The Qwen endpoint is the mainland DashScope endpoint used by the formal runs.

You can also install this directory with `pip install ./trajstoryagent_v5` and
use the `trajstory-v5` entry point. Supply `--repo-root` explicitly when installed.
Generation uses real model requests; the offline tests do not.

## Identity, provenance and outputs

- Framework: `v5_multi_role_insight_workflow`.
- Implementation: `formal_two_round_refinement_memory`.
- Defaults: two rounds, three questions per round, three case-internal analysis
  workers, `n_min=1`, partition penalty 1, distortion weight 1, tolerance 0.25.
- `run_manifest.json` records input/source hashes and configuration.
- `source_snapshot/trajstoryagent_v5/` preserves the source used for each new run.
- `early/` contains first-round refinement; `refinement/` contains new
  second-round refinement. `03_memory.json` and
  `03_actual_planning_bundle.json` record the handoff.
- `predictions.json`, `06_report.json` and `result.json` expose the same final
  insights. Scoring is separate: two independent GPT judge rounds are used by
  the experiment evaluation pipeline.

Identifiers such as `raw_reduction.v1` and `trajstory.analysis_tools.v2` describe
evidence schemas, not alternative framework implementations. Their values are
preserved to retain the validated evidence-checking protocol.

## Offline validation

```powershell
python -m pytest --confcutdir=trajstoryagent_v5 --rootdir=trajstoryagent_v5 --import-mode=importlib trajstoryagent_v5/tests -q
```

Tests cover input selection, workflow ordering, refined-insight handoff,
summary/prediction consistency, refinement fallbacks, numerical compression,
execution, entry identity, and external-package isolation. Offline equivalence
does not promise identical future stochastic model responses or reproduce live
benchmark scores without model calls.


## Independent batch entry (2026-10-25)

From the repository root:
```powershell
python -m trajstoryagent_v5.batch --model gpt --output results/v5_fresh_gpt --dry-run
python -m trajstoryagent_v5.batch --model gpt --output results/v5_fresh_gpt
python -m trajstoryagent_v5.batch --model qwen --output results/v5_fresh_qwen
```
The default selection is all manifest cases; `--case-ids ID ...` selects a subset.
The batch uses only the canonical v5 package and Python standard library,
freezes package sources, and launches isolated case processes. Each case receives
one fresh attempt. Existing nonempty output directories are rejected. Failures
remain recorded; rerun explicitly into a new directory. Generation only: scoring
is separate, and this command does not claim to regenerate historical scores.
Historical `scripts/run_v5_*_insightbench125.py` entries retain their original
frozen implementation chain and are not the current v5 batch interface.

## Independent historical-protocol experiments and scoring

The following entries import only this package, not old version packages or
`scripts`. Install the optional scorer dependency with
`pip install "./trajstoryagent_v5[evaluation]"`.

```powershell
python -m trajstoryagent_v5.experiment --source-run PATH_TO_FROZEN_CASE_RUN --model qwen --variant without_compression --output results/new_qwen_no_compression_case --dry-run
python -m trajstoryagent_v5.experiment --source-run PATH_TO_FROZEN_CASE_RUN --model gpt --variant hyperparameter --partition-penalty 3 --output results/new_gpt_lp3_case
python -m trajstoryagent_v5.batch --model qwen --experiment without_compression --checkpoint-manifest results/v5_ablation_result/qwen/experiment/manifest.json --output results/new_qwen_no_compression --dry-run
python -m trajstoryagent_v5.evaluate --case PATH_TO_CASE_JSON --result PATH_TO_NEW_RESULT_JSON --output results/new_case_evaluation
python -m trajstoryagent_v5.evaluate_batch --generation-root results/new_qwen_no_compression/generation --output results/new_qwen_no_compression_scores --dry-run
```

Remove `--dry-run` after reviewing the plan to execute paid generation.
Variants: `without_conditional_refinement`, `without_compression`,
`without_memory`, `without_data_profiling`, `hyperparameter`.
Mathematical settings: `--partition-penalty`, `--min-support`, `--tolerance-sd`,
`--distortion-weight`. Change one at a time for the original sensitivity design.
Defaults are 1, 1, 0.25, 1 respectively. Batch accepts the same settings.
Prefer `--checkpoint-manifest` to preserve the exact original case-to-checkpoint
selection and verify its frozen hashes. `--source-generation` instead explicitly
selects the checkpoints referenced by that generation directory's result files.

Historical protocol is retained: D+C and C reuse the first-round checkpoint;
C retains the original uncompressed first-round realization; memory removal
reuses first-round refinement and hands off raw records; sensitivity reuses
first-round observations without re-exporting them. Removing data profiling
is the historical exception: both rounds run afresh with only column names,
raw types and the first three rows. The replay entries reject checkpoints with
admitted first-round conflicts. Input hashes are checked, checkpoint hashes
are recorded and rechecked, and existing outputs cannot be overwritten.

The scorer embeds the original InsightBench G-Eval prompt text. It uses two
independent GPT-6 Luna rounds, discrete rating/10, per-reference maxima and
coverage threshold 0.6 separately in each round before averaging. It does not
adopt or replace scores. Empty generations are rejected rather than silently
scored as zero. These are migrated entry points, not proof of live score
equivalence. Historical result manifests and source snapshots remain unchanged.
