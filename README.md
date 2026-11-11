# TrajStoryBench

TrajStoryBench is a benchmark for discovering insights from trajectory data. It contains 125 analytical cases and 423 reference insights across five dataset families. This repository also provides TrajStoryAgent and adapters for six baseline methods.

TrajStoryAgent explores data in two rounds, with three planned questions per round by default. Its workflow combines data profiling, code-based analysis and verification, conditional decomposition and compression, a memory manager, and final summarization.

## Dataset

| Family | Cases |
|---|---:|
| Danish AIS | 25 |
| LaDe | 25 |
| MTL Trajet | 25 |
| Porto Taxi | 25 |
| WorldTrace | 25 |
| **Total** | **125** |

The case index is `trajstory_examples/insight/manifest.json`. Each entry identifies a case JSON containing the analytical goal, reference insights, and data references. Case directories also contain CSV tables, an analysis notebook, and supporting figures.

Formal generation uses the projected `input_*.csv` tables selected by the case loader. Other case tables support the authored analyses and evidence. Keep the case directory structure intact so that relative data and figure paths resolve correctly.

## Repository layout

```text
trajstory_examples/insight/   Benchmark cases, tables, reference insights, and evidence
trajstoryagent_v5/           Independent TrajStoryAgent implementation and evaluation
baselines/                  Baseline implementations, adapters, and shared utilities
```

`trajstoryagent_v5` contains the current independent agent implementation. Generation, evaluation, ablation, and hyperparameter entry points are owned by this package. Baseline implementations and shared utilities are grouped under `baselines/`.

## Installation

Use Python 3.11 or later. Run the following commands from the repository root, preferably in a dedicated virtual environment:

```bash
git clone https://github.com/Richardkit/TrajStoryBench.git
cd TrajStoryBench
python -m pip install "./trajstoryagent_v5[evaluation,test]"
```

This installs the dependencies for TrajStoryAgent, its evaluator, and offline tests. Baselines have additional framework dependencies and may require separate environments; the agent installation does not install all baseline frameworks.

## Run TrajStoryAgent

### One case

```bash
python -m trajstoryagent_v5 --case trajstory_examples/insight/Danish_AIS/01_shared_corridor_day/insight.json --repo-root . --model gpt --output results/example_gpt_ais01
```

Use `--model qwen` and a different output directory for Qwen. Each run requires a new, empty output directory.

The run saves input/source hashes, configuration, generated analysis and evidence, finalized insights, and a source snapshot. `result.json` provides the insight strings consumed by the evaluator.

### All 125 cases

Validate the inputs without generation:

```bash
python -m trajstoryagent_v5.batch --model gpt --output results/full_gpt --dry-run
```

Run a generation batch:

```bash
python -m trajstoryagent_v5.batch --model gpt --output results/full_gpt
python -m trajstoryagent_v5.batch --model qwen --output results/full_qwen
```

Use `--case-ids CASE_ID ...` to select a subset and `--workers N` to control concurrent cases. Generation and scoring are separate steps.

## Evaluate generated insights

Score one case:

```bash
python -m trajstoryagent_v5.evaluate --case trajstory_examples/insight/Danish_AIS/01_shared_corridor_day/insight.json --result results/example_gpt_ais01/result.json --output results/example_gpt_ais01_score
```

Score a completed batch:

```bash
python -m trajstoryagent_v5.evaluate_batch --generation-root results/full_gpt/generation --output results/full_gpt_scores
```

The evaluator uses two independent GPT judge rounds with temperature 0 and reasoning disabled. For each reference insight, it takes the maximum similarity rating over generated insights. G-Eval averages these reference-level scores. Coverage@0.6 thresholds the scores separately in each judging round before averaging the two round results.

The batch summary weights successfully evaluated cases by their number of reference insights. It reports failures explicitly; inspect completeness before comparing methods. New API runs can differ from reported results because model responses and provider implementations can change.

## Ablation studies

Run both rounds from scratch, or reuse a checkpoint from your own completed full-v5 run. The goal is to reproduce the workflow and evaluation protocol; generated outputs and scores can vary.

| Variant | Change |
|---|---|
| `without_conditional_refinement` | Disable conditional decomposition and compression |
| `without_compression` | Retain conditional decomposition and disable compression |
| `without_memory` | Disable the memory manager |
| `without_data_profiling` | Disable profiling; run both rounds afresh using the restricted schema/row preview |

Fresh ablation preflight:

```bash
python -m trajstoryagent_v5.ablation --mode fresh --case trajstory_examples/insight/Danish_AIS/01_shared_corridor_day/insight.json --repo-root . --model gpt --variant without_compression --output results/fresh_no_compression --dry-run
```

To reuse the full-v5 run generated in the example above:

```bash
python -m trajstoryagent_v5.ablation --mode checkpoint --source-run results/example_gpt_ais01 --model gpt --variant without_compression --output results/checkpoint_no_compression --dry-run
```

Remove `--dry-run` to execute. For batches, use `--experiment VARIANT --mode fresh`, or `--experiment VARIANT --mode checkpoint --source-generation results/full_gpt/generation` with the batch entry. The data-profiling ablation only supports fresh mode, so no profiled checkpoint information is reused. Evaluate the outputs with the same scoring commands above.

## Validation

Run the independent agent's offline tests with:

```bash
python -m pytest --confcutdir=trajstoryagent_v5 --rootdir=trajstoryagent_v5 --import-mode=importlib trajstoryagent_v5/tests -q
```

Offline tests check implementation behavior and do not call the generation or judge APIs.

## Data sources and third-party code

The benchmark draws on the five source families listed above. Preserve the case metadata and provenance when using the data. Third-party baseline code retains its upstream license and copyright notices; those notices apply to the corresponding components.
