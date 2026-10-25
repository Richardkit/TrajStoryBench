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
scripts/                    Selected baseline runners and paper figure utilities
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

On Windows, use a short checkout path such as `C:\TrajStoryBench` to accommodate deeply nested checkpoint paths.

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

These studies preserve the frozen-checkpoint continuation protocol. The checkpoint files and the manifest selecting them must be supplied together. A fresh full-agent run is a different experiment from continuing the fixed checkpoints.

| Variant | Change |
|---|---|
| `without_conditional_refinement` | Disable conditional decomposition and compression |
| `without_compression` | Retain conditional decomposition and disable compression |
| `without_memory` | Disable the memory manager |
| `without_data_profiling` | Disable profiling; run both rounds afresh using the restricted schema/row preview |

Example ablation preflight, once the checkpoint artifacts are available:

```bash
python -m trajstoryagent_v5.batch --model qwen --experiment without_compression --checkpoint-manifest results/v5_ablation_result/qwen/experiment/manifest.json --output results/qwen_without_compression --dry-run
```

Remove `--dry-run` to run the experiment after validation. Evaluate the generated results with the scoring entry points above.

## Paper figures and validation

The Figure 3, ablation, and hyperparameter plotting utilities are `scripts/fig3.py`, `scripts/ablation_fig6.py`, and `scripts/hyperparameter_fig6c.py`. They require their corresponding statistical input files. Figure 6 uses reference-insight weighting; the hyperparameter plot retains its documented equal-case aggregation. Plotting saved statistics does not rerun generation or judging.

Run the independent agent's offline tests with:

```bash
python -m pytest --confcutdir=trajstoryagent_v5 --rootdir=trajstoryagent_v5 --import-mode=importlib trajstoryagent_v5/tests -q
```

Offline tests check implementation behavior and do not call the generation or judge APIs.

## Release status

The current upload selection contains the benchmark data and agent/baseline implementation directories. Selected runner scripts, frozen experimental inputs, and figure input files are being assembled for the complete reproduction package. Commands requiring those artifacts become usable once the artifacts are included. Ordinary historical run logs are not prerequisites for fresh full-agent generation.

## Data sources and third-party code

The benchmark draws on the five source families listed above. Preserve the case metadata and provenance when using the data. Third-party baseline code retains its upstream license and copyright notices; those notices apply to the corresponding components.
