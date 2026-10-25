# Independent LLM-only baseline

Run from the repository root with the experiment environment containing pandas,
numpy and OpenAI (OpenAI is needed only for live generation or scoring).

```powershell
python -m baselines llm-only --case trajstory_examples/insight/Danish_AIS/01_shared_corridor_day/insight.json --model gpt --output results/new_llm_only_ais01
```

This prepares deterministic full-column statistics and head/seed-0 sample rows,
request JSON, input hashes and source snapshots without calling a model. Add
`--run` to generate. Use `--model qwen` for Qwen. Use a fresh output directory.
Credentials are read from the environment or existing local credential settings;
never put keys in source files. One semantic response is used; transient transport
errors can be retried. No score-based response selection is performed here.

Score the resulting `result.json` with:

```powershell
python -m baselines evaluate --case trajstory_examples/insight/Danish_AIS/01_shared_corridor_day/insight.json --result results/new_llm_only_ais01/result.json --output results/new_llm_only_ais01_score
```

Scoring calls GPT twice with the existing evaluation protocol. Current inputs and
requests were checked against all 125 cases for both generators (250 exact request
matches, including historical attempt/retry subdirectories). This is an offline
compatibility check, not a claim that new stochastic responses reproduce scores.
The package depends on `baseline_support`, not any agent version or old retry runner.
