# WorldTrace NYC Insight Examples

These are five small InsightBench-format cases built from the previously
prepared WorldTrace NYC subset. Each case contains at most 20 trajectories and
is designed around a high-level goal rather than a single question.

Each JSON file follows the native InsightBench structure:

- `dataset_csv_path` and `user_dataset_csv_path`;
- `metadata.goal`, role, category, description, and header;
- `insight_list`, where every item contains `data_type`, `insight`,
  `insight_value`, `plot`, `question`, `actionable_insight`, and `code`;
- the flattened `insights` list and final `summary`.

The CSV files are derived views. The original WorldTrace archive and the
previous NYC tables are unchanged. Each case's `dataset_csv_path` points to
one CSV containing all trajectories for that example. Per-trajectory maps
remain available as auxiliary WorldTrace assets, but are not added as a custom
JSON field.

Cases:

1. `01_main_patterns_and_exceptions` — dominant patterns and selected exceptions.
2. `02_user_consistency_and_variation` — three users, observed mode composition, a shared driving corridor, and contrasting driving-speed profiles. This case uses one point-level CSV so the maps and speed behaviour are directly inspectable.
3. `03_unusual_trajectories` — 21 typical comparison trajectories plus speed exceptions A/B and an outside-city distance exception C.
4. `04_geographic_group_differences` — balanced comparison of four coarse regions.
5. `05_context_conditioned_patterns` — 64 vehicle trajectories across four consecutive days in the same South Brooklyn area, compared by time period and cold-day daytime activity.
6. `06_user_normal_pattern_and_anomaly` — 40 point-level trajectories from one user, with a dominant bidirectional Area A–Area B corridor and one closed-loop exception.

The cases are intentionally small pilot examples. They are suitable for
checking the task format and model execution process before scaling up.

Case 02 visualizations are in
`visualizations/02_user_consistency_and_variation/overview.html`. The driving
and walking maps are separate offline PNGs with cached OpenStreetMap context.

The normalization utility is:

```bash
python scripts/normalize_worldtrace_insightbench_json.py
```
