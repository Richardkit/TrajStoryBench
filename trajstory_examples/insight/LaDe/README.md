# LaDe InsightBench-Style Examples

This directory contains 25 LaDe-style courier trajectory examples. The first
examples originate from the downloaded LaDe Shanghai delivery data; later
examples also include purpose-built records that realize their documented
pre-schema constraints. The original source files are unchanged.

Each example has one InsightBench-style `insight.json`. It contains the goal,
insight list, questions, plot specifications, plotting code, and summary. The
CSV files referenced by `dataset_csv_path` and `input_references` are kept in
the same example directory.

Cases:

1. `01_time_window_order_handling`: temporal organization of orders and courier movement.
2. `02_regional_courier_order_state`: regional workload, movement, and delivery-state variation.
3. `03_delivery_pattern_sequence`: delivery sequence and route structure.
4. `04_experience_change`: changes associated with courier experience.
5. `05_area_time_task_organization`: spatial and temporal task organization.
6. `06_distinctive_service_patterns`: recurring service patterns and an exception.
7. `07_pickup_time_window_reliability`: pickup timing relative to service windows.
8. `08_pickup_conditions_and_constraints`: pickup behavior under operational constraints.
9. `09_multi_day_delivery_cadence`: cadence across days, route positions, and route lengths.
10. `10_time_window_route_effects`: time-window effects on route execution.
11. `11_workday_time_allocation`: travel time, visit time, and per-delivery visit efficiency.
12. `12_clockwise_route_delivery_sequence`: route orientation, task ordering, and cluster movement.
13. `13_accepted_pickup_task_dynamics`: pending workload, spatial priority, and a completion pause.
14. `14_aoi_type_delivery_activity`: AOI-specific delivery batching, GPS travel efficiency, and time-of-day allocation.
15. `15_delivery_acceptance_bursts`: bursty task acceptance, spatial completion sequences, and burst splitting.
16. `16_pending_delivery_prioritization`: backlog-dependent prioritization by destination proximity and task age.
17. `17_contextual_delivery_prioritization`: backlog, deadline, and acceptance-location effects on delivery priority.
18. `18_multi_courier_pickup_ordering`: matched-workday courier ordering, deadline-modulated deferral, AOI-set changes, and localized pickup-event GPS errors.
19. `19_regional_pickup_workload_distribution`: task-count-matched route/active-time differences, busy-AOI staffing, and late-finishing courier-days.
20. `20_aoi_service_periods_and_courier_subregions`: separated versus continuous AOI service periods, relative service-window timing, and stable courier-specific pickup subregions with shared access roads.
21. `21_road_class_delivery_dynamics`: native road classes, delivery intensity and within-class movement speeds.
22. `22_regular_aoi_work_pattern`: fourteen workdays, repeated residential AOIs, late direct routing and one day with unvisited available tasks.
23. `23_oneway_detour_dynamics`: fourteen workdays of one-way exposure, route detours and class-transition dynamics. The schema value 2.01 is provisionally interpreted as percent, pending confirmation.
24. `24_bridge_tunnel_movement`: matched early/late crossings of the same mapped bridge and tunnel, with independently verified approach classes.
25. `25_service_region_road_comparison`: fourteen matched workdays in two explicitly constructed Shanghai service regions.

Examples 21–25 preserve native OSM road attributes and directed edge geometry.
Their task events, schedules, speeds, AOIs and windows are constructed extensions,
not raw LaDe observations. Moving GPS spacing is at most 22 m. Tunnel points are
map-based interpolation; turn-restriction relations are not modeled. Each notebook
contains one evidence plot per insight and recomputes its evidence from local CSVs.
Validation is recorded in `dataset/LaDe_ex21_25_consistency_audit.json`.

Canonical data files are under `dataset/construct_lade/`.
