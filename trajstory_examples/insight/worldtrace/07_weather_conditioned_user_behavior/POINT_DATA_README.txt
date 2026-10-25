The primary CSV contains 68 trajectory summaries. trajectory_points.csv contains all 35,117 reconstructed points for those trajectories.
Join on trajectory_id and order by point_index. timestamp is the point time. The first point has no incoming speed/bearing; the first two points have no heading-change value.
Distances, mean speeds and detour ratios can be recomputed from these points. mean_yaw_change_deg is the mean absolute wrapped change of consecutive GPS segment bearings, a horizontal heading-change proxy rather than measured IMU yaw.
Coordinates are synthetic reconstructions fitted to the existing figure values, not recovered original WorldTrace observations or road-matched paths.
See reconstruction_provenance.json for metric definitions and reconstruction details. The notebook loads the point table as user_df.

Current adjustment: wt7_0043 through wt7_0046 are reconstructed to 1793, 2176, 2426 and 3187 m. Cold-rain car tracks now comprise six short (<1500 m), four medium (1500 to <4000 m) and two long (>=4000 m) tracks. See results/dataset_quality/worldtrace07_short6_20261008/repair.json.
