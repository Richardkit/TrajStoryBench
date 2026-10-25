# Porto Taxi Insight examples

The five examples correspond to Overall Characterization, Anomaly and
Exception, Segmentation Structure, demand and driver-level service time, and
the role of taxi stands in daily demand.
Each example is represented by one
InsightBench-style `insight.json`, with one question, plot specification and
analysis code for every insight. The referenced trip and trajectory CSV files
are kept beside the JSON.

Example 1 contains 64 constructed one-day trajectories in a selected Porto
area. Its `station_locations.csv` records the two case-designed taxi stands
used for the station-origin comparison; these coordinates are not official
station coordinates supplied by the UCI dataset.

Example 4 contains 278 completed taxi request events for 24 drivers on one
observed day. Its secondary `driver_day_summary.csv` is a driver-level table,
so the waiting-time comparison is not limited to an overall order average.

Example 5 is derived from 120 trips on 2013-11-07 in the original UCI Porto
Taxi data. It retains the original taxi IDs, request-source fields and GPS
points. The case does not treat `CALL_TYPE` as evidence that a taxi was
physically waiting at a stand; it compares taxi-stand service with the other
recorded request sources.
