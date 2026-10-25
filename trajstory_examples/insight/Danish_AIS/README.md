# Danish AIS Insight examples

The collection contains eleven maritime trajectory examples, including cases
in the Fehmarn Belt and Kattegat geographic contexts. They follow the
single-file InsightBench-style format:
each case directory contains one `insight.json`, one trajectory-point CSV, one
vessel-attribute CSV, and matching plots. The visible cases are constructed
derived cases; the downloaded official AIS archives remain unchanged.

Example 1 is a one-day corridor organization case with movement regimes and a
reported-draught comparison. Example 2 compares two days of corridor passage,
direction, anchoring, ship type, and cross-day vessel recurrence. Example 3
follows one constructed cargo vessel across four days of repeated eastbound and
westbound passages and examines a missed turn, a corrective turn, and a
temporary off-corridor slowdown. Example 4 compares same-period traffic in
Bornholmsgat and the Great Belt, including draught, vessel type, ferry
presence, corridor concentration, and transverse track geometry.

Example 11 compares 30 repeated corridor passages for each of five cargo
vessels. It contrasts the stable routes and close heading–COG alignment of
Vessels A–C with the wider route dispersion, sustained heading–COG offsets,
and frequent within-passage anchoring of Vessels D–E. Seventy supplied points
belonging to segments that crossed or closely followed the supplied map coastline
were minimally shifted toward open water;
the case-local `data_repair_report.json` records every adjustment.
