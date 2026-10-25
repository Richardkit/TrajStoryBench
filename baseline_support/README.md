# Baseline support

Version-neutral case loading, projected input resolution, JSON handling, analysis
helpers, transport auditing, deterministic profiling and judge utilities.
Current CodeGen, Pandas, DSAEval, InsightBench and AgentOrchestra adapters import
this package; it imports no TrajStory agent controller or versioned package.
The original implementations were copied from the prior shared v1 modules;
only the import namespace changed. Evidence schema identifiers remain unchanged.

Historical result directories and snapshots remain unmodified. Their frozen
source hashes must not be changed to accept this migration. Use fresh output
directories with current runners. Original changed sources are backed up in the
task's baseline independence audit deliverables.
