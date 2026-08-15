# Analysis outputs

The stopping-rule result of record is `summary.json`, with each interim decision retained as
`summary_stage_XX.json`. Its supporting condition and paired-replication tables use the same
stage suffix. These files are produced by `../analyze.py` under the frozen `../PROTOCOL.md`.

The remaining `metrics.csv`, `regressions.csv`, and plot files are the generic factorial summaries
that `run_parallel.py` emits for every run directory. They are useful descriptive checks, but they
do not enter the zero-bias estimands or the support/refute/extend decision and should not be used as
an alternative stopping analysis.

The final analyzer output at 30 replications per cell is still `extend`. Because
`../PROTOCOL.md` caps the design at 30, this is interpreted operationally as
inconclusive at the planned maximum, not as authorization for another stage.
