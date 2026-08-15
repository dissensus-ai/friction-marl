# Incomplete exploratory run — do not analyze as a result

This directory preserves an interrupted 6 August 2026 exploratory zero-bias run for provenance.
It is not a completed factorial and is not evidence for or against the bias-initialization
explanation.

- The requested design was 25 epsilon-zero cells x 30 replications (750 runs).
- `replication_results.csv` contains only 240 rows: eight complete cells and one partial cell,
  covering nine of the 25 requested cells.
- The run predates `--seed-grid-replications` and therefore used a compact restricted-grid seed
  schedule rather than the matched 5 x 5 x 5 schedule of the CPU arm of record.
- It has no frozen protocol or stopping-rule decision file.

The result of record is the staged, matched-seed diagnostic under
`results/zero_bias_eps0/`, whose `PROTOCOL.md` was frozen before outcome inspection. Preserve this
directory, but do not merge its rows into that diagnostic or cite it as a completed experiment.
