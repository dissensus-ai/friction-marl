"""Merge the partial-sharing CSVs after an infrastructure interruption.

The first full-grid run (seed 42, --overlaps 0 1 2 3) completed overlaps k=0,1 for
both modes and k=2 for IQL before the background processes were killed at ~59 min.
The incomplete overlaps were re-run PER-OVERLAP (--overlaps 2 3, same
seed/rhos/sigmas, identical args across modes so k=2,3 stay mode-matched).

This merge keeps:
  * k in {0,1} from the original full-grid run  (_orig_partial_{mode}.csv)
  * k in {2,3} from the per-overlap resume run   (partial_{mode}_resume.csv)
and writes the canonical partial_{mode}.csv (4 overlaps x 16 conds x 30 reps).

Seeding note (honest): k=0,1 used full-grid cond-seeds; k=2,3 used per-overlap
cond-seeds. Within EVERY overlap the IQL/VDN draws are matched (same args), so the
per-(mode,w) slope analysis is unaffected; overlaps are never seed-compared to each
other, so the mixed draw-order is immaterial.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
EXPECT_RHOS = 8
EXPECT_SIG = 2
EXPECT_REPS = 30

for mode in ["iql", "vdn"]:
    orig = pd.read_csv(HERE / f"_orig_partial_{mode}.csv")
    resume = pd.read_csv(HERE / f"partial_{mode}_resume.csv")
    keep_orig = orig[orig.overlap_k.isin([0, 1])].copy()
    keep_res = resume[resume.overlap_k.isin([2, 3])].copy()
    final = pd.concat([keep_orig, keep_res], ignore_index=True)
    final = final.sort_values(["overlap_k", "rho", "sigma", "replication"]).reset_index(drop=True)
    # integrity checks
    for k in [0, 1, 2, 3]:
        sub = final[final.overlap_k == k]
        nconds = sub.groupby(["rho", "sigma"]).ngroups
        reps = sub.groupby(["rho", "sigma"]).size()
        assert nconds == EXPECT_RHOS * EXPECT_SIG, f"{mode} k={k}: {nconds} conds (want {EXPECT_RHOS*EXPECT_SIG})"
        assert reps.min() == EXPECT_REPS and reps.max() == EXPECT_REPS, \
            f"{mode} k={k}: reps {reps.min()}-{reps.max()} (want {EXPECT_REPS})"
    n_expect = 4 * EXPECT_RHOS * EXPECT_SIG * EXPECT_REPS
    assert len(final) == n_expect, f"{mode}: {len(final)} rows (want {n_expect})"
    final.to_csv(HERE / f"partial_{mode}.csv", index=False)
    print(f"{mode}: merged -> partial_{mode}.csv  ({len(final)} rows; "
          f"k0/1 from full-grid, k2/3 from per-overlap resume) OK")

print("MERGE COMPLETE")
