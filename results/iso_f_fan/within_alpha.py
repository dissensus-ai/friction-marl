"""Iso-F tie test restricted to WITHIN-alpha comparisons.

Written 5 Aug 2026 in response to a review objection that was correct.

The original iso-F fan pooled cells of equal F = sigma(1+eps)/(1+alpha) across
the whole grid, including across the first-stage alignment axis. But section 4.3
disowns that axis in terms: the first-stage alpha is sign-blind, its realised
correlation is even in |alpha|, and a monotone 1/(1+alpha) denominator was
GUARANTEED to misfit it. Rejecting the composite on that grid is circular.

This restricts the test to tie sets WITHIN a single alpha level. There, F is
monotone in sigma*(1+eps), so cells sharing a value of sigma*(1+eps) share F --
and the broken axis is never crossed. Any outcome difference between them is a
difference the composite cannot express, on data section 4.3 does not disown.

Because this produces 25 within-alpha tie-set tests, the output reports the raw
p-values together with Holm family-wise and Benjamini-Hochberg false-discovery-
rate adjustments. The adjusted counts, rather than the raw count alone, are the
robustness summary used in the manuscript.

Run:  python3 results/iso_f_fan/within_alpha.py
"""
from pathlib import Path

import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "results/gpu_factorial_postfix/replication_results.csv"


def main():
    d = pd.read_csv(DATA)
    d["gap"] = -d["mean_reward"]
    d["k"] = (d.sigma * (1 + d.epsilon)).round(6)   # F rank within a fixed alpha

    rows = []
    for a, ga in d.groupby("alpha"):
        for k, g in ga.groupby("k"):
            cells = g.groupby(["sigma", "epsilon"])
            if cells.ngroups < 2:
                continue
            groups = [x.gap.values for _, x in cells]
            F, p = stats.f_oneway(*groups)
            grand = g.gap.mean()
            ss_b = sum(len(x) * (x.mean() - grand) ** 2 for x in groups)
            rows.append({"alpha": a, "sigma(1+eps)": k, "n_cells": cells.ngroups,
                         "F": round(float(F), 3), "p": float(p),
                         "eta_sq": round(float(ss_b / ((g.gap - grand) ** 2).sum()), 4)})

    r = pd.DataFrame(rows).sort_values(["alpha", "sigma(1+eps)"]).reset_index(drop=True)
    reject_holm, p_holm, _, _ = multipletests(r["p"], alpha=0.05, method="holm")
    reject_bh, p_bh, _, _ = multipletests(r["p"], alpha=0.05, method="fdr_bh")
    r["p_holm"] = p_holm
    r["p_bh"] = p_bh
    r["reject_holm_05"] = reject_holm
    r["reject_bh_05"] = reject_bh
    r = r[["alpha", "sigma(1+eps)", "n_cells", "F", "p", "p_holm", "p_bh",
           "reject_holm_05", "reject_bh_05", "eta_sq"]]
    out = Path(__file__).resolve().parent / "within_alpha.csv"
    r.to_csv(out, index=False)
    print(r.to_string(index=False))
    print(f"\ntie sets: {len(r)} across {r.alpha.nunique()} alpha levels")
    print(f"raw p < .05: {(r.p < .05).sum()}/{len(r)}")
    print(f"Holm FWER < .05: {r.reject_holm_05.sum()}/{len(r)}")
    print(f"BH FDR < .05: {r.reject_bh_05.sum()}/{len(r)}")
    print(f"min p: {r.p.min():.3g}   median eta^2: {r.eta_sq.median():.3f}   "
          f"max eta^2: {r.eta_sq.max():.3f}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
