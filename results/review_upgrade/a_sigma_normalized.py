"""Deliverable A: sigma-normalized re-analysis of the friction-MARL factorial.

Tests whether the 'stakes dominate' headline (eta^2 ~ 0.74 for sigma) is MECHANICAL:
sigma scales the reward weights w ~ N(sigma, 0.05), and the reward gap
  dR = (1/n) sum_i sum_j w_ij (s_j - tau_ij)^2
is therefore ~degree-1 homogeneous in sigma (doubling sigma ~doubles dR for a fixed
policy). Normalizing the gap by sigma (dR/sigma) removes this mechanical scaling and
exposes the *structural* effect of stakes on the LEARNED policy.

Outputs:
  - anova_raw_replication.csv / anova_norm_replication.csv  (3750-row ANOVA, type II)
  - anova_raw_condition.csv   / anova_norm_condition.csv     (125-mean ANOVA, type II)
  - alpha_symmetry.csv  (gap by alpha level, symmetric vs antisymmetric decomposition)
  - interaction_table.csv (gap_norm by alpha x sigma -- does the U deepen with stakes?)
  - A_SUMMARY.txt
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DATA = REPO / "results" / "gpu_factorial" / "replication_results.csv"
OUT = HERE


def anova_eta(df: pd.DataFrame, dv: str, typ: int = 2) -> pd.DataFrame:
    # Balanced full factorial => type-I == type-II for orthogonal terms. At the
    # condition level the 3-way model saturates (0 residual df), so use type-I
    # (sequential SS) which is well-defined there.
    model = ols(f"{dv} ~ C(alpha) * C(sigma) * C(epsilon)", data=df).fit()
    tbl = anova_lm(model, typ=typ)
    ss_total = tbl["sum_sq"].sum()
    tbl["eta_sq"] = tbl["sum_sq"] / ss_total
    tbl["eta_sq_pct"] = tbl["eta_sq"] * 100.0
    return tbl


def main() -> None:
    df = pd.read_csv(DATA)
    # Gap (positive): optimal reward (0) minus realized mean reward.
    df["gap"] = -df["mean_reward"]
    # sigma-normalized gap: gap per unit stake. weights ~ N(sigma,0.05) => mean weight ~ sigma,
    # so dR/sigma ~ mean squared deviation, free of the mechanical weight-scaling.
    df["gap_norm"] = df["gap"] / df["sigma"]
    df["abs_alpha"] = df["alpha"].abs()

    lines = []
    def p(s=""):
        print(s)
        lines.append(s)

    p("=" * 78)
    p("DELIVERABLE A: sigma-normalized re-analysis")
    p(f"rows={len(df)}  conditions={df.groupby(['alpha','sigma','epsilon']).ngroups}")
    p("=" * 78)

    # ----- Replication-level ANOVA (3750 rows), raw gap vs normalized gap -----
    raw_rep = anova_eta(df, "gap")
    norm_rep = anova_eta(df, "gap_norm")
    raw_rep.to_csv(OUT / "anova_raw_replication.csv", float_format="%.6f")
    norm_rep.to_csv(OUT / "anova_norm_replication.csv", float_format="%.6f")

    def eta(tbl, key):
        return float(tbl.loc[key, "eta_sq"]) if key in tbl.index else float("nan")

    p("\n--- REPLICATION-LEVEL eta^2 (3750 rows) ---")
    p(f"{'effect':<28}{'RAW gap':>12}{'NORM gap/sigma':>16}")
    for key in ["C(alpha)", "C(sigma)", "C(epsilon)",
                "C(alpha):C(sigma)", "C(alpha):C(epsilon)",
                "C(sigma):C(epsilon)", "C(alpha):C(sigma):C(epsilon)", "Residual"]:
        p(f"{key:<28}{eta(raw_rep,key):>12.4f}{eta(norm_rep,key):>16.4f}")

    # ----- Condition-level ANOVA (125 means) -----
    cond = (df.groupby(["alpha", "sigma", "epsilon"])
              [["gap", "gap_norm"]].mean().reset_index())
    raw_cond = anova_eta(cond, "gap", typ=1)
    norm_cond = anova_eta(cond, "gap_norm", typ=1)
    raw_cond.to_csv(OUT / "anova_raw_condition.csv", float_format="%.6f")
    norm_cond.to_csv(OUT / "anova_norm_condition.csv", float_format="%.6f")

    p("\n--- CONDITION-LEVEL eta^2 (125 condition means) ---")
    p(f"{'effect':<28}{'RAW gap':>12}{'NORM gap/sigma':>16}")
    for key in ["C(alpha)", "C(sigma)", "C(epsilon)",
                "C(alpha):C(sigma)", "C(alpha):C(epsilon)",
                "C(sigma):C(epsilon)", "C(alpha):C(sigma):C(epsilon)", "Residual"]:
        p(f"{key:<28}{eta(raw_cond,key):>12.4f}{eta(norm_cond,key):>16.4f}")

    # ----- alpha symmetry: is the U-shape symmetric in sign (=> |alpha| effect)? -----
    p("\n--- ALPHA SYMMETRY (normalized gap means by alpha) ---")
    gn = df.groupby("alpha")["gap_norm"].agg(["mean", "std", "count"])
    gr = df.groupby("alpha")["gap"].mean()
    sym = pd.DataFrame({
        "gap_raw_mean": gr,
        "gap_norm_mean": gn["mean"],
        "gap_norm_std": gn["std"],
        "n": gn["count"],
    })
    sym.to_csv(OUT / "alpha_symmetry.csv", float_format="%.6f")
    for a, row in sym.iterrows():
        p(f"  alpha={a:+.1f}  raw_gap={row['gap_raw_mean']:.4f}  "
          f"norm_gap={row['gap_norm_mean']:.4f} (sd {row['gap_norm_std']:.3f})")

    # Symmetric (|alpha|, alpha^2) vs antisymmetric (alpha) decomposition on normalized gap.
    # Compare a SIGNED-linear model vs a SYMMETRIC (quadratic in |alpha|) model.
    m_lin = ols("gap_norm ~ alpha", data=df).fit()
    m_quad = ols("gap_norm ~ alpha + I(alpha**2)", data=df).fit()
    m_abs = ols("gap_norm ~ abs_alpha", data=df).fit()
    m_full = ols("gap_norm ~ alpha + I(alpha**2)", data=df).fit()
    p("\n  Variance of normalized gap explained by alpha terms (R^2):")
    p(f"    signed linear (alpha)        R^2 = {m_lin.rsquared:.4f}")
    p(f"    symmetric |alpha|            R^2 = {m_abs.rsquared:.4f}")
    p(f"    alpha + alpha^2 (full)       R^2 = {m_quad.rsquared:.4f}")
    p(f"    => antisymmetric(sign) share = {m_lin.rsquared:.4f} of {m_quad.rsquared:.4f}"
      f"  ({100*m_lin.rsquared/max(m_quad.rsquared,1e-9):.1f}% of alpha-variance is sign)")
    # Direct symmetric test: compare +a vs -a means.
    sym_pairs = []
    for a in [0.4, 0.8]:
        hi = df.loc[df.alpha == a, "gap_norm"].mean()
        lo = df.loc[df.alpha == -a, "gap_norm"].mean()
        sym_pairs.append((a, lo, hi, hi - lo))
        p(f"    |alpha|={a}: gap_norm(-{a})={lo:.4f}  gap_norm(+{a})={hi:.4f}  "
          f"diff={hi-lo:+.4f}")

    # ----- alpha x sigma interaction: does the U deepen with stakes (normalized)? -----
    p("\n--- ALPHA x SIGMA on NORMALIZED gap (U-depth vs stakes) ---")
    piv = df.groupby(["sigma", "alpha"])["gap_norm"].mean().unstack()
    piv.to_csv(OUT / "interaction_table.csv", float_format="%.6f")
    p("  rows=sigma, cols=alpha; values=normalized gap")
    p("  " + "        ".join([f"a={a:+.1f}" for a in piv.columns]))
    for sig, row in piv.iterrows():
        p(f"  sigma={sig:.1f}: " + "  ".join([f"{v:7.4f}" for v in row.values]))
    # U-depth = gap_norm(alpha=0) - mean(gap_norm(alpha=+-0.8)); larger => deeper U.
    p("\n  U-depth (normalized) = gap_norm(a=0) - mean(gap_norm(a=+-0.8)), by sigma:")
    for sig, row in piv.iterrows():
        depth = row[0.0] - 0.5 * (row[-0.8] + row[0.8])
        p(f"    sigma={sig:.1f}: U-depth = {depth:+.4f}")
    # Same on RAW gap for contrast.
    piv_raw = df.groupby(["sigma", "alpha"])["gap"].mean().unstack()
    p("\n  U-depth (RAW) = gap(a=0) - mean(gap(a=+-0.8)), by sigma:")
    for sig, row in piv_raw.iterrows():
        depth = row[0.0] - 0.5 * (row[-0.8] + row[0.8])
        p(f"    sigma={sig:.1f}: U-depth = {depth:+.4f}")

    (OUT / "A_SUMMARY.txt").write_text("\n".join(lines))
    p("\nSaved A_SUMMARY.txt + 6 csv tables to results/review_upgrade/")


if __name__ == "__main__":
    main()
