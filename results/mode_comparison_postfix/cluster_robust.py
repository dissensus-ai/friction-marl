"""Cluster-robust slopes for the 29-30 Jul post-fix arms.

Naive OLS over all rows treats each (condition, replication) row as independent. It is not:
replication index r reuses the same preference-draw / seed stream across conditions, so residuals
are correlated within a replication. Prior work on this paper already showed the cost -- a rho x
overlap interaction read p ~ 0.02 naively and p ~ 0.09 once clustered at the run level.

This script reports naive and cluster-robust (CR1, clustered on replication) standard errors side
by side so the difference is visible rather than asserted. sigma enters as a fixed effect because
the adversarial designs sweep two stake levels and the claim of interest is the rho slope.

Run:  PYTHONPATH=<repo> python3 results/mode_comparison_postfix/cluster_robust.py
"""
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

HERE = Path(__file__).resolve().parent
MODES = ["iql", "vdn", "mappo"]


def fit(df, xvar):
    """Return (beta, naive_p, cluster_p, n_obs, n_clusters) for eval_rew ~ xvar + C(sigma)."""
    formula = f"eval_rew ~ {xvar} + C(sigma)"
    naive = smf.ols(formula, data=df).fit()
    clus = smf.ols(formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["replication"]}
    )
    return (
        clus.params[xvar],
        naive.pvalues[xvar],
        clus.pvalues[xvar],
        int(clus.nobs),
        df["replication"].nunique(),
    )


def report(fname, xvar, label):
    path = HERE / fname
    if not path.exists():
        print(f"[skip] {fname} not found")
        return
    d = pd.read_csv(path)
    # b_vdn_runner / adv_runner store eval_gap as the NEGATED eval reward, not a gap.
    d["eval_rew"] = -d["eval_gap"]
    print(f"\n=== {label}  ({fname})")
    print(f"{'mode':<7}{'slope':>9}{'p_naive':>11}{'p_cluster':>11}{'n':>7}{'clusters':>10}  verdict")
    for m in MODES:
        s = d[d["mode"] == m]
        if s.empty:
            continue
        b, pn, pc, n, k = fit(s, xvar)
        moved = "" if pc < 0.05 else ("  <- LOST at .05" if pn < 0.05 else "")
        print(f"{m:<7}{b:>+9.3f}{pn:>11.2e}{pc:>11.2e}{n:>7}{k:>10}{moved}")


if __name__ == "__main__":
    report("adv_equicorr.csv", "rho", "Adversarial equicorrelated DGP - slope on signed rho")
    report("adv_team.csv", "rho", "Adversarial team/blocked DGP - slope on signed rho")
    report("modes_iql_vdn_mappo.csv", "alpha", "Mode comparison grid - slope on alpha")
