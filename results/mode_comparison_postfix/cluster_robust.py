"""Estimator ladder for the 29-31 Jul post-fix arms.

v2 (31 Jul), superseding the 30 Jul version after the fleet-review methods audit
(_reviews/FLEET_REVIEW_30JUL2026.md section 3). Two corrections:

1. The v1 docstring asserted residual correlation within a replication because "the
   same preference-draw / seed stream" is reused across conditions. FALSE for these
   datasets: adv_runner/b_vdn_runner draw an independent seed per condition
   (cond_seeds), so replication index carries no shared component -- measured ICC is
   ~0 (-0.01..+0.004). Clustering on replication is therefore a PLACEBO, not a
   correction, and the v1 headline "MAPPO loses significance under clustering" was an
   artifact of the wrong cluster variable.

2. One specification is not an answer. This version reports the ladder -- naive OLS /
   HC3 / CR1 clustered on condition cell (G small => t(G-1) critical values) /
   OLS on condition means -- plus replication-CR1 as the labelled placebo. For the
   team arm it also reports the design-implied predictor (between-team correlation
   -opp^2; the linear-on-opp call was form-dependent, p .037 vs .053) and a rank
   check (Wilcoxon on per-replication slopes), since the -1.404 mean slope is
   carried by a failure tail (15/30 negative), not a median shift.

Run:  PYTHONPATH=<repo> python3 results/mode_comparison_postfix/cluster_robust.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

HERE = Path(__file__).resolve().parent
MODES = ["iql", "vdn", "mappo"]


def ladder(df, xvar):
    """Return dict of (label -> (beta, p)) for eval_rew ~ xvar + C(sigma)."""
    formula = f"eval_rew ~ {xvar} + C(sigma)"
    out = {}
    m = smf.ols(formula, data=df)
    naive = m.fit()
    out["naive"] = (naive.params[xvar], naive.pvalues[xvar])
    hc3 = m.fit(cov_type="HC3")
    out["HC3"] = (hc3.params[xvar], hc3.pvalues[xvar])
    df = df.assign(cond=list(zip(df[xvar], df.sigma)))
    cc = m.fit(cov_type="cluster", cov_kwds={"groups": df["cond"]}, use_t=True)
    out[f"CR-cond(G={df['cond'].nunique()})"] = (cc.params[xvar], cc.pvalues[xvar])
    agg = df.groupby("cond", as_index=False).agg(eval_rew=("eval_rew", "mean"),
                                                sigma=("sigma", "first"),
                                                **{xvar: (xvar, "first")})
    ma = smf.ols(formula, data=agg).fit()
    out["cond-means"] = (ma.params[xvar], ma.pvalues[xvar])
    cr = m.fit(cov_type="cluster", cov_kwds={"groups": df["replication"]}, use_t=True)
    out["CR-rep [placebo]"] = (cr.params[xvar], cr.pvalues[xvar])
    return out


def per_rep_slopes(df, xvar):
    """Per-replication OLS slope of eval_rew on xvar (sigma pooled by demeaning)."""
    d = df.copy()
    d["y"] = d.eval_rew - d.groupby("sigma").eval_rew.transform("mean")
    slopes = []
    for _, g in d.groupby("replication"):
        if g[xvar].nunique() > 1:
            slopes.append(np.polyfit(g[xvar], g["y"], 1)[0])
    return np.array(slopes)


def report(fname, xvars, label):
    path = HERE / fname
    if not path.exists():
        print(f"[skip] {fname} not found")
        return
    d = pd.read_csv(path)
    # Prefer the unambiguous column; legacy b_vdn-family CSVs store eval_gap as the
    # NEGATED eval reward, not a gap (see results/DATA_DICTIONARY.md).
    d["eval_rew"] = d["eval_reward"] if "eval_reward" in d.columns else -d["eval_gap"]
    print(f"\n=== {label}  ({fname})")
    for xvar, xdesc in xvars:
        if xvar not in d.columns and xvar == "neg_opp_sq":
            d["neg_opp_sq"] = -d["opp"] ** 2
        print(f"-- predictor: {xvar}  ({xdesc})")
        hdr = f"{'mode':<7}" + "".join(f"{k:>22}" for k in
              ["naive", "HC3", "CR-cond", "cond-means", "CR-rep[placebo]", "Wilcoxon"])
        print(hdr)
        for m in MODES:
            s = d[d["mode"] == m]
            if s.empty:
                continue
            lad = ladder(s, xvar)
            cells = []
            for k in list(lad):
                b, p = lad[k]
                cells.append(f"{b:+.3f} p={p:.3f}")
            sl = per_rep_slopes(s, xvar)
            try:
                wp = stats.wilcoxon(sl).pvalue
            except ValueError:
                wp = float("nan")
            cells.append(f"med={np.median(sl):+.3f} p={wp:.3f}")
            print(f"{m:<7}" + "".join(f"{c:>22}" for c in cells))


if __name__ == "__main__":
    report("adv_equicorr.csv", [("rho", "signed pairwise correlation")],
           "Adversarial equicorrelated DGP")
    report("adv_team_opp.csv",
           [("opp", "opposition strength, linear (form-dependent -- see v2 note)"),
            ("neg_opp_sq", "between-team correlation -opp^2 (design-implied)")],
           "Adversarial team/block DGP")
    report("modes_iql_vdn_mappo.csv", [("alpha", "|alpha| magnitude DGP (sign-blind)")],
           "Mode comparison grid")
