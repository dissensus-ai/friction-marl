"""TASK A2 -- categorical-w refit + baseline-difficulty confound test.

DV: gap_norm = -mean_reward / sigma   (matches analyze_partial.load)
predictor: signed rho. factor: w in {0,1/3,2/3,1}. pooled over sigma in {0.6,1.0}.

All scipy/statsmodels results accessed via NAMED attributes only.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

REPO = Path("/home/purrpower/Resurrexi/projects/papers/github-repos/friction-marl")
PS = REPO / "results" / "partial_sharing"
OUTDIR = REPO / "results" / "review2_jun27"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT = []
def p(s=""):
    print(s); OUT.append(str(s))


def load(path):
    df = pd.read_csv(path)
    df["gap"] = -df["mean_reward"]
    df["gap_norm"] = df["gap"] / df["sigma"]
    return df


def per_w_slope_from_categorical(model, w_levels, ref_w):
    """Per-w rho-slope via linear combinations of the C(w)*rho params, robust cov.
    Returns dict[w] -> dict(slope, ci_lo, ci_hi, p) using NAMED ContrastResults attrs."""
    names = list(model.model.exog_names)
    idx = {nm: i for i, nm in enumerate(names)}
    # locate the main rho coef and per-level interaction coefs
    rho_i = idx["rho"]
    out = {}
    for w in w_levels:
        c = np.zeros(len(names))
        c[rho_i] = 1.0
        if w != ref_w:
            # find interaction term matching this level
            match = [nm for nm in names if nm.endswith(":rho") and (f"[T.{w}]" in nm or f"[T.{repr(w)}]" in nm)]
            if not match:
                # fall back: match by float compare embedded in label
                match = [nm for nm in names if nm.endswith(":rho") and _label_float(nm) is not None
                         and abs(_label_float(nm) - w) < 1e-9]
            assert len(match) == 1, f"could not uniquely match interaction for w={w}: {match}"
            c[idx[match[0]]] = 1.0
        tt = model.t_test(c)
        ci = tt.conf_int(alpha=0.05)
        out[w] = dict(slope=float(np.ravel(tt.effect)[0]),
                      se=float(np.ravel(tt.sd)[0]),
                      p=float(np.ravel(tt.pvalue)[0]),
                      ci_lo=float(ci[0, 0]), ci_hi=float(ci[0, 1]))
    return out


def per_w_baseline_from_categorical(model, w_levels, ref_w):
    """Per-w intercept (predicted gap_norm at rho=0) via C(w) main-effect combos."""
    names = list(model.model.exog_names)
    idx = {nm: i for i, nm in enumerate(names)}
    int_i = idx["Intercept"]
    out = {}
    for w in w_levels:
        c = np.zeros(len(names))
        c[int_i] = 1.0
        if w != ref_w:
            match = [nm for nm in names if nm.startswith("C(w)[T.") and ":rho" not in nm
                     and _label_float(nm) is not None and abs(_label_float(nm) - w) < 1e-9]
            assert len(match) == 1, f"baseline match fail w={w}: {match}"
            c[idx[match[0]]] = 1.0
        tt = model.t_test(c)
        ci = tt.conf_int(alpha=0.05)
        out[w] = dict(intercept=float(np.ravel(tt.effect)[0]),
                      se=float(np.ravel(tt.sd)[0]),
                      ci_lo=float(ci[0, 0]), ci_hi=float(ci[0, 1]))
    return out


def _label_float(nm):
    """extract the float inside C(w)[T.<float>]..."""
    try:
        s = nm.split("[T.")[1].split("]")[0]
        return float(s)
    except Exception:
        return None


def linregress_per_w(sub):
    r = stats.linregress(sub["rho"].to_numpy(float), sub["gap_norm"].to_numpy(float))
    n = len(sub)
    tcrit = float(stats.t.ppf(0.975, n - 2))
    return dict(slope=float(r.slope), se=float(r.stderr), p=float(r.pvalue),
                r2=float(r.rvalue**2), n=n,
                ci_lo=float(r.slope - tcrit * r.stderr),
                ci_hi=float(r.slope + tcrit * r.stderr))


def main():
    data = {"IQL": load(PS / "partial_iql.csv"), "VDN": load(PS / "partial_vdn.csv")}
    w_levels = sorted(data["IQL"]["w"].unique())
    ref_w = w_levels[0]
    p("#" * 90)
    p("# TASK A2 -- CATEGORICAL-w REFIT (misspec fix) + BASELINE-DIFFICULTY CONFOUND TEST")
    p("#" * 90)
    p(f"DV = gap_norm = -mean_reward/sigma | predictor = signed rho | factor w={[round(w,3) for w in w_levels]}")
    p(f"pooled over sigma={sorted(data['IQL']['sigma'].unique())} ; rho grid="
      f"{sorted(data['IQL']['rho'].unique())} ; 30 reps/cell ; n=1920 per learner")
    p(f"python/scipy/statsmodels named-attr access; robust HC3 cov on pooled categorical model")

    slope_rows, baseline_rows, comp_rows = [], [], []

    for learner, df in data.items():
        p("\n" + "=" * 90)
        p(f"[{learner}]")
        p("=" * 90)

        # ---- categorical interaction model (robust HC3) ----
        m_cat = smf.ols("gap_norm ~ C(w) * rho", data=df).fit(cov_type="HC3")
        slopes_cat = per_w_slope_from_categorical(m_cat, w_levels, ref_w)
        base_cat = per_w_baseline_from_categorical(m_cat, w_levels, ref_w)

        # ---- per-w independent OLS cross-check ----
        p("\n(i) PER-w rho-SLOPE  [categorical C(w)*rho model, HC3]  vs  [independent per-w OLS]")
        p(f"   {'w':>6} {'k':>2} | {'slope_cat':>9} {'95% CI (cat)':>22} {'p_cat':>10} |"
          f" {'slope_OLS':>9} {'95% CI (OLS)':>22} {'p_OLS':>10}  {'R2':>5}")
        for w in w_levels:
            k = int(round(w * 3))
            sc = slopes_cat[w]
            so = linregress_per_w(df[df["w"] == w])
            p(f"   {w:6.3f} {k:>2} | {sc['slope']:+9.3f} [{sc['ci_lo']:+7.3f},{sc['ci_hi']:+7.3f}] "
              f"{sc['p']:10.2e} | {so['slope']:+9.3f} [{so['ci_lo']:+7.3f},{so['ci_hi']:+7.3f}] "
              f"{so['p']:10.2e}  {so['r2']:5.3f}")
            slope_rows.append(dict(learner=learner, w=round(w, 4), k=k,
                                   slope_cat=sc['slope'], ci_lo_cat=sc['ci_lo'], ci_hi_cat=sc['ci_hi'],
                                   p_cat=sc['p'], slope_ols=so['slope'], ci_lo_ols=so['ci_lo'],
                                   ci_hi_ols=so['ci_hi'], p_ols=so['p'], r2_ols=so['r2']))

        # threshold shape summary
        sl = np.array([slopes_cat[w]["slope"] for w in w_levels])
        p(f"\n   threshold shape (signed slope by w): "
          + "  ".join(f"w={w:.2f}:{slopes_cat[w]['slope']:+.3f}" for w in w_levels))
        p(f"   |slope| jump w=1/3->2/3 : {abs(sl[2])-abs(sl[1]):+.3f}   "
          f"(low-w mean |slope|={np.mean(np.abs(sl[:2])):.3f}, high-w mean |slope|={np.mean(np.abs(sl[2:])):.3f})")

        # ---- model comparisons ----
        p("\n(i) MODEL COMPARISONS")
        m_add = smf.ols("gap_norm ~ C(w) + rho", data=df).fit()          # common slope
        m_catO = smf.ols("gap_norm ~ C(w) * rho", data=df).fit()          # per-w slopes (OLS for F)
        m_lin = smf.ols("gap_norm ~ rho * w", data=df).fit()             # single LINEAR rho:w
        m_linO = m_lin

        # (A) does the interaction exist? categorical interaction vs additive (no interaction)
        aov_A = anova_lm(m_add, m_catO)
        F_A = float(aov_A["F"].iloc[-1]); pF_A = float(aov_A["Pr(>F)"].iloc[-1])
        ddf_A = float(aov_A["df_diff"].iloc[-1])
        # (B) is the single LINEAR interaction an adequate summary? linear(rho*w) vs categorical
        aov_B = anova_lm(m_lin, m_catO)
        F_B = float(aov_B["F"].iloc[-1]); pF_B = float(aov_B["Pr(>F)"].iloc[-1])
        ddf_B = float(aov_B["df_diff"].iloc[-1])

        # the single linear interaction coefficient (the misspecified summary)
        b_lin = float(m_lin.params["rho:w"]); p_lin = float(m_lin.pvalues["rho:w"])
        ci_lin = m_lin.conf_int().loc["rho:w"]

        p(f"   single LINEAR rho:w coef (misspecified summary) = {b_lin:+.3f} "
          f"[95% CI {float(ci_lin.iloc[0]):+.3f},{float(ci_lin.iloc[1]):+.3f}] p={p_lin:.2e}")
        p(f"   (A) C(w)*rho  vs  C(w)+rho   [do per-w slopes DIFFER?]   "
          f"F({ddf_A:.0f}, {float(aov_A['df_resid'].iloc[-1]):.0f})={F_A:.2f}  p={pF_A:.3e}")
        p(f"   (B) C(w)*rho  vs  rho*w(lin) [is LINEAR summary adequate?] "
          f"F({ddf_B:.0f}, {float(aov_B['df_resid'].iloc[-1]):.0f})={F_B:.2f}  p={pF_B:.3e}")
        p(f"   AIC: linear rho*w = {m_lin.aic:.1f} | categorical C(w)*rho = {m_catO.aic:.1f} "
          f"(Delta={m_lin.aic - m_catO.aic:+.1f}, lower=better)")
        comp_rows.append(dict(learner=learner,
                              lin_interaction=b_lin, lin_interaction_p=p_lin,
                              F_interaction_exists=F_A, p_interaction_exists=pF_A, df_A=ddf_A,
                              F_linear_inadequate=F_B, p_linear_inadequate=pF_B, df_B=ddf_B,
                              aic_linear=float(m_lin.aic), aic_categorical=float(m_catO.aic)))

        # ---- (ii) BASELINE-DIFFICULTY CONFOUND ----
        p("\n(ii) BASELINE (rho=0) gap_norm by w  [does difficulty raise the LEVEL?]")
        p(f"   {'w':>6} {'k':>2} | {'intercept(model)':>16} {'95% CI':>20} | {'rho=0 cell mean':>15} {'95% CI(raw)':>22} {'n':>4}")
        base_means = []
        for w in w_levels:
            k = int(round(w * 3))
            bc = base_cat[w]
            cell = df[(df["w"] == w) & (df["rho"] == 0.0)]["gap_norm"].to_numpy(float)
            mu = float(cell.mean()); sd = float(cell.std(ddof=1)); n = cell.size
            se = sd / np.sqrt(n); tcrit = float(stats.t.ppf(0.975, n - 1))
            lo, hi = mu - tcrit * se, mu + tcrit * se
            base_means.append(mu)
            p(f"   {w:6.3f} {k:>2} | {bc['intercept']:16.3f} [{bc['ci_lo']:+6.3f},{bc['ci_hi']:+6.3f}] |"
              f" {mu:15.3f} [{lo:+7.3f},{hi:+7.3f}] {n:>4}")
            baseline_rows.append(dict(learner=learner, w=round(w, 4), k=k,
                                      intercept_model=bc['intercept'], int_ci_lo=bc['ci_lo'],
                                      int_ci_hi=bc['ci_hi'], rho0_cellmean=mu, raw_ci_lo=lo,
                                      raw_ci_hi=hi, n=n))
        # trend test: baseline rises with w?  (regress rho=0 obs on w)
        d0 = df[df["rho"] == 0.0]
        tr = stats.linregress(d0["w"].to_numpy(float), d0["gap_norm"].to_numpy(float))
        tcrit0 = float(stats.t.ppf(0.975, len(d0) - 2))
        sp0 = stats.spearmanr(d0["w"].to_numpy(float), d0["gap_norm"].to_numpy(float))
        p(f"\n   baseline trend (gap_norm@rho=0 ~ w): slope={float(tr.slope):+.3f} "
          f"[95% CI {float(tr.slope-tcrit0*tr.stderr):+.3f},{float(tr.slope+tcrit0*tr.stderr):+.3f}] "
          f"p={float(tr.pvalue):.3e}  Spearman rho={float(sp0.statistic):+.3f} p={float(sp0.pvalue):.3e}")
        base_rise = base_means[-1] - base_means[0]
        # rho-driven swing at high w for comparison (slope * rho range)
        rho_range = float(df["rho"].max() - df["rho"].min())
        swing_w1 = slopes_cat[w_levels[-1]]["slope"] * rho_range
        p(f"   baseline rise w=0->w=1 : {base_rise:+.3f} (gap_norm units)")
        p(f"   rho-driven swing at w=1 (slope*rho_range, range={rho_range:.2f}) : {swing_w1:+.3f}")
        p(f"   INSULATION: slope is dgap/drho at FIXED w; an additive level shift contributes EXACTLY 0 "
          f"to the within-w slope -> 0% of the {abs(sl[2]+sl[3])-abs(sl[0]+sl[1]):.2f}-unit slope steepening "
          f"is attributable to the baseline level rise.")

    # ---- write CSVs ----
    pd.DataFrame(slope_rows).to_csv(OUTDIR / "a2_per_w_slopes.csv", index=False)
    pd.DataFrame(baseline_rows).to_csv(OUTDIR / "a2_baseline_by_w.csv", index=False)
    pd.DataFrame(comp_rows).to_csv(OUTDIR / "a2_model_comparisons.csv", index=False)
    (OUTDIR / "a2_console.txt").write_text("\n".join(OUT))
    p("\nWrote: a2_per_w_slopes.csv, a2_baseline_by_w.csv, a2_model_comparisons.csv, a2_console.txt")


if __name__ == "__main__":
    main()
