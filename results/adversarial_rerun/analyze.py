"""STEP 4: re-analysis -- does structured OPPOSITION beat indifference?

Loads the signed-DGP re-run (rho = genuine cross-agent correlation) for IQL & VDN
and answers:
  1. gap vs rho curve for both learners.
  2. Is rho=0 (indifference) still the worst case?
  3. Do GENUINE-adversarial (rho<0) conditions coordinate BETTER than neutral
     (real U-shape w/ a real adversarial arm) or is it MONOTONE (coop best,
     adversarial worst)?  <-- HEADLINE
  4. eta^2 (raw + sigma-normalized); does the OLD |alpha|-symmetry survive or
     does SIGN now matter?
  5. Does VDN preserve the shape?

Usage:  python analyze.py --csvs a.csv b.csv ... --tag equicorr
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

HERE = Path(__file__).resolve().parent


def anova_eta(df, dv, formula_rhs):
    model = ols(f"{dv} ~ {formula_rhs}", data=df).fit()
    tbl = anova_lm(model, typ=2)
    ss_total = tbl["sum_sq"].sum()
    tbl["eta_sq"] = tbl["sum_sq"] / ss_total
    return tbl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="+", required=True)
    ap.add_argument("--tag", default="equicorr")
    args = ap.parse_args()

    df = pd.concat([pd.read_csv(c) for c in args.csvs], ignore_index=True)
    df["gap"] = -df["mean_reward"]
    df["gap_norm"] = df["gap"] / df["sigma"]
    df["abs_rho"] = df["rho"].abs()

    lines = []
    def p(s=""):
        print(s); lines.append(s)

    modes = sorted(df["mode"].unique())
    rhos = sorted(df["rho"].unique())
    sigmas = sorted(df["sigma"].unique())
    nrep = df.groupby(["mode", "rho", "sigma"]).size()
    p("=" * 84)
    p(f"ADVERSARIAL RE-ANALYSIS [{args.tag}]  -- signed DGP (rho = real cross-agent corr)")
    p(f"modes={modes}  rhos={rhos}  sigmas={sigmas}  reps={nrep.min()}-{nrep.max()}")
    p("=" * 84)

    # ---- (1) gap vs rho table ----
    for dv, lab in [("gap", "RAW gap"), ("gap_norm", "NORMALIZED gap (/sigma)")]:
        p(f"\n### {lab}: mean (SEM) by rho, per sigma x mode ###")
        for s in sigmas:
            p(f"\n  sigma={s}")
            p("    rho:    " + "  ".join(f"{r:+.2f}" for r in rhos))
            for m in modes:
                cells = []
                for r in rhos:
                    v = df[(df["mode"] == m) & (df["sigma"] == s) & (df["rho"] == r)][dv]
                    cells.append(f"{v.mean():5.2f}")
                p(f"    {m:<5}: " + "  ".join(cells))

    # ---- (2)+(3) worst case + monotone-vs-U headline ----
    p("\n" + "=" * 84)
    p("### HEADLINE: indifference (rho=0) vs genuine opposition (rho<0) ###")
    p("=" * 84)
    head_rows = []
    for m in modes:
        sub = df[df["mode"] == m]
        g_by_rho = sub.groupby("rho")["gap_norm"].agg(["mean", "std", "count"])
        worst_rho = g_by_rho["mean"].idxmax()
        best_rho = g_by_rho["mean"].idxmin()
        g0 = sub[sub["rho"] == 0.0]["gap_norm"]
        neg_rhos = [r for r in rhos if r < 0]
        adv_extreme = min(neg_rhos)               # most adversarial available
        gadv = sub[sub["rho"] == adv_extreme]["gap_norm"]
        gcoop = sub[sub["rho"] == max(rhos)]["gap_norm"]
        # is neutral worse than adversarial?  (U-shape signature)
        t_adv, p_adv = stats.ttest_ind(g0, gadv, equal_var=False)
        u_adv, pu_adv = stats.mannwhitneyu(g0, gadv, alternative="two-sided")
        t_coop, p_coop = stats.ttest_ind(g0, gcoop, equal_var=False)
        p(f"\n  [{m.upper()}]  worst rho = {worst_rho:+.2f}   best rho = {best_rho:+.2f}")
        p(f"    gap_norm(rho=0,neutral)      = {g0.mean():.3f} (sd {g0.std():.3f}, n {len(g0)})")
        p(f"    gap_norm(rho={adv_extreme:+.2f},adversarial)= {gadv.mean():.3f} (sd {gadv.std():.3f})")
        p(f"    gap_norm(rho={max(rhos):+.2f},cooperative) = {gcoop.mean():.3f} (sd {gcoop.std():.3f})")
        p(f"    neutral - adversarial = {g0.mean()-gadv.mean():+.3f}  "
          f"(Welch t={t_adv:+.2f}, p={p_adv:.2e}; MWU p={pu_adv:.2e})")
        p(f"    neutral - cooperative = {g0.mean()-gcoop.mean():+.3f}  "
          f"(Welch t={t_coop:+.2f}, p={p_coop:.2e})")
        # verdict
        if g0.mean() > gadv.mean() and p_adv < 0.05:
            verdict = ("U-SHAPE with a REAL adversarial arm: structured OPPOSITION "
                       "coordinates BETTER than indifference.")
        elif gadv.mean() > g0.mean() and p_adv < 0.05:
            verdict = ("MONOTONE: cooperative best, adversarial WORST; opposition does "
                       "NOT beat indifference (adversarial is even worse).")
        else:
            verdict = ("FLAT between neutral & adversarial (no significant difference)."
                       )
        p(f"    => {verdict}")
        head_rows.append({"tag": args.tag, "mode": m, "worst_rho": worst_rho,
                          "best_rho": best_rho, "gap_neutral": g0.mean(),
                          "gap_adversarial": gadv.mean(), "gap_cooperative": gcoop.mean(),
                          "neutral_minus_adv": g0.mean()-gadv.mean(), "p_adv": p_adv,
                          "neutral_minus_coop": g0.mean()-gcoop.mean(), "p_coop": p_coop,
                          "verdict": verdict})
    pd.DataFrame(head_rows).to_csv(HERE / f"headline_{args.tag}.csv", index=False)

    # ---- (4) eta^2 + sign test ----
    p("\n" + "=" * 84)
    p("### eta^2 (does SIGN of rho now matter, vs the old |alpha|-symmetry?) ###")
    p("=" * 84)
    # pooled over modes? do per mode to be clean, plus combined.
    for m in modes:
        sub = df[df["mode"] == m].copy()
        raw = anova_eta(sub, "gap", "C(rho)*C(sigma)")
        nrm = anova_eta(sub, "gap_norm", "C(rho)*C(sigma)")
        def e(t, k): return float(t.loc[k, "eta_sq"]) if k in t.index else float("nan")
        p(f"\n  [{m.upper()}] replication-level eta^2")
        p(f"    {'effect':<22}{'RAW':>10}{'NORM':>10}")
        for k in ["C(rho)", "C(sigma)", "C(rho):C(sigma)", "Residual"]:
            p(f"    {k:<22}{e(raw,k):>10.4f}{e(nrm,k):>10.4f}")

        # signed-linear vs symmetric(|rho|) vs quadratic on gap_norm
        m_lin = ols("gap_norm ~ rho", data=sub).fit()
        m_abs = ols("gap_norm ~ abs_rho", data=sub).fit()
        m_quad = ols("gap_norm ~ rho + I(rho**2)", data=sub).fit()
        p(f"    R^2  signed-linear(rho)={m_lin.rsquared:.4f}   "
          f"symmetric(|rho|)={m_abs.rsquared:.4f}   quad(rho+rho^2)={m_quad.rsquared:.4f}")
        share = 100*m_lin.rsquared/max(m_quad.rsquared, 1e-9)
        p(f"         => signed(monotone) explains {share:.0f}% of the (quad) rho-variance"
          f"  [linear slope={m_lin.params['rho']:+.3f}]")
        # direct sign test on the one symmetric magnitude available (+-0.2)
        if (-0.2 in rhos) and (0.2 in rhos):
            lo = sub[sub["rho"] == -0.2]["gap_norm"]
            hi = sub[sub["rho"] == 0.2]["gap_norm"]
            ts, ps = stats.ttest_ind(lo, hi, equal_var=False)
            p(f"         sign test |rho|=0.2: gap_norm(-0.2)={lo.mean():.3f} vs "
              f"gap_norm(+0.2)={hi.mean():.3f}  diff={hi.mean()-lo.mean():+.3f} "
              f"(t={ts:+.2f}, p={ps:.2e})  "
              f"[{'SIGN MATTERS' if ps<0.05 else 'symmetric'}]")

    # ---- (5) VDN preserves shape? ----
    if "iql" in modes and "vdn" in modes:
        p("\n" + "=" * 84)
        p("### VDN vs IQL: same shape over rho? (centralization effect) ###")
        p("=" * 84)
        ceff = []
        for r in rhos:
            iq = df[(df["mode"] == "iql") & (df["rho"] == r)]["gap_norm"]
            vd = df[(df["mode"] == "vdn") & (df["rho"] == r)]["gap_norm"]
            t, pv = stats.ttest_ind(iq, vd, equal_var=False)
            ceff.append({"rho": r, "iql_gap": iq.mean(), "vdn_gap": vd.mean(),
                         "iql_minus_vdn": iq.mean()-vd.mean(), "t": t, "p": pv})
            p(f"  rho={r:+.2f}: IQL={iq.mean():.3f}  VDN={vd.mean():.3f}  "
              f"IQL-VDN={iq.mean()-vd.mean():+.3f} (t={t:+.2f}, p={pv:.2e})")
        pd.DataFrame(ceff).to_csv(HERE / f"centralization_{args.tag}.csv", index=False)
        # correlation of the two shape-curves
        iq_curve = [df[(df["mode"]=="iql")&(df["rho"]==r)]["gap_norm"].mean() for r in rhos]
        vd_curve = [df[(df["mode"]=="vdn")&(df["rho"]==r)]["gap_norm"].mean() for r in rhos]
        shape_corr = np.corrcoef(iq_curve, vd_curve)[0, 1]
        p(f"\n  IQL vs VDN gap-vs-rho SHAPE correlation = {shape_corr:.3f}  "
          f"({'VDN preserves the shape' if shape_corr>0.9 else 'shapes differ'})")

    # ---- plot ----
    fig, axes = plt.subplots(1, len(sigmas), figsize=(5.2*len(sigmas), 4.4), sharey=True)
    if len(sigmas) == 1:
        axes = [axes]
    colors = {"iql": "#7a1f2b", "vdn": "#1f4e79"}
    for ax, s in zip(axes, sigmas):
        for m in modes:
            sub = df[(df["mode"] == m) & (df["sigma"] == s)]
            g = sub.groupby("rho")["gap_norm"]
            mean = g.mean(); sem = g.std()/np.sqrt(g.count())
            ax.errorbar(mean.index, mean.values, yerr=1.96*sem.values, marker="o",
                        capsize=3, lw=2, label=m.upper(), color=colors.get(m))
        ax.axvline(0, color="grey", ls=":", lw=1, label="rho=0 (indifference)")
        ax.axvline(-1/3, color="b", ls=":", lw=0.8, alpha=0.6)
        ax.set_title(f"sigma={s}")
        ax.set_xlabel(r"$\rho$  (cross-agent correlation: <0 adversarial, >0 cooperative)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"normalized reward gap $\Delta R/\sigma$")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"[{args.tag}] reward gap vs GENUINE preference correlation: "
                 "cooperative -> neutral -> adversarial", y=1.02)
    fig.tight_layout()
    fig.savefig(HERE / f"gap_vs_rho_{args.tag}.png", dpi=150, bbox_inches="tight")
    fig.savefig(HERE / f"gap_vs_rho_{args.tag}.pdf", bbox_inches="tight")
    p(f"\nSaved gap_vs_rho_{args.tag}.png/.pdf")

    (HERE / f"ANALYSIS_{args.tag}.txt").write_text("\n".join(lines))
    p(f"Saved ANALYSIS_{args.tag}.txt + headline/centralization CSVs")


if __name__ == "__main__":
    main()
