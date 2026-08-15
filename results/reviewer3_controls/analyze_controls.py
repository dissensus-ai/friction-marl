"""Clean analysis for the two reviewer3 control experiments.

All test statistics are extracted via the named scipy result attributes
(`.statistic` / `.pvalue`) -- no positional tuple unpacking -- to avoid the
tuple-vs-float ambiguity flagged in the prior adversarial_rerun/analyze.py.

(A) SEPARABLE-RESOURCE CONTROL: does the signed-correlation law (gap tracks rho)
    SURVIVE when each agent has its own independent resource pool, or does the
    gap-vs-rho curve go FLAT (=> the friction was shared-state contention)?
    Compared head-to-head with the shared-pool equicorr run.

(B) n=2 LATENT-TARGET STRONG OPPOSITION: at target rho=-1 (feasible-optimum
    correlation -0.468 after box constraints, plus an agent-count confound), is
    there a detected difference from indifference? Welch contrast + 95% CI + p.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

HERE = Path(__file__).resolve().parent
ADV = HERE.parent / "adversarial_rerun"

OUT_LINES = []
def p(s=""):
    print(s)
    OUT_LINES.append(s)


# --------------------------- clean stat helpers --------------------------- #
def welch(a, b):
    """Welch t-test + 95% CI of (mean(a)-mean(b)); all via named attributes."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    res = stats.ttest_ind(a, b, equal_var=False)
    t = float(res.statistic); pv = float(res.pvalue)
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = float(np.sqrt(va / na + vb / nb))
    diff = float(a.mean() - b.mean())
    # Welch-Satterthwaite dof
    dof = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    tcrit = float(stats.t.ppf(0.975, dof))
    lo, hi = diff - tcrit * se, diff + tcrit * se
    mwu = stats.mannwhitneyu(a, b, alternative="two-sided")
    return dict(diff=diff, t=t, p=pv, ci_lo=lo, ci_hi=hi, dof=float(dof),
                mwu_p=float(mwu.pvalue), na=na, nb=nb,
                mean_a=float(a.mean()), mean_b=float(b.mean()),
                sd_a=float(a.std(ddof=1)), sd_b=float(b.std(ddof=1)))


def lin_slope(x, y):
    """OLS slope via scipy.linregress -- named attributes only."""
    r = stats.linregress(np.asarray(x, float), np.asarray(y, float))
    return dict(slope=float(r.slope), intercept=float(r.intercept),
                r2=float(r.rvalue ** 2), p=float(r.pvalue), se=float(r.stderr))


def anova_across_rho(df, dv="gap_norm"):
    """One-way ANOVA of dv across rho levels (f_oneway) -- named attributes."""
    groups = [g[dv].values for _, g in df.groupby("rho")]
    res = stats.f_oneway(*groups)
    return dict(F=float(res.statistic), p=float(res.pvalue), k=len(groups))


def load(*paths):
    df = pd.concat([pd.read_csv(x) for x in paths], ignore_index=True)
    df["gap"] = -df["mean_reward"]
    df["gap_norm"] = df["gap"] / df["sigma"]
    return df


# ============================ EXPERIMENT 1 ============================ #
def experiment1():
    p("#" * 84)
    p("# EXPERIMENT 1 -- SEPARABLE-RESOURCE CONTROL (reviewer3 Critical #1)")
    p("#" * 84)
    sep = load(HERE / "separable_iql.csv", HERE / "separable_vdn.csv")
    sha = load(ADV / "equicorr_iql.csv", ADV / "equicorr_vdn.csv")
    sep["arch"] = "separable"; sha["arch"] = "shared"

    rhos = sorted(sep["rho"].unique())
    modes = sorted(sep["mode"].unique())
    p(f"\nseparable: rhos={rhos}  modes={modes}  "
      f"reps/cond={sep.groupby(['mode','rho','sigma']).size().min()}-"
      f"{sep.groupby(['mode','rho','sigma']).size().max()}  sigmas={sorted(sep['sigma'].unique())}")
    p(f"shared   : rhos={sorted(sha['rho'].unique())}  "
      f"reps/cond={sha.groupby(['mode','rho','sigma']).size().min()}-"
      f"{sha.groupby(['mode','rho','sigma']).size().max()}")

    # gap-vs-rho table (normalized, pooled over sigma)
    p("\n--- normalized gap dR/sigma vs rho (pooled over sigma) ---")
    p("  arch/mode :  " + "  ".join(f"{r:+.2f}" for r in rhos))
    curves = {}
    for arch, d in [("shared", sha), ("separable", sep)]:
        for m in modes:
            row = []
            for r in rhos:
                v = d[(d["mode"] == m) & (d["rho"] == r)]["gap_norm"]
                row.append(v.mean())
            curves[(arch, m)] = np.array(row)
            p(f"  {arch[:4]}/{m:<3} :  " + "  ".join(f"{x:5.2f}" for x in row))

    # per-mode tests
    head = []
    for m in modes:
        p("\n" + "=" * 72)
        p(f"[{m.upper()}]  separable vs shared structure")
        p("=" * 72)
        for arch, d in [("shared", sha), ("separable", sep)]:
            sub = d[d["mode"] == m]
            sl = lin_slope(sub["rho"], sub["gap_norm"])
            av = anova_across_rho(sub, "gap_norm")
            g0 = sub[sub["rho"] == 0.0]["gap_norm"]
            gcoop = sub[sub["rho"] == max(rhos)]["gap_norm"]
            gadv = sub[sub["rho"] == min(rhos)]["gap_norm"]
            coop = welch(gcoop, g0)   # cooperation - indifference (expect <0 if coop helps)
            adv = welch(gadv, g0)     # adversarial - indifference
            p(f"\n  {arch.upper()}")
            p(f"    slope(gap_norm~rho) = {sl['slope']:+.3f}  (R^2={sl['r2']:.3f}, p={sl['p']:.2e})")
            p(f"    one-way ANOVA across rho: F={av['F']:.2f}, p={av['p']:.2e}  (k={av['k']} levels)")
            p(f"    cooperation (rho={max(rhos):+.2f}) - indifference (rho=0): "
              f"{coop['diff']:+.3f}  [95% CI {coop['ci_lo']:+.3f},{coop['ci_hi']:+.3f}]  "
              f"Welch t={coop['t']:+.2f} p={coop['p']:.2e}  MWU p={coop['mwu_p']:.2e}")
            p(f"    opposition  (rho={min(rhos):+.2f}) - indifference (rho=0): "
              f"{adv['diff']:+.3f}  [95% CI {adv['ci_lo']:+.3f},{adv['ci_hi']:+.3f}]  "
              f"Welch t={adv['t']:+.2f} p={adv['p']:.2e}")
            head.append(dict(experiment="separable_control", arch=arch, mode=m,
                             slope=sl['slope'], slope_r2=sl['r2'], slope_p=sl['p'],
                             anova_F=av['F'], anova_p=av['p'],
                             coop_minus_indiff=coop['diff'], coop_ci_lo=coop['ci_lo'],
                             coop_ci_hi=coop['ci_hi'], coop_p=coop['p'],
                             adv_minus_indiff=adv['diff'], adv_p=adv['p']))
        # shape correlation
        sc = float(np.corrcoef(curves[("shared", m)], curves[("separable", m)])[0, 1])
        p(f"\n  shape corr (shared vs separable gap-vs-rho curve) = {sc:+.3f}")

    # ---- verdict ----
    p("\n" + "=" * 72)
    p("VERDICT (Experiment 1)")
    p("=" * 72)
    hd = pd.DataFrame(head)
    for m in modes:
        sep_row = hd[(hd["arch"] == "separable") & (hd["mode"] == m)].iloc[0]
        sha_row = hd[(hd["arch"] == "shared") & (hd["mode"] == m)].iloc[0]
        sep_tracks = (sep_row["slope_p"] < 0.05 and sep_row["slope"] < 0) or (sep_row["coop_p"] < 0.05 and sep_row["coop_minus_indiff"] < 0)
        sha_tracks = (sha_row["slope_p"] < 0.05 and sha_row["slope"] < 0) or (sha_row["coop_p"] < 0.05 and sha_row["coop_minus_indiff"] < 0)
        if sha_tracks and not sep_tracks:
            v = ("FLATTENS -> the signed-correlation law is driven by SHARED-STATE "
                 "CONTENTION; with separable pools the gap is flat across rho. "
                 "Central claim is BOUNDED to common-pool friction.")
        elif sha_tracks and sep_tracks:
            v = ("SURVIVES -> cooperation still lowers the gap with independent pools; "
                 "the law is about preference correlation and generalizes.")
        else:
            v = ("INCONCLUSIVE for this mode (shared tracking weak too); see numbers.")
        p(f"  [{m.upper()}] shared_tracks={sha_tracks}  separable_tracks={sep_tracks}")
        p(f"           slope shared={sha_row['slope']:+.3f}(p={sha_row['slope_p']:.1e}) "
          f"separable={sep_row['slope']:+.3f}(p={sep_row['slope_p']:.1e})")
        p(f"           => {v}")

    hd.to_csv(HERE / "exp1_separable_headline.csv", index=False)

    # ---- overlay plot ----
    fig, axes = plt.subplots(1, len(modes), figsize=(5.4 * len(modes), 4.4), sharey=False)
    if len(modes) == 1:
        axes = [axes]
    col = {"shared": "#7a1f2b", "separable": "#1f4e79"}
    for ax, m in zip(axes, modes):
        for arch, d in [("shared", sha), ("separable", sep)]:
            sub = d[d["mode"] == m]
            g = sub.groupby("rho")["gap_norm"]
            mean = g.mean(); sem = g.std() / np.sqrt(g.count())
            ax.errorbar(mean.index, mean.values, yerr=1.96 * sem.values, marker="o",
                        capsize=3, lw=2, label=arch, color=col[arch])
        ax.axvline(0, color="grey", ls=":", lw=1)
        ax.set_title(f"{m.upper()}: gap vs rho")
        ax.set_xlabel(r"$\rho$ (cross-agent preference corr)")
        ax.grid(alpha=0.3); ax.legend(fontsize=9)
    axes[0].set_ylabel(r"normalized reward gap $\Delta R/\sigma$")
    fig.suptitle("Exp1: shared-pool vs separable-resource -- does the signed-corr law survive?", y=1.02)
    fig.tight_layout()
    fig.savefig(HERE / "exp1_shared_vs_separable.png", dpi=150, bbox_inches="tight")
    p(f"\nSaved exp1_shared_vs_separable.png + exp1_separable_headline.csv")
    return curves


# ============================ EXPERIMENT 2 ============================ #
def experiment2():
    p("\n\n" + "#" * 84)
    p("# EXPERIMENT 2 -- n=2 LATENT-TARGET STRONG OPPOSITION (reviewer3 Critical #2)")
    p("#" * 84)
    d = load(HERE / "n2_iql.csv", HERE / "n2_vdn.csv")
    feasible_table = pd.read_csv(ADV / "feasible_correlation.csv")
    feasible_row = feasible_table[
        (feasible_table["n_agents"] == 2)
        & np.isclose(feasible_table["rho_nominal"], -1.0)
    ]
    if len(feasible_row) != 1:
        raise ValueError("expected one n=2, rho_nominal=-1 feasible-correlation row")
    feasible_corr = float(feasible_row.iloc[0]["corr_feasible"])
    rhos = sorted(d["rho"].unique())
    modes = sorted(d["mode"].unique())
    p(f"\nn=2 shared-pool: rhos={rhos}  modes={modes}  sigmas={sorted(d['sigma'].unique())}  "
      f"reps/cond={d.groupby(['mode','rho','sigma']).size().min()}-"
      f"{d.groupby(['mode','rho','sigma']).size().max()}")
    p(f"scope: latent target rho=-1; feasible-optimum corr={feasible_corr:+.4f}; "
      "n=2 introduces an agent-count confound; non-detection is not equivalence")

    p("\n--- normalized gap dR/sigma vs rho (pooled over sigma) ---")
    p("  mode :  " + "  ".join(f"{r:+.2f}" for r in rhos))
    for m in modes:
        row = [d[(d["mode"] == m) & (d["rho"] == r)]["gap_norm"].mean() for r in rhos]
        p(f"  {m:<4} :  " + "  ".join(f"{x:5.2f}" for x in row))

    head = []
    for m in modes:
        p("\n" + "=" * 72)
        p(f"[{m.upper()}]  HEADLINE: latent-target rho=-1 vs indifference (rho=0)")
        p("=" * 72)
        sub = d[d["mode"] == m]
        g_opp = sub[sub["rho"] == -1.0]["gap_norm"]
        g0 = sub[sub["rho"] == 0.0]["gap_norm"]
        gcoop = sub[sub["rho"] == max(rhos)]["gap_norm"]
        w_opp = welch(g_opp, g0)      # opposition - indifference
        w_coop = welch(gcoop, g0)     # cooperation - indifference
        # worst point
        gbyr = sub.groupby("rho")["gap_norm"].mean()
        worst = gbyr.idxmax(); best = gbyr.idxmin()
        p(f"  gap_norm(rho=-1, opposition)  = {w_opp['mean_a']:.3f} (sd {w_opp['sd_a']:.3f}, n {w_opp['na']})")
        p(f"  gap_norm(rho= 0, indifference)= {w_opp['mean_b']:.3f} (sd {w_opp['sd_b']:.3f}, n {w_opp['nb']})")
        p(f"  gap_norm(rho=+{max(rhos):.1f}, cooperation) = {w_coop['mean_a']:.3f} (sd {w_coop['sd_a']:.3f})")
        p(f"\n  OPPOSITION - INDIFFERENCE = {w_opp['diff']:+.3f}  "
          f"[95% CI {w_opp['ci_lo']:+.3f}, {w_opp['ci_hi']:+.3f}]")
        p(f"     Welch t={w_opp['t']:+.2f} (dof {w_opp['dof']:.0f}), p={w_opp['p']:.3e};  "
          f"Mann-Whitney p={w_opp['mwu_p']:.3e}")
        p(f"  COOPERATION - INDIFFERENCE = {w_coop['diff']:+.3f}  "
          f"[95% CI {w_coop['ci_lo']:+.3f}, {w_coop['ci_hi']:+.3f}]  "
          f"Welch t={w_coop['t']:+.2f}, p={w_coop['p']:.3e}")
        p(f"  worst rho = {worst:+.2f}   best rho = {best:+.2f}")

        if w_opp['diff'] > 0 and w_opp['p'] < 0.05:
            verdict = ("pooled Welch detects a HIGHER mean gap at latent-target rho=-1; "
                       "no opposition advantage is detected. This is not a clean n=4 "
                       "confirmation: feasible corr is attenuated and agent count changes.")
        elif w_opp['diff'] < 0 and w_opp['p'] < 0.05:
            verdict = ("pooled Welch detects a LOWER mean gap at latent-target rho=-1; "
                       "interpret only within n=2 and feasible-correlation qualifications.")
        else:
            verdict = ("NO DETECTED DIFFERENCE at latent-target rho=-1; this is a "
                       "non-detection, not equivalence, and feasible corr is attenuated.")
        p(f"  => VERDICT [{m.upper()}]: {verdict}")
        head.append(dict(experiment="n2_strong_opposition", mode=m,
                         gap_opp=w_opp['mean_a'], gap_indiff=w_opp['mean_b'],
                         gap_coop=w_coop['mean_a'],
                         opp_minus_indiff=w_opp['diff'], opp_ci_lo=w_opp['ci_lo'],
                         opp_ci_hi=w_opp['ci_hi'], opp_t=w_opp['t'], opp_p=w_opp['p'],
                         opp_mwu_p=w_opp['mwu_p'], opp_dof=w_opp['dof'],
                         coop_minus_indiff=w_coop['diff'], coop_p=w_coop['p'],
                         worst_rho=worst, best_rho=best,
                         latent_target_rho=-1.0,
                         feasible_optimum_corr=feasible_corr,
                         agent_count_confounded=True,
                         equivalence_tested=False,
                         verdict=verdict))
    pd.DataFrame(head).to_csv(HERE / "exp2_n2_headline.csv", index=False)

    # plot
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    col = {"iql": "#7a1f2b", "vdn": "#1f4e79"}
    for m in modes:
        sub = d[d["mode"] == m]
        g = sub.groupby("rho")["gap_norm"]
        mean = g.mean(); sem = g.std() / np.sqrt(g.count())
        ax.errorbar(mean.index, mean.values, yerr=1.96 * sem.values, marker="o",
                    capsize=3, lw=2, label=m.upper(), color=col.get(m))
    ax.axvline(0, color="grey", ls=":", lw=1, label="rho=0 (indifference)")
    ax.axvline(-1.0, color="k", ls="--", lw=0.8, alpha=0.6,
               label="latent target rho=-1")
    ax.set_xlabel(r"latent target $\rho$ (n=2; feasible $\rho=-1$ corr $=-0.468$)")
    ax.set_ylabel(r"normalized reward gap $\Delta R/\sigma$")
    ax.set_title("Exp2: n=2 latent-target opposition (feasible corr at -1 = -0.468)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "exp2_n2_gap_vs_rho.png", dpi=150, bbox_inches="tight")
    p(f"\nSaved exp2_n2_gap_vs_rho.png + exp2_n2_headline.csv")


def main():
    experiment1()
    experiment2()
    (HERE / "ANALYSIS.txt").write_text("\n".join(OUT_LINES))
    p(f"\nSaved ANALYSIS.txt")


if __name__ == "__main__":
    main()
