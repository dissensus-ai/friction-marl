"""STEP 4 (robustness): STRONG-opposition (two-team) analysis.

In the team DGP the control knob is `opp` in [0,1] (stored in the `rho` column):
  between-team corr = -opp^2  (reaches -1.0 at opp=1 -> genuine strong opposition),
  within-team  corr = +opp^2,
  mean pairwise corr = -opp^2/3.
opp=0 == pure indifference (all agents independent). So the test is direct:
does adding STRONG structured opposition lower the gap below the opp=0 baseline
(U-shape / opposition helps) or raise it (monotone / opposition hurts)?
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

HERE = Path(__file__).resolve().parent


def main():
    df = pd.concat([pd.read_csv(HERE / "team_iql.csv"),
                    pd.read_csv(HERE / "team_vdn.csv")], ignore_index=True)
    df = df.rename(columns={"rho": "opp"})
    df["gap"] = -df["mean_reward"]
    df["gap_norm"] = df["gap"] / df["sigma"]
    df["between_corr"] = -df["opp"] ** 2          # strongest pairwise opposition

    lines = []
    def p(s=""):
        print(s); lines.append(s)

    modes = sorted(df["mode"].unique())
    opps = sorted(df["opp"].unique())
    p("=" * 80)
    p("TEAM (STRONG-OPPOSITION) ROBUSTNESS  -- between-team corr = -opp^2 (down to -1)")
    p("=" * 80)
    p(f"\n  opp:           " + "  ".join(f"{o:.2f}" for o in opps))
    p(f"  between_corr:  " + "  ".join(f"{-o*o:+.2f}" for o in opps))
    for m in modes:
        cells = [f"{df[(df['mode']==m)&(df['opp']==o)]['gap'].mean():.2f}" for o in opps]
        p(f"  gap [{m}]:     " + "  ".join(cells))

    p("\n  HEADLINE: does STRONG opposition beat indifference (opp=0)?")
    rows = []
    for m in modes:
        sub = df[df["mode"] == m]
        g0 = sub[sub["opp"] == 0.0]["gap"]                 # indifference
        gmax = sub[sub["opp"] == max(opps)]["gap"]         # strongest opposition
        t, pv = stats.ttest_ind(g0, gmax, equal_var=False)
        u, pu = stats.mannwhitneyu(g0, gmax, alternative="two-sided")
        # monotone trend: regress gap on opp
        slope, intercept, r, ptrend, se = stats.linregress(sub["opp"], sub["gap"])
        worst_opp = sub.groupby("opp")["gap"].mean().idxmax()
        if gmax.mean() > g0.mean() and pv < 0.05:
            verdict = ("opposition HURTS: strong opposition gap > indifference gap "
                       "(MONOTONE worse). Opposition does NOT beat indifference.")
        elif gmax.mean() < g0.mean() and pv < 0.05:
            verdict = "opposition HELPS: strong opposition beats indifference (real U)."
        else:
            verdict = "no significant difference between strong opposition and indifference."
        p(f"\n  [{m.upper()}] indifference gap(opp=0)={g0.mean():.3f}  "
          f"strong-opp gap(opp=1)={gmax.mean():.3f}  diff={gmax.mean()-g0.mean():+.3f}")
        p(f"        Welch t={t:+.2f} p={pv:.2e}  MWU p={pu:.2e}  worst_opp={worst_opp:.2f}")
        p(f"        gap~opp linear slope={slope:+.3f} (r={r:+.2f}, p_trend={ptrend:.2e})")
        p(f"        => {verdict}")
        rows.append({"mode": m, "gap_indiff": g0.mean(), "gap_strongopp": gmax.mean(),
                     "diff": gmax.mean()-g0.mean(), "p": pv, "slope": slope,
                     "p_trend": ptrend, "worst_opp": worst_opp, "verdict": verdict})
    pd.DataFrame(rows).to_csv(HERE / "headline_team.csv", index=False)

    # plot gap vs between_corr
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"iql": "#7a1f2b", "vdn": "#1f4e79"}
    for m in modes:
        sub = df[df["mode"] == m]
        g = sub.groupby("opp")
        x = [-o * o for o in opps]                  # between-team corr
        mean = [g.get_group(o)["gap"].mean() for o in opps]
        sem = [g.get_group(o)["gap"].std() / np.sqrt(g.get_group(o)["gap"].count()) for o in opps]
        ax.errorbar(x, mean, yerr=[1.96 * s for s in sem], marker="o", capsize=3,
                    lw=2, label=m.upper(), color=colors.get(m))
    ax.axvline(0, color="grey", ls=":", lw=1, label="opp=0 (indifference)")
    ax.set_xlabel(r"between-team correlation $-\mathrm{opp}^2$  (0 = indifference, $-1$ = max opposition)")
    ax.set_ylabel(r"raw reward gap $\Delta R$  (sigma=1.0)")
    ax.set_title("Strong structured opposition vs indifference (two-team DGP)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "gap_vs_opposition_team.png", dpi=150, bbox_inches="tight")
    p("\n  Saved gap_vs_opposition_team.png + headline_team.csv")
    (HERE / "ANALYSIS_team.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
