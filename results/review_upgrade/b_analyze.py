"""Deliverable B analysis: does the U-shape over alpha survive centralized training?

Compares IQL (independent learners) vs VDN (centralized value-decomposition) on the
matched targeted grid. Verdict logic:
  - U preserved : alpha=0 remains the worst (highest gap) under VDN, symmetric in sign.
  - U flattened : VDN closes the alpha=0 penalty relative to the |alpha|=0.8 extremes.
  - U inverted  : VDN makes an extreme (e.g. adversarial-coded alpha<0) the worst.

Also tests the key interaction: does centralized training help MORE where preferences
are dispersed (alpha=0) than where they are structured (|alpha|=0.8)?
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
CSV = HERE / "vdn_vs_iql.csv"


def main() -> None:
    df = pd.read_csv(CSV)
    df["gap"] = -df["mean_reward"]
    df["gap_norm"] = df["gap"] / df["sigma"]

    lines = []
    def p(s=""):
        print(s); lines.append(s)

    modes = sorted(df["mode"].unique())
    alphas = sorted(df["alpha"].unique())
    sigmas = sorted(df["sigma"].unique())
    p("=" * 80)
    p("DELIVERABLE B: VDN (centralized) vs IQL (independent) -- U-shape over alpha")
    p(f"modes={modes} alphas={alphas} sigmas={sigmas} "
      f"reps={df.groupby(['mode','alpha','sigma']).size().min()}-"
      f"{df.groupby(['mode','alpha','sigma']).size().max()}")
    p("=" * 80)

    # --- per-condition gap means (normalized) ---
    for dv, lab in [("gap", "RAW gap"), ("gap_norm", "NORMALIZED gap (/sigma)")]:
        p(f"\n### {lab}: mean by alpha (rows), per sigma x mode ###")
        for s in sigmas:
            p(f"\n  sigma={s}")
            head = "    alpha:  " + "  ".join(f"{a:+.1f}" for a in alphas)
            p(head)
            for m in modes:
                vals = []
                for a in alphas:
                    v = df[(df["mode"] == m) & (df["sigma"] == s) & (df["alpha"] == a)][dv].mean()
                    vals.append(v)
                p(f"    {m:<5}: " + "  ".join(f"{v:5.3f}" for v in vals))

    # --- U-shape verdict per mode/sigma ---
    p("\n### U-shape diagnostics (normalized gap) ###")
    p("  U-depth = gap(a=0) - mean(gap(a=-0.8), gap(a=+0.8)); >0 => U (mid worst)")
    rows = []
    for m in modes:
        for s in sigmas:
            sub = df[(df["mode"] == m) & (df["sigma"] == s)]
            g = {a: sub[sub["alpha"] == a]["gap_norm"].mean() for a in alphas}
            mid = g[0.0]
            ext = 0.5 * (g[-0.8] + g[0.8])
            depth = mid - ext
            argmax_a = max(g, key=g.get)
            sign_asym = g[0.8] - g[-0.8]
            rows.append({"mode": m, "sigma": s, "gap_a0": mid, "gap_ext": ext,
                         "U_depth": depth, "worst_alpha": argmax_a,
                         "sign_asym_+vs-": sign_asym})
            p(f"  {m:<5} sigma={s}: U-depth={depth:+.3f}  worst_alpha={argmax_a:+.1f}  "
              f"(+0.8 vs -0.8 diff={sign_asym:+.3f})")
    pd.DataFrame(rows).to_csv(HERE / "b_ushape_diagnostics.csv", index=False)

    # --- key interaction: does centralization help more at alpha=0 than at |alpha|=0.8? ---
    p("\n### Centralization effect: IQL gap - VDN gap (positive => VDN coordinates better) ###")
    p("    (normalized gap, pooled over sigma)")
    inter_rows = []
    for a in alphas:
        iql = df[(df["mode"] == "iql") & (df["alpha"] == a)]["gap_norm"]
        vdn = df[(df["mode"] == "vdn") & (df["alpha"] == a)]["gap_norm"]
        diff = iql.mean() - vdn.mean()
        t, pv = stats.ttest_ind(iql, vdn, equal_var=False)
        inter_rows.append({"alpha": a, "iql_gap": iql.mean(), "vdn_gap": vdn.mean(),
                           "iql_minus_vdn": diff, "welch_t": t, "p": pv})
        p(f"  alpha={a:+.1f}: IQL={iql.mean():.3f}  VDN={vdn.mean():.3f}  "
          f"IQL-VDN={diff:+.3f}  (Welch t={t:+.2f}, p={pv:.3g})")
    pd.DataFrame(inter_rows).to_csv(HERE / "b_centralization_effect.csv", index=False)

    # Interaction contrast: (IQL-VDN at a=0) vs mean(IQL-VDN at |a|=0.8)
    d0 = (df[(df.mode=="iql")&(df.alpha==0.0)]["gap_norm"].mean()
          - df[(df.mode=="vdn")&(df.alpha==0.0)]["gap_norm"].mean())
    d8 = 0.5*((df[(df.mode=="iql")&(df.alpha==-0.8)]["gap_norm"].mean()
               - df[(df.mode=="vdn")&(df.alpha==-0.8)]["gap_norm"].mean())
              +(df[(df.mode=="iql")&(df.alpha==0.8)]["gap_norm"].mean()
               - df[(df.mode=="vdn")&(df.alpha==0.8)]["gap_norm"].mean()))
    p(f"\n  Centralization benefit at alpha=0 : {d0:+.3f}")
    p(f"  Centralization benefit at |alpha|=0.8: {d8:+.3f}")
    p(f"  => differential (mid - extremes) = {d0-d8:+.3f}  "
      f"({'VDN helps the dispersed midpoint MORE (flattens U)' if d0-d8>0.05 else 'roughly uniform (U preserved)' if abs(d0-d8)<=0.05 else 'VDN helps extremes more'})")

    # --- VERDICT ---
    p("\n### VERDICT ###")
    vdn_depths = [r["U_depth"] for r in rows if r["mode"] == "vdn"]
    iql_depths = [r["U_depth"] for r in rows if r["mode"] == "iql"]
    p(f"  mean U-depth  IQL={np.mean(iql_depths):+.3f}  VDN={np.mean(vdn_depths):+.3f}")
    if np.mean(vdn_depths) > 0.05:
        verd = "U-shape PRESERVED under centralized training (alpha=0 still worst)."
    elif np.mean(vdn_depths) < -0.05:
        verd = "U-shape INVERTED under centralized training."
    else:
        verd = "U-shape FLATTENED under centralized training."
    p(f"  {verd}")
    if np.mean(iql_depths) > 0:
        p(f"  U-depth retained fraction = {np.mean(vdn_depths)/np.mean(iql_depths)*100:.0f}% of IQL's.")

    # --- plot: side-by-side gap_norm vs alpha, faceted by sigma ---
    fig, axes = plt.subplots(1, len(sigmas), figsize=(4.2*len(sigmas), 4.0), sharey=True)
    if len(sigmas) == 1:
        axes = [axes]
    colors = {"iql": "#7a1f2b", "vdn": "#1f4e79"}
    for ax, s in zip(axes, sigmas):
        for m in modes:
            sub = df[(df["mode"] == m) & (df["sigma"] == s)]
            g = sub.groupby("alpha")["gap_norm"]
            mean = g.mean()
            sem = g.std() / np.sqrt(g.count())
            ax.errorbar(mean.index, mean.values, yerr=1.96*sem.values, marker="o",
                        capsize=3, label=m.upper(), color=colors.get(m, None), lw=2)
        ax.set_title(f"sigma = {s}")
        ax.set_xlabel(r"$\alpha$ (preference alignment)")
        ax.axvline(0, color="grey", ls=":", lw=0.8)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"normalized reward gap $\Delta R / \sigma$")
    axes[0].legend(title="learner")
    fig.suptitle("U-shape over preference alignment: independent (IQL) vs centralized (VDN)",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(HERE / "b_ushape_iql_vs_vdn.png", dpi=150, bbox_inches="tight")
    fig.savefig(HERE / "b_ushape_iql_vs_vdn.pdf", bbox_inches="tight")
    p(f"\n  Saved plot: b_ushape_iql_vs_vdn.png/.pdf")

    (HERE / "B_SUMMARY.txt").write_text("\n".join(lines))
    p("  Saved B_SUMMARY.txt + diagnostics CSVs")


if __name__ == "__main__":
    main()
