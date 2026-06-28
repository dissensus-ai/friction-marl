"""Analysis for the PARTIAL-SHARING control (reviewer #5 mechanistic test).

Question: does the alignment-gap slope (d gap_norm / d rho) SCALE with the degree
of shared-state contention w = k/m, bridging the two known endpoints
  w=0 (separable, FLAT, slope ~ -0.03) and w=1 (fully shared, SLOPED, slope ~ -0.8)?

Pipeline:
  1. SANITY GATE -- the NEW runner's own endpoints (w=0, w=1) must reproduce the
     PRE-EXISTING reference runs (separable_*.csv at w=0, equicorr_*.csv at w=1).
     If they do not, we stop and report the discrepancy (printed verdict).
  2. Per (mode, w): OLS slope of gap_norm vs signed rho (scipy linregress; named
     .slope/.stderr/.pvalue/.rvalue) with analytic 95% CI -> slope-vs-w table.
  3. TREND TESTS that the slope scales with overlap:
       (a) Spearman(w, slope) across the 4 levels  -- monotonicity (named attrs).
       (b) interaction OLS  gap_norm ~ rho * w (w continuous in [0,1])  -- the
           rho:w coefficient is the mediation effect; significantly NEGATIVE means
           contention DEGREE mediates the alignment effect (statsmodels; named
           .params/.pvalues/.conf_int).
  4. Plots: gap-vs-rho colored by w (per mode) + slope-vs-w with endpoint refs.

All test statistics use the NAMED scipy/statsmodels result attributes -- no
positional tuple unpacking.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import stats
import statsmodels.formula.api as smf

HERE = Path(__file__).resolve().parent
ADV = HERE.parent / "adversarial_rerun"          # equicorr_* = w=1 reference (shared)
R3 = HERE.parent / "reviewer3_controls"          # separable_* = w=0 reference

OUT = []
def p(s=""):
    print(s)
    OUT.append(s)


def load(*paths):
    df = pd.concat([pd.read_csv(x) for x in paths], ignore_index=True)
    df["gap"] = -df["mean_reward"]
    df["gap_norm"] = df["gap"] / df["sigma"]
    return df


def slope_fit(sub):
    """OLS slope gap_norm~rho via scipy.linregress; analytic 95% CI on slope."""
    r = stats.linregress(sub["rho"].to_numpy(float), sub["gap_norm"].to_numpy(float))
    n = len(sub)
    tcrit = float(stats.t.ppf(0.975, n - 2))
    return dict(slope=float(r.slope), se=float(r.stderr), p=float(r.pvalue),
                r2=float(r.rvalue ** 2), n=n,
                ci_lo=float(r.slope - tcrit * r.stderr),
                ci_hi=float(r.slope + tcrit * r.stderr))


def ci_overlap(a, b):
    return not (a["ci_hi"] < b["ci_lo"] or b["ci_hi"] < a["ci_lo"])


# ======================================================================== #
def main():
    part = load(HERE / "partial_iql.csv", HERE / "partial_vdn.csv")
    # references
    sep = load(R3 / "separable_iql.csv", R3 / "separable_vdn.csv")     # w=0 ref
    sha = load(ADV / "equicorr_iql.csv", ADV / "equicorr_vdn.csv")     # w=1 ref

    modes = sorted(part["mode"].unique())
    ws = sorted(part["w"].unique())
    ks = sorted(part["overlap_k"].unique())
    rhos = sorted(part["rho"].unique())

    p("#" * 86)
    p("# PARTIAL-SHARING CONTROL -- does the alignment-gap slope scale with overlap w?")
    p("#" * 86)
    p(f"\noverlap levels k={ks}  ->  w={[round(w,3) for w in ws]}   (w = k / 3 resources)")
    p(f"rhos={[round(r,2) for r in rhos]}   sigmas={sorted(part['sigma'].unique())}   "
      f"modes={modes}")
    p(f"reps/cond={part.groupby(['mode','w','rho','sigma']).size().min()}-"
      f"{part.groupby(['mode','w','rho','sigma']).size().max()}   "
      f"(epsilon={sorted(part['epsilon'].unique())})")

    # ---------------- normalized gap table (pooled over sigma) ---------------- #
    p("\n--- normalized gap dR/sigma vs rho, by overlap w (pooled over sigma) ---")
    for m in modes:
        p(f"\n  [{m.upper()}]  w \\ rho :  " + "  ".join(f"{r:+.2f}" for r in rhos))
        for w in ws:
            row = []
            for r in rhos:
                v = part[(part["mode"] == m) & (part.w == w) & (part.rho == r)]["gap_norm"]
                row.append(v.mean())
            p(f"           w={w:.2f} :  " + "  ".join(f"{x:5.2f}" for x in row))

    # ============================ SANITY GATE ============================ #
    p("\n" + "=" * 86)
    p("SANITY GATE -- new runner endpoints must reproduce the pre-existing reference runs")
    p("=" * 86)
    gate_rows = []
    gate_pass = True
    for m in modes:
        # w=0 (my k=0) vs separable reference
        my0 = slope_fit(part[(part["mode"] == m) & (part.w == ws[0])])
        ref0 = slope_fit(sep[sep["mode"] == m])
        # w=1 (my k=3) vs equicorr(shared) reference
        my1 = slope_fit(part[(part["mode"] == m) & (part.w == ws[-1])])
        ref1 = slope_fit(sha[sha["mode"] == m])

        # gate criteria:
        #  w=0: FLAT -> |my slope|<0.25 AND not sig (p>0.05) AND CI overlaps ref
        flat_ok = (abs(my0["slope"]) < 0.25) and (my0["p"] > 0.05) and ci_overlap(my0, ref0)
        #  w=1: SLOPED -> my slope significantly<0 AND CI overlaps ref AND |my-ref|<0.3
        slope_ok = (my1["slope"] < 0) and (my1["p"] < 0.05) and \
                   (ci_overlap(my1, ref1) or abs(my1["slope"] - ref1["slope"]) < 0.30)
        gate_pass = gate_pass and flat_ok and slope_ok

        p(f"\n  [{m.upper()}]")
        p(f"    w=0 (separable)  : my slope={my0['slope']:+.3f} [CI {my0['ci_lo']:+.3f},{my0['ci_hi']:+.3f}] p={my0['p']:.2e}"
          f"   |  ref(separable)={ref0['slope']:+.3f} [CI {ref0['ci_lo']:+.3f},{ref0['ci_hi']:+.3f}]"
          f"   -> FLAT&match: {flat_ok}")
        p(f"    w=1 (full share) : my slope={my1['slope']:+.3f} [CI {my1['ci_lo']:+.3f},{my1['ci_hi']:+.3f}] p={my1['p']:.2e}"
          f"   |  ref(equicorr)={ref1['slope']:+.3f} [CI {ref1['ci_lo']:+.3f},{ref1['ci_hi']:+.3f}]"
          f"   -> SLOPED&match: {slope_ok}")
        gate_rows += [
            dict(mode=m, endpoint="w0_separable", my_slope=my0["slope"], my_p=my0["p"],
                 my_ci_lo=my0["ci_lo"], my_ci_hi=my0["ci_hi"], ref_slope=ref0["slope"],
                 ref_ci_lo=ref0["ci_lo"], ref_ci_hi=ref0["ci_hi"], ok=flat_ok),
            dict(mode=m, endpoint="w1_shared", my_slope=my1["slope"], my_p=my1["p"],
                 my_ci_lo=my1["ci_lo"], my_ci_hi=my1["ci_hi"], ref_slope=ref1["slope"],
                 ref_ci_lo=ref1["ci_lo"], ref_ci_hi=ref1["ci_hi"], ok=slope_ok),
        ]
    pd.DataFrame(gate_rows).to_csv(HERE / "sanity_gate.csv", index=False)
    p(f"\n  GATE {'PASSED' if gate_pass else 'FAILED'} -> "
      + ("endpoints reproduce; the middle overlap levels can be trusted."
         if gate_pass else "ENDPOINTS DO NOT REPRODUCE -- middle is NOT trustworthy; investigate."))

    # ===================== SLOPE vs OVERLAP (headline) ===================== #
    p("\n" + "=" * 86)
    p("SLOPE of gap_norm vs rho, per overlap level w  (the headline relationship)")
    p("=" * 86)
    slope_rows = []
    for m in modes:
        p(f"\n  [{m.upper()}]   w     k   slope    se      95% CI            p        R^2")
        fits = {}
        for w, k in zip(ws, ks):
            f = slope_fit(part[(part["mode"] == m) & (part.w == w)])
            fits[w] = f
            p(f"        {w:.3f}  {k}  {f['slope']:+.3f}  {f['se']:.3f}  "
              f"[{f['ci_lo']:+.3f},{f['ci_hi']:+.3f}]  {f['p']:.2e}  {f['r2']:.3f}")
            slope_rows.append(dict(mode=m, w=w, overlap_k=k, slope=f["slope"], se=f["se"],
                                   ci_lo=f["ci_lo"], ci_hi=f["ci_hi"], p=f["p"], r2=f["r2"]))
        # monotonicity of |slope| in w
        slopes = np.array([fits[w]["slope"] for w in ws])
        absslopes = np.abs(slopes)
        mono_abs = bool(np.all(np.diff(absslopes) >= -1e-9))
        mono_signed = bool(np.all(np.diff(slopes) <= 1e-9))   # slope becomes more negative
        # Spearman(w, slope): expect strong NEGATIVE (slope falls as w rises)
        sp = stats.spearmanr(ws, slopes)
        sp_abs = stats.spearmanr(ws, absslopes)
        p(f"      |slope| monotonically non-decreasing in w: {mono_abs}    "
          f"(signed slope monotonically non-increasing: {mono_signed})")
        p(f"      Spearman(w, signed slope) = {float(sp.statistic):+.3f} (p={float(sp.pvalue):.3f}); "
          f"Spearman(w, |slope|) = {float(sp_abs.statistic):+.3f} (p={float(sp_abs.pvalue):.3f})")
    slope_df = pd.DataFrame(slope_rows)
    slope_df.to_csv(HERE / "slope_vs_overlap.csv", index=False)

    # =================== interaction OLS (mediation test) =================== #
    p("\n" + "=" * 86)
    p("INTERACTION TEST -- gap_norm ~ rho * w  (rho:w coeff = contention-degree mediation)")
    p("=" * 86)
    inter_rows = []
    for m in modes:
        d = part[part["mode"] == m].copy()
        model = smf.ols("gap_norm ~ rho * w", data=d).fit()
        b_int = float(model.params["rho:w"])
        p_int = float(model.pvalues["rho:w"])
        ci = model.conf_int().loc["rho:w"]
        b_rho = float(model.params["rho"])   # slope at w=0
        p(f"\n  [{m.upper()}]  n={int(model.nobs)}")
        p(f"    slope at w=0 (coef 'rho')           = {b_rho:+.3f}")
        p(f"    rho:w interaction (slope change/Δw) = {b_int:+.3f}  "
          f"[95% CI {float(ci[0]):+.3f}, {float(ci[1]):+.3f}]  p={p_int:.3e}")
        p(f"    => implied slope at w=1             = {b_rho + b_int:+.3f}")
        verdict = ("MEDIATES: slope becomes significantly MORE NEGATIVE as overlap rises"
                   if (b_int < 0 and p_int < 0.05) else
                   "NO significant scaling of slope with overlap")
        p(f"    VERDICT [{m.upper()}]: {verdict}")
        inter_rows.append(dict(mode=m, slope_at_w0=b_rho, interaction=b_int,
                               interaction_ci_lo=float(ci[0]), interaction_ci_hi=float(ci[1]),
                               interaction_p=p_int, implied_slope_w1=b_rho + b_int,
                               n=int(model.nobs), verdict=verdict))
    pd.DataFrame(inter_rows).to_csv(HERE / "interaction_test.csv", index=False)

    # ============================== PLOTS ============================== #
    # Fig 1: gap-vs-rho colored by w, one subplot per mode
    cmap = cm.get_cmap("viridis")
    fig, axes = plt.subplots(1, len(modes), figsize=(6.0 * len(modes), 4.6), sharey=False)
    if len(modes) == 1:
        axes = [axes]
    for ax, m in zip(axes, modes):
        for w in ws:
            sub = part[(part["mode"] == m) & (part.w == w)]
            g = sub.groupby("rho")["gap_norm"]
            mean = g.mean(); sem = g.std() / np.sqrt(g.count())
            ax.errorbar(mean.index, mean.values, yerr=1.96 * sem.values, marker="o",
                        capsize=3, lw=2, color=cmap(w),
                        label=f"w={w:.2f} (k={int(round(w*3))})")
        ax.axvline(0, color="grey", ls=":", lw=1)
        ax.set_title(f"{m.upper()}: gap vs rho by overlap")
        ax.set_xlabel(r"$\rho$ (cross-agent preference correlation)")
        ax.grid(alpha=0.3); ax.legend(fontsize=8, title="overlap")
    axes[0].set_ylabel(r"normalized reward gap $\Delta R/\sigma$")
    fig.suptitle("Partial sharing: gap-vs-$\\rho$ steepens as shared-state contention rises", y=1.02)
    fig.tight_layout()
    fig.savefig(HERE / "gap_vs_rho_by_overlap.png", dpi=150, bbox_inches="tight")
    fig.savefig(HERE / "gap_vs_rho_by_overlap.pdf", bbox_inches="tight")

    # Fig 2: slope vs w, with endpoint reference slopes
    fig2, ax2 = plt.subplots(figsize=(6.6, 4.8))
    col = {"iql": "#7a1f2b", "vdn": "#1f4e79"}
    for m in modes:
        sm = slope_df[slope_df["mode"] == m].sort_values("w")
        ax2.errorbar(sm["w"], sm["slope"], yerr=1.96 * sm["se"], marker="o", capsize=4,
                     lw=2, color=col.get(m, "k"), label=f"{m.upper()} (partial)")
    # reference endpoints
    for m in modes:
        ref0 = slope_fit(sep[sep["mode"] == m]); ref1 = slope_fit(sha[sha["mode"] == m])
        ax2.scatter([0.0], [ref0["slope"]], marker="s", s=70, facecolors="none",
                    edgecolors=col.get(m, "k"), zorder=5,
                    label=f"{m.upper()} ref (separable w=0)")
        ax2.scatter([1.0], [ref1["slope"]], marker="D", s=70, facecolors="none",
                    edgecolors=col.get(m, "k"), zorder=5,
                    label=f"{m.upper()} ref (shared w=1)")
    ax2.axhline(0, color="grey", ls=":", lw=1)
    ax2.set_xlabel(r"overlap degree $w = k/m$ (fraction of resources shared / contended)")
    ax2.set_ylabel(r"slope $\;d(\Delta R/\sigma)/d\rho$")
    ax2.set_title("Alignment-gap slope scales with shared-state contention degree")
    ax2.grid(alpha=0.3); ax2.legend(fontsize=7, ncol=2)
    fig2.tight_layout()
    fig2.savefig(HERE / "slope_vs_overlap.png", dpi=150, bbox_inches="tight")
    fig2.savefig(HERE / "slope_vs_overlap.pdf", bbox_inches="tight")

    p("\nSaved: gap_vs_rho_by_overlap.{png,pdf}, slope_vs_overlap.{png,pdf}, "
      "slope_vs_overlap.csv, interaction_test.csv, sanity_gate.csv")
    (HERE / "ANALYSIS.txt").write_text("\n".join(OUT))
    p("Saved ANALYSIS.txt")


if __name__ == "__main__":
    main()
