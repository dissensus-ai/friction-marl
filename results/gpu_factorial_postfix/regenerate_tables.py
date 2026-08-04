"""Single source of truth for every factorial-derived number in the manuscript.

Written 5 Aug 2026, after a read-through found that repointing tab:main-effects
to the post-fix factorial had reached the TABLE and nothing else: the prose
F-statistics, tab:interactions and tab:model-comparison were all still pre-fix,
so the paper reported the same ANOVA twice with different numbers.

Hand-patching is what caused that. This script regenerates all three tables plus
the prose values from one place, so they cannot drift apart again.

VALIDATION. The pre-fix run survives only as condition means
(results/gpu_factorial/analysis/condition_statistics.csv); its per-replication
CSV was never committed. Model comparison is condition-level, so it can be fit on
BOTH runs -- and running this with --validate refits the pre-fix means and checks
the result against the numbers currently printed in the manuscript. If those
reproduce, the model specification here is the one that produced the original
table, and the post-fix numbers below are a like-for-like replacement rather than
a different analysis. (The ANOVA is per-replication and therefore post-fix only.)

Run:  python3 results/gpu_factorial_postfix/regenerate_tables.py [--validate]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

REPO = Path(__file__).resolve().parents[2]
POST = REPO / "results/gpu_factorial_postfix/replication_results.csv"
PRE_MEANS = REPO / "results/gpu_factorial/analysis/condition_statistics.csv"
FACTORS = ["alpha", "sigma", "epsilon"]
AGENTS = [f"agent_{i}_reward" for i in range(4)]

# What the manuscript currently prints for tab:model-comparison (pre-fix fit).
MANUSCRIPT_PREFIX_MODELS = {
    "M1": (0.1084, 322.3, 327.9, 0.8643),
    "M2": (0.1586, 315.0, 320.7, 0.8396),
    "M3": (0.1860, 310.9, 316.6, 0.8258),
    "M4": (0.7533, 165.7, 177.0, 0.4547),
}


def gini(row):
    """Gini over agent rewards. Rewards are negative, so magnitudes are used --
    this matches the definition in generate_appendix_latex.py that produced the
    manuscript's original Gini rows."""
    x = np.abs(np.asarray(row, dtype=float))
    if x.sum() == 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def anova(df, dv):
    m = smf.ols(f"{dv} ~ C(alpha)*C(sigma)*C(epsilon)", data=df).fit()
    tab = sm.stats.anova_lm(m, typ=2)
    tab["eta_sq"] = tab["sum_sq"] / tab["sum_sq"].sum()
    return tab


def condition_eta_sq(cond, dv):
    """Condition-level eta^2. The factorial model is SATURATED at 125 means
    (zero residual df), so anova_lm cannot form F tests; the design is balanced,
    so the SS partition is taken from the marginal means instead."""
    g = cond[dv].mean()
    m = {f: cond.groupby(f)[dv].mean() for f in FACTORS}
    pairs = [("alpha", "sigma"), ("alpha", "epsilon"), ("sigma", "epsilon")]
    m2 = {p: cond.groupby(list(p))[dv].mean() for p in pairs}
    nl = {f: cond[f].nunique() for f in FACTORS}
    out, total = {}, ((cond[dv] - g) ** 2).sum()
    for f in FACTORS:
        out[f"C({f})"] = (len(cond) / nl[f]) * ((m[f] - g) ** 2).sum()
    for (a, b) in pairs:
        idx = m2[(a, b)].index
        resid = m2[(a, b)] - idx.get_level_values(a).map(m[a]) \
                           - idx.get_level_values(b).map(m[b]) + g
        out[f"C({a}):C({b})"] = (len(cond) / (nl[a] * nl[b])) * (resid ** 2).sum()
    out["C(alpha):C(sigma):C(epsilon)"] = total - sum(out.values())
    return {k: v / total for k, v in out.items()}


def model_comparison(cond, dv="gap"):
    """M1-M3 single composite predictors (2 params); M4 three independent (4)."""
    d = cond.copy()
    d["m1"] = d.sigma * (1 + d.epsilon) / (1 + d.alpha)
    d["m2"] = d.sigma + d.epsilon - d.alpha
    d["m3"] = d.sigma * d.epsilon * (1 - d.alpha)
    specs = {
        "M1": f"{dv} ~ m1",
        "M2": f"{dv} ~ m2",
        "M3": f"{dv} ~ m3",
        "M4": f"{dv} ~ alpha + sigma + epsilon",
    }
    rows = {}
    for name, f in specs.items():
        r = smf.ols(f, data=d).fit()
        rmse = float(np.sqrt(np.mean(r.resid ** 2)))
        rows[name] = (float(r.rsquared), float(r.aic), float(r.bic), rmse)
    return rows


def show_models(title, rows):
    print(f"\n{title}")
    print(f"{'model':<6}{'R^2':>10}{'AIC':>10}{'BIC':>10}{'RMSE':>10}")
    for k, (r2, aic, bic, rmse) in rows.items():
        print(f"{k:<6}{r2:>10.4f}{aic:>10.1f}{bic:>10.1f}{rmse:>10.4f}")


def validate():
    pre = pd.read_csv(PRE_MEANS)
    pre["gap"] = -pre["reward_mean"]
    rows = model_comparison(pre)
    show_models("PRE-FIX condition means, refit here:", rows)
    print("\nmanuscript currently prints (pre-fix):")
    ok = True
    for k, (r2, aic, bic, rmse) in MANUSCRIPT_PREFIX_MODELS.items():
        g = rows[k]
        hit = (abs(g[0] - r2) < 5e-4 and abs(g[1] - aic) < 0.2
               and abs(g[2] - bic) < 0.2 and abs(g[3] - rmse) < 5e-4)
        ok &= hit
        print(f"  {k}: {r2} {aic} {bic} {rmse}   -> {'MATCH' if hit else 'MISMATCH'}")
    print("\nSPEC VALIDATED - post-fix numbers are like-for-like." if ok else
          "\nSPEC DOES NOT REPRODUCE the manuscript table; do not swap the numbers.")
    return ok


def main():
    if "--validate" in sys.argv:
        validate()
        print("\n" + "=" * 64)

    df = pd.read_csv(POST)
    df["gap"] = -df["mean_reward"]
    df["gini"] = df[AGENTS].apply(lambda r: gini(r.values), axis=1)
    cond = df.groupby(FACTORS, as_index=False).agg(gap=("gap", "mean"))

    a_gap, a_gini = anova(df, "gap"), anova(df, "gini")
    c_gap = condition_eta_sq(df.groupby(FACTORS, as_index=False)["gap"].mean(), "gap")

    label = {"C(alpha)": "alpha", "C(sigma)": "sigma", "C(epsilon)": "epsilon",
             "C(alpha):C(sigma)": "alpha x sigma",
             "C(alpha):C(epsilon)": "alpha x epsilon",
             "C(sigma):C(epsilon)": "sigma x epsilon",
             "C(alpha):C(sigma):C(epsilon)": "alpha x sigma x epsilon"}

    print("POST-FIX  n(per-rep)=%d  n(cond)=%d" % (len(df), len(cond)))
    print("\ntab:main-effects + prose values  (DV: gap == -mean reward)")
    print(f"{'term':<26}{'df':>4}{'F':>10}{'eta2':>9}{'p':>12}{'eta2_cond':>11}")
    for k, name in label.items():
        r = a_gap.loc[k]
        print(f"{name:<26}{int(r['df']):>4}{r['F']:>10.2f}{r['eta_sq']:>9.4f}"
              f"{r['PR(>F)']:>12.3g}{c_gap[k]:>11.4f}")

    mains = sum(c_gap[f"C({f})"] for f in FACTORS)
    print(f"\ncaption sums: main effects {mains:.4f}, interactions {1-mains:.4f}")

    print("\ntab:interactions, DV: Gini coefficient")
    print(f"{'term':<26}{'df':>4}{'F':>10}{'eta2':>9}{'p':>12}")
    for k in ["C(alpha):C(sigma)", "C(alpha):C(epsilon)",
              "C(sigma):C(epsilon)", "C(alpha):C(sigma):C(epsilon)"]:
        r = a_gini.loc[k]
        print(f"{label[k]:<26}{int(r['df']):>4}{r['F']:>10.2f}{r['eta_sq']:>9.4f}{r['PR(>F)']:>12.3g}")

    show_models("tab:model-comparison, POST-FIX condition means:",
                model_comparison(cond))


if __name__ == "__main__":
    main()
