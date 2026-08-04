"""Regenerate the manuscript's headline main-effects ANOVA (tab:main-effects).

Written 4 Aug 2026. The table in the manuscript was computed on the PRE-fix
factorial, i.e. before the two 29-30 Jul corrections (observation-noise units
eps vs sqrt(eps), and the replay leak that stored a clean next observation).
Both are fixed at HEAD and the factorial was re-run into this directory; this
script recomputes the table from that re-run so the manuscript can cite it.

DV = the coordination gap. Because R* = 0 (the per-agent unconstrained ideal),
the gap is -1 x mean training reward, which is what the manuscript reports.

Two aggregations, both in the table:
  per-replication  n = 3750, retains within-condition replication variance
  condition-level  n = 125 condition means, the headline aggregation

Run:  python3 results/gpu_factorial_postfix/main_effects_table.py
"""
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

HERE = Path(__file__).resolve().parent
FACTORS = ["alpha", "sigma", "epsilon"]
LABEL = {"alpha": "Alignment", "sigma": "Stakes", "epsilon": "Entropy"}


def anova(df, dv="gap"):
    """Type-II ANOVA on the full 3-way factorial, returning eta^2 per term."""
    formula = f"{dv} ~ C(alpha)*C(sigma)*C(epsilon)"
    model = smf.ols(formula, data=df).fit()
    tab = sm.stats.anova_lm(model, typ=2)
    ss_total = tab["sum_sq"].sum()
    tab["eta_sq"] = tab["sum_sq"] / ss_total
    return tab


def condition_eta_sq(cond, dv="gap"):
    """eta^2 at the condition level, where the factorial model is SATURATED.

    125 condition means against a full 5x5x5 model leaves zero residual df, so
    anova_lm cannot form F tests. The design is balanced, so the sum-of-squares
    partition is orthogonal and each term's SS can be taken directly from the
    marginal means -- which is all eta^2 = SS/SS_total needs.
    """
    g = cond[dv].mean()
    m = {f: cond.groupby(f)[dv].mean() for f in FACTORS}
    m2 = {p: cond.groupby(list(p))[dv].mean()
          for p in [("alpha", "sigma"), ("alpha", "epsilon"), ("sigma", "epsilon")]}
    n_lvl = {f: cond[f].nunique() for f in FACTORS}

    out, total = {}, ((cond[dv] - g) ** 2).sum()
    for f in FACTORS:                                    # main effects
        reps = len(cond) / n_lvl[f]
        out[f"C({f})"] = reps * ((m[f] - g) ** 2).sum()
    for (a, b) in m2:                                    # two-way interactions
        reps = len(cond) / (n_lvl[a] * n_lvl[b])
        resid = m2[(a, b)] - m2[(a, b)].index.get_level_values(a).map(m[a]) \
                           - m2[(a, b)].index.get_level_values(b).map(m[b]) + g
        out[f"C({a}):C({b})"] = reps * (resid ** 2).sum()
    out["C(alpha):C(sigma):C(epsilon)"] = total - sum(out.values())
    return {k: v / total for k, v in out.items()}


def term_name(raw):
    """C(alpha):C(sigma) -> alpha x sigma"""
    parts = [p.replace("C(", "").replace(")", "") for p in raw.split(":")]
    return " x ".join(parts)


def main():
    df = pd.read_csv(HERE / "replication_results.csv")
    df["gap"] = -df["mean_reward"]

    per_rep = anova(df)
    cond = condition_eta_sq(df.groupby(FACTORS, as_index=False)["gap"].mean())

    print(f"n(per-replication) = {len(df)}   n(condition-level) = "
          f"{df.groupby(FACTORS).ngroups}\n")
    hdr = f"{'Source':<28}{'df':>4}{'F':>12}{'eta2':>10}{'p':>12}{'eta2 cond':>12}"
    print(hdr)
    print("-" * len(hdr))
    for raw in per_rep.index:
        if raw == "Residual":
            continue
        name = term_name(raw)
        r = per_rep.loc[raw]
        c_eta = cond.get(raw, float("nan"))
        print(f"{name:<28}{int(r['df']):>4}{r['F']:>12.2f}{r['eta_sq']:>10.4f}"
              f"{r['PR(>F)']:>12.3g}{c_eta:>12.4f}")


if __name__ == "__main__":
    main()
