"""Iso-F fan: the direct test of the composite-index claim on LEARNED agents.

F = sigma(1+eps)/(1+alpha) asserts outcome = f(F) for SOME monotone f. Therefore any
two factorial cells with equal F but different (sigma, alpha, eps) must have equal
outcome -- no functional form for f needed, no model comparison, no fitting. This
script finds every exact F-tie in the 5x5x5 post-fix factorial and tests whether cell
identity matters within an F-level.

Two DVs: mean_reward (training, the paper's gap DV negated) and eval_reward_clean
(held-out frozen eval, noise-free). Per F-level: one-way ANOVA across cells on
replication-level data + eta^2(cell | F fixed). Omnibus: R^2 of DV ~ C(F_level)
versus DV ~ C(cell) on condition means -- the difference is variance the index F
cannot carry by construction.

Provenance note: the 30 Jul closed-form derivation (results/regret_closed_form/)
already retracted the composite for EXACT best-response agents; this is the
complementary test on the corpus's own trained learners, usable as R1 evidence
independent of the derivation's solution concepts.

Run:  PYTHONPATH=<repo> python3 results/iso_f_fan/analysis.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
CSV = HERE.parent / "gpu_factorial_postfix" / "replication_results.csv"

df = pd.read_csv(CSV)
df["F"] = (df.sigma * (1 + df.epsilon) / (1 + df.alpha)).round(6)
df["cell"] = list(zip(df.alpha, df.sigma, df.epsilon))

rows = []
print(f"factorial: {df.cell.nunique()} cells, {len(df)} rows")
levels = sorted(f for f, g in df.groupby("F") if g.cell.nunique() >= 2)
print(f"F-levels with >=2 tied cells: {len(levels)} "
      f"(>=3 cells: {sum(df[df.F == f].cell.nunique() >= 3 for f in levels)})\n")

for dv in ("mean_reward", "eval_reward_clean"):
    print(f"=== DV = {dv}")
    print(f"{'F':>8} {'cells':>6} {'spread':>8} {'anova_F':>9} {'p':>11} {'eta2_cell':>10}")
    for f in levels:
        g = df[df.F == f]
        groups = [c[dv].values for _, c in g.groupby("cell")]
        cm = [x.mean() for x in groups]
        aF, p = stats.f_oneway(*groups)
        gm = g[dv].mean()
        ss_b = sum(len(x) * (x.mean() - gm) ** 2 for x in groups)
        ss_t = ((g[dv] - gm) ** 2).sum()
        rows.append(dict(dv=dv, F=f, n_cells=len(groups), spread=max(cm) - min(cm),
                         anova_F=aF, p=p, eta2_cell=ss_b / ss_t))
        print(f"{f:8.3f} {len(groups):6d} {max(cm)-min(cm):8.3f} {aF:9.2f} {p:11.2e} {ss_b/ss_t:10.3f}")

    # omnibus on condition means: how much variance can ANY f(F) carry vs cell identity?
    cond = df.groupby(["cell", "F"])[dv].mean().reset_index()
    ss = lambda x: ((x - x.mean()) ** 2).sum()
    r2_F = 1 - cond.groupby("F")[dv].transform(lambda x: x - x.mean()).pipe(ss) / ss(cond[dv])
    print(f"  omnibus on {len(cond)} condition means: R2 upper bound for ANY f(F) = {r2_F:.3f}"
          f"  (1 - R2 = {1 - r2_F:.3f} of between-cell variance is invisible to F)\n")

pd.DataFrame(rows).to_csv(HERE / "iso_f_fan_results.csv", index=False)
print(f"-> {HERE / 'iso_f_fan_results.csv'}")
