"""A15.  CROSS-IMPLEMENTATION VALIDATION against the numericist's independent
grid (../G_grid_*.csv), which was written from a separate codebase (../core.py)
that this library deliberately does not import.

WHAT SHOULD AGREE, AND WHY
  * The numericist's `naive_bayes` rule is the same object as this library's
    `protocol="naive", clamp_aware=True`: myopic best response, others treated as
    static, posterior N(shat, eps), expected loss taken through the clamp.
  * Their CESARO design removes the planner's own travelling transient by
    benchmarking against the stationary optimum; this library removes it by
    running the planner's exact T=100 backward induction FROM THE SAME START, so
    both G's are "steady-state delegated loss minus attainable-optimum loss".
    Their T100 design does NOT remove it (they report the planner transient as
    0.535/step), so their T100 = their Cesaro + transient.  We therefore compare
    against the CESARO column.
  * eps = 0 exactly is a tie-convention knife-edge (a13 P2): the two codebases
    resolve ties differently, so eps=0 is expected to DISAGREE.  That expectation
    is itself the test: agreement for every eps>0, disagreement at eps=0 only.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from a00_lib import draw_problem, gap_batch, rho_of_alpha

HERE = Path(__file__).resolve().parent
GRID = HERE.parent
R = 20000
OUT = {}

num = pd.read_csv(GRID / "G_grid_naive_bayes_cesaro.csv")
print("numericist columns:", list(num.columns)[:12])
fams = sorted(num.family.unique())
print("families:", fams, " sigmas:", sorted(num.sigma.unique()),
      " eps:", sorted(num.eps.unique()))


def mine(dgp, knob, sigma, eps, protocol="naive", seed=20260730):
    rng = np.random.default_rng(seed)
    tot, var = 0.0, 0.0
    for _ in range(3):
        t, w, x0 = draw_problem(rng, R, dgp, knob, sigma)
        g = gap_batch(t, w, x0, eps, protocol=protocol)
        tot += g["G"].mean()
        var += g["G"].var() / R
    return tot, np.sqrt(var)


rows = []
print("\n" + "=" * 96)
print("X1  naive + Bayesian beliefs.  mine (independent code, T=100 paired planner)")
print("    vs numericist (core.py, Cesaro).   sigma = 0.6")
print("    family        control   eps  |   mine +- se     numericist +- se   | "
      " ratio   |diff|/comb.se")
for fam in fams:
    sub = num[(num.sigma == 0.6) & (num.family == fam)]
    for _, r in sub.iterrows():
        g, se = mine(fam, float(r.control), 0.6, float(r.eps))
        comb = np.hypot(se, float(r.G_se))
        z = abs(g - float(r.G_mean)) / max(comb, 1e-12)
        flag = "  <-- eps=0 knife-edge" if r.eps == 0 else ""
        print(f"    {fam:13s}{r.control:+6.3f} {r.eps:5.2f} | {g:7.4f}+-{se:.4f}"
              f"   {r.G_mean:7.4f}+-{r.G_se:.4f} | {g/max(r.G_mean,1e-9):6.3f}"
              f"   {z:8.1f}{flag}")
        rows.append(dict(family=fam, control=float(r.control), eps=float(r.eps),
                         mine=float(g), mine_se=float(se),
                         theirs=float(r.G_mean), theirs_se=float(r.G_se),
                         ratio=float(g / max(r.G_mean, 1e-9))))
OUT["X1"] = rows
d = pd.DataFrame(rows)
pos = d[d.eps > 0]
print(f"\n    eps > 0  ({len(pos)} cells): median ratio {pos.ratio.median():.4f}, "
      f"IQR [{pos.ratio.quantile(.25):.4f}, {pos.ratio.quantile(.75):.4f}], "
      f"range [{pos.ratio.min():.4f}, {pos.ratio.max():.4f}]")
zer = d[d.eps == 0]
print(f"    eps = 0  ({len(zer)} cells): median ratio {zer.ratio.median():.4f}, "
      f"range [{zer.ratio.min():.4f}, {zer.ratio.max():.4f}]")
OUT["X1_summary"] = dict(
    eps_pos_median=float(pos.ratio.median()),
    eps_pos_min=float(pos.ratio.min()), eps_pos_max=float(pos.ratio.max()),
    eps0_median=float(zer.ratio.median()),
    eps0_min=float(zer.ratio.min()), eps0_max=float(zer.ratio.max()))

print("\n" + "=" * 96)
print("X2  sigma-homogeneity cross-check: G(sigma)/G(0.6) vs sigma_eff(sigma)/0.6")
from scipy.stats import norm


def sigma_eff(s):
    a, b = (0 - s) / 0.05, (1 - s) / 0.05
    return (s * (norm.cdf(b) - norm.cdf(a)) - 0.05 * (norm.pdf(b) - norm.pdf(a))
            + (1 - norm.cdf(b)))


base, _ = mine("implemented", 0.0, 0.6, 0.25)
print("     sigma |   mine G   ratio to 0.6 |  sigma_eff/0.6  | numericist ratio")
nb = num[(num.family == "implemented") & (num.control == 0.0) & (num.eps == 0.25)]
nbase = float(nb[nb.sigma == 0.6].G_mean.iloc[0])
srow = []
for s in (0.2, 0.4, 0.6, 0.8, 1.0):
    g, _ = mine("implemented", 0.0, s, 0.25)
    nr = float(nb[nb.sigma == s].G_mean.iloc[0]) / nbase
    print(f"      {s:.1f}  |  {g:7.4f}   {g/base:7.4f}     |    {sigma_eff(s)/0.6:7.4f}"
          f"      |   {nr:7.4f}")
    srow.append(dict(sigma=s, mine=float(g), ratio=float(g / base),
                     pred=float(sigma_eff(s) / 0.6), theirs=nr))
OUT["X2"] = srow

with open("a15_results.json", "w") as fh:
    json.dump(OUT, fh, indent=1)
print("\nwrote a15_results.json")
