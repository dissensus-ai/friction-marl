"""A15b.  Repair of a15's X2 (the numericist's grid carries only sigma in
{0.2, 0.6, 1.0}, not the full five levels).  Also re-serialises a15's X1 table
from its saved stdout so the comparison survives in JSON.
"""
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm
from a00_lib import draw_problem, gap_batch

HERE = Path(__file__).resolve().parent
GRID = HERE.parent
OUT = {}

# ---- re-serialise X1 from a15_out.txt --------------------------------------- #
pat = re.compile(r"^\s+(\w+)\s+([-+][\d.]+)\s+([\d.]+)\s*\|\s*([\d.]+)\+-([\d.]+)"
                 r"\s+([\d.]+)\+-([\d.]+)\s*\|\s*([\d.]+)")
rows = []
for line in (HERE / "a15_out.txt").read_text().splitlines():
    m = pat.match(line)
    if m:
        rows.append(dict(family=m.group(1), control=float(m.group(2)),
                         eps=float(m.group(3)), mine=float(m.group(4)),
                         mine_se=float(m.group(5)), theirs=float(m.group(6)),
                         theirs_se=float(m.group(7)), ratio=float(m.group(8))))
d = pd.DataFrame(rows)
OUT["X1"] = rows
print(f"recovered {len(d)} X1 comparison rows from a15_out.txt")
pos, zer = d[d.eps > 0], d[d.eps == 0]
print(f"  eps>0 ({len(pos)}): median ratio {pos.ratio.median():.4f}  "
      f"IQR [{pos.ratio.quantile(.25):.4f},{pos.ratio.quantile(.75):.4f}]  "
      f"range [{pos.ratio.min():.4f},{pos.ratio.max():.4f}]")
print(f"  eps=0 ({len(zer)}): median ratio {zer.ratio.median():.4f}  "
      f"range [{zer.ratio.min():.4f},{zer.ratio.max():.4f}]")
print("  largest departures from 1:")
for _, r in d.reindex(d.ratio.sub(1).abs().sort_values(ascending=False).index).head(5).iterrows():
    z = abs(r.mine - r.theirs) / np.hypot(r.mine_se, r.theirs_se)
    print(f"    {r.family:13s}{r.control:+6.3f} eps={r.eps:4.2f}: {r.mine:7.4f} vs "
          f"{r.theirs:7.4f}  ratio {r.ratio:.4f}  |diff|/comb.SE = {z:.1f}")
OUT["X1_summary"] = dict(n_pos=int(len(pos)), pos_median=float(pos.ratio.median()),
                         pos_lo=float(pos.ratio.min()), pos_hi=float(pos.ratio.max()),
                         pos_iqr=[float(pos.ratio.quantile(.25)),
                                  float(pos.ratio.quantile(.75))],
                         n_zero=int(len(zer)), zero_median=float(zer.ratio.median()),
                         zero_lo=float(zer.ratio.min()), zero_hi=float(zer.ratio.max()))

# ---- X2 with the sigmas the numericist actually ran -------------------------- #
R = 20000


def sigma_eff(s):
    a, b = (0 - s) / 0.05, (1 - s) / 0.05
    return (s * (norm.cdf(b) - norm.cdf(a)) - 0.05 * (norm.pdf(b) - norm.pdf(a))
            + (1 - norm.cdf(b)))


def mine(dgp, knob, sigma, eps, seed=20260730):
    rng = np.random.default_rng(seed)
    tot = 0.0
    for _ in range(3):
        t, w, x0 = draw_problem(rng, R, dgp, knob, sigma)
        tot += gap_batch(t, w, x0, eps, protocol="naive")["G"].mean()
    return tot


num = pd.read_csv(GRID / "G_grid_naive_bayes_cesaro.csv")
print("\nX2  sigma-homogeneity cross-check (sigma in {0.2, 0.6, 1.0} = what exists)")
x2 = []
for eps in (0.25, 1.0):
    nb = num[(num.family == "implemented") & (num.control == 0.0) & (num.eps == eps)]
    nbase = float(nb[nb.sigma == 0.6].G_mean.iloc[0])
    base = mine("implemented", 0.0, 0.6, eps)
    print(f"  --- eps = {eps} ---")
    print("     sigma |   mine G   ratio to 0.6 |  sigma_eff/0.6 (predicted) | numericist ratio")
    for s in (0.2, 0.6, 1.0):
        g = mine("implemented", 0.0, s, eps)
        nr = float(nb[nb.sigma == s].G_mean.iloc[0]) / nbase
        print(f"      {s:.1f}  |  {g:7.4f}   {g/base:7.4f}     |    {sigma_eff(s)/0.6:7.4f}"
              f"                  |   {nr:7.4f}")
        x2.append(dict(eps=eps, sigma=s, mine=float(g), ratio=float(g / base),
                       pred=float(sigma_eff(s) / 0.6), theirs=nr))
OUT["X2"] = x2
print("  predicted ratios are sigma_eff(sigma)/sigma_eff(0.6); the residual excess at")
print("  sigma=0.2 is the O((0.05/sigma)^2) weight-dispersion term of eq. (4.3).")

with open("a15b_results.json", "w") as fh:
    json.dump(OUT, fh, indent=1)
print("\nwrote a15b_results.json")
