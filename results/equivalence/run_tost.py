"""Bound the epsilon null: TOST equivalence on the post-fix factorial.

The paper currently hedges "failure to detect, not equivalence" (main.tex:1370,:1780)
and its TOST module was dead code. This driver puts a number on the hedge: for the
frozen held-out clean-eval DV, how large an epsilon effect (eps=0 vs eps=1.0, the
full manipulation range) can be RULED OUT, per (alpha, sigma) cell and pooled?

Margin anchor: the sigma main effect on the same DV at eps=0 (pooled mean at
sigma=1.0 minus at sigma=0.2) -- the design's own largest effect. TOST at margins
{10%, 25%, 50%} of the anchor, plus the exact equivalence bound (the smallest margin
at which TOST rejects at alpha=.05 = the half-width of the 90% CI around the diff).

Interpretation guard (fleet review, 30 Jul): the epsilon effect is real and monotone
(p ~ 1e-4 across specifications); TOST here bounds its SIZE, it does not license
"epsilon is inert".

Run:  PYTHONPATH=<repo> python3 results/equivalence/run_tost.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tost import tost_equivalence  # noqa: E402

CSV = HERE.parent / "gpu_factorial_postfix" / "replication_results.csv"
DV = "eval_reward_clean"

df = pd.read_csv(CSV)
e0 = df[df.epsilon == 0.0]
anchor = abs(e0[e0.sigma == 1.0][DV].mean() - e0[e0.sigma == 0.2][DV].mean())
print(f"DV = {DV}; sigma anchor effect (eps=0, sigma 1.0 vs 0.2) = {anchor:.3f}\n")

FRACS = (0.10, 0.25, 0.50)


def row(label, x, y):
    d = x.mean() - y.mean()
    se = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
    dfree = (x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y)) ** 2 / (
        (x.var(ddof=1) / len(x)) ** 2 / (len(x) - 1) + (y.var(ddof=1) / len(y)) ** 2 / (len(y) - 1))
    hw90 = stats.t.ppf(0.95, dfree) * se
    bound = abs(d) + hw90  # smallest margin at which TOST rejects at .05
    ps = [tost_equivalence(x, y, f * anchor).p_tost for f in FRACS]
    cells = "".join(f"{p:>12.4f}" for p in ps)
    print(f"{label:<22} diff={d:+.3f}  eq-bound={bound:.3f} ({bound/anchor*100:4.1f}% of anchor){cells}")
    return bound


hdr = "".join(f"{f'p@{int(f*100)}%':>12}" for f in FRACS)
print(f"{'cell':<22} {'':<40}{hdr}")
bounds = []
for (a, s), g in df.groupby(["alpha", "sigma"]):
    x = g[g.epsilon == 0.0][DV].values
    y = g[g.epsilon == 1.0][DV].values
    bounds.append(row(f"alpha={a:+.1f} sig={s:.1f}", x, y))

x = df[df.epsilon == 0.0][DV].values
y = df[df.epsilon == 1.0][DV].values
print()
row("POOLED (all cells)", x, y)
print(f"\nper-cell equivalence bounds: median {np.median(bounds):.3f}, "
      f"worst {max(bounds):.3f} (vs anchor {anchor:.3f})")
