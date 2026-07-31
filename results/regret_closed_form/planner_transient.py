"""Exact T=100 planner benchmark vs the static (Cesaro) lattice optimum.

The Cesaro G uses the planner's long-run value, which IS the static best
lattice point (validate_lattice.py V5). The environment, though, pays a 100-step
episode from a Uniform[0,10]^3 reset, and the planner needs up to
ceil(10/4) = 3 steps to reach its optimum. This script computes the planner's
EXACT finite-horizon value by per-resource backward induction from the actual
start state (legitimate by separability fact S1, validated in V4) and reports
how much of the reported T=100 gap is really the planner's own transient.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core as K       # noqa: E402
import run_grid as R   # noqa: E402

T = K.EPISODE_LENGTH


def planner_T_value(L, sstar, Wbar, k0, T=T):
    """Exact undiscounted T-step backward induction, vectorised over draws.
    Returns mean per-step planner welfare (tracking part only) from x0."""
    idx = R.TIDX                                   # (21, 9), f-independent
    r = -Wbar[:, None] * (L - sstar[:, None]) ** 2  # (D, 21) on the POST state
    V = np.zeros_like(r)
    for _ in range(T):
        V = (r[:, idx] + V[:, idx]).max(axis=2)
    return V[np.arange(L.shape[0]), 2 * k0 + 1] / T


def main(ndraw=1000):
    com = R.draw_commons(ndraw, R.SEED)
    Lm = [R.lattices(com["f"][:, m]) for m in range(K.N_RESOURCES)]
    rows = []
    for fam, controls in [("implemented", R.ALPHAS), ("equicorr", R.RHOS)]:
        for ctrl in controls:
            tau = (K.dgp_implemented(com["b"], com["xi"], ctrl)
                   if fam == "implemented" else K.dgp_equicorr(com["z_eq"], ctrl))
            for sigma in R.SIGMAS:
                w = K.weights_from_z(com["z_w"], sigma)
                Wbar = w.sum(axis=1)
                sstar = (w * tau).sum(axis=1) / Wbar
                stat = np.zeros(ndraw)
                fin = np.zeros(ndraw)
                for m in range(K.N_RESOURCES):
                    L = Lm[m]
                    ss = sstar[:, m]
                    stat += -Wbar[:, m] * ((L - ss[:, None]) ** 2).min(axis=1)
                    fin += planner_T_value(L, ss, Wbar[:, m], com["k0"][:, m])
                rows.append(dict(family=fam, control=ctrl, sigma=sigma,
                                 planner_static_perstep=float(stat.mean()),
                                 planner_T100_perstep=float(fin.mean()),
                                 planner_transient_perstep=float((stat - fin).mean())))
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "PLANNER_TRANSIENT.csv", index=False)
    print(df.round(4).to_string(index=False))
    print("\nmean planner transient per step:",
          round(float(df.planner_transient_perstep.mean()), 4))
    return df


if __name__ == "__main__":
    main()
