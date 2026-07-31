"""Stage-Nash equilibrium diagnostics.

(1) Exhaustive census of PURE equilibria over all 3^4 = 81 per-resource action
    profiles: does one exist, is it unique, how often does the Gauss-Seidel
    iterated best response land on it.
(2) Belief-model sensitivity: certainty-equivalent vs exact-Bayes agents on a
    subgrid (the full grid runs CE because the Bayes objective is ~15x more
    expensive; the naive rule runs BOTH on the full grid, so the size of the
    CE/Bayes wedge is separately known there).
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


def main(ndraw=200, n_noise=128):
    com = R.draw_commons(ndraw, R.SEED + 3)
    Lm = [R.lattices(com["f"][:, m]) for m in range(K.N_RESOURCES)]
    starts = [R.start_dist(com["k0"][:, m]) for m in range(K.N_RESOURCES)]
    rows, wedge = [], []
    for fam, ctrl in [("implemented", 0.0), ("implemented", 0.8),
                      ("equicorr", -0.33), ("equicorr", 0.8)]:
        tau = (K.dgp_implemented(com["b"], com["xi"], ctrl) if fam == "implemented"
               else K.dgp_equicorr(com["z_eq"], ctrl))
        for eps in (0.0, 0.25, 1.0):
            cen = K.stage_nash_pure_ne_census(
                Lm[0], tau[:, :, 0], eps, np.random.default_rng(1),
                n_noise=n_noise)
            cen.update(dict(family=fam, control=ctrl, eps=eps))
            rows.append(cen)
            # CE vs Bayes G on the same draws
            gs = {}
            for bay in (False, True):
                occ = []
                for m in range(K.N_RESOURCES):
                    nn = 1 if eps <= 0 else n_noise
                    pmf, _ = K.stage_nash_pmf(Lm[m], tau[:, :, m], eps,
                                              np.random.default_rng(7 + m),
                                              n_noise=nn, bayes=bay)
                    P = K.transition_from_pmf(pmf, R.TIDX)
                    occ.append(K.cesaro_occupation(P, starts[m]))
                w = K.weights_from_z(com["z_w"], 0.6)
                G, _, _ = R.cell_metrics(tau, w, occ, Lm)
                gs["bayes" if bay else "ce"] = float(G.mean())
            # equilibrium-SELECTION sensitivity: start the BR dynamic at the
            # naive-myopic profile instead of the all-zero profile
            occ = []
            for m in range(K.N_RESOURCES):
                nn = 1 if eps <= 0 else n_noise
                pmf, _ = K.stage_nash_pmf(Lm[m], tau[:, :, m], eps,
                                          np.random.default_rng(7 + m),
                                          n_noise=nn, start="naive")
                P = K.transition_from_pmf(pmf, R.TIDX)
                occ.append(K.cesaro_occupation(P, starts[m]))
            w = K.weights_from_z(com["z_w"], 0.6)
            Gn, _, _ = R.cell_metrics(tau, w, occ, Lm)
            wedge.append(dict(family=fam, control=ctrl, eps=eps,
                              G_ce=gs["ce"], G_bayes=gs["bayes"],
                              pct_change_bayes=100 * (gs["bayes"] / gs["ce"] - 1),
                              G_ce_start_naive=float(Gn.mean()),
                              pct_change_selection=100 * (float(Gn.mean()) /
                                                          gs["ce"] - 1)))
    a = pd.DataFrame(rows)
    b = pd.DataFrame(wedge)
    a.to_csv(HERE / "NASH_NE_CENSUS.csv", index=False)
    b.to_csv(HERE / "NASH_BELIEF_WEDGE.csv", index=False)
    print(a.round(4).to_string(index=False))
    print()
    print(b.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
