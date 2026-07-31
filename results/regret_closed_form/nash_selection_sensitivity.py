"""Equilibrium-SELECTION sensitivity for the stage-Nash arm.

13-31% of (state, signal) cells admit more than one pure equilibrium
(NASH_NE_CENSUS.csv), so the Gauss-Seidel start matters in principle. Measured:
  * start = all-zero  vs  start = naive-myopic profile -> IDENTICAL (0.0)
  * start = uniformly random profile (3 seeds)         -> quantified below
Same signal draws in every arm (the noise is drawn before the start branch).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core as K, run_grid as R

def G_nash(com, Lm, starts, tau, eps, start, seed, n_noise=256):
    occ = []
    for m in range(K.N_RESOURCES):
        nn = 1 if eps <= 0 else n_noise
        pmf, _ = K.stage_nash_pmf(Lm[m], tau[:, :, m], eps,
                                  np.random.default_rng(seed + m),
                                  n_noise=nn, start=start)
        occ.append(K.cesaro_occupation(K.transition_from_pmf(pmf, R.TIDX), starts[m]))
    w = K.weights_from_z(com["z_w"], 0.6)
    G, _, _ = R.cell_metrics(tau, w, occ, Lm)
    return float(G.mean())

def main(ndraw=200):
    com = R.draw_commons(ndraw, R.SEED + 3)
    Lm = [R.lattices(com["f"][:, m]) for m in range(K.N_RESOURCES)]
    st = [R.start_dist(com["k0"][:, m]) for m in range(K.N_RESOURCES)]
    rows = []
    for fam, ctrl in [("implemented", 0.0), ("implemented", 0.8),
                      ("equicorr", -0.33), ("equicorr", 0.8)]:
        tau = (K.dgp_implemented(com["b"], com["xi"], ctrl) if fam == "implemented"
               else K.dgp_equicorr(com["z_eq"], ctrl))
        for eps in (0.0, 0.25, 1.0):
            g0 = G_nash(com, Lm, st, tau, eps, "zeros", 7)
            gn = G_nash(com, Lm, st, tau, eps, "naive", 7)
            gr = [G_nash(com, Lm, st, tau, eps, "random", 7 + 100 * k)
                  for k in range(3)]
            rows.append(dict(family=fam, control=ctrl, eps=eps, G_zeros=g0,
                             G_naive_start=gn, G_random_mean=float(np.mean(gr)),
                             G_random_min=float(np.min(gr)),
                             G_random_max=float(np.max(gr)),
                             pct_random_vs_zeros=100 * (np.mean(gr) / g0 - 1)))
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "NASH_SELECTION_SENSITIVITY.csv", index=False)
    print(df.round(4).to_string(index=False))

if __name__ == "__main__":
    main()
