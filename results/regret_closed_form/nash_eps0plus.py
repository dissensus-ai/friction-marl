"""Stage-Nash at eps -> 0+ : the same clamp knife-edge as the naive arm (§2.2a).

At eps = 0 exactly the clamped losses tie for any agent with tau_i in (-1/2, 0)
and the tie rule decides; at eps = 0+ the private signal splits the tie. The
eps = 0 column of the stage-Nash grid is therefore convention-dependent too, and
this quantifies by how much.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core as K, run_grid as R

def G_nash(com, Lm, st, tau, eps, n_noise, seed=31):
    occ = []
    for m in range(K.N_RESOURCES):
        nn = 1 if eps <= 0 else n_noise
        pmf, _ = K.stage_nash_pmf(Lm[m], tau[:, :, m], eps,
                                  np.random.default_rng(seed + m), n_noise=nn)
        occ.append(K.cesaro_occupation(K.transition_from_pmf(pmf, R.TIDX), st[m]))
    w = K.weights_from_z(com["z_w"], 0.6)
    G, _, _ = R.cell_metrics(tau, w, occ, Lm)
    return float(G.mean())

def main(ndraw=1000, n_noise=512):
    com = R.draw_commons(ndraw, R.SEED)
    Lm = [R.lattices(com["f"][:, m]) for m in range(K.N_RESOURCES)]
    st = [R.start_dist(com["k0"][:, m]) for m in range(K.N_RESOURCES)]
    rows = []
    for fam, ctrl in [("implemented", 0.0), ("implemented", 0.8),
                      ("equicorr", -0.33), ("equicorr", 0.8)]:
        tau = (K.dgp_implemented(com["b"], com["xi"], ctrl) if fam == "implemented"
               else K.dgp_equicorr(com["z_eq"], ctrl))
        g0 = G_nash(com, Lm, st, tau, 0.0, n_noise)
        gp = G_nash(com, Lm, st, tau, 1e-10, n_noise)
        rows.append(dict(family=fam, control=ctrl, G_eps0_convention=g0,
                         G_eps_0plus=gp, jump=g0 - gp,
                         jump_pct=100 * (g0 / gp - 1)))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(HERE / "NASH_EPS0PLUS.csv", index=False)

if __name__ == "__main__":
    main()
