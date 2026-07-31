"""Closed form for the planner's QUANTIZATION FLOOR, and its validation.

    floor = sum_m Wbar_m * min_{x in L(f_m)} (x - sstar_m)^2 .

Claim Q1 (interior, generic).  For sstar far from both boundaries and uniform
mod 1, the reachable set is two interleaved unit grids offset by f ~ U(0,1).
Given f, sstar falls in a gap of length f w.p. f and of length 1-f w.p. 1-f;
inside a gap of length g the distance to the nearer endpoint is U[0, g/2], so
    E[d^2 | f] = (f^3 + (1-f)^3)/12,      E_f[d^2] = 1/24 = 0.0416667.

Claim Q2 (boundary).  sstar <= 0 is absorbed by the lattice point 0 exactly, so
the floor contribution is sstar^2. Under every DGP here sstar is symmetric
about 0, so P(sstar<0) = 1/2 and E[sstar^2 1{sstar<0}] = Var(sstar)/2.

Claim Q3 (whole).  E[min dist^2] ~= Var(sstar)/2 + 1/24, valid once
Var(sstar) is large enough that the sstar>0 branch is generically interior.
Var(sstar) itself is closed form (equal weights, n = 4 agents):
    implemented   tau_i = a b + (1-|a|) xi_i    -> Var = a^2 + (1-|a|)^2/4
    renormalised  divide by sqrt(a^2+(1-|a|)^2) -> Var = (a^2 + (1-|a|)^2/4)
                                                        / (a^2 + (1-|a|)^2)
    equicorrelated, pairwise corr rho           -> Var = (1 + 3 rho)/4
      (=> exactly 0 at the PSD boundary rho = -1/3: the four targets sum to a
       constant, so the ensemble optimum is deterministic and the floor
       vanishes -- which is why the equicorr rho = -0.33 column has floor ~0.03
       rather than the generic ~1/24 x E[Wbar] x 3.)
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


def q1_check(n=4_000_000, seed=7):
    rng = np.random.default_rng(seed)
    f = rng.uniform(0, 1, n)
    s = rng.uniform(3, 7, n)                      # interior, uniform mod 1
    d_int = np.abs(s - np.round(s))
    k = np.round(s - f)
    d_sh = np.abs(s - (f + k))
    d2 = np.minimum(d_int, d_sh) ** 2
    return float(d2.mean()), 1.0 / 24.0


def var_sstar_analytic(fam, ctrl):
    if fam == "implemented":
        return ctrl ** 2 + (1 - abs(ctrl)) ** 2 / 4.0
    if fam == "renormalised":
        return (ctrl ** 2 + (1 - abs(ctrl)) ** 2 / 4.0) / (
            ctrl ** 2 + (1 - abs(ctrl)) ** 2)
    return (1.0 + 3.0 * ctrl) / 4.0


def main(ndraw=20000):
    mc, exact = q1_check()
    print(f"Q1  E[min dist^2] interior: MC {mc:.6f}   closed form 1/24 = {exact:.6f}")
    com = R.draw_commons(ndraw, R.SEED + 17)
    Lm = [R.lattices(com["f"][:, m]) for m in range(K.N_RESOURCES)]
    rows = []
    for fam, controls in [("implemented", R.ALPHAS), ("renormalised", R.ALPHAS),
                          ("equicorr", R.RHOS)]:
        for ctrl in controls:
            tau = (K.dgp_implemented(com["b"], com["xi"], ctrl) if fam == "implemented"
                   else K.dgp_renormalised(com["b"], com["xi"], ctrl)
                   if fam == "renormalised" else K.dgp_equicorr(com["z_eq"], ctrl))
            for sigma in R.SIGMAS:
                w = K.weights_from_z(com["z_w"], sigma)
                Wbar = w.sum(axis=1)
                sstar = (w * tau).sum(axis=1) / Wbar
                fl = np.zeros(ndraw)
                d2s = []
                for m in range(K.N_RESOURCES):
                    d2 = ((Lm[m] - sstar[:, [m]]) ** 2).min(axis=1)
                    d2s.append(d2)
                    fl += Wbar[:, m] * d2
                v_emp = float(sstar.var())
                v_th = var_sstar_analytic(fam, ctrl)
                pred = (v_th / 2.0 + 1.0 / 24.0) * float(Wbar.mean()) * 3
                rows.append(dict(family=fam, control=ctrl, sigma=sigma,
                                 floor_exact=float(fl.mean()),
                                 floor_closed_form=pred,
                                 rel_err_pct=100 * (pred / fl.mean() - 1),
                                 var_sstar_emp=v_emp, var_sstar_theory=v_th,
                                 E_min_dist2=float(np.mean(d2s)),
                                 E_Wbar_per_resource=float(Wbar.mean())))
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "FLOOR_CLOSED_FORM.csv", index=False)
    print(df.round(4).to_string(index=False))
    return df


if __name__ == "__main__":
    main()
