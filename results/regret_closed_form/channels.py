"""Exact channel decomposition of G on the main grid (naive-myopic rule).

    G = sum_m Wbar_m [ E_pi (x_m - sstar_m)^2 - (xlat_m - sstar_m)^2 ]
      = sum_m Wbar_m [ Var_pi(x_m) ]                          <- VARIANCE channel
      + sum_m Wbar_m [ (E_pi x_m - sstar_m)^2 - (xlat_m - sstar_m)^2 ]
                                                              <- BIAS channel

The VARIANCE channel is exactly the "noise-response variance" object: at
eps = 0 it is whatever limit-cycle amplitude the deterministic delegated
dynamics carries; its increment over eps = 0 is the part injected by agents
responding to noisy observations.

The floor  sum_m Wbar_m (xlat_m - sstar_m)^2  is the planner's QUANTIZATION
FLOOR: it is the welfare the principal itself cannot reach because the joint
action set is {-4..4}^3 on a unit lattice. It is reported but it CANCELS out of
G by construction -- which is the whole content of the "irreducible sigma/2
baseline cannot survive a difference-against-benchmark definition" argument.
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


def main(ndraw=1000):
    com = R.draw_commons(ndraw, R.SEED)
    Lm = [R.lattices(com["f"][:, m]) for m in range(K.N_RESOURCES)]
    starts = [R.start_dist(com["k0"][:, m]) for m in range(K.N_RESOURCES)]
    rows = []
    fams = [("implemented", R.ALPHAS), ("renormalised", R.ALPHAS),
            ("equicorr", R.RHOS)]
    for fam, controls in fams:
        for ctrl in controls:
            tau = (K.dgp_implemented(com["b"], com["xi"], ctrl) if fam == "implemented"
                   else K.dgp_renormalised(com["b"], com["xi"], ctrl)
                   if fam == "renormalised" else K.dgp_equicorr(com["z_eq"], ctrl))
            for eps in R.EPSILONS:
                occ = []
                for m in range(K.N_RESOURCES):
                    th1, th2 = K.naive_thresholds(tau[:, :, m], eps, bayes=False)
                    p = K.naive_action_probs(Lm[m][:, None, :], th1, th2, eps)
                    P = K.transition_from_pmf(K.aggregate_delta_pmf(p), R.TIDX)
                    occ.append(K.cesaro_occupation(P, starts[m]))
                for sigma in R.SIGMAS:
                    w = K.weights_from_z(com["z_w"], sigma)
                    Wbar = w.sum(axis=1)
                    degen = Wbar <= 0
                    sstar = np.where(degen, 0.0, (w * tau).sum(axis=1) /
                                     np.where(degen, 1.0, Wbar))
                    var_ch = np.zeros(ndraw)
                    bias_ch = np.zeros(ndraw)
                    floor = np.zeros(ndraw)
                    absbias = np.zeros(ndraw)
                    for m in range(K.N_RESOURCES):
                        o, L = occ[m], Lm[m]
                        mx = (o * L).sum(1)
                        mx2 = (o * L ** 2).sum(1)
                        v = mx2 - mx ** 2
                        ss = sstar[:, m]
                        d2 = (L - ss[:, None]) ** 2
                        j = d2.argmin(1)
                        xlat = L[np.arange(ndraw), j]
                        fl = (xlat - ss) ** 2
                        var_ch += Wbar[:, m] * v
                        bias_ch += Wbar[:, m] * ((mx - ss) ** 2 - fl)
                        floor += Wbar[:, m] * fl
                        absbias += Wbar[:, m] * (mx - ss) ** 2
                    rows.append(dict(family=fam, control=ctrl, sigma=sigma,
                                     eps=eps,
                                     G_mean=float((var_ch + bias_ch).mean()),
                                     var_channel=float(var_ch.mean()),
                                     bias_channel=float(bias_ch.mean()),
                                     bias_raw=float(absbias.mean()),
                                     floor=float(floor.mean())))
    df = pd.DataFrame(rows)
    # noise-response variance increment relative to the same cell at eps = 0
    base = df[df.eps == 0].set_index(["family", "control", "sigma"])["var_channel"]
    df["var_channel_eps0"] = [base.loc[(f, c, s)] for f, c, s in
                              zip(df.family, df.control, df.sigma)]
    df["var_channel_noise_increment"] = df.var_channel - df.var_channel_eps0
    df.to_csv(HERE / "CHANNELS.csv", index=False)
    print(df.groupby(["family", "eps"])[
        ["G_mean", "var_channel", "bias_channel", "floor",
         "var_channel_noise_increment"]].mean().round(4).to_string())
    return df


if __name__ == "__main__":
    main()
