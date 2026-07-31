"""The noise-response variance channel, derived and then validated.

DERIVATION (naive-myopic, certainty-equivalent, one resource, interior state)
---------------------------------------------------------------------------
Agent i's decision rule is a two-threshold step function of its private signal
shat_i = x + N(0, eps) (core.naive_thresholds; monotonicity asserted there):

    delta_i = +1  if shat_i < th1_i,   0 if th1_i < shat_i < th2_i,   -1 else,
    th1_i = tau_i - 1/2 + O(boundary),  th2_i = tau_i + 1/2 + O(boundary).

Therefore, EXACTLY,

    p_i^+(x) = Phi((th1_i - x)/sqrt(eps))
    p_i^0(x) = Phi((th2_i - x)/sqrt(eps)) - p_i^+(x)
    p_i^-(x) = 1 - Phi((th2_i - x)/sqrt(eps))

and, because the signals are independent across agents,

    mD(x)  = E[D | x]   = sum_i (p_i^+ - p_i^-)
    vD(x)  = Var[D | x] = sum_i (p_i^+ + p_i^- - (p_i^+ - p_i^-)^2).

Linearise the drift at the delegated rest point xbar (mD(xbar) = 0):

    kappa = -mD'(xbar) = sum_i [ phi((th1_i-xbar)/sqrt(eps))
                               + phi((th2_i-xbar)/sqrt(eps)) ] / sqrt(eps)

so x_{t+1} - xbar ~= (1-kappa)(x_t - xbar) + noise of variance vD(xbar), giving

    Var_pi(x) ~= vD(xbar) / (kappa (2 - kappa))        [AR(1) prediction]

and the variance channel of G is sum_m Wbar_m Var_pi(x_m).

SMALL-eps LAW.  For a FIXED preference draw, every p_i^+ and p_i^- is either
0 or 1 up to exp(-d^2 / 2 eps) with d the distance from x to the nearest
threshold, so vD -> 0 faster than any power of eps: dG/deps^k = 0 for all k at
eps = 0+.  Averaged over the preference DGP, d has a bounded density with
p(0) > 0, and
    E_tau[ Phi(-d/sqrt(eps)) ] ~= p(0) sqrt(eps) int_0^inf Phi(-u) du
                                = p(0) sqrt(eps) / sqrt(2 pi),
so E[G](eps) - E[G](0) = Theta(sqrt(eps)):  a square-root cusp, NOT a linear
term.  expansion.py::P2 measures the exponent; this script checks the AR(1)
variance prediction against the exact chain.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core as K       # noqa: E402
import run_grid as R   # noqa: E402


def main(ndraw=400):
    com = R.draw_commons(ndraw, R.SEED + 5)
    L = R.lattices(com["f"][:, 0])
    start = R.start_dist(com["k0"][:, 0])
    rows = []
    for fam, ctrl in [("implemented", 0.0), ("implemented", 0.8),
                      ("equicorr", -0.33), ("equicorr", 0.8)]:
        tau3 = (K.dgp_implemented(com["b"], com["xi"], ctrl) if fam == "implemented"
                else K.dgp_equicorr(com["z_eq"], ctrl))
        tau = tau3[:, :, 0]
        for eps in (0.0, 0.01, 0.05, 0.25, 0.5, 1.0):
            th1, th2 = K.naive_thresholds(tau, eps, bayes=False)
            p = K.naive_action_probs(L[:, None, :], th1, th2, eps)  # (D,A,21,3)
            P = K.transition_from_pmf(K.aggregate_delta_pmf(p), R.TIDX)
            occ = K.cesaro_occupation(P, start)
            mx = (occ * L).sum(1)
            var_exact = (occ * L ** 2).sum(1) - mx ** 2

            pp = p[..., 2]                     # P(+1) per (draw, agent, state)
            pm = p[..., 0]
            mD = (pp - pm).sum(1)              # (D, 21)   E[D | x]  exact
            vD = (pp + pm - (pp - pm) ** 2).sum(1)  #        Var[D|x] exact
            j = occ.argmax(1)                  # the chain's own modal state
            ar = np.arange(ndraw)
            xbar = L[ar, j]
            if eps > 0:
                sd = np.sqrt(eps)
                kap = ((norm.pdf((th1 - xbar[:, None]) / sd)
                        + norm.pdf((th2 - xbar[:, None]) / sd)).sum(1)) / sd
            else:
                kap = np.full(ndraw, np.nan)
            den = kap * (2 - kap)
            good = np.isfinite(den) & (den > 1e-3)
            var_ar1 = np.where(good, vD[ar, j] / np.where(good, den, 1.0), np.nan)
            rat = var_ar1 / np.maximum(var_exact, 1e-12)
            rows.append(dict(
                family=fam, control=ctrl, eps=eps,
                var_exact=float(var_exact.mean()),
                vD_at_mode=float(vD[ar, j].mean()),
                kappa_med=float(np.nanmedian(kap)) if eps > 0 else np.nan,
                kappa_frac_ge2=float(np.nanmean(kap >= 2.0)) if eps > 0 else np.nan,
                ar1_usable_frac=float(good.mean()),
                ar1_ratio_median=float(np.nanmedian(rat)) if good.any() else np.nan,
                ar1_ratio_q25=float(np.nanquantile(rat, .25)) if good.any() else np.nan,
                ar1_ratio_q75=float(np.nanquantile(rat, .75)) if good.any() else np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "NOISE_CHANNEL.csv", index=False)
    print(df.round(4).to_string(index=False))
    return df


if __name__ == "__main__":
    main()
