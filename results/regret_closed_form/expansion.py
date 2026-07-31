"""Local behaviour of E[G] about (alpha, eps) = (0, 0) at fixed sigma.

This is deliverable 2: is  sigma (1+eps)/(1+alpha)  the first-order expansion?
Four separable questions, each answered with its own numerical probe:

  P1  degree-1 homogeneity in sigma            -> fit  log E[G] ~ p log sigma
  P2  the eps slope at eps = 0+                 -> fit  log (E[G](eps)-E[G](0))
                                                        ~ q log eps on eps <= 1e-2
  P3  the alpha slope at alpha = 0, its SIGN and its SHAPE
        -- exact +/-alpha symmetry test for the implemented and renormalised
           DGPs (a distributional identity, checked to machine precision, not
           by Monte Carlo)
        -- signed rho slope for the equicorrelated DGP, where rho really is the
           pairwise correlation
  P4  candidate shapes 1/(1+a), (1-|a|)^2, |a| fitted on the alpha axis

Everything uses the naive-myopic rule, whose noise integration is EXACT
(Gaussian CDF thresholds + convolution), so the only Monte Carlo here is over
preference draws -- and every eps/alpha comparison is PAIRED on the same draws.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core as K          # noqa: E402
import run_grid as R      # noqa: E402

NDRAW = 2000
SEED = 20260730


def G_for(tau, com, Lm, starts, eps, sigma, design="cesaro"):
    occ = []
    for m in range(K.N_RESOURCES):
        th1, th2 = K.naive_thresholds(tau[:, :, m], eps, bayes=False)
        p = K.naive_action_probs(Lm[m][:, None, :], th1, th2, eps)
        P = K.transition_from_pmf(K.aggregate_delta_pmf(p), R.TIDX)
        occ.append(K.cesaro_occupation(P, starts[m]) if design == "cesaro"
                   else K.horizon_occupation(P, starts[m]))
    w = K.weights_from_z(com["z_w"], sigma)
    G, Gi, fl = R.cell_metrics(tau, w, occ, Lm)
    return G, Gi, fl


def main():
    com = R.draw_commons(NDRAW, SEED)
    Lm = [R.lattices(com["f"][:, m]) for m in range(K.N_RESOURCES)]
    starts = [R.start_dist(com["k0"][:, m]) for m in range(K.N_RESOURCES)]
    out = {}

    # ---------------- P1  sigma homogeneity -------------------------------- #
    rows = []
    for (fam, ctrl) in [("implemented", 0.0), ("implemented", 0.8),
                        ("equicorr", 0.0), ("equicorr", 0.8)]:
        tau = (K.dgp_implemented(com["b"], com["xi"], ctrl) if fam == "implemented"
               else K.dgp_equicorr(com["z_eq"], ctrl))
        for eps in (0.0, 0.5):
            for sigma in (0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2):
                G, _, fl = G_for(tau, com, Lm, starts, eps, sigma)
                rows.append(dict(family=fam, control=ctrl, eps=eps, sigma=sigma,
                                 G_mean=float(G.mean()),
                                 G_se=float(G.std(ddof=1) / np.sqrt(G.size)),
                                 G_over_sigma=float(G.mean() / sigma),
                                 floor_mean=float(fl.mean())))
    p1 = pd.DataFrame(rows)
    p1.to_csv(HERE / "P1_sigma_homogeneity.csv", index=False)
    fits = []
    for (fam, ctrl, eps), g in p1.groupby(["family", "control", "eps"]):
        c = np.polyfit(np.log(g.sigma), np.log(g.G_mean), 1)
        fits.append(dict(family=fam, control=ctrl, eps=eps,
                         log_log_slope=float(c[0]),
                         G_over_sigma_min=float(g.G_over_sigma.min()),
                         G_over_sigma_max=float(g.G_over_sigma.max()),
                         spread_pct=float(100 * (g.G_over_sigma.max() /
                                                 g.G_over_sigma.min() - 1))))
    out["P1_sigma"] = fits

    # ---------------- P2  the eps slope at 0+ ------------------------------ #
    eps_grid = [0.0, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2,
                0.1, 0.25, 0.5, 1.0]
    rows = []
    for (fam, ctrl) in [("implemented", 0.0), ("implemented", 0.4),
                        ("implemented", 0.8), ("equicorr", -0.33),
                        ("equicorr", 0.0), ("equicorr", 0.8)]:
        tau = (K.dgp_implemented(com["b"], com["xi"], ctrl) if fam == "implemented"
               else K.dgp_equicorr(com["z_eq"], ctrl))
        base = None
        for eps in eps_grid:
            G, _, _ = G_for(tau, com, Lm, starts, eps, 0.6)
            if base is None:
                base = G
            d = G - base
            rows.append(dict(family=fam, control=ctrl, eps=eps,
                             G_mean=float(G.mean()),
                             dG_mean=float(d.mean()),
                             dG_se_paired=float(d.std(ddof=1) / np.sqrt(d.size))
                             if eps > 0 else 0.0,
                             frac_draws_changed=float((np.abs(d) > 1e-12).mean())))
    p2 = pd.DataFrame(rows)
    p2.to_csv(HERE / "P2_eps_expansion.csv", index=False)
    fits = []
    for (fam, ctrl), g in p2.groupby(["family", "control"]):
        sm = g[(g.eps > 0) & (g.eps <= 1e-2)]
        y = np.abs(sm.dG_mean.values)
        ok = y > 0
        slope = float(np.polyfit(np.log(sm.eps.values[ok]), np.log(y[ok]), 1)[0]) \
            if ok.sum() >= 2 else float("nan")
        fits.append(dict(family=fam, control=ctrl,
                         small_eps_loglog_slope=slope,
                         sign_of_dG_at_eps_1e_2=float(np.sign(
                             g[g.eps == 1e-2].dG_mean.values[0])),
                         dG_at_eps_0p25=float(g[g.eps == 0.25].dG_mean.values[0]),
                         dG_at_eps_1p0=float(g[g.eps == 1.0].dG_mean.values[0])))
    out["P2_eps"] = fits

    # ---------------- P3  alpha: exact symmetry + slope -------------------- #
    sym = []
    for fam in ("implemented", "renormalised"):
        fn = K.dgp_implemented if fam == "implemented" else K.dgp_renormalised
        for a in (0.05, 0.2, 0.4, 0.8):
            t_pos = fn(com["b"], com["xi"], a)
            t_neg = fn(-com["b"], com["xi"], -a)
            sym.append(dict(family=fam, alpha=a,
                            max_abs_tau_diff=float(np.abs(t_pos - t_neg).max())))
    out["P3_alpha_pm_symmetry"] = sym

    rows = []
    a_grid = [-0.8, -0.6, -0.4, -0.2, -0.1, -0.05, -0.02, -0.01, 0.0,
              0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
    for fam in ("implemented", "renormalised"):
        fn = K.dgp_implemented if fam == "implemented" else K.dgp_renormalised
        for eps in (0.0, 0.5):
            for a in a_grid:
                tau = fn(com["b"], com["xi"], a)
                G, _, _ = G_for(tau, com, Lm, starts, eps, 0.6)
                rows.append(dict(family=fam, eps=eps, control=a,
                                 G_mean=float(G.mean()),
                                 G_se=float(G.std(ddof=1) / np.sqrt(G.size)),
                                 target_corr=float(a ** 2 /
                                                   (a ** 2 + (1 - abs(a)) ** 2))))
    r_grid = [-0.33, -0.25, -0.15, -0.05, 0.0, 0.05, 0.15, 0.25, 0.4, 0.6, 0.8, 0.95]
    for eps in (0.0, 0.5):
        for r in r_grid:
            tau = K.dgp_equicorr(com["z_eq"], r)
            G, _, _ = G_for(tau, com, Lm, starts, eps, 0.6)
            rows.append(dict(family="equicorr", eps=eps, control=r,
                             G_mean=float(G.mean()),
                             G_se=float(G.std(ddof=1) / np.sqrt(G.size)),
                             target_corr=float(r)))
    p3 = pd.DataFrame(rows)
    p3.to_csv(HERE / "P3_alpha_axis.csv", index=False)

    # local slopes
    slopes = []
    for (fam, eps), g in p3.groupby(["family", "eps"]):
        g = g.sort_values("control")
        c = g.control.values
        y = g.G_mean.values
        if fam == "equicorr":
            loc = g[(np.abs(c) <= 0.25)]
            sl = float(np.polyfit(loc.control, loc.G_mean, 1)[0])
            slopes.append(dict(family=fam, eps=eps, kind="two_sided_linear",
                               slope=sl))
        else:
            rt = g[(c >= 0) & (c <= 0.2)]
            lf = g[(c <= 0) & (c >= -0.2)]
            slopes.append(dict(family=fam, eps=eps, kind="right_derivative",
                               slope=float(np.polyfit(rt.control, rt.G_mean, 1)[0])))
            slopes.append(dict(family=fam, eps=eps, kind="left_derivative",
                               slope=float(np.polyfit(lf.control, lf.G_mean, 1)[0])))
    out["P3_local_slopes"] = slopes

    (HERE / "EXPANSION.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
