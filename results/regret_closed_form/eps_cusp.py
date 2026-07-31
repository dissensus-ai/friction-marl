"""The behaviour of E[G] as eps -> 0+, resolved properly.

Two separate facts, which P2_eps_expansion.csv shows are easy to confuse:

(A) A KNIFE-EDGE AT eps = 0 EXACTLY.  The lower clamp x = 0 is a lattice point
    AND a decision threshold: for any agent with tau_i in (-1/2, 0) the moves
    delta = -1 and delta = 0 both land the post-state on 0, so the clamped
    losses tie exactly, and the tie rule decides. At eps = 0 the tie is broken
    by convention (T1 -> delta = 0); at eps = 0+ the Gaussian signal splits it
    50/50. Since s* < 0 for about half of all resources, the chain spends a lot
    of time on that boundary, so
        E[G](0)  !=  lim_{eps->0+} E[G](eps).
    Measured below. This is a property of the environment, not of the code.

(B) THE SHAPE OF THE LIMIT.  For a FIXED preference draw every decision
    probability is 0 or 1 up to exp(-d^2/2 eps) with d the distance from the
    state to the nearest threshold, so E[G] is flat to all orders in eps at
    0+ for a.e. draw. Averaged over the DGP, d has a bounded density with
    p(0) > 0 and E_tau[Phi(-d/sqrt(eps))] ~ p(0) sqrt(eps)/sqrt(2 pi), giving a
    SQUARE-ROOT CUSP: E[G](eps) - E[G](0+) = Theta(sqrt(eps)).  There is no
    linear term, so no first-order expansion in eps exists.

Both are measured here against the eps -> 0+ base (eps = 1e-12), on paired
draws, using the fact that the certainty-equivalent decision thresholds do not
depend on eps (so they are computed once and reused at every eps).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core as K       # noqa: E402
import run_grid as R   # noqa: E402

NDRAW = 8000
EPS = [0.0, 1e-12, 1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]


def main():
    com = R.draw_commons(NDRAW, R.SEED)
    Lm = [R.lattices(com["f"][:, m]) for m in range(K.N_RESOURCES)]
    st = [R.start_dist(com["k0"][:, m]) for m in range(K.N_RESOURCES)]
    w = K.weights_from_z(com["z_w"], 0.6)
    rows = []
    for fam, ctrl in [("implemented", 0.0), ("implemented", 0.8),
                      ("equicorr", -0.33), ("equicorr", 0.8)]:
        tau = (K.dgp_implemented(com["b"], com["xi"], ctrl) if fam == "implemented"
               else K.dgp_equicorr(com["z_eq"], ctrl))
        # CE thresholds do NOT depend on eps -- compute once per resource
        th = [K.naive_thresholds(tau[:, :, m], 0.0, bayes=False)
              for m in range(K.N_RESOURCES)]
        base = None
        for eps in EPS:
            occ = []
            for m in range(K.N_RESOURCES):
                p = K.naive_action_probs(Lm[m][:, None, :], th[m][0], th[m][1], eps)
                P = K.transition_from_pmf(K.aggregate_delta_pmf(p), R.TIDX)
                occ.append(K.cesaro_occupation(P, st[m]))
            G, _, _ = R.cell_metrics(tau, w, occ, Lm)
            if eps == 1e-12:
                base = G.copy()
            rows.append(dict(family=fam, control=ctrl, eps=eps,
                             G_mean=float(G.mean()),
                             G_se=float(G.std(ddof=1) / np.sqrt(G.size)),
                             _G=G))
        for r in rows:
            if r["family"] == fam and r["control"] == ctrl and base is not None:
                d = r.pop("_G") - base
                r["dG_from_0plus"] = float(d.mean())
                r["dG_se_paired"] = float(d.std(ddof=1) / np.sqrt(d.size))
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "EPS_CUSP.csv", index=False)
    print(df.round(6).to_string(index=False))

    out = []
    for (fam, ctrl), g in df.groupby(["family", "control"]):
        g = g.sort_values("eps")
        knife = float(g[g.eps == 0].G_mean.values[0]
                      - g[g.eps == 1e-12].G_mean.values[0])
        sm = g[(g.eps >= 1e-8) & (g.eps <= 1e-3)]
        y = np.abs(sm.dG_from_0plus.values)
        ok = y > 1e-12
        slope = float(np.polyfit(np.log(sm.eps.values[ok]), np.log(y[ok]), 1)[0]) \
            if ok.sum() >= 3 else float("nan")
        # linear-model falsification: if E[G] had a linear term c*eps, then
        # dG/eps would be constant; under sqrt it scales as eps^-1/2
        ratio = {f"dG_over_eps@{e:g}":
                 float(g[g.eps == e].dG_from_0plus.values[0] / e)
                 for e in (1e-6, 1e-4, 1e-2)}
        out.append(dict(family=fam, control=ctrl,
                        G_at_eps0_convention=float(g[g.eps == 0].G_mean.values[0]),
                        G_at_0plus=float(g[g.eps == 1e-12].G_mean.values[0]),
                        knife_edge_jump=knife,
                        knife_edge_pct=100 * knife / float(
                            g[g.eps == 1e-12].G_mean.values[0]),
                        cusp_loglog_slope_1e8_to_1e3=slope, **ratio))
    o = pd.DataFrame(out)
    o.to_csv(HERE / "EPS_CUSP_FITS.csv", index=False)
    print()
    print(o.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
