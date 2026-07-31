"""The G surface: Monte Carlo over preference draws, exact in everything else.

Design (all declared)
---------------------
D1  NDRAW = 1000 preference draws per cell (brief requires >= 500).
D2  COMMON RANDOM NUMBERS. One set of standard normals (b, xi, z_eq, z_w) and
    one set of start states (f, k0) is drawn ONCE and reused across every cell,
    so cross-cell contrasts in (sigma, alpha/rho, eps) are not MC-confounded.
D3  Because the delegated policy is exactly independent of the weights (core.py
    fact S4), the delegated Markov chain is computed once per
    (DGP, control, eps, resource) and reused across all three sigma values.
D4  Start state x0 ~ Uniform[0,10]^3 exactly as the environment resets: the
    integer part k0 ~ U{0..9} indexes the lattice point and the fractional part
    f ~ U(0,1) fixes the lattice offset.
D5  Long-run W(realised) = Cesaro limit of the per-step welfare (primary).
    The T = 100 episode average (what the environment actually pays) is also
    reported for every cell.
D6  Objects reported per cell:
      G          = W*(planner) - W(realised)         [signed delegation gap]
      F = G_+    , B = G_-                            [friction / surplus]
      G_i        = per-principal gap (planner benchmark = the ENSEMBLE-optimal
                   lattice point, agent i's own welfare there)
      Ginc       = stakes-weighted incidence, sum_i s_i G_i / sum_i s_i with
                   s_i = sum_m w_im
      floor      = the planner's quantization floor, sum_m Wbar_m (xlat-sstar)^2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core as K  # noqa: E402

NDRAW = 1000
N_NOISE = 512
SIGMAS = [0.2, 0.6, 1.0]
ALPHAS = [-0.8, -0.4, 0.0, 0.4, 0.8]
RHOS = [-0.33, 0.0, 0.4, 0.8]
EPSILONS = [0.0, 0.25, 0.5, 1.0]
SEED = 20260730


def draw_commons(n=NDRAW, seed=SEED):
    rng = np.random.default_rng(seed)
    b = rng.standard_normal((n, K.N_RESOURCES))
    xi = rng.standard_normal((n, K.N_AGENTS, K.N_RESOURCES))
    z_eq = rng.standard_normal((n, K.N_AGENTS, K.N_RESOURCES))
    z_w = rng.standard_normal((n, K.N_AGENTS, K.N_RESOURCES))
    f = rng.uniform(1e-6, 1 - 1e-6, size=(n, K.N_RESOURCES))
    k0 = rng.integers(0, 10, size=(n, K.N_RESOURCES))
    return dict(b=b, xi=xi, z_eq=z_eq, z_w=z_w, f=f, k0=k0)


def lattices(f):
    """(NDRAW, 21) lattice values. Ordering is the same for every f in (0,1):
    position 2k -> integer k (k=0..10), position 2k+1 -> f+k (k=0..9).
    Verified in build_lattice_check()."""
    n = f.shape[0]
    L = np.empty((n, 21))
    for k in range(11):
        L[:, 2 * k] = k
    for k in range(10):
        L[:, 2 * k + 1] = f + k
    return L


def build_lattice_check():
    """The transition index is f-independent (see core.py S2). Assert it."""
    agg = np.arange(-K.N_AGENTS, K.N_AGENTS + 1, dtype=float)
    ref = None
    for f in (0.01, 0.3719, 0.5, 0.9999):
        L = K.lattice(f)
        Lm = lattices(np.array([f]))[0]
        assert np.abs(np.sort(L) - Lm).max() < 1e-12, "lattice ordering mismatch"
        idx = K.lattice_transition_index(Lm, agg)
        if ref is None:
            ref = idx
        assert np.array_equal(idx, ref), "transition index depends on f"
    return ref


TIDX = build_lattice_check()


def start_dist(k0_col):
    """One-hot at lattice position 2*k0+1, i.e. state value f + k0 = x0."""
    n = k0_col.shape[0]
    s = np.zeros((n, 21))
    s[np.arange(n), 2 * k0_col + 1] = 1.0
    return s


# --------------------------------------------------------------------------- #
def chain_for(tau_m, L, k0_col, eps, rule, bayes, rng):
    """Delegated transition matrix for one resource. tau_m: (NDRAW, A)."""
    if rule == "naive":
        th1, th2 = K.naive_thresholds(tau_m, eps, bayes)
        p = K.naive_action_probs(L[:, None, :], th1, th2, eps)   # (D,A,21,3)
        pmf = K.aggregate_delta_pmf(p)
        diag = {"nonconverged_frac": 0.0, "n_noise": 0}
    elif rule == "nash":
        nn = 1 if eps <= 0 else N_NOISE
        pmf, diag = K.stage_nash_pmf(L, tau_m, eps, rng, n_noise=nn, bayes=bayes)
    else:
        raise ValueError(rule)
    return K.transition_from_pmf(pmf, TIDX), diag


def cell_metrics(tau, w, occ, L):
    """occ: list of 3 (NDRAW, 21) occupation measures. Returns dict of arrays."""
    Wbar = w.sum(axis=1)                                   # (D, M)
    # Degenerate draw: w ~ N(sigma, 0.05) clipped at 0 can put ALL FOUR weights
    # on a resource at exactly 0 (probability Phi(-sigma/0.05)^4; ~6e-4 at
    # sigma = 0.05, ~1e-18 at sigma = 0.2, so it never occurs on the main grid).
    # That resource then contributes nothing to anyone's welfare: set sstar = 0
    # so the zero multiplier propagates instead of a NaN.
    degen = Wbar <= 0
    sstar = np.where(degen, 0.0, (w * tau).sum(axis=1) / np.where(degen, 1.0, Wbar))
    G = np.zeros(tau.shape[0])
    Gi = np.zeros((tau.shape[0], K.N_AGENTS))
    floor = np.zeros(tau.shape[0])
    for m in range(K.N_RESOURCES):
        o = occ[m]
        mx = (o * L[m]).sum(axis=1)
        mx2 = (o * L[m] ** 2).sum(axis=1)
        ss = sstar[:, m]
        e_track = mx2 - 2 * ss * mx + ss ** 2               # E_pi (x - sstar)^2
        d2 = (L[m] - ss[:, None]) ** 2
        j = d2.argmin(axis=1)
        xlat = L[m][np.arange(L[m].shape[0]), j]
        fl = (xlat - ss) ** 2
        G += Wbar[:, m] * (e_track - fl)
        floor += Wbar[:, m] * fl
        ti = tau[:, :, m]
        e_i = mx2[:, None] - 2 * ti * mx[:, None] + ti ** 2
        Gi += w[:, :, m] * (e_i - (xlat[:, None] - ti) ** 2)
    return G, Gi, floor


def summarise(name, G, Gi, floor, extra):
    s = {"design": name}
    s.update(extra)
    n = G.size
    s["G_mean"] = float(G.mean())
    s["G_se"] = float(G.std(ddof=1) / np.sqrt(n))
    s["G_median"] = float(np.median(G))
    s["F_mean"] = float(np.maximum(G, 0).mean())
    s["B_mean"] = float(np.maximum(-G, 0).mean())
    s["P_G_neg"] = float((G < 0).mean())
    s["floor_mean"] = float(floor.mean())
    for i in range(K.N_AGENTS):
        s[f"G_{i}_mean"] = float(Gi[:, i].mean())
    s["Gi_min_mean"] = float(Gi.min(axis=1).mean())
    s["Gi_max_mean"] = float(Gi.max(axis=1).mean())
    s["P_any_Gi_neg"] = float((Gi < 0).any(axis=1).mean())
    s["Gi_sum_check_maxabs"] = float(np.abs(Gi.sum(axis=1) - G).max())
    return s


def run(rule="naive", bayes=False, out_tag="naive", ndraw=NDRAW):
    com = draw_commons(ndraw)
    # one lattice per (draw, resource): offset f[d, m]
    Lm = [lattices(com["f"][:, m]) for m in range(K.N_RESOURCES)]
    starts = [start_dist(com["k0"][:, m]) for m in range(K.N_RESOURCES)]
    rows_long, rows_T = [], []
    diags = []
    rng = np.random.default_rng(SEED + 991)

    families = [("implemented", ALPHAS), ("renormalised", ALPHAS),
                ("equicorr", RHOS)]
    for fam, controls in families:
        for ctrl in controls:
            if fam == "implemented":
                tau = K.dgp_implemented(com["b"], com["xi"], ctrl)
            elif fam == "renormalised":
                tau = K.dgp_renormalised(com["b"], com["xi"], ctrl)
            else:
                tau = K.dgp_equicorr(com["z_eq"], ctrl)
            for eps in EPSILONS:
                t0 = time.time()
                occ_long, occ_T = [], []
                for m in range(K.N_RESOURCES):
                    P, dg = chain_for(tau[:, :, m], Lm[m], com["k0"][:, m],
                                      eps, rule, bayes, rng)
                    occ_long.append(K.cesaro_occupation(P, starts[m]))
                    occ_T.append(K.horizon_occupation(P, starts[m]))
                    dg.update(dict(family=fam, control=ctrl, eps=eps, m=m,
                                   rule=rule))
                    diags.append(dg)
                for sigma in SIGMAS:
                    w = K.weights_from_z(com["z_w"], sigma)
                    ex = dict(family=fam, control=ctrl, sigma=sigma, eps=eps,
                              rule=rule, belief="bayes" if bayes else "ce",
                              ndraw=ndraw)
                    G, Gi, fl = cell_metrics(tau, w, occ_long, Lm)
                    rows_long.append(summarise("cesaro", G, Gi, fl, ex))
                    G, Gi, fl = cell_metrics(tau, w, occ_T, Lm)
                    rows_T.append(summarise("T100", G, Gi, fl, ex))
                print(f"  {rule:5s} {fam:12s} c={ctrl:+.2f} eps={eps:.2f} "
                      f"{time.time()-t0:5.1f}s", flush=True)
    df = pd.DataFrame(rows_long)
    df.to_csv(HERE / f"G_grid_{out_tag}_cesaro.csv", index=False)
    pd.DataFrame(rows_T).to_csv(HERE / f"G_grid_{out_tag}_T100.csv", index=False)
    pd.DataFrame(diags).to_csv(HERE / f"diagnostics_{out_tag}.csv", index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", default="naive", choices=["naive", "nash"])
    ap.add_argument("--bayes", action="store_true")
    ap.add_argument("--ndraw", type=int, default=NDRAW)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    tag = a.tag or (a.rule + ("_bayes" if a.bayes else "_ce"))
    t0 = time.time()
    df = run(a.rule, a.bayes, tag, a.ndraw)
    print(f"\n{tag}: {len(df)} cells in {(time.time()-t0)/60:.1f} min")
