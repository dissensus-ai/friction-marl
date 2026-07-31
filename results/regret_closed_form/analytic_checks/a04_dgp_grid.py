"""E[G](sigma, alpha, eps) on the 5x5x5 design grid under the THREE DGPs, plus
the first-order expansion at (alpha, eps) = (0,0) and the head-to-head against
the canonical functional F = sigma (1+eps)/(1+alpha).

DGP (i)   implemented   : tau_i = alpha b + (1-|alpha|) xi_i
DGP (ii)  renormalised  : tau_i = (alpha b + (1-|alpha|) xi_i)/sqrt(alpha^2+(1-|alpha|)^2)
DGP (iii) equicorr      : tau ~ N(0, (1-rho) I + rho 11'), rho IS the pairwise corr

Everything is the EXACT 21-state chain per resource; the only stochastic element
is the Monte Carlo average over preference draws (SE reported).
"""
from __future__ import annotations

import json
import numpy as np

from a00_lib import (CAP, N_AGENTS, N_RES, gap_batch, targets_equicorr,
                     targets_implemented, targets_renormalised, rho_of_alpha)

ALPHAS = [-0.8, -0.4, 0.0, 0.4, 0.8]
SIGMAS = [0.2, 0.4, 0.6, 0.8, 1.0]
EPSILONS = [0.0, 0.25, 0.5, 0.75, 1.0]
RHOS = [-1 / 3, -0.2, 0.0, 0.2, 0.3077, 0.6, 0.9412]
R = 20000
EPS0 = 1e-9
OUT = {}


def make_targets(dgp, knob, rng, R, n=N_AGENTS):
    if dgp == "implemented":
        b = rng.standard_normal(R)
        xi = rng.standard_normal((R, n))
        return targets_implemented(b, xi, knob)
    if dgp == "renormalised":
        b = rng.standard_normal(R)
        xi = rng.standard_normal((R, n))
        return targets_renormalised(b, xi, knob)
    if dgp == "equicorr":
        return targets_equicorr(rng.standard_normal((R, n)), knob)
    raise ValueError(dgp)


def cell(dgp, knob, sigma, eps, protocol, seed=777, R=R):
    """Total E[G] over all 3 resources (they are i.i.d. replicas)."""
    rng = np.random.default_rng(seed)
    t = make_targets(dgp, knob, rng, R)
    w = np.clip(sigma + 0.05 * rng.standard_normal((R, N_AGENTS)), 0.0, 1.0)
    x0 = rng.uniform(0.0, CAP, size=R)
    # SNE is used for EVERY eps including 0 (evaluated at EPS0 = 1e-9, its
    # eps->0+ limit, which is a pure Nash selection) so that the eps profile has
    # no protocol discontinuity.  The exact Nash selection bracket at eps=0 is
    # reported separately in a05.
    proto = "sne" if protocol == "nash" else "naive"
    g = gap_batch(t, w, x0, max(eps, 1e-9) if proto == "sne" else eps,
                  protocol=proto)
    G = g["G"] * N_RES                      # 3 i.i.d. resources
    return G.mean(), G.std(ddof=1) / np.sqrt(R)


# --------------------------------------------------------------------------- #
print("=" * 78)
print("1.  E[G] on the 5x5x5 grid, IMPLEMENTED DGP, protocol = stage-Nash")
grid = {}
for proto in ("nash", "naive"):
    print(f"\n  --- protocol = {proto} ---")
    print("   sigma \\ (alpha,eps)")
    for alpha in ALPHAS:
        for eps in EPSILONS:
            for sigma in SIGMAS:
                m, se = cell("implemented", alpha, sigma, eps, proto)
                grid[f"{proto}|{alpha}|{sigma}|{eps}"] = (m, se)
        row = "  ".join(f"{grid[f'{proto}|{alpha}|0.6|{e}'][0]:7.3f}" for e in EPSILONS)
        print(f"   alpha={alpha:+.1f} (sigma=0.6): eps-> {row}")
OUT["grid_implemented"] = grid

# --------------------------------------------------------------------------- #
print("\n" + "=" * 78)
print("2.  ALPHA PROFILE at sigma=0.6, eps=0 under all three DGPs")
print("      knob      implemented    renormalised     equicorr(rho=knob)")
prof = {}
for proto in ("nash", "naive"):
    print(f"  --- {proto} ---")
    for a in ALPHAS:
        r1 = cell("implemented", a, 0.6, 0.0, proto)
        r2 = cell("renormalised", a, 0.6, 0.0, proto)
        prof[f"{proto}|{a}"] = (r1, r2)
        print(f"      a={a:+.1f}   {r1[0]:8.4f}+-{r1[1]:.4f}   {r2[0]:8.4f}+-{r2[1]:.4f}")
    for rho in RHOS:
        r3 = cell("equicorr", rho, 0.6, 0.0, proto)
        prof[f"{proto}|rho{rho:.4f}"] = r3
        print(f"      rho={rho:+.4f} equicorr:  {r3[0]:8.4f}+-{r3[1]:.4f}")
OUT["alpha_profiles"] = prof

# --------------------------------------------------------------------------- #
print("\n" + "=" * 78)
print("3.  FIRST-ORDER EXPANSION AT (alpha, eps) = (0,0), sigma fixed at 0.6")
h = 0.02
for proto in ("nash", "naive"):
    print(f"  --- {proto} ---")
    base = cell("implemented", 0.0, 0.6, 0.0, proto, R=60000)
    for dgp in ("implemented", "renormalised"):
        lo = cell(dgp, -h, 0.6, 0.0, proto, R=60000)
        hi = cell(dgp, +h, 0.6, 0.0, proto, R=60000)
        d_r = (hi[0] - base[0]) / h
        d_l = (base[0] - lo[0]) / h
        print(f"      [{dgp:12s}] E[G](0)={base[0]:.4f}   right-slope in alpha ="
              f" {d_r:+.4f}   left-slope = {d_l:+.4f}"
              f"   two-sided? {'YES' if abs(d_r-d_l) < 3*(base[1]+hi[1]+lo[1])/h else 'NO (KINK)'}")
    lo = cell("equicorr", -h, 0.6, 0.0, proto, R=60000)
    hi = cell("equicorr", +h, 0.6, 0.0, proto, R=60000)
    print(f"      [equicorr rho] slope dE[G]/drho = {(hi[0]-lo[0])/(2*h):+.4f}"
          f"   (right {(hi[0]-base[0])/h:+.4f}, left {(base[0]-lo[0])/h:+.4f})")
    # eps slope
    for e in (0.001, 0.004, 0.01, 0.04):
        v = cell("implemented", 0.0, 0.6, e, proto, R=60000)
        print(f"      eps={e:6.3f}: E[G]={v[0]:.4f}  dE/deps={(v[0]-base[0])/e:9.3f}"
              f"   dE/d(sqrt eps)={(v[0]-base[0])/np.sqrt(e):8.4f}")

# --------------------------------------------------------------------------- #
print("\n" + "=" * 78)
print("4.  HEAD-TO-HEAD vs F = sigma (1+eps)/(1+alpha) on the design grid")
for proto in ("nash", "naive"):
    y, Fc, cols = [], [], []
    for alpha in ALPHAS:
        for sigma in SIGMAS:
            for eps in EPSILONS:
                m, _ = grid[f"{proto}|{alpha}|{sigma}|{eps}"]
                y.append(m)
                Fc.append(sigma * (1 + eps) / (1 + alpha))
                cols.append((alpha, sigma, eps))
    y = np.array(y)
    Fc = np.array(Fc)
    A = np.column_stack([np.ones_like(Fc), Fc])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ beta
    R2 = 1 - r.var() / y.var()
    print(f"  --- {proto} ---  best AFFINE reparameterisation a + b*F :"
          f" a={beta[0]:+.4f} b={beta[1]:+.4f}  R^2 = {R2:.4f}")
    # partial: does F get the SIGN of each factor right?
    for name, idx in (("alpha", 0), ("sigma", 1), ("eps", 2)):
        lv = sorted(set(c[idx] for c in cols))
        means = [np.mean([y[k] for k in range(len(y)) if cols[k][idx] == v]) for v in lv]
        fmeans = [np.mean([Fc[k] for k in range(len(y)) if cols[k][idx] == v]) for v in lv]
        print(f"      {name:6s}: E[G] marginal means "
              f"{['%.3f' % m for m in means]}")
        print(f"      {'':6s}  F     marginal means "
              f"{['%.3f' % m for m in fmeans]}")
    OUT[f"affine_{proto}"] = dict(a=float(beta[0]), b=float(beta[1]), R2=float(R2))

# --------------------------------------------------------------------------- #
print("\n" + "=" * 78)
print("5.  ISO-F FAN TEST: do cells with equal F have equal E[G]?")
for proto in ("nash",):
    buckets = {}
    for alpha in ALPHAS:
        for sigma in SIGMAS:
            for eps in EPSILONS:
                F = round(sigma * (1 + eps) / (1 + alpha), 6)
                buckets.setdefault(F, []).append(
                    (grid[f"{proto}|{alpha}|{sigma}|{eps}"][0], (alpha, sigma, eps)))
    big = {k: v for k, v in buckets.items() if len(v) >= 3}
    print(f"  {len(big)} iso-F levels with >=3 cells (as the measurement spec claims)")
    wr, tot = [], []
    for F, vs in sorted(big.items()):
        g = np.array([v[0] for v in vs])
        wr.append(g.std())
        tot.extend(g)
        print(f"      F={F:6.3f}  n={len(vs)}  E[G] range [{g.min():7.3f},{g.max():7.3f}]"
              f"  within-F SD={g.std():6.3f}")
    print(f"  mean within-iso-F SD = {np.mean(wr):.4f} vs total SD = {np.std(tot):.4f}"
          f"  -> iso-F does NOT collapse E[G]" if np.mean(wr) > 0.3 * np.std(tot)
          else "  -> iso-F collapses E[G]")

with open("a04_results.json", "w") as f:
    json.dump(OUT, f, indent=1, default=float)
print("\nwrote a04_results.json")
