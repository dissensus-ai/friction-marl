"""Corrected three-term decomposition, protocol continuity at eps -> 0,
Nash-selection bracketing, the lambda^2 law, and the sqrt(eps) law.

DECOMPOSITION (exact identity, proved in the memo):
    G_m / Wbar_m = Var_pi(x)          [noise-response variance]
                 + bdisp^2            [contention / attractor displacement]
                 + 2 * qres * bdisp   [quantization-floor x displacement]
  where qres = xstar - mu  (signed quantization/boundary residual of the planner)
        bdisp = E_pi[x] - xstar       (displacement of the delegated attractor)
"""
from __future__ import annotations

import json
import numpy as np

from a00_lib import (CAP, N_AGENTS, agent_probs, gap_batch, state_values,
                     targets_equicorr, targets_renormalised, rho_of_alpha)

R = 8000
OUT = {}
EPS0 = 1e-4          # SNE evaluated here = the eps -> 0+ limit (a Nash selection)


def draw_lg(rng, R, lam, gamma, sigma, n=N_AGENTS):
    z = rng.standard_normal((R, n))
    e = lam * (z - z.mean(axis=1, keepdims=True)) / np.sqrt((n - 1) / n)
    c = gamma * rng.standard_normal((R, 1))
    t = c + e
    w = np.clip(sigma + 0.05 * rng.standard_normal((R, n)), 0.0, 1.0)
    return t, w, rng.uniform(0.0, CAP, size=R)


def run(lam, gamma, eps, sigma=0.6, protocol="sne", seed=1, R=R, **kw):
    rng = np.random.default_rng(seed)
    t, w, x0 = draw_lg(rng, R, lam, gamma, sigma)
    e = EPS0 if (protocol == "sne" and eps == 0) else eps
    return gap_batch(t, w, x0, e, protocol=protocol, **kw)


# --------------------------------------------------------------------------- #
print("=" * 78)
print("D0  SNE fixed-point convergence, and continuity of SNE at eps -> 0+")
rng = np.random.default_rng(11)
t, w, x0 = draw_lg(rng, 2000, 1.0, 1.0, 0.6)
xv = state_values(x0 - np.floor(x0))
for eps in (1e-4, 0.01, 0.25, 1.0):
    _, _, res = agent_probs(t, xv, eps, protocol="sne", report_conv=True)
    print(f"    eps={eps:9.2e}  max |c_new - c| after 600 damped iters = {res:.3e}")
print("    E[G] as eps -> 0+ (SNE) vs the exact Nash bracket at eps = 0:")
for eps in (1e-4, 1e-3, 1e-2, 5e-2):
    g = gap_batch(t, w, x0, eps, protocol="sne")["G"].mean()
    print(f"      SNE eps={eps:7.1e}: {g:.5f}")
gb = gap_batch(t, w, x0, 0.0, protocol="nash", selection="welfare_best")["G"].mean()
gw = gap_batch(t, w, x0, 0.0, protocol="nash", selection="welfare_worst")["G"].mean()
gm = gap_batch(t, w, x0, 0.0, protocol="nash", selection="minimal")["G"].mean()
print(f"      exact Nash, welfare-BEST  selection : {gb:.5f}")
print(f"      exact Nash, MINIMAL-move  selection : {gm:.5f}")
print(f"      exact Nash, welfare-WORST selection : {gw:.5f}   <-- selection matters")
OUT["nash_bracket"] = dict(best=gb, minimal=gm, worst=gw)

# --------------------------------------------------------------------------- #
print("=" * 78)
print("D1  EXACT DECOMPOSITION.  G/Wbar = Var + bdisp^2 + 2 qres bdisp")
print("  proto  lam  gamma  eps |   E[G]   |   Wbar*Var   Wbar*bdisp^2  2Wbar*q*b |"
      " identity")
rows = []
for proto in ("sne", "naive"):
    for lam, gamma in ((0.87, 0.5), (0.21, 0.98)):
        for eps in (0.0, 0.25, 1.0):
            g = run(lam, gamma, eps, protocol=proto, seed=51)
            Wb = g["Wbar"]
            A = (Wb * g["var"]).mean()
            B = (Wb * g["bdisp"] ** 2).mean()
            Cx = (2 * Wb * g["qres"] * g["bdisp"]).mean()
            tot = A + B + Cx
            # exact identity holds pointwise up to the planner's transient variance
            resid = (g["G"] - Wb * (g["var"] + g["bdisp"] ** 2
                                    + 2 * g["qres"] * g["bdisp"])).mean()
            print(f"  {proto:5s} {lam:5.2f} {gamma:5.2f} {eps:4.2f} | {g['G'].mean():8.4f} |"
                  f" {A:11.4f} {B:12.4f} {Cx:11.4f} | sum={tot:8.4f}"
                  f" resid={resid:+.4f}")
            rows.append(dict(proto=proto, lam=lam, gamma=gamma, eps=eps,
                             G=g["G"].mean(), var=A, disp=B, cross=Cx,
                             resid=resid))
OUT["decomposition"] = rows
print("  (resid = the planner's own transient variance Wbar*var_p, i.e. the cost of")
print("   the <=6 steps it spends travelling to its optimum; it is NOT part of the")
print("   steady-state decomposition.)")

# --------------------------------------------------------------------------- #
print("=" * 78)
print("D2  THE lambda^2 LAW (eps = 0, SNE) after removing the lattice-access offset")
lams = [0.0, 0.15, 0.3, 0.45, 0.6, 0.87, 1.2, 1.6, 2.2]
vals = []
for lam in lams:
    g = run(lam, 0.5, 0.0, protocol="sne", seed=61)
    vals.append(g["G"].mean())
    print(f"    lam={lam:4.2f}  E[G]={vals[-1]:8.5f}   (excess over lam=0:"
          f" {vals[-1]-vals[0]:8.5f})")
L = np.array(lams[1:])
Y = np.array(vals[1:]) - vals[0]
p = np.polyfit(np.log(L), np.log(Y), 1)
print(f"    power-law on the OFFSET-REMOVED excess: exponent = {p[0]:.3f}"
      f"   (theory: 2)")
m = L >= 0.6
p2 = np.polyfit(np.log(L[m]), np.log(Y[m]), 1)
print(f"    restricted to lam >= 0.6 (deadzone negligible): exponent = {p2[0]:.3f}")
OUT["lambda_law"] = dict(lams=lams, vals=vals, exp_all=float(p[0]),
                         exp_large=float(p2[0]))

# --------------------------------------------------------------------------- #
print("=" * 78)
print("D3  THE eps CHANNEL under a SINGLE protocol (no eps=0 discontinuity)")
epss = [0.0, 0.005, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
for proto in ("sne", "naive"):
    print(f"  --- {proto} ---")
    ys = []
    for eps in epss:
        g = run(0.87, 0.5, eps, protocol=proto, seed=71)
        ys.append(g["G"].mean())
        print(f"    eps={eps:5.3f} sd={np.sqrt(eps):5.3f}  E[G]={ys[-1]:8.4f}"
              f"   Var-term={(g['Wbar']*g['var']).mean():7.4f}"
              f"   disp-term={(g['Wbar']*g['bdisp']**2).mean():7.4f}")
    E = np.array(epss)
    Y = np.array(ys)
    d = Y - Y[0]
    m = (E > 0) & (E <= 0.05)
    if np.all(d[m] > 0):
        q = np.polyfit(np.log(E[m]), np.log(d[m]), 1)
        print(f"    small-eps exponent p in (E[G](eps)-E[G](0)) ~ eps^p : {q[0]:.3f}"
              f"    [p = 1/2  <=>  LINEAR IN THE NOISE SD sqrt(eps)]")
        OUT[f"eps_exponent_{proto}"] = float(q[0])
    else:
        print(f"    increment is NEGATIVE at small eps -> dE[G]/deps < 0 for {proto}")
        OUT[f"eps_exponent_{proto}"] = None
    OUT[f"eps_profile_{proto}"] = [float(v) for v in Y]

# --------------------------------------------------------------------------- #
print("=" * 78)
print("D4  IDENTITY: renormalised DGP == equicorr DGP at rho(alpha)?")
for alpha in (0.4, 0.8):
    rho = rho_of_alpha(alpha)
    r1 = np.random.default_rng(5)
    b = r1.standard_normal(120000)
    xi = r1.standard_normal((120000, 4))
    t1 = targets_renormalised(b, xi, alpha)
    r2 = np.random.default_rng(6)
    t2 = targets_equicorr(r2.standard_normal((120000, 4)), rho)
    print(f"    alpha={alpha}: rho={rho:.4f}  renorm(var,corr)="
          f"({t1.var():.4f},{np.corrcoef(t1.T)[0,1]:.4f})  "
          f"equicorr=({t2.var():.4f},{np.corrcoef(t2.T)[0,1]:.4f})")
    for proto in ("sne",):
        w = np.clip(0.6 + 0.05 * r1.standard_normal((20000, 4)), 0, 1)
        x0 = r1.uniform(0, CAP, 20000)
        g1 = gap_batch(t1[:20000], w, x0, EPS0, protocol=proto)["G"]
        g2 = gap_batch(t2[:20000], w, x0, EPS0, protocol=proto)["G"]
        print(f"      E[G] renormalised={g1.mean():.5f}+-{g1.std()/141:.5f}   "
              f"equicorr={g2.mean():.5f}+-{g2.std()/141:.5f}")

with open("a05_results.json", "w") as f:
    json.dump(OUT, f, indent=1, default=float)
print("\nwrote a05_results.json")
