"""The reduced gap functional Psi(lambda, gamma, eps) and its scaling laws.

Translation structure.  Write tau_i = c + e_i, c = mean_i tau_i, sum_i e_i = 0.
In the INTERIOR the whole problem is translation-equivariant in (x, tau), so G
depends on the target draw only through the centred deviations e.  The clamp at
[0, C] breaks this, and re-introduces a dependence on the common level c.  So the
reduced object is
        E[G] = n * sigma_eff(sigma) * Psi(lambda, gamma, eps) * (#resources)
with lambda = sd of the centred deviations scale, gamma = sd of the common level.

Checks run here
  C1  sigma-homogeneity of E[G] (degree 1), and the size of the clip correction.
  C2  the delegated behaviour is EXACTLY independent of sigma (chain identity).
  C3  Psi(lambda, ., 0): shape in lambda for both protocols.
  C4  the eps channel: is the leading term sqrt(eps) (= noise SD) or eps?
  C5  the three-term decomposition bias^2 / variance / quantization floor.
  C6  Nash post-state uniqueness + selection invariance + tie-rule sensitivity.
"""
from __future__ import annotations

import json
import numpy as np

from a00_lib import (CAP, N_AGENTS, gap_batch, draw_problem, nash_pmf_wrap,
                     state_values, N_RES)

rng = np.random.default_rng(4242)
R = 12000
RES = {}


def draw_lam_gamma(rng, R, lam, gamma, sigma, n=N_AGENTS):
    """Targets with centred-deviation scale lam and common-level sd gamma."""
    z = rng.standard_normal((R, n))
    e = lam * (z - z.mean(axis=1, keepdims=True))
    c = gamma * rng.standard_normal((R, 1))
    t = c + e
    w = np.clip(sigma + 0.05 * rng.standard_normal((R, n)), 0.0, 1.0)
    x0 = rng.uniform(0.0, CAP, size=R)
    return t, w, x0


def EG(lam, gamma, eps, sigma=0.6, protocol="naive", seed=1, R=R, **kw):
    r = np.random.default_rng(seed)
    t, w, x0 = draw_lam_gamma(r, R, lam, gamma, sigma)
    g = gap_batch(t, w, x0, eps, protocol=protocol, **kw)
    return g


# --------------------------------------------------------------------------- #
print("=" * 78)
print("C6  Nash correspondence: post-state uniqueness, selection, tie rule")
t, w, x0 = draw_lam_gamma(np.random.default_rng(5), 2000, 1.0, 1.0, 0.6)
xv = state_values(x0 - np.floor(x0))
mu = (w * t).sum(1) / w.sum(1)
_, isna = nash_pmf_wrap(t, xv, mu)
# distinct POST-STATES among the Nash aggregates
ys = np.stack([np.clip(xv + D, 0.0, CAP) for D in range(-4, 5)], axis=-1)
ny = np.array([[len(set(np.round(ys[r, s][isna[r, s]], 9)))
                for s in range(xv.shape[1])] for r in range(200)])
print(f"    #distinct Nash AGGREGATES : mean {isna.sum(2).mean():.3f}  max {isna.sum(2).max()}")
print(f"    #distinct Nash POST-STATES: mean {ny.mean():.4f}  max {ny.max()}"
      f"   -> multiplicity is payoff-irrelevant (absorbed by the clamp)")
gb = gap_batch(t, w, x0, 0.0, protocol="nash", selection="welfare_best")
gw = gap_batch(t, w, x0, 0.0, protocol="nash", selection="welfare_worst")
print(f"    E[G] welfare-BEST selection  = {gb['G'].mean():.6f}")
print(f"    E[G] welfare-WORST selection = {gw['G'].mean():.6f}"
      f"   (identical: {np.allclose(gb['G'], gw['G'])})")
for eps in (0.0, 0.5):
    a = gap_batch(t, w, x0, eps, protocol="naive", clamp_aware=True)["G"].mean()
    b = gap_batch(t, w, x0, eps, protocol="naive", clamp_aware=False)["G"].mean()
    print(f"    eps={eps}: E[G] clamp-aware {a:.4f}  vs clamp-unaware {b:.4f}"
          f"  ({100*(b/a-1):+.1f}%)")

# --------------------------------------------------------------------------- #
print("=" * 78)
print("C1 / C2  sigma-homogeneity")
rows = []
for proto in ("naive", "nash"):
    base = None
    for sigma in (0.2, 0.4, 0.6, 0.8, 1.0):
        g = EG(1.0, 1.0, 0.0, sigma=sigma, protocol=proto, seed=9, R=8000)
        v = g["G"].mean()
        if base is None:
            base = v / 0.2
        rows.append((proto, sigma, v, v / (sigma * base)))
        print(f"    [{proto:5s}] sigma={sigma:.1f}  E[G]={v:9.5f}"
              f"   E[G]/(sigma*k)={v/(sigma*base):.5f}")
RES["sigma_homogeneity"] = rows

# chain identity: the transition matrix must not depend on sigma at all
t1, w1, x01 = draw_lam_gamma(np.random.default_rng(3), 500, 1.0, 1.0, 0.2)
t2, w2, _ = draw_lam_gamma(np.random.default_rng(3), 500, 1.0, 1.0, 1.0)
o1 = gap_batch(t1, w1, x01, 0.5, protocol="naive")["occ_d"]
o2 = gap_batch(t2, w2, x01, 0.5, protocol="naive")["occ_d"]
print(f"    C2 delegated occupancy identical across sigma=0.2 vs 1.0: "
      f"{np.allclose(o1, o2)}  (max|diff| = {np.abs(o1-o2).max():.2e})")

# --------------------------------------------------------------------------- #
print("=" * 78)
print("C3  Psi(lambda) at eps = 0, gamma = 1  (per resource, sigma = 0.6)")
lams = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.83, 1.0, 1.15, 1.4]
tab = []
for lam in lams:
    gn = EG(lam, 1.0, 0.0, protocol="naive", seed=17)
    ge = EG(lam, 1.0, 0.0, protocol="nash", seed=17)
    tab.append((lam, gn["G"].mean(), ge["G"].mean(),
                gn["bias2"].mean(), gn["var"].mean(), gn["q2"].mean(),
                ge["bias2"].mean(), ge["var"].mean(), ge["q2"].mean()))
    print(f"    lam={lam:5.2f}  naive E[G]={tab[-1][1]:8.4f}   Nash E[G]={tab[-1][2]:8.5f}")
RES["psi_lambda_eps0"] = tab

# fit the Nash branch to a*lam^p
L = np.array([r[0] for r in tab][1:])
Y = np.array([r[2] for r in tab][1:])
p = np.polyfit(np.log(L), np.log(Y), 1)
print(f"    Nash branch power-law fit over lam in [0.05,1.4]: E[G] ~ lam^{p[0]:.3f}")
sub = (L >= 0.4)
p2 = np.polyfit(np.log(L[sub]), np.log(Y[sub]), 1)
print(f"    Nash branch, lam >= 0.4 only:                    E[G] ~ lam^{p2[0]:.3f}")
RES["nash_power"] = [float(p[0]), float(p2[0])]

# --------------------------------------------------------------------------- #
print("=" * 78)
print("C4  the eps channel: sqrt(eps) (noise SD) or eps (noise variance)?")
epss = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
tab2 = []
for eps in epss:
    gn = EG(1.0, 1.0, eps, protocol="naive", seed=23)
    proto2 = "nash" if eps == 0 else "sne"
    ge = EG(1.0, 1.0, eps, protocol=proto2, seed=23)
    tab2.append((eps, gn["G"].mean(), ge["G"].mean(), ge["var"].mean(),
                 ge["bias2"].mean()))
    print(f"    eps={eps:5.3f} (sd={np.sqrt(eps):5.3f})  naive E[G]={tab2[-1][1]:8.4f}"
          f"   Nash/SNE E[G]={tab2[-1][2]:8.4f}   [var={tab2[-1][3]:7.4f}"
          f" bias2={tab2[-1][4]:7.4f}]")
RES["psi_eps"] = tab2

E = np.array([r[0] for r in tab2])
Yn = np.array([r[1] for r in tab2])
Ye = np.array([r[2] for r in tab2])
for nm, Y in (("naive", Yn), ("SNE", Ye)):
    m = (E > 0) & (E <= 0.1)
    q = np.polyfit(np.log(E[m]), np.log(Y[m] - Y[0]), 1)
    print(f"    [{nm}] small-eps increment (E[G](eps)-E[G](0)) ~ eps^{q[0]:.3f}"
          f"   -> exponent 0.5 means LINEAR IN THE NOISE SD")
    RES[f"eps_power_{nm}"] = float(q[0])

# --------------------------------------------------------------------------- #
print("=" * 78)
print("C5  three-term decomposition at the grid corners (sigma=0.6, gamma=1)")
print("     protocol  lam   eps    E[G]     E[Wbar*bias^2]  E[Wbar*var]  -E[Wbar*q^2]")
for proto, eps in (("nash", 0.0), ("sne", 0.25), ("sne", 1.0),
                   ("naive", 0.0), ("naive", 0.25), ("naive", 1.0)):
    for lam in (0.24, 1.0):
        g = EG(lam, 1.0, eps, protocol=proto, seed=31)
        Wb = g["Wbar"]
        print(f"     {proto:8s} {lam:4.2f} {eps:5.2f} {g['G'].mean():8.4f}"
              f"   {(Wb*g['bias2']).mean():12.4f} {(Wb*g['var']).mean():12.4f}"
              f" {-(Wb*g['q2']).mean():12.4f}")

# --------------------------------------------------------------------------- #
print("=" * 78)
print("C3b  gamma (common-level) sensitivity -- how much does the wall matter?")
for gamma in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5):
    gn = EG(1.0, gamma, 0.0, protocol="naive", seed=41)
    ge = EG(1.0, gamma, 0.0, protocol="nash", seed=41)
    print(f"    gamma={gamma:4.2f}  naive E[G]={gn['G'].mean():8.4f}"
          f"   Nash E[G]={ge['G'].mean():8.5f}")

with open("a03_results.json", "w") as f:
    json.dump(RES, f, indent=1, default=float)
print("\nwrote a03_results.json")
