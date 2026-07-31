"""A12.  THE lambda-REDUCTION: the (sigma, alpha, eps) surface is really
(sigma_eff) x Psi(lambda, v_tau, eps), and for any UNIT-MARGINAL-VARIANCE DGP it
is the ONE-parameter Psi~(lambda, eps).

DEFINITIONS (all closed form, all checked here)
  v_tau  = Var(tau_i)                       marginal target variance
  rho    = Corr(tau_i, tau_j), i != j       cross-agent target correlation
  lambda^2 = v_tau (1 - rho)                the CONTENTION scale
           => Var(tau_i - taubar) = (n-1)/n * lambda^2 = (3/4) lambda^2  (n=4)
  Var(mu) = v_tau - (n-1)/n * lambda^2      the LOCATION scale (equal weights)

  implemented   tau_i = a b + (1-|a|) xi_i : v_tau = a^2 + (1-|a|)^2
                                             rho   = a^2 / v_tau
                                             lambda = 1 - |a|
                                             Var(mu) = a^2 + (1-|a|)^2 / 4
  renormalised  the same / sqrt(v_tau)     : v_tau = 1, lambda = sqrt(1-rho)
  equicorr      Sigma = (1-r)I + r 11'     : v_tau = 1, rho = r, lambda = sqrt(1-r)

CLAIM C1 (verified in R1).  The above are exact.
CLAIM C2 (verified in R2).  The delegated chain and the planner both depend on
  the draw only through (tau - mu) and mu, and the reward scale factorises, so
      E[G] = 3 * E[Wbar] * Psi(lambda, v_tau; eps)   with   E[Wbar] = n sigma_eff.
  For v_tau = 1 (renormalised, equicorr) Psi collapses to Psi~(lambda; eps).
  Test: renormalised at alpha and equicorr at rho(alpha) must give the SAME
  E[G] -- the same lambda, the same v_tau -- even though the DGPs differ.
CLAIM C3 (verified in R3).  Psi~(lambda; eps) = kappa_c(eps) lambda^2 + a0(eps)
  on the regime lambda >~ 0.45, with kappa_c bracketed by the order-statistic
  constants of a07.  Below that the perfect-coordination knife-edge takes over.
"""
import json
import numpy as np
from a00_lib import (CAP, N_AGENTS, draw_problem, gap_batch, rho_of_alpha,
                     targets_implemented, targets_renormalised, targets_equicorr)

OUT = {}
R = 30000
SIG = 0.6
n = N_AGENTS


def closed_form(dgp, knob):
    if dgp == "implemented":
        a = abs(knob)
        v = a ** 2 + (1 - a) ** 2
        lam = 1 - a
        rho = a ** 2 / v
    elif dgp == "renormalised":
        a = abs(knob)
        v = 1.0
        rho = a ** 2 / (a ** 2 + (1 - a) ** 2)
        lam = np.sqrt(1 - rho)
    elif dgp == "equicorr":
        v, rho = 1.0, float(knob)
        lam = np.sqrt(1 - rho)
    return v, rho, lam, v - (n - 1) / n * lam ** 2


print("=" * 84)
print("R1  the closed forms for (v_tau, rho, lambda, Var(mu)) vs Monte Carlo")
print("     dgp          knob |   v_tau          rho           lambda^2*3/4    Var(mu)")
rows = []
rng = np.random.default_rng(1212)
for dgp, knobs in (("implemented", (-0.8, -0.4, 0.0, 0.4, 0.8)),
                   ("renormalised", (0.0, 0.4, 0.8)),
                   ("equicorr", (-1 / 3, 0.0, 0.4, 0.8, 0.95))):
    for k in knobs:
        if dgp == "implemented":
            t = targets_implemented(rng.standard_normal(400000),
                                    rng.standard_normal((400000, n)), k)
        elif dgp == "renormalised":
            t = targets_renormalised(rng.standard_normal(400000),
                                     rng.standard_normal((400000, n)), k)
        else:
            t = targets_equicorr(rng.standard_normal((400000, n)), k)
        v, rho, lam, vmu = closed_form(dgp, k)
        C = np.cov(t.T)
        ev = np.mean(np.diag(C))
        erho = (C.sum() - np.trace(C)) / (n * (n - 1)) / ev
        tb = t.mean(1)
        edisp = ((t - tb[:, None]) ** 2).mean()
        print(f"    {dgp:13s}{k:+6.3f} | {v:6.4f}/{ev:6.4f}  {rho:+6.4f}/{erho:+6.4f}   "
              f"{0.75*lam**2:6.4f}/{edisp:6.4f}   {vmu:6.4f}/{tb.var():6.4f}")
        rows.append(dict(dgp=dgp, knob=float(k), v=v, rho=rho, lam=lam, vmu=vmu,
                         v_mc=float(ev), rho_mc=float(erho),
                         disp_mc=float(edisp), vmu_mc=float(tb.var())))
OUT["R1"] = rows

print("=" * 84)
print("R2  COLLAPSE: renormalised(alpha) vs equicorr(rho(alpha)) -- same (lambda, v_tau)")
print("    and implemented(alpha), which has the same lambda but a DIFFERENT v_tau.")
print("     alpha   lambda  |  v_tau  |     E[G] renorm     E[G] equicorr    E[G] implemented")


def cellG(dgp, knob, eps, protocol, seed=909, R=R, sigma=SIG):
    rng = np.random.default_rng(seed)
    tot, tot2 = 0.0, 0.0
    for _ in range(3):
        t, w, x0 = draw_problem(rng, R, dgp, knob, sigma)
        g = gap_batch(t, w, x0, eps, protocol=protocol)
        tot += g["G"].mean()
        tot2 += g["G"].var() / R
    return tot, np.sqrt(tot2)


coll = []
for proto in ("naive", "sne"):
    for eps in (0.0, 0.25):
        print(f"  --- protocol={proto}  eps={eps} ---")
        for a in (0.2, 0.4, 0.6, 0.8):
            rho = rho_of_alpha(a)
            lam = np.sqrt(1 - rho)
            gr, sr = cellG("renormalised", a, eps, proto)
            ge, se = cellG("equicorr", rho, eps, proto)
            gi, si = cellG("implemented", a, eps, proto)
            z = (gr - ge) / np.hypot(sr, se)
            print(f"     {a:+.2f}   {lam:6.4f}  |  1.000  |  {gr:7.4f}+-{sr:.4f}"
                  f"   {ge:7.4f}+-{se:.4f}   {gi:7.4f}+-{si:.4f}   |z|={abs(z):5.2f}")
            coll.append(dict(proto=proto, eps=eps, alpha=a, rho=float(rho),
                             lam=float(lam), G_renorm=gr, G_equi=ge, G_impl=gi,
                             se_r=sr, se_e=se, z=float(z)))
OUT["R2"] = coll

print("=" * 84)
print("R3  Psi~(lambda; eps) on the UNIT-VARIANCE family (equicorr), and the")
print("    quadratic law.  Psi~ := E[G] / (3 * E[Wbar]),  E[Wbar] = 4*sigma_eff.")
EWbar = 4 * 0.6                                   # sigma_eff(0.6) = 0.6 exactly
lams = np.array([0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.87, 1.00, 1.15])
rhos = 1 - lams ** 2                              # equicorr knob; rho >= -1/3 ok
fits = {}
for proto in ("naive", "sne"):
    for eps in (0.0, 0.01, 0.25, 1.0):
        Y = []
        for r in rhos:
            if r < -1 / 3 - 1e-12:
                Y.append(np.nan)
                continue
            g, _ = cellG("equicorr", float(r), eps, proto)
            Y.append(g / (3 * EWbar))
        Y = np.array(Y)
        m = (lams >= 0.45) & np.isfinite(Y)
        A = np.column_stack([np.ones(m.sum()), lams[m] ** 2])
        b, *_ = np.linalg.lstsq(A, Y[m], rcond=None)
        pred = A @ b
        r2 = 1 - ((Y[m] - pred) ** 2).sum() / ((Y[m] - Y[m].mean()) ** 2).sum()
        maxres = np.abs(Y[m] - pred).max()
        print(f"  [{proto:5s} eps={eps:4.2f}]  Psi~ = " +
              " ".join(f"{y:6.4f}" for y in Y))
        print(f"  {'':16s}  a0={b[0]:+7.4f}  kappa_c={b[1]:+7.4f}  R^2={r2:7.5f}"
              f"  max|resid|={maxres:.4f}  (rel {100*maxres/max(Y[m].max(),1e-9):.1f}%)")
        fits[f"{proto}|{eps}"] = dict(a0=float(b[0]), kappa_c=float(b[1]),
                                      r2=float(r2), maxres=float(maxres),
                                      lams=lams.tolist(), Psi=Y.tolist())
OUT["R3"] = fits
print("    order-statistic bracket for kappa_c (a07): Var(med-mean)=0.0482 (attractor")
print("    at the median of tau) .. E[(tau_(2)-mean)^2]=0.1986 (attractor at an order")
print("    statistic); anything ABOVE 0.1986 is the variance channel, not the bias one.")

with open("a12_results.json", "w") as fh:
    json.dump(OUT, fh, indent=1)
print("\nwrote a12_results.json")
