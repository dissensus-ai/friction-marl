"""Exact pure-strategy Nash correspondence (eps = 0) on the THREE real DGPs.

No fixed-point iteration -> no convergence problem.  Three selection rules give
the bracket.  This answers: does strategic sophistication flip the sign of the
alignment effect that the naive protocol shows?
"""
import numpy as np
from a00_lib import (CAP, N_AGENTS, N_RES, gap_batch, targets_equicorr,
                     targets_implemented, targets_renormalised, rho_of_alpha)

R, SIG, SEED = 12000, 0.6, 20260730
ALPHAS = [-0.8, -0.4, 0.0, 0.4, 0.8]
RHOS = [-1/3, -0.25, -0.1, 0.0, 0.1, 0.3077, 0.5, 0.7, 0.9412, 0.99]

def targets(dgp, k, rng, R, n=N_AGENTS):
    if dgp == "implemented":
        return targets_implemented(rng.standard_normal(R), rng.standard_normal((R,n)), k)
    if dgp == "renormalised":
        return targets_renormalised(rng.standard_normal(R), rng.standard_normal((R,n)), k)
    return targets_equicorr(rng.standard_normal((R,n)), k)

def run(dgp, k, sel):
    rng = np.random.default_rng(SEED)
    t = targets(dgp, k, rng, R)
    w = np.clip(SIG + 0.05*rng.standard_normal((R, N_AGENTS)), 0, 1)
    x0 = rng.uniform(0, CAP, R)
    g = gap_batch(t, w, x0, 0.0, protocol="nash", selection=sel)["G"]*N_RES
    return g.mean(), g.std(ddof=1)/np.sqrt(R)

print("="*76)
print("N1  EXACT stage-Nash, eps=0, sigma=0.6.  E[G] under three selection rules")
print("      knob            welfare-BEST        MINIMAL-move       welfare-WORST")
for dgp, knobs in (("implemented", ALPHAS), ("renormalised", ALPHAS),
                   ("equicorr", RHOS)):
    print(f"  --- {dgp} ---")
    for k in knobs:
        vals = [run(dgp, k, s) for s in ("welfare_best", "minimal", "welfare_worst")]
        rho = rho_of_alpha(k) if dgp != "equicorr" else k
        print(f"    {k:+.4f} (rho={rho:+.4f})  " + "  ".join(
            f"{m:9.4f}+-{s:.4f}" for m, s in vals))

print("="*76)
print("N2  |alpha|-SYMMETRY of the implemented and renormalised DGPs")
print("    Exact theorem: tau_i = alpha*b + (1-|alpha|)*xi_i with b ~ N(0,1)")
print("    symmetric  =>  law of (tau_1..tau_n) is INVARIANT under alpha -> -alpha.")
print("    Hence E[G](alpha) = E[G](-alpha) identically, for ANY protocol and any eps.")
for dgp in ("implemented", "renormalised"):
    for a in (0.4, 0.8):
        p = run(dgp, +a, "welfare_best")
        m = run(dgp, -a, "welfare_best")
        print(f"    {dgp:12s} alpha=+{a}: {p[0]:8.4f}+-{p[1]:.4f}   "
              f"alpha=-{a}: {m[0]:8.4f}+-{m[1]:.4f}   "
              f"|diff|/SE = {abs(p[0]-m[0])/np.sqrt(p[1]**2+m[1]**2):.2f}")
