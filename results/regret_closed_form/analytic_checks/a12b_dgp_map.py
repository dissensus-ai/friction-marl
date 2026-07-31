"""A12b.  The (v_tau, rho, lambda, Var(mu)) map for all three DGPs, closed form vs
Monte Carlo.  (The full a12 -- which re-ran the chain to confirm the collapse --
was retired: its algebra is verified symbolically in a19 (19/19) and its collapse
claim is established directly by a05 D4, which shows renormalised(alpha) and
equicorr(rho(alpha)) give the same E[G] to within 0.3-0.7 SE.  This keeps the
cheap part: that the closed forms describe the samplers.)
"""
import json
import numpy as np
from a00_lib import (N_AGENTS, targets_implemented, targets_renormalised,
                     targets_equicorr)

n = N_AGENTS
N = 600_000
OUT = []


def closed_form(dgp, knob):
    if dgp == "implemented":
        a = abs(knob)
        v = a ** 2 + (1 - a) ** 2
        rho = a ** 2 / v
        lam2 = (1 - a) ** 2
    elif dgp == "renormalised":
        a = abs(knob)
        v = 1.0
        rho = a ** 2 / (a ** 2 + (1 - a) ** 2)
        lam2 = 1 - rho
    else:
        v, rho = 1.0, float(knob)
        lam2 = 1 - rho
    return v, rho, lam2, v - (n - 1) / n * lam2


print("=" * 104)
print("R1  closed form vs Monte Carlo (600k draws per row).  lambda^2 = v_tau (1-rho);")
print("    Var(tau_i - taubar) = (3/4) lambda^2 ;  Var(mu) = v_tau - (3/4) lambda^2")
print("  dgp             knob |   v_tau  (MC)    |    rho    (MC)    | (3/4)lam^2 (MC)  |"
      "  Var(mu)   (MC)   | max rel err")
rng = np.random.default_rng(1212)
worst = 0.0
for dgp, knobs in (("implemented", (-0.8, -0.4, 0.0, 0.4, 0.8)),
                   ("renormalised", (-0.8, 0.0, 0.4, 0.8)),
                   ("equicorr", (-1 / 3, -0.1, 0.0, 0.4, 0.8, 0.95))):
    for k in knobs:
        if dgp == "implemented":
            t = targets_implemented(rng.standard_normal(N), rng.standard_normal((N, n)), k)
        elif dgp == "renormalised":
            t = targets_renormalised(rng.standard_normal(N), rng.standard_normal((N, n)), k)
        else:
            t = targets_equicorr(rng.standard_normal((N, n)), k)
        v, rho, lam2, vmu = closed_form(dgp, k)
        Cv = np.cov(t.T)
        ev = float(np.mean(np.diag(Cv)))
        erho = float((Cv.sum() - np.trace(Cv)) / (n * (n - 1)) / ev)
        tb = t.mean(1)
        edisp = float(((t - tb[:, None]) ** 2).mean())
        evmu = float(tb.var())
        errs = [abs(ev - v) / max(v, 1e-9),
                abs(erho - rho) / max(abs(rho), 1e-2),
                abs(edisp - 0.75 * lam2) / max(0.75 * lam2, 1e-9),
                abs(evmu - vmu) / max(vmu, 1e-2)]
        worst = max(worst, max(errs))
        print(f"  {dgp:13s}{k:+7.3f} | {v:7.4f} ({ev:6.4f}) | {rho:+7.4f} ({erho:+6.4f}) |"
              f" {0.75*lam2:7.4f} ({edisp:6.4f}) | {vmu:7.4f} ({evmu:6.4f}) |"
              f" {max(errs)*100:8.3f}%")
        OUT.append(dict(dgp=dgp, knob=float(k), v_tau=v, rho=rho, lam2=lam2, vmu=vmu,
                        v_mc=ev, rho_mc=erho, disp_mc=edisp, vmu_mc=evmu,
                        max_rel_err=float(max(errs))))
print(f"  worst relative error over all {len(OUT)} rows: {100*worst:.3f}%")
print("\n  NOTE the collapse: renormalised and equicorr both have v_tau = 1, so both are")
print("  ONE-parameter families in lambda -- and renormalised(alpha) has the same lambda")
print("  as equicorr(rho(alpha)).  a05 D4 confirms they give the same E[G]:")
print("    alpha=0.4 (rho=0.3077): renorm 0.25310+-0.00427  equicorr 0.25172+-0.00430")
print("    alpha=0.8 (rho=0.9412): renorm 0.11439+-0.00341  equicorr 0.11805+-0.00354")
print("  The implemented DGP does NOT collapse onto them: same lambda, different v_tau.")

with open("a12b_results.json", "w") as fh:
    json.dump(dict(rows=OUT, worst_rel_err=float(worst)), fh, indent=1)
print("\nwrote a12b_results.json")
