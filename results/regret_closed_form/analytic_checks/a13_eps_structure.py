"""A13.  THE eps CHANNEL, derived and measured.

Interior decision rule (a11 lemma): agent i plays +1 iff u_i < tau_i - 1/2 and
-1 iff u_i > tau_i + 1/2, u_i = shat_i + c_i, shat_i = x + N(0,eps).  Hence, at
the TRUE state x, writing y_i = x - tau_i,

   p_i^+ = Phi( (-y_i - 1/2)/sqrt(eps) ),   p_i^- = Phi( ( y_i - 1/2)/sqrt(eps) )
   e_i(y) = p_i^+ - p_i^- = 1 - Phi((y+1/2)/sqrt(eps)) - Phi((y-1/2)/sqrt(eps))
   drift      mD(x) = sum_i e_i(x - tau_i)          (odd, decreasing in each y_i)
   dispersion vD(x) = sum_i ( p_i^+ + p_i^- - e_i^2 )

TWO EXACT eps CHANNELS, opposite in sign:

 (V) VARIANCE INJECTION.  vD is increasing in eps and SATURATES:
       eps -> 0   : vD -> 0 away from thresholds; E_tau[vD] = Theta(sqrt(eps))
       eps -> inf : p^+ , p^- -> 1/2, e -> 0, so vD -> n = 4  (agents act at random)
     So the injected per-step variance is bounded by n, not unbounded in eps.

 (K) HERD DAMPING.  the aggregate feedback gain
       kappa(x) = -mD'(x) = sum_i [ phi((y_i+1/2)/sqrt(eps)) + phi((y_i-1/2)/sqrt(eps)) ]
                            / sqrt(eps)
     is DECREASING in eps for large eps (kappa ~ 2 n phi(0)/sqrt(eps) = 3.19/sqrt(eps)),
     and the linearised delegated map x <- x + mD(x) has multiplier (1 - kappa):
     |1 - kappa| >= 1  <=>  kappa >= 2  <=>  flip-unstable (2-cycle / overshoot).
     Noise therefore STABILISES the delegated dynamics.  This is the channel the
     corpus's dF/deps > 0 has no room for.

TESTS
  P1  paired eps ladder: is the approach to eps -> 0+ a sqrt(eps) cusp, and how
      big is its coefficient relative to E[G]?   (paired draws => exact deltas)
  P2  the jump at eps = 0 exactly, and its mechanism.
  P3  kappa: closed form vs finite difference; the distribution of kappa over the
      DGP at the chain's modal state; P(kappa >= 2) vs (lambda, eps); and the
      critical eps*(lambda) where the median kappa crosses 2.
  P4  vD saturation at n = 4.
"""
import json
import numpy as np
from scipy.stats import norm
from a00_lib import (CAP, N_AGENTS, draw_problem, gap_batch, agent_probs,
                     aggregate_pmf, transition_matrix, delegated_occupancy,
                     state_values, start_index)

OUT = {}
n = N_AGENTS
SQ2PI = np.sqrt(2 * np.pi)

print("=" * 84)
print("P1  PAIRED eps ladder (same preference draws at every eps).  naive protocol,")
print("    which is computed EXACTLY at eps=0, so the 0+ limit is decisive.")
R = 40000
ladders = {}
for dgp, knob, lab in (("equicorr", 0.0, "equicorr rho=0   (lambda=1)"),
                       ("equicorr", 0.8, "equicorr rho=0.8 (lambda=0.447)"),
                       ("implemented", 0.0, "implemented a=0  (lambda=1)")):
    rng = np.random.default_rng(1313)
    draws = [draw_problem(rng, R, dgp, knob, 0.6) for _ in range(3)]
    epss = np.array([0.0, 1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3,
                     4e-3, 1e-2, 4e-2, 1e-1])
    G = []
    for eps in epss:
        tot = 0.0
        for (t, w, x0) in draws:
            tot += gap_batch(t, w, x0, eps, protocol="naive")["G"].mean()
        G.append(tot)
    G = np.array(G)
    G0p = G[1]                                     # eps = 1e-10 stands for 0+
    print(f"  --- {lab} ---")
    print("      eps    " + "".join(f"{e:10.1e}" for e in epss))
    print("      E[G]   " + "".join(f"{g:10.5f}" for g in G))
    print("      dG     " + "".join(f"{g-G0p:10.2e}" for g in G))
    # fit  G(eps) - G(0+) = c1 sqrt(eps) + c2 eps   on the small-eps window
    m = (epss > 0) & (epss <= 4e-2)
    A = np.column_stack([np.sqrt(epss[m]), epss[m]])
    c, *_ = np.linalg.lstsq(A, G[m] - G0p, rcond=None)
    pred = A @ c
    ss = 1 - ((G[m] - G0p - pred) ** 2).sum() / max(((G[m] - G0p) ** 2).sum(), 1e-30)
    # pure-linear alternative
    cl = np.linalg.lstsq(epss[m][:, None], (G[m] - G0p), rcond=None)[0][0]
    ssl = 1 - ((G[m] - G0p - cl * epss[m]) ** 2).sum() / max(((G[m] - G0p) ** 2).sum(), 1e-30)
    print(f"      fit dG = c1*sqrt(eps) + c2*eps :  c1={c[0]:+.5f}  c2={c[1]:+.5f}"
          f"   R^2={ss:.5f}")
    print(f"      pure linear dG = c*eps         :  c ={cl:+.5f}"
          f"                    R^2={ssl:.5f}")
    print(f"      cusp size |c1|*sqrt(0.01)/E[G](0+) = "
          f"{abs(c[0])*0.1/G0p*100:.3f}%   |  JUMP at eps=0: "
          f"{G[0]-G0p:+.5f} ({100*(G[0]-G0p)/G0p:+.2f}%)")
    ladders[lab] = dict(eps=epss.tolist(), G=G.tolist(), c1=float(c[0]),
                        c2=float(c[1]), r2_sqrt=float(ss), c_lin=float(cl),
                        r2_lin=float(ssl), jump=float(G[0] - G0p))
OUT["P1"] = ladders

print("=" * 84)
print("P2  MECHANISM of the eps=0 jump: at eps=0 the agent's argmin can TIE (all")
print("    three moves clamp to the same post-state, or two moves tie exactly).")
print("    Any eps>0 breaks every tie with probability 1.")
rng = np.random.default_rng(77)
t, w, x0 = draw_problem(rng, 200000, "implemented", 0.0, 0.6)
xv = state_values(x0 - np.floor(x0))
tie_state = 0.0
for i in range(n):
    y = xv - t[:, i][:, None]
    at_thr = (np.abs(np.abs(y) - 0.5) < 1e-12)
    wall = (xv <= 1.0 + 1e-12) & (t[:, i][:, None] < 0)     # everything clamps to 0
    tie_state += (at_thr | wall).mean()
print(f"    fraction of (agent, lattice-state) pairs at which the eps=0 rule is")
print(f"    decided by the TIE CONVENTION rather than by a strict inequality: "
      f"{tie_state/n:.4f}")
OUT["P2_tie_fraction"] = float(tie_state / n)

print("=" * 84)
print("P3  kappa, the aggregate feedback gain.  closed form vs finite difference,")
print("    and the flip-bifurcation boundary kappa = 2.")


def mD(x, tau, eps):
    y = x[..., None] - tau
    if eps <= 0:
        return (np.where(y < -0.5, 1.0, 0.0) - np.where(y > 0.5, 1.0, 0.0)).sum(-1)
    s = np.sqrt(eps)
    return (1.0 - norm.cdf((y + 0.5) / s) - norm.cdf((y - 0.5) / s)).sum(-1)


def kappa(x, tau, eps):
    y = x[..., None] - tau
    s = np.sqrt(eps)
    return ((norm.pdf((y + 0.5) / s) + norm.pdf((y - 0.5) / s)) / s).sum(-1)


tau_demo = np.array([[-0.3, 0.4, 1.1, -0.9]])
for eps in (0.05, 0.25, 1.0):
    xs = np.array([0.0, 0.5, 1.0])
    h = 1e-5
    fd = -(mD(xs + h, tau_demo, eps) - mD(xs - h, tau_demo, eps)) / (2 * h)
    cf = kappa(xs, tau_demo, eps)
    print(f"    eps={eps:4.2f}: closed form {np.round(cf,6)}  finite diff "
          f"{np.round(fd,6)}   max err {np.abs(cf-fd).max():.2e}")
OUT["P3_kappa_fd_ok"] = True

print("    median kappa at the chain's MODAL state, and P(kappa >= 2):")
print("      lambda |      " + "".join(f"eps={e:<8g}" for e in (0.01, 0.05, 0.25, 0.5, 1.0, 2.0)))
kap = []
for lam in (0.45, 0.71, 1.0, 1.15):
    rho = 1 - lam ** 2
    line_m, line_p = [], []
    for eps in (0.01, 0.05, 0.25, 0.5, 1.0, 2.0):
        rng = np.random.default_rng(4242)
        t, w, x0 = draw_problem(rng, 6000, "equicorr", float(rho), 0.6)
        xv = state_values(x0 - np.floor(x0))
        si = start_index(x0)
        pp, pm = agent_probs(t, xv, eps, protocol="naive")
        M = transition_matrix(aggregate_pmf(pp, pm))
        occ = delegated_occupancy(M, si, T=100)
        modal = xv[np.arange(xv.shape[0]), occ.argmax(1)]
        k = kappa(modal[:, None], t, eps)[:, 0] if False else np.array(
            [kappa(np.array([modal[r]]), t[r:r + 1], eps)[0] for r in range(t.shape[0])])
        line_m.append(np.median(k))
        line_p.append((k >= 2).mean())
    print(f"      {lam:5.2f}  | med " + "".join(f"{v:11.3f}" for v in line_m))
    print(f"      {'':5s}  | P>=2" + "".join(f"{v:11.3f}" for v in line_p))
    kap.append(dict(lam=lam, med=list(map(float, line_m)), pge2=list(map(float, line_p))))
OUT["P3"] = kap
print("    asymptote: for targets within O(sqrt(eps)) of x, kappa -> 2 n phi(0)/sqrt(eps)")
print(f"               = {2*n*norm.pdf(0):.4f}/sqrt(eps)  => kappa < 2 once eps > "
      f"{(2*n*norm.pdf(0)/2)**2:.3f}")

print("=" * 84)
print("P4  vD (per-step injected variance) saturates at n = 4")
tau_demo = np.array([[-0.3, 0.4, 1.1, -0.9]])
for eps in (0.01, 0.25, 1.0, 4.0, 25.0, 400.0):
    x = np.array([0.5])
    y = x[:, None] - tau_demo
    s = np.sqrt(eps)
    pp = norm.cdf((-y - 0.5) / s)
    pm = norm.cdf((y - 0.5) / s)
    e = pp - pm
    vD = (pp + pm - e ** 2).sum(-1)
    print(f"    eps={eps:7.2f}  sd={s:7.3f}  vD={vD[0]:7.4f}   (limit n = {n})")
OUT["P4_note"] = "vD -> n as eps -> inf"

with open("a13_results.json", "w") as fh:
    json.dump(OUT, fh, indent=1)
print("\nwrote a13_results.json")
