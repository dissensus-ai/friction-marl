"""A11.  THE BELIEF-IRRELEVANCE LEMMA, and the claim that the whole
belief-model wedge (and with it the sign ambiguity of dE[G]/deps) is a
BOUNDARY artefact of the clamp, not a property of noisy delegation.

LEMMA (belief irrelevance off the clamp).
  Agent i observes shat_i = s + eta_i, eta_i ~ N(0, eps); flat prior => the
  posterior is s | shat_i ~ N(shat_i, eps).  Let c_i be its conjecture about the
  others' aggregate move and let the post-state be x' = s + c_i + d.  Then

      E[ (x' - tau_i)^2 | shat_i ]  =  (shat_i + c_i + d - tau_i)^2  +  eps .

  The eps is ADDITIVE and does not depend on d, so
      argmin_d E[ (x'-tau_i)^2 | shat_i ]  =  argmin_d (shat_i+c_i+d-tau_i)^2 ,
  i.e. the Bayes rule and the certainty-equivalent rule are the SAME MAP.
  Both equal  d = clip(round(tau_i - shat_i - c_i), -1, 1):  a unit deadband
  centred on tau_i.

  Hence, in the interior, eps enters ONLY through the randomness of shat_i, and
  never through the shape of the decision rule.  A belief-model wedge can arise
  only where x' = clamp(s+c_i+d, 0, C) is non-linear on the posterior support.

TESTS
  L1  the additive-eps identity, symbolically (sympy) and numerically.
  L2  where the clamp-aware Bayes argmin differs from the CE argmin: is the
      disagreement set contained in a O(sqrt(eps)) neighbourhood of a wall?
  L3  the wedge in E[G]:  Bayes vs CE, with targets at their NATIVE location
      (mean 0, so ~half the resource optima sit ON the lower wall) vs the same
      DGP SHIFTED to the middle of the box (clamp slack).  If the lemma is the
      explanation, the wedge must collapse under the shift.
  L4  the same shift applied to the sign of dE[G]/deps.
"""
import json
import numpy as np
from a00_lib import (CAP, N_AGENTS, expected_clamped_loss, gap_batch,
                     draw_problem, _delta_star)

OUT = {}
np.set_printoptions(precision=5, suppress=True)

print("=" * 78)
print("L1  the additive-eps identity  E[(Z+d-tau)^2] = (m+d-tau)^2 + eps,  Z~N(m,eps)")
try:
    import sympy as sp
    m, tau, e, d = sp.symbols("m tau eps d", real=True)
    z = sp.Symbol("z", real=True)
    pdf = sp.exp(-(z - m) ** 2 / (2 * e)) / sp.sqrt(2 * sp.pi * e)
    expr = sp.integrate((z + d - tau) ** 2 * pdf, (z, -sp.oo, sp.oo),
                        conds="none")
    simp = sp.simplify(sp.expand(expr) - ((m + d - tau) ** 2 + e))
    print(f"    sympy: E[.] - [(m+d-tau)^2 + eps] simplifies to {simp}")
    OUT["L1_sympy_residual"] = str(simp)
except Exception as ex:                                   # pragma: no cover
    print(f"    sympy unavailable/slow ({ex}); numeric only")
    OUT["L1_sympy_residual"] = "skipped"

rng = np.random.default_rng(11)
mm, tt, ee = rng.normal(0, 3, 5), rng.normal(0, 2, 5), np.array([.01, .1, .25, .5, 1.])
worst = 0.0
for mi, ti, ei in zip(mm, tt, ee):
    for dd in (-1.0, 0.0, 1.0):
        Z = rng.normal(mi + dd, np.sqrt(ei), 4_000_000)
        lhs = ((Z - ti) ** 2).mean()
        rhs = (mi + dd - ti) ** 2 + ei
        worst = max(worst, abs(lhs - rhs) / max(abs(rhs), 1e-9))
print(f"    numeric (4e6 draws, unclamped): worst relative error {worst:.2e}")
OUT["L1_numeric_worst_relerr"] = float(worst)
print("    => in the interior, Bayes rule == certainty-equivalent rule, EXACTLY.")

print("=" * 78)
print("L2  disagreement set of the two rules, vs distance to the nearer wall")
print("    (u = shat + c is the agent's point estimate of its own pre-move position)")
rows = []
for eps in (0.01, 0.05, 0.25, 0.5, 1.0):
    u = np.linspace(-2.0, CAP + 2.0, 4801)
    tau = rng.normal(0.0, 1.0, 4000)
    U = np.repeat(u[:, None], tau.size, axis=1)
    T = np.repeat(tau[None, :], u.size, axis=0)
    d_bayes = _delta_star(U, T, eps)                       # clamp-aware Bayes
    d_ce = np.clip(np.round(T - U), -1, 1).astype(int)     # CE / interior rule
    diff = d_bayes != d_ce
    wall = np.minimum(np.abs(U), np.abs(CAP - U))          # distance to a wall
    frac = diff.mean()
    if diff.any():
        q = np.quantile(wall[diff], [0.5, 0.9, 0.99, 1.0])
    else:
        q = np.zeros(4)
    print(f"    eps={eps:4.2f} sd={np.sqrt(eps):5.3f} | disagree on {100*frac:6.3f}% "
          f"of (u,tau) | wall-distance of disagreements: "
          f"med={q[0]:5.2f} p90={q[1]:5.2f} p99={q[2]:5.2f} max={q[3]:5.2f}"
          f" | max/sd = {q[3]/np.sqrt(eps):5.2f}")
    rows.append(dict(eps=eps, frac=float(frac), med=float(q[0]), p99=float(q[2]),
                     mx=float(q[3]), mx_over_sd=float(q[3] / np.sqrt(eps))))
OUT["L2"] = rows
print("    => every disagreement lies within a few noise SDs of a wall.  The two")
print("       belief models are the SAME rule except in the clamp boundary layer.")

print("=" * 78)
print("L3/L4  E[G] under Bayes vs CE, at the NATIVE target location and SHIFTED")
print("       to the centre of the box (shift = +5, so the clamp is slack).")


def cell(dgp, knob, sigma, eps, protocol, clamp_aware, shift, R=20000, seed=515):
    rng = np.random.default_rng(seed)
    tot = 0.0
    for _ in range(3):                                     # 3 resources
        t, w, x0 = draw_problem(rng, R, dgp, knob, sigma)
        t = t + shift
        g = gap_batch(t, w, x0, eps, protocol=protocol, clamp_aware=clamp_aware)
        tot += g["G"].mean()
    return tot


res = {}
for shift, lab in ((0.0, "native (mu ~ N(0,.))"), (5.0, "shifted +5 (clamp slack)")):
    print(f"  --- {lab} ---")
    for proto in ("naive", "sne"):
        line_b, line_c = [], []
        for eps in (0.0, 0.01, 0.05, 0.25, 0.5, 1.0):
            gb = cell("implemented", 0.0, 0.6, eps, proto, True, shift)
            gc = cell("implemented", 0.0, 0.6, eps, proto, False, shift)
            line_b.append(gb)
            line_c.append(gc)
            res[f"{lab}|{proto}|{eps}"] = (gb, gc)
        e = [0.0, 0.01, 0.05, 0.25, 0.5, 1.0]
        print(f"    [{proto:5s}] eps      " + "".join(f"{x:9.3g}" for x in e))
        print(f"    [{proto:5s}] Bayes    " + "".join(f"{x:9.4f}" for x in line_b))
        print(f"    [{proto:5s}] CE       " + "".join(f"{x:9.4f}" for x in line_c))
        wed = [100 * (b - c) / c if c else np.nan for b, c in zip(line_b, line_c)]
        print(f"    [{proto:5s}] wedge %  " + "".join(f"{x:9.2f}" for x in wed))
        sgn_b = np.sign(line_b[3] - line_b[1])
        sgn_c = np.sign(line_c[3] - line_c[1])
        print(f"    [{proto:5s}] sign of G(0.25)-G(0.01):  Bayes {sgn_b:+.0f}   CE {sgn_c:+.0f}")
OUT["L3"] = {k: list(map(float, v)) for k, v in res.items()}

with open("a11_results.json", "w") as fh:
    json.dump(OUT, fh, indent=1)
print("\nwrote a11_results.json")
