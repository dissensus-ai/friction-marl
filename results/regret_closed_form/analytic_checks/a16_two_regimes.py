"""A16.  THE TWO REGIMES OF Psi~(lambda), and where the design sits in them.

Psi~ is NOT monotone in lambda.  It has

  * a CONTENTION branch at large lambda:  Psi~ ~ kappa_c lambda^2   (a08 E2, R^2=0.9997)
  * a HERDING branch at small lambda: when the targets agree, all n agents see the
    same sign, |D| in {0, n} against a deadband of width 1, and the chain cannot
    stop (a02 T6: tau=5 for all four, x0=3.4 -> 3.4, 7.4, 3.4, 7.4, ... amplitude 4).

so there is an interior MINIMUM at some lambda*(protocol, eps).  The claim tested
here is that the "sign of the alignment effect" is not a sign at all -- it is the
question of which side of lambda* the design sits on:

    naive delegation : lambda* lies ABOVE the whole reachable range, so
                       dG/drho > 0 everywhere in the design.
    stage-Nash       : lambda* lies INSIDE it, so the sign flips within the design.

REACHABILITY.  The equicorrelated family is capped by positive semi-definiteness,
rho >= -1/(n-1) = -1/3, i.e. lambda^2 <= n/(n-1) = 4/3, lambda <= 1.1547 for n=4.
To probe past it we use the direct-dispersion DGP tau_i = lambda * z_i (which has
the same centred configuration law, up to the location scale), exactly as a08 did.
"""
import json
import numpy as np
from a00_lib import CAP, N_AGENTS, gap_batch

n = N_AGENTS
OUT = {}
R = 25000
SIG = 0.6


def draw(rng, R, lam, gamma):
    """centred dispersion scale lambda, location scale gamma (as in a08)."""
    z = rng.standard_normal((R, n))
    e = lam * (z - z.mean(1, keepdims=True)) / np.sqrt((n - 1) / n)
    t = gamma * rng.standard_normal((R, 1)) + e
    w = np.clip(SIG + 0.05 * rng.standard_normal((R, n)), 0, 1)
    return t, w, rng.uniform(0, CAP, R)


print("=" * 90)
print("W1  Psi~(lambda) over a range that STRADDLES the equicorrelated PSD cap")
print("    (cap: lambda <= sqrt(n/(n-1)) = %.4f).  Psi~ = E[G] / (3 E[Wbar]), E[Wbar]=2.4"
      % np.sqrt(n / (n - 1)))
lams = [0.05, 0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.87, 1.00, 1.1547,
        1.35, 1.60, 2.00, 2.50, 3.00]
EW = 3 * 4 * 0.6
tab = {}
for proto in ("naive", "sne"):
    for eps in (0.0, 0.01, 0.25, 1.0):
        if proto == "sne" and eps == 0.0:
            eps_use = 1e-6            # the SNE fixed point is ill-posed at eps=0
            tag = "1e-6"
        else:
            eps_use, tag = eps, f"{eps}"
        ys = []
        for lam in lams:
            rng = np.random.default_rng(1616)
            t, w, x0 = draw(rng, R, lam, 0.5)
            g = gap_batch(t, w, x0, eps_use, protocol=proto)
            ys.append(3 * g["G"].mean() / EW)
        ys = np.array(ys)
        j = int(np.argmin(ys))
        print(f"  [{proto:5s} eps={tag:>5s}]  " + " ".join(f"{y:6.3f}" for y in ys))
        print(f"  {'':17s}  lambda* = {lams[j]:.4f}  (Psi~min={ys[j]:.4f});  "
              f"reachable cap = 1.1547 -> lambda* is "
              f"{'INSIDE' if lams[j] < 1.1547 else 'ABOVE'} the design range")
        tab[f"{proto}|{tag}"] = dict(lams=lams, psi=ys.tolist(),
                                     lam_star=float(lams[j]), psi_min=float(ys[j]))
OUT["W1"] = tab

print("=" * 90)
print("W2  the quadratic law refit on the CONTENTION branch only (lambda > lambda*)")
for key, d in tab.items():
    L = np.array(d["lams"])
    Y = np.array(d["psi"])
    m = L > d["lam_star"] + 1e-9
    if m.sum() < 3:
        print(f"  [{key:14s}] contention branch has <3 points; skipped")
        continue
    A = np.column_stack([np.ones(m.sum()), L[m] ** 2])
    b, *_ = np.linalg.lstsq(A, Y[m], rcond=None)
    pred = A @ b
    r2 = 1 - ((Y[m] - pred) ** 2).sum() / ((Y[m] - Y[m].mean()) ** 2).sum()
    e = np.polyfit(np.log(L[m]), np.log(np.maximum(Y[m] - b[0], 1e-12)), 1)[0]
    print(f"  [{key:14s}] a0={b[0]:+7.4f}  kappa_c={b[1]:+7.4f}  R^2={r2:7.5f}"
          f"  log-log exponent={e:5.3f} (theory 2)   n={m.sum()}")
    tab[key].update(a0=float(b[0]), kappa_c=float(b[1]), r2=float(r2), expo=float(e))

print("=" * 90)
print("W3  the herding branch: |D| in {0, n} at lambda = 0, and the cycle amplitude")
rng = np.random.default_rng(3)
for lam in (0.0, 0.1, 0.3, 0.6, 1.0):
    t, w, x0 = draw(rng, 4000, lam, 0.5)
    from a00_lib import agent_probs, aggregate_pmf, state_values
    xv = state_values(x0 - np.floor(x0))
    pp, pm = agent_probs(t, xv, 0.0, protocol="naive")
    pmf = aggregate_pmf(pp, pm)                      # (R,S,9), D = -4..4
    mass_extreme = pmf[:, :, [0, 8]].sum() / pmf.sum()
    mass_zero = pmf[:, :, 4].sum() / pmf.sum()
    print(f"    lambda={lam:4.2f}: P(|D| = 4) = {mass_extreme:.4f}   P(D = 0) = "
          f"{mass_zero:.4f}   P(0<|D|<4) = {1-mass_extreme-mass_zero:.4f}")
    OUT.setdefault("W3", []).append(dict(lam=lam, p_extreme=float(mass_extreme),
                                         p_zero=float(mass_zero)))
print("    => at lambda -> 0 the aggregate is a 4-or-nothing instrument against a")
print("       deadband of width 1: the state cannot come to rest.  That is the")
print("       herding branch, and it is why ALIGNMENT hurts uncoordinated delegates.")

with open("a16_results.json", "w") as fh:
    json.dump(OUT, fh, indent=1)
print("\nwrote a16_results.json")
