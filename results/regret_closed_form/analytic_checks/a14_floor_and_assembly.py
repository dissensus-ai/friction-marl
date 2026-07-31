"""A14.  (i) the quantization floor in closed form; (ii) assembly of the closed
form for E[G] and a like-for-like head-to-head against the canonical
F = sigma (1+eps) / (1+alpha) on the corpus's own 5 x 5 x 5 design.

(i) THE FLOOR.  The planner cannot sit at mu; it sits at the nearest point of
    L(f) = {0..10} u {f, f+1, ..., f+9}.  Its irreducible per-step loss is
        Floor = sum_m Wbar_m * d(mu_m, L(f_m))^2 ,
    which enters BOTH W* and W(realised)'s benchmark and therefore CANCELS in
    G = W* - W.  It is nonetheless the object that the corpus's "irreducible
    sigma/2 baseline" is trying to name, so we give it exactly.

    INTERIOR ASYMPTOTIC (mu far from both walls).  L(f) restricted to a unit
    cell is the two-point set {0, f} mod 1, which cuts the circle into arcs of
    length f and 1-f.  Inside an arc of length g the distance to the nearer
    endpoint is U[0, g/2], so E[d^2 | arc] = g^2/12, and weighting each arc by
    its own length,
        E[d^2 | f] = ( f^3 + (1-f)^3 ) / 12 ,
        E_f~U(0,1)[ d^2 ] = (1/12) * (1/4 + 1/4) = 1/24 = 0.0416667   (exact).

    BOUNDARY BRANCH.  mu <= 0 is absorbed by the lattice point 0 exactly, so
    d = |mu|; mu >= 10 likewise.  Under every DGP here mu is symmetric about 0,
    so this branch carries probability 1/2 and contributes Var(mu)/2.

    The often-quoted  E[floor] ~ 3 E[Wbar] ( Var(mu)/2 + 1/24 )  is therefore an
    UPPER approximation: it double counts the region 0 < mu < 1, where the wall
    at 0 is itself the nearest lattice point.  Exact value by quadrature below.

(ii) THE CLOSED FORM.
        E[G](sigma, knob, eps)  =  3 * n * sigma_eff(sigma) * Psi(lambda, v_tau; eps)
        sigma_eff(sigma) = E[clip(N(sigma, 0.05^2), 0, 1)]           (exact, a07)
        lambda^2 = v_tau (1 - rho)                                    (a12)
        Psi(lambda, 1; eps) = kappa_c(eps) lambda^2 + a0(eps)  for lambda >~ 0.45
"""
import json
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad, dblquad
from a00_lib import CAP, N_AGENTS, draw_problem, gap_multi_sigma, rho_of_alpha

OUT = {}
n = N_AGENTS

print("=" * 84)
print("F1  the interior constant 1/24, three ways")
v1 = quad(lambda f: (f ** 3 + (1 - f) ** 3) / 12.0, 0, 1)[0]
print(f"    analytic integral  int_0^1 (f^3+(1-f)^3)/12 df = {v1:.10f}   (1/24 = {1/24:.10f})")


def d_lat(mu, f):
    """distance from mu to L(f) = {0..10} u {f,...,f+9}, elementwise in (mu, f)."""
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    f = np.atleast_1d(np.asarray(f, dtype=float)) * np.ones_like(mu)
    d1 = np.min(np.abs(mu[:, None] - np.arange(11.0)[None, :]), axis=1)
    d2 = np.min(np.abs(mu[:, None] - (f[:, None] + np.arange(10.0)[None, :])), axis=1)
    return np.minimum(d1, d2)


def mc_d2(mu, f, chunk=500_000):
    tot, cnt = 0.0, 0
    for i in range(0, mu.size, chunk):
        d = d_lat(mu[i:i + chunk], f[i:i + chunk])
        tot += float((d ** 2).sum())
        cnt += d.size
    return tot / cnt


rng = np.random.default_rng(14)
mu = rng.uniform(2.0, 8.0, 4_000_000)              # deep interior
f = rng.uniform(0, 1, 4_000_000)
print(f"    Monte Carlo, mu ~ U(2,8) (deep interior): {mc_d2(mu,f):.7f}")
v2 = dblquad(lambda m, ff: d_lat(np.array([m]), ff)[0] ** 2, 0, 1,
             lambda ff: 4.0, lambda ff: 5.0)[0]
print(f"    2-D quadrature over one interior cell:    {v2:.7f}")
OUT["F1"] = dict(analytic=v1, quad=float(v2))

print("=" * 84)
print("F2  the EXACT floor for mu ~ N(0, v), f ~ U(0,1), vs the Var(mu)/2 + 1/24 form")
print("     Var(mu)  |  exact E[d^2]  |  approx v/2 + 1/24  |  rel err  |  which DGP")
tags = {0.0625: "equicorr rho=-1/3+ (near-degenerate)",
        0.25: "alpha=0 / rho=0",
        0.4375: "equicorr rho=+0.25",
        0.625: "equicorr rho=+0.5",
        0.8125: "equicorr rho=+0.75",
        0.955: "equicorr rho=+0.94 ~ |alpha|=0.8 renormalised"}
rows = []
for v in (0.0, 0.0625, 0.25, 0.4375, 0.625, 0.8125, 0.955, 1.0):
    if v == 0:
        ex = 0.0
    else:
        mu = rng.normal(0, np.sqrt(v), 4_000_000)
        f = rng.uniform(0, 1, 4_000_000)
        ex = mc_d2(mu, f)
    ap = v / 2 + 1 / 24
    print(f"      {v:6.4f}  |   {ex:9.6f}   |     {ap:9.6f}      | "
          f"{100*(ap-ex)/max(ex,1e-12):+8.2f}%  | {tags.get(v,'')}")
    rows.append(dict(v=v, exact=ex, approx=ap))
OUT["F2"] = rows
print("    => the closed form OVER-predicts by 8-25% because for 0 < mu < 1 the wall")
print("       at 0 is itself a lattice point.  Exact alternative, one line:")
print("         E[d^2] = E[mu^2 1{mu<0}] + E[d_int(mu,f)^2 1{mu>0}]   (quadrature above)")
print("    Note Var(mu) = v_tau - (3/4) lambda^2, so the floor is a function of the")
print("    SAME two shape parameters as G itself.")

print("=" * 84)
print("F3  ASSEMBLY.  Predicted vs exact E[G] on the corpus's own 5 x 5 x 5 design.")
ALPHAS = [-0.8, -0.4, 0.0, 0.4, 0.8]
SIGMAS = [0.2, 0.4, 0.6, 0.8, 1.0]
EPSILONS = [0.0, 0.25, 0.5, 0.75, 1.0]
R = 12000


def sigma_eff(s):
    a, b = (0 - s) / 0.05, (1 - s) / 0.05
    return (s * (norm.cdf(b) - norm.cdf(a)) - 0.05 * (norm.pdf(b) - norm.pdf(a))
            + 1.0 * (1 - norm.cdf(b)))


def lam_vtau(dgp, alpha):
    if dgp == "implemented":
        a = abs(alpha)
        return 1 - a, a ** 2 + (1 - a) ** 2
    a = abs(alpha)
    rho = a ** 2 / (a ** 2 + (1 - a) ** 2)
    return np.sqrt(1 - rho), 1.0


grids = {}
for proto in ("naive", "sne"):
    for dgp in ("implemented", "renormalised"):
        cells = {}
        for alpha in ALPHAS:
            for eps in EPSILONS:
                rng = np.random.default_rng(31415)
                acc = {s: 0.0 for s in SIGMAS}
                for _ in range(3):
                    t, w0, x0 = draw_problem(rng, R, dgp, alpha, 0.6)
                    res = gap_multi_sigma(t, x0, SIGMAS, eps, rng, protocol=proto)
                    for s in SIGMAS:
                        acc[s] += res[s]["G"].mean()
                for s in SIGMAS:
                    cells[(alpha, s, eps)] = acc[s]
        grids[(proto, dgp)] = cells
        print(f"  [{proto} | {dgp}] grid done ({len(cells)} cells)")

# --- fit the closed form:  G = 3 n sigma_eff * (kc(eps) lam^2 + a0(eps)) ------
report = {}
for (proto, dgp), cells in grids.items():
    keys = list(cells)
    y = np.array([cells[k] for k in keys])
    al = np.array([k[0] for k in keys])
    sg = np.array([k[1] for k in keys])
    ep = np.array([k[2] for k in keys])
    lam = np.array([lam_vtau(dgp, a)[0] for a in al])
    vt = np.array([lam_vtau(dgp, a)[1] for a in al])
    se = sigma_eff(sg)
    # design: 3n*se * [ 1, lam^2 ] x [ 1, eps, eps^2 ]  (kc and a0 quadratic in eps)
    base = 3 * n * se
    cols = []
    for pw in (np.ones_like(ep), ep, ep ** 2):
        cols += [base * pw, base * lam ** 2 * pw]
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    # canonical F, scaled and affine
    F = sg * (1 + ep) / (1 + al)
    r2_scaled = 1 - ((y - F * (F @ y) / (F @ F)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    Xa = np.column_stack([np.ones_like(F), F])
    ba, *_ = np.linalg.lstsq(Xa, y, rcond=None)
    r2_aff = 1 - ((y - Xa @ ba) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    k = X.shape[1]
    N = len(y)
    def aicc(resid, kk):
        rss = (resid ** 2).sum()
        return N * np.log(rss / N) + 2 * kk + 2 * kk * (kk + 1) / max(N - kk - 1, 1)
    print(f"\n  === {proto} | {dgp} ===")
    print(f"    closed form  3n sigma_eff (kc(eps) lam^2 + a0(eps)), k={k}: "
          f"R^2={r2:.4f}  AICc={aicc(y-pred,k):8.2f}")
    print(f"    canonical    c * sigma(1+eps)/(1+alpha),           k=1: "
          f"R^2={r2_scaled:+.4f}  AICc={aicc(y-F*(F@y)/(F@F),1):8.2f}")
    print(f"    canonical    a + b sigma(1+eps)/(1+alpha),         k=2: "
          f"R^2={r2_aff:+.4f}  AICc={aicc(y-Xa@ba,2):8.2f}   (b={ba[1]:+.4f})")
    print(f"    fitted  a0(eps) = {beta[0]:+.5f} {beta[2]:+.5f} eps {beta[4]:+.5f} eps^2")
    print(f"    fitted  kc(eps) = {beta[1]:+.5f} {beta[3]:+.5f} eps {beta[5]:+.5f} eps^2")
    report[f"{proto}|{dgp}"] = dict(
        r2_closed=float(r2), r2_canon_scaled=float(r2_scaled),
        r2_canon_affine=float(r2_aff), beta=beta.tolist(),
        aicc_closed=float(aicc(y - pred, k)),
        aicc_canon=float(aicc(y - Xa @ ba, 2)),
        cells={f"{a}|{s}|{e}": float(v) for (a, s, e), v in cells.items()})
OUT["F3"] = report

with open("a14_results.json", "w") as fh:
    json.dump(OUT, fh, indent=1)
print("\nwrote a14_results.json")
