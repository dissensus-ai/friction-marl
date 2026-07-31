"""A14b.  HEAD-TO-HEAD on the corpus's own 5 x 5 x 5 design: the derived closed form
against the canonical F = sigma (1+eps)/(1+alpha).

Naive-myopic delegation only.  That is not a convenience: a10 X2 shows the trained
IQL agents land ON the naive-myopic value at alpha=0 (-2.323 vs -2.287, 1.6%) and
are bracketed by the two concepts elsewhere, and run_gpu.py:79-99 trains n
independent Q-networks with no joint-action conditioning.  Naive-myopic IS the
family the corpus's experiments implement.  (The SNE arm of a14 was retired: its
damped fixed point costs ~70-150 s per cell, and the numericist's nash_ce grid
already covers the equilibrium concept -- FORM_FITS.csv.)

MODELS COMPARED (all fitted by OLS on the same 125 cell means):
  M_closed  : 3 n sigma_eff(sigma) * ( a0(eps) + kc(eps) lambda^2 ),
              a0, kc quadratic in eps           -> 6 free coefficients
  M_closed1 : the same with a0, kc LINEAR in eps -> 4 free coefficients
  M_canon_s : c * sigma (1+eps)/(1+alpha)        -> 1
  M_canon_a : a + b * sigma (1+eps)/(1+alpha)    -> 2
  M_const   : a                                  -> 1   (the floor any model must beat)
Everything the closed form uses -- sigma_eff, lambda(knob) -- is DERIVED (a07, a12b,
a19), not fitted; only a0(eps) and kc(eps) are free.
"""
import json
import numpy as np
from scipy.stats import norm
from a00_lib import draw_problem, gap_multi_sigma

OUT = {}
ALPHAS = [-0.8, -0.4, 0.0, 0.4, 0.8]
SIGMAS = [0.2, 0.4, 0.6, 0.8, 1.0]
EPSILONS = [0.0, 0.25, 0.5, 0.75, 1.0]
R = 15000
n = 4


def sigma_eff(s):
    a, b = (0 - s) / 0.05, (1 - s) / 0.05
    return (s * (norm.cdf(b) - norm.cdf(a)) - 0.05 * (norm.pdf(b) - norm.pdf(a))
            + (1 - norm.cdf(b)))


def lam_of(dgp, alpha):
    a = abs(alpha)
    if dgp == "implemented":
        return 1 - a
    return (1 - a) / np.sqrt(a ** 2 + (1 - a) ** 2)


def aicc(resid, k, N):
    rss = float((resid ** 2).sum())
    return N * np.log(rss / N) + 2 * k + 2 * k * (k + 1) / max(N - k - 1, 1)


def fit(X, y, k):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    r2 = 1 - (r ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return b, float(r2), aicc(r, k, len(y))


for dgp in ("implemented", "renormalised"):
    print("=" * 92)
    print(f"GRID  dgp = {dgp},  naive-myopic, R = {R} draws x 3 resources per cell")
    cells = {}
    for alpha in ALPHAS:
        for eps in EPSILONS:
            rng = np.random.default_rng(31415)
            acc = {s: 0.0 for s in SIGMAS}
            for _ in range(3):
                t, _w, x0 = draw_problem(rng, R, dgp, alpha, 0.6)
                res = gap_multi_sigma(t, x0, SIGMAS, eps, rng, protocol="naive")
                for s in SIGMAS:
                    acc[s] += res[s]["G"].mean()
            for s in SIGMAS:
                cells[(alpha, s, eps)] = acc[s]
    keys = list(cells)
    y = np.array([cells[k] for k in keys])
    al = np.array([k[0] for k in keys])
    sg = np.array([k[1] for k in keys])
    ep = np.array([k[2] for k in keys])
    lam = np.array([lam_of(dgp, a) for a in al])
    base = 3 * n * sigma_eff(sg)
    N = len(y)

    X6 = np.column_stack(sum([[base * p, base * lam ** 2 * p]
                              for p in (np.ones_like(ep), ep, ep ** 2)], []))
    X4 = np.column_stack(sum([[base * p, base * lam ** 2 * p]
                              for p in (np.ones_like(ep), ep)], []))
    F = sg * (1 + ep) / (1 + al)
    X1 = F[:, None]
    X2 = np.column_stack([np.ones_like(F), F])
    X0 = np.ones((N, 1))

    res = {}
    for name, X, k in (("M_closed  (6)", X6, 6), ("M_closed1 (4)", X4, 4),
                       ("M_canon_s (1)", X1, 1), ("M_canon_a (2)", X2, 2),
                       ("M_const   (1)", X0, 1)):
        b, r2, ai = fit(X, y, k)
        res[name] = (r2, ai, b.tolist())
    best = min(v[1] for v in res.values())
    print(f"  model                 R^2        AICc     dAICc")
    for name, (r2, ai, b) in res.items():
        print(f"  {name:16s} {r2:+8.4f}  {ai:9.2f}  {ai-best:8.2f}")
    b6 = res["M_closed  (6)"][2]
    print(f"  fitted a0(eps) = {b6[0]:+.5f} {b6[2]:+.5f} eps {b6[4]:+.5f} eps^2")
    print(f"  fitted kc(eps) = {b6[1]:+.5f} {b6[3]:+.5f} eps {b6[5]:+.5f} eps^2")
    bc = res["M_canon_a (2)"][2]
    print(f"  canonical affine slope b = {bc[1]:+.4f}   (the canonical form asserts b>0)")
    # marginal means, for the record
    print("  marginal means of E[G]:")
    for nm, arr, lv in (("alpha", al, ALPHAS), ("sigma", sg, SIGMAS), ("eps", ep, EPSILONS)):
        print(f"    {nm:6s} {lv}")
        print(f"    E[G]   {[round(float(y[arr==v].mean()),3) for v in lv]}")
        print(f"    F      {[round(float(F[arr==v].mean()),3) for v in lv]}")
    OUT[dgp] = dict(models={k: (v[0], v[1], v[2]) for k, v in res.items()},
                    cells={f"{a}|{s}|{e}": float(v) for (a, s, e), v in cells.items()})

with open("a14b_results.json", "w") as fh:
    json.dump(OUT, fh, indent=1)
print("\nwrote a14b_results.json")
