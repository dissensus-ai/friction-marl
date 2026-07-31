"""R4. REFEREE TEST: the analyst's closed form against the numericist's surface.

The analyst's claim (DERIVATION sec 0):

    E[G](sigma, knob, eps) = m * n * sigma_eff(sigma) * Psi(lambda, v_tau; eps)

with Psi FREE OF SIGMA EXACTLY, collapsing to a one-parameter Psi~(lambda; eps)
for any unit-marginal-variance DGP, and on the contention branch

    Psi~(lambda; eps) = kappa_c(eps) * lambda^2 + a0(eps).

Tested here as three separate, falsifiable claims:
  H1  sigma-collapse    : G / sigma_eff(sigma) is constant across sigma
  H2  lambda-collapse   : renormalised(alpha) == equicorr(rho(alpha)) cell by cell
                          (same lambda, same v_tau=1, different sampler)
  H3  v_tau-irrelevance : does the implemented DGP (v_tau != 1) land on the same
                          Psi~(lambda) as the unit-variance DGPs?
  H4  the lambda^2 law  : fit Psi~ = kc*lam^2 + a0 and report residuals
  H5  head-to-head      : analyst form vs canonical sigma(1+eps)/(1+alpha) vs
                          constant, per cell, on the numericist's grid
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

M_RES, N_AG = 3, 4


def sigma_eff(s, sd=0.05):
    a, b = (0.0 - s) / sd, (1.0 - s) / sd
    return s * (norm.cdf(b) - norm.cdf(a)) + sd * (norm.pdf(a) - norm.pdf(b)) \
        + (1.0 - norm.cdf(b))


def dgp_params(fam, c):
    """returns (v_tau, rho, lambda^2)"""
    if fam == "implemented":
        v = c ** 2 + (1 - abs(c)) ** 2
        rho = c ** 2 / v
        return v, rho, v * (1 - rho)
    if fam == "renormalised":
        v0 = c ** 2 + (1 - abs(c)) ** 2
        rho = c ** 2 / v0
        return 1.0, rho, 1.0 - rho
    if fam == "equicorr":
        return 1.0, c, 1.0 - c
    raise ValueError(fam)


def load(tag, design="cesaro"):
    d = pd.read_csv(f"../G_grid_{tag}_{design}.csv")
    p = d.apply(lambda r: dgp_params(r.family, r.control), axis=1, result_type="expand")
    d["v_tau"], d["rho"], d["lam2"] = p[0], p[1], p[2]
    d["lam"] = np.sqrt(d.lam2)
    d["seff"] = sigma_eff(d.sigma)
    d["Psi"] = d.G_mean / (M_RES * N_AG * d.seff)
    return d


def h1_sigma(d, name):
    print(f"\n--- H1  sigma-collapse ({name}) ---")
    print("  max spread of G/sigma_eff across sigma within (family,control,eps):")
    g = d.groupby(["family", "control", "eps"])
    rel = g.apply(lambda x: (x.Psi.max() - x.Psi.min()) / x.Psi.mean(),
                  include_groups=False)
    print(f"    median {rel.median()*100:6.2f}%   p90 {rel.quantile(.9)*100:6.2f}%"
          f"   max {rel.max()*100:6.2f}%   (n={len(rel)} groups, sigma in "
          f"{sorted(d.sigma.unique())})")
    worst = rel.idxmax()
    print(f"    worst group: {worst}  spread {rel.max()*100:.2f}%")
    return rel


def h2_lambda(d, name):
    print(f"\n--- H2  lambda-collapse: renormalised(alpha) vs equicorr(rho(alpha)) "
          f"({name}) ---")
    print(f"  {'alpha':>6} {'rho':>7} {'lam2':>7} {'eps':>5} "
          f"{'renorm G':>10} {'equi G(interp)':>15} {'ratio':>7}")
    eq = d[d.family == "equicorr"]
    rows = []
    for a in [0.0, 0.4, 0.8]:
        _, rho, l2 = dgp_params("renormalised", a)
        for eps in sorted(d.eps.unique()):
            r = d[(d.family == "renormalised") & (np.isclose(d.control, a))
                  & (np.isclose(d.sigma, 0.6)) & (np.isclose(d.eps, eps))]
            e = eq[(np.isclose(eq.sigma, 0.6)) & (np.isclose(eq.eps, eps))]
            e = e.sort_values("control")
            if len(r) == 0 or len(e) < 2:
                continue
            gi = np.interp(rho, e.control.values, e.G_mean.values)
            gr = float(r.G_mean.iloc[0])
            rows.append((a, rho, l2, eps, gr, gi, gr / gi))
            print(f"  {a:6.2f} {rho:7.4f} {l2:7.4f} {eps:5.2f} "
                  f"{gr:10.4f} {gi:15.4f} {gr/gi:7.4f}")
    r = np.array([x[6] for x in rows])
    print(f"  ratio: median {np.median(r):.4f}  min {r.min():.4f}  max {r.max():.4f}")
    return rows


def h3_vtau(d, name):
    print(f"\n--- H3  v_tau irrelevance: implemented(alpha) vs equicorr at the SAME "
          f"lambda ({name}) ---")
    print(f"  implemented lam2=(1-|a|)^2 but v_tau=a^2+(1-|a|)^2 != 1.")
    print(f"  {'alpha':>6} {'lam2':>7} {'v_tau':>7} {'eps':>5} "
          f"{'impl G':>9} {'equi G@same lam':>16} {'ratio':>7}")
    eq = d[(d.family == "equicorr") & (np.isclose(d.sigma, 0.6))]
    for a in [0.4, 0.8]:
        v, _, l2 = dgp_params("implemented", a)
        rho_eq = 1.0 - l2                # equicorr with the SAME lambda^2
        for eps in sorted(d.eps.unique()):
            r = d[(d.family == "implemented") & (np.isclose(d.control, a))
                  & (np.isclose(d.sigma, 0.6)) & (np.isclose(d.eps, eps))]
            e = eq[np.isclose(eq.eps, eps)].sort_values("control")
            if len(r) == 0 or len(e) < 2:
                continue
            gi = np.interp(rho_eq, e.control.values, e.G_mean.values)
            gr = float(r.G_mean.iloc[0])
            print(f"  {a:6.2f} {l2:7.4f} {v:7.4f} {eps:5.2f} {gr:9.4f} "
                  f"{gi:16.4f} {gr/gi:7.4f}")


def h5_headtohead(d, name):
    print(f"\n--- H5  head-to-head on the numericist's grid ({name}, "
          f"{len(d)} cells) ---")
    y = d.G_mean.values
    # canonical uses the DGP's own knob as "alpha"; for equicorr use rho
    knob = d.control.values
    Fcan = d.sigma.values * (1 + d.eps.values) / (1 + knob)
    epsu = sorted(d.eps.unique())
    # analyst: m n seff (a0(eps) + kc(eps) lam^2), one (a0,kc) per eps level
    X = []
    for e in epsu:
        ind = (d.eps.values == e).astype(float)
        X.append(M_RES * N_AG * d.seff.values * ind)
        X.append(M_RES * N_AG * d.seff.values * d.lam2.values * ind)
    Xa = np.column_stack(X)

    def fit(Xm, y, k):
        beta, *_ = np.linalg.lstsq(Xm, y, rcond=None)
        res = y - Xm @ beta
        sse = (res ** 2).sum()
        sst = ((y - y.mean()) ** 2).sum()
        n = len(y)
        aicc = n * np.log(sse / n) + 2 * k + 2 * k * (k + 1) / max(n - k - 1, 1)
        return 1 - sse / sst, aicc, res, beta

    models = {
        "analyst  m n seff (a0(e)+kc(e) lam^2)": (Xa, Xa.shape[1] + 1),
        "canonical  c * sigma(1+e)/(1+a)": (Fcan[:, None], 2),
        "canonical  A + B sigma(1+e)/(1+a)": (np.column_stack([np.ones_like(y), Fcan]), 3),
        "constant": (np.ones((len(y), 1)), 2),
        "seff only  c * m n seff": ((M_RES * N_AG * d.seff.values)[:, None], 2),
    }
    out = {}
    print(f"  {'model':42s} {'k':>3} {'R2':>8} {'AICc':>10} {'MAPE%':>8}")
    for nm, (Xm, k) in models.items():
        r2, aicc, res, beta = fit(Xm, y, k)
        mape = 100 * np.mean(np.abs(res / np.maximum(np.abs(y), 1e-9)))
        out[nm] = (r2, aicc, res, beta)
        print(f"  {nm:42s} {k:3d} {r2:8.4f} {aicc:10.2f} {mape:8.1f}")
    base = min(v[1] for v in out.values())
    print(f"  dAICc vs best: " + "; ".join(
        f"{nm.split()[0]}={v[1]-base:.1f}" for nm, v in out.items()))
    return out


def h4_lambda2(d, name):
    print(f"\n--- H4  the lambda^2 law: Psi~ = kc(eps) lam^2 + a0(eps), unit-var "
          f"DGPs only ({name}) ---")
    u = d[(d.family.isin(["renormalised", "equicorr"])) & (np.isclose(d.sigma, 0.6))]
    print(f"  {'eps':>5} {'kc':>9} {'a0':>9} {'R2':>8} {'n':>4}  "
          f"{'max|resid|/Psi':>15}")
    for e in sorted(u.eps.unique()):
        s = u[np.isclose(u.eps, e)]
        A = np.column_stack([s.lam2.values, np.ones(len(s))])
        beta, *_ = np.linalg.lstsq(A, s.Psi.values, rcond=None)
        res = s.Psi.values - A @ beta
        r2 = 1 - (res ** 2).sum() / ((s.Psi.values - s.Psi.values.mean()) ** 2).sum()
        print(f"  {e:5.2f} {beta[0]:9.4f} {beta[1]:9.4f} {r2:8.4f} {len(s):4d}  "
              f"{np.abs(res).max()/s.Psi.mean():15.3f}")


if __name__ == "__main__":
    for tag in ["naive_ce", "naive_bayes", "nash_ce"]:
        d = load(tag)
        print("\n" + "=" * 86)
        print(f"###  {tag}  (Cesaro), {len(d)} cells")
        print("=" * 86)
        h1_sigma(d, tag)
        h2_lambda(d, tag)
        h3_vtau(d, tag)
        h4_lambda2(d, tag)
        h5_headtohead(d, tag)
