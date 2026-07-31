"""R6. Remaining referee checks.

C1  lambda-collapse at EXACTLY matched (v_tau, rho): renormalised(alpha) vs
    equicorr(rho(alpha)).  (r4 used np.interp, which CLAMPS -- invalid at
    alpha=0.8 where rho=0.9412 is outside the grid's rho<=0.8.)
C2  Is sigma_eff really needed, or does plain sigma do as well?
C3  kappa, the aggregate feedback gain, at the chain's modal state
    (numericist: median 4.73 at eps=0.01; analyst: 2.52 at the same lambda)
C4  STEELMAN: at alpha = 0 exactly -- the expansion point -- how good is the
    canonical (1+eps) factor under CE beliefs?
C5  Are the three DGPs the SAME two-parameter Gaussian family? (algebraic claim)
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from r2_bruteforce import run_cell, tau_implemented, tau_renormalised, tau_equicorr

print("=" * 84)
print("C5  the three DGPs are one two-parameter Gaussian family N(0, v_tau*Sigma(rho))")
rng = np.random.default_rng(5)
for a in [0.4, 0.8]:
    v = a ** 2 + (1 - abs(a)) ** 2
    rho = a ** 2 / v
    ti = tau_implemented(rng, 400000, a)[:, :, 0]
    tr = tau_renormalised(rng, 400000, a)[:, :, 0]
    te = tau_equicorr(rng, 400000, rho)[:, :, 0]
    def stats(t):
        C = np.cov(t.T)
        return C[0, 0], (C[0, 1] + C[0, 2] + C[1, 2]) / 3
    for nm, t in [("implemented", ti), ("renormalised", tr), ("equicorr(rho)", te)]:
        vv, cc = stats(t)
        print(f"  alpha={a}  {nm:14s} var={vv:.4f} cov={cc:.4f} corr={cc/vv:.4f}"
              f"   [theory: impl v={v:.4f} rho={rho:.4f}]")

print("\n" + "=" * 84)
print("C1  lambda-collapse at EXACTLY matched (v_tau=1, rho).  brute force, D=2500")
rows = []
for a in [0.4, 0.8]:
    v0 = a ** 2 + (1 - abs(a)) ** 2
    rho = a ** 2 / v0
    for eps in [0.0, 0.25, 1.0]:
        gr = run_cell("renormalised", a, 0.6, eps, D=2500, T=1500, burn=300, seed=31)
        ge = run_cell("equicorr", rho, 0.6, eps, D=2500, T=1500, burn=300, seed=31)
        rows.append(dict(alpha=a, rho=rho, eps=eps, renorm=gr["G_mean"],
                         equi=ge["G_mean"], ratio=gr["G_mean"] / ge["G_mean"],
                         se_r=gr["G_se"], se_e=ge["G_se"]))
        print(f"  alpha={a} rho={rho:.4f} eps={eps:.2f} | renorm {gr['G_mean']:8.4f}"
              f"+-{gr['G_se']:.4f}  equicorr {ge['G_mean']:8.4f}+-{ge['G_se']:.4f}"
              f"  ratio {gr['G_mean']/ge['G_mean']:.4f}", flush=True)
pd.DataFrame(rows).to_csv("R6_C1_lambda_collapse.csv", index=False)

print("\n" + "=" * 84)
print("C2  sigma_eff vs plain sigma as the homogeneity normaliser")


def sigma_eff(s, sd=0.05):
    a, b = (0.0 - s) / sd, (1.0 - s) / sd
    return s * (norm.cdf(b) - norm.cdf(a)) + sd * (norm.pdf(a) - norm.pdf(b)) \
        + (1.0 - norm.cdf(b))


d = pd.read_csv("../G_grid_naive_ce_cesaro.csv")
d["seff"] = sigma_eff(d.sigma)
for norm_col, lab in [("sigma", "plain sigma"), ("seff", "sigma_eff")]:
    r = d.groupby(["family", "control", "eps"]).apply(
        lambda x: ((x.G_mean / x[norm_col]).max() - (x.G_mean / x[norm_col]).min())
        / (x.G_mean / x[norm_col]).mean(), include_groups=False)
    print(f"  {lab:12s}: median spread {r.median()*100:5.2f}%  p90 {r.quantile(.9)*100:5.2f}%"
          f"  max {r.max()*100:5.2f}%")
# the specific sigma=0.2 vs 0.6 excess the analyst predicts at +4.69%
s02 = d[np.isclose(d.sigma, 0.2)].set_index(["family", "control", "eps"]).G_mean
s06 = d[np.isclose(d.sigma, 0.6)].set_index(["family", "control", "eps"]).G_mean
ratio = (s02 / s06) / (sigma_eff(0.2) / sigma_eff(0.6))
print(f"  G(0.2)/G(0.6) / [seff(0.2)/seff(0.6)] : median {ratio.median():.4f} "
      f"(analyst predicts 1.0469 from the weight-dispersion correction)")
print(f"  seff(0.2)={sigma_eff(0.2):.6f} seff(0.6)={sigma_eff(0.6):.6f} "
      f"seff(1.0)={sigma_eff(1.0):.6f}  [1-0.05/sqrt(2pi)={1-0.05/np.sqrt(2*np.pi):.6f}]")

print("\n" + "=" * 84)
print("C3  kappa = -mD'(x), the aggregate feedback gain, over the DGP")
for lab, gen in [("implemented a=0 (lam=1)", lambda r, n: r.standard_normal((n, 4))),
                 ("equicorr rho=0.8 (lam=0.447)",
                  lambda r, n: r.standard_normal((n, 4)) @
                  (lambda S: (lambda ev, U: (U * np.sqrt(np.clip(ev, 0, None))) @ U.T)
                   (*np.linalg.eigh(S)))(0.2 * np.eye(4) + 0.8 * np.ones((4, 4))).T)]:
    print(f"  --- {lab} ---")
    print(f"    {'eps':>6} {'med kappa':>10} {'P(k>=2)':>9}  (x = weighted mean of targets)")
    for eps in [0.01, 0.05, 0.25, 0.5, 1.0, 2.0]:
        r = np.random.default_rng(7)
        t = gen(r, 200000)
        x = t.mean(axis=1)                      # evaluate at the ensemble optimum
        y = x[:, None] - t
        k = (norm.pdf((y + 0.5) / np.sqrt(eps)) +
             norm.pdf((y - 0.5) / np.sqrt(eps))).sum(axis=1) / np.sqrt(eps)
        print(f"    {eps:6.2f} {np.median(k):10.3f} {np.mean(k >= 2):9.3f}")

print("\n" + "=" * 84)
print("C4  STEELMAN of the canonical (1+eps) factor at alpha = 0 (the expansion point),")
print("    naive-myopic, CE beliefs, sigma=0.6, implemented alpha=0.")
g = d[(d.family == "implemented") & (np.isclose(d.control, 0.0))
      & (np.isclose(d.sigma, 0.6))].sort_values("eps")
G0p = 1.8829     # numericist EPS_CUSP eps->0+ (my brute force: 1.9078)
print(f"    {'eps':>6} {'G(exact)':>10} {'G(0+)*(1+eps)':>15} {'ratio':>8}")
for _, r in g.iterrows():
    print(f"    {r.eps:6.2f} {r.G_mean:10.4f} {G0p*(1+r.eps):15.4f} "
          f"{r.G_mean/(G0p*(1+r.eps)):8.4f}")
print("    -> at the expansion point, under CE, the canonical (1+eps) shape is")
print("       within ~10% on eps in [0.25,1.0].  It fails BELOW eps~0.01 (sqrt cusp)")
print("       and it fails at every |alpha|>0 (sign flip) and under Bayes beliefs.")
