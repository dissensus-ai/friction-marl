"""Fit candidate functional forms to the exact E[G] surface.

Every model is fitted by OLS on the CELL MEANS (the MC SE of a cell mean is
reported alongside and is two orders of magnitude below the between-cell
spread, so weighting is immaterial -- checked). AICc uses the Gaussian
log-likelihood with k = (#coefficients + 1) for the variance.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def aicc(y, yhat, k):
    n = y.size
    rss = float(((y - yhat) ** 2).sum())
    ll = -0.5 * n * (np.log(2 * np.pi * rss / n) + 1.0)
    a = 2 * k - 2 * ll
    if n - k - 1 > 0:
        a += 2 * k * (k + 1) / (n - k - 1)
    return a, rss


def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, X @ beta


def design_matrices(s, a, e):
    one = np.ones_like(s)
    can = s * (1.0 + e) / (1.0 + a)
    return {
        "M0_constant":            (np.c_[one], "c"),
        "M1_canonical_scaled":    (np.c_[can], "c*sigma(1+e)/(1+a)"),
        "M1a_canonical_affine":   (np.c_[one, can], "A + B*sigma(1+e)/(1+a)"),
        "M2_sigma_only":          (np.c_[s], "c*sigma"),
        "M2a_sigma_affine":       (np.c_[one, s], "A + B*sigma"),
        "M3_additive":            (np.c_[one, s, e, a], "A+B s+C e+D a"),
        "M4_sigma_x_absalpha":    (np.c_[s, s * np.abs(a), s * e],
                                   "sigma(c0 + c1|a| + c2 e)"),
        "M5_sigma_x_1malpha2":    (np.c_[s, s * (1 - np.abs(a)) ** 2, s * e],
                                   "sigma(c0 + c1(1-|a|)^2 + c2 e)"),
        "M6_sigma_x_inv1plusa":   (np.c_[s, s / (1 + a), s * e],
                                   "sigma(c0 + c1/(1+a) + c2 e)"),
        "M7_sigma_x_quad":        (np.c_[s, s * np.abs(a), s * a ** 2, s * e,
                                         s * e ** 2, s * np.abs(a) * e],
                                   "sigma(c0+c1|a|+c2 a^2+c3 e+c4 e^2+c5|a|e)"),
        "M8_sigma_x_sqrt_e":      (np.c_[s, s * np.abs(a), s * np.sqrt(e)],
                                   "sigma(c0 + c1|a| + c2 sqrt(e))"),
    }


def fit_family(df, label, rows):
    s = df["sigma"].values
    a = df["control"].values
    e = df["eps"].values
    y = df["G_mean"].values
    for name, (X, expr) in design_matrices(s, a, e).items():
        beta, yhat = ols(X, y)
        A, rss = aicc(y, yhat, X.shape[1] + 1)
        r2 = 1 - rss / float(((y - y.mean()) ** 2).sum())
        rows.append(dict(subset=label, model=name, expr=expr,
                         k=X.shape[1], R2=r2, AICc=A,
                         rmse=float(np.sqrt(rss / y.size)),
                         coefs=np.round(beta, 5).tolist()))
    # log-log free-exponent version of the canonical form
    pos = y > 0
    X = np.c_[np.ones(pos.sum()), np.log(s[pos]), np.log1p(e[pos]),
              np.log1p(a[pos])]
    beta, yhat_log = ols(X, np.log(y[pos]))
    yhat = np.exp(yhat_log)
    A, rss = aicc(y[pos], yhat, 5)
    r2 = 1 - rss / float(((y[pos] - y[pos].mean()) ** 2).sum())
    rows.append(dict(subset=label, model="M9_free_exponent_canonical",
                     expr="c*sigma^p (1+e)^q (1+a)^r  [canonical => p=1,q=1,r=-1]",
                     k=4, R2=r2, AICc=A, rmse=float(np.sqrt(rss / pos.sum())),
                     coefs=dict(log_c=round(float(beta[0]), 5),
                                p_sigma=round(float(beta[1]), 5),
                                q_eps=round(float(beta[2]), 5),
                                r_alpha=round(float(beta[3]), 5))))


def main():
    rows = []
    files = sorted(HERE.glob("G_grid_*_cesaro.csv")) + \
        sorted(HERE.glob("G_grid_*_T100.csv"))
    for f in files:
        df = pd.read_csv(f)
        tag = f.stem.replace("G_grid_", "")
        for fam, g in df.groupby("family"):
            fit_family(g, f"{tag}|{fam}", rows)
        fit_family(df[df.family != "equicorr"], f"{tag}|impl+renorm", rows)
    out = pd.DataFrame(rows)
    out["dAICc"] = out.groupby("subset")["AICc"].transform(lambda x: x - x.min())
    out = out.sort_values(["subset", "AICc"])
    out.to_csv(HERE / "FORM_FITS.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_colwidth", 40):
        for sub, g in out.groupby("subset"):
            print(f"\n=== {sub} ===")
            print(g[["model", "k", "R2", "AICc", "dAICc", "rmse"]].to_string(index=False))
    return out


if __name__ == "__main__":
    main()
