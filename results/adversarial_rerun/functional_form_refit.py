#!/usr/bin/env python3
"""
functional_form_refit.py
========================
Re-estimate the H4 functional-form model comparison (M1--M4 of the consent-friction
functional) on the VALID, genuinely SIGNED equicorrelation data, in place of the
discredited first-stage sign-blind n=125 factorial.

Background
----------
The paper's Table `tab:model-comparison` (M1 R^2=0.108, AIC=322.3; M4 R^2=0.753,
AIC=165.7; Delta-AIC=156.6) was computed on the FIRST-STAGE sign-blind factorial,
whose alignment axis realizes corr = alpha^2/(alpha^2+(1-|alpha|)^2) -- a |alpha|-only
U-shape. Rejecting the monotone 1/(1+alpha) denominator THERE is circular: it was
guaranteed to misfit an even-in-alpha artifact. This script re-estimates the same
four candidate forms on the equicorrelation re-run, where the signed cross-agent
correlation rho is genuine (rho<0 is real opposition, not the |alpha| mirror).

Estimator class (identical to the paper)
----------------------------------------
- DV: condition-level reward gap. Gap g = R* - mean_reward = -mean_reward (R*=0).
  Condition = (rho, sigma) cell; DV = mean over 30 reps. -> 16 conditions/learner.
- OLS via statsmodels; AIC/BIC are statsmodels OLS.aic/.bic
  (= -2*llf + 2*k_mean_params, k = # regressors incl. intercept; variance not
   separately counted). RMSE = sqrt(RSS/n). This convention reproduces the paper's
   M1 AIC 322.3 and M4 AIC 165.7 to the decimal on the original n=125 data.

Honest scope constraint
-----------------------
The equicorr data has epsilon=0 and only two sigma levels {0.6,1.0}. So this refit
cleanly re-tests the SIGNED-ALIGNMENT (rho) and STAKES (sigma) dependence of the
functional form -- NOT the entropy term. That is exactly the right scope: the
alpha/U-shape term is what the artifact corrupted; the epsilon term was never the
issue. Consequences for the four forms at epsilon=0:
    M1 Friction      F = sigma*(1+eps)/(1+rho)        -> sigma/(1+rho)   [identifiable]
    M2 Additive      C = sigma + eps - rho            -> sigma - rho     [identifiable]
    M3 Multiplicative C = sigma*eps*(1-rho)           -> 0 (eps=0)       [DEGENERATE: not identifiable]
    M4 Independent   gap = g0 + g_rho*rho + g_sig*sigma (+ g_eps*eps)    [eps term absorbed -> 3 params]
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

HERE = Path(__file__).resolve().parent


def condition_means(csv_path):
    """Return condition-level (rho, sigma) mean reward gap = -mean_reward."""
    d = pd.read_csv(csv_path)
    d["gap"] = -d["mean_reward"]                       # R* = 0 -> gap = -mean_reward
    g = (d.groupby(["rho", "sigma"], as_index=False)
           .agg(gap=("gap", "mean"),
                eps=("epsilon", "first"),
                n_reps=("gap", "size")))
    return g


def fit_ols(y, X_no_const, label):
    """OLS with intercept; return dict of fit stats matching paper convention."""
    X = sm.add_constant(X_no_const, has_constant="add")
    m = sm.OLS(np.asarray(y, float), np.asarray(X, float)).fit()
    rss = float(np.sum(m.resid ** 2))
    n = len(y)
    rmse = float(np.sqrt(rss / n))                    # paper: sqrt(RSS/n)
    return {
        "model": label,
        "n": n,
        "k_params": int(X.shape[1]),                  # mean params incl. intercept
        "R2": float(m.rsquared),
        "AIC": float(m.aic),
        "BIC": float(m.bic),
        "RMSE": rmse,
        "slope_or_coefs": m.params[1:].round(4).tolist(),
    }


def refit_learner(name, csv_path):
    g = condition_means(csv_path)
    assert (g["eps"] == 0).all(), "expected epsilon=0 throughout"
    rho, sig, y = g["rho"].values, g["sigma"].values, g["gap"].values

    rows = []
    # M1 Friction: sigma*(1+eps)/(1+rho) = sigma/(1+rho)  (eps=0)
    F1 = sig * (1.0 + 0.0) / (1.0 + rho)
    rows.append(fit_ols(y, F1.reshape(-1, 1), "M1: Friction sigma/(1+rho)"))

    # M2 Additive: sigma + eps - rho = sigma - rho  (eps=0)
    C2 = sig + 0.0 - rho
    rows.append(fit_ols(y, C2.reshape(-1, 1), "M2: Additive sigma-rho"))

    # M3 Multiplicative: sigma*eps*(1-rho) = 0  (eps=0) -> DEGENERATE
    C3 = sig * 0.0 * (1.0 - rho)
    if np.allclose(C3, 0.0):
        rows.append({
            "model": "M3: Multiplicative sigma*eps*(1-rho)",
            "n": len(y), "k_params": np.nan, "R2": np.nan,
            "AIC": np.nan, "BIC": np.nan, "RMSE": np.nan,
            "slope_or_coefs": "NOT IDENTIFIABLE (composite==0 at eps=0)",
        })
    else:
        rows.append(fit_ols(y, C3.reshape(-1, 1), "M3: Multiplicative"))

    # M4 Independent: g0 + g_rho*rho + g_sig*sigma  (eps constant -> dropped)
    X4 = np.column_stack([rho, sig])
    rows.append(fit_ols(y, X4, "M4: Independent rho+sigma"))

    df = pd.DataFrame(rows)
    df.insert(0, "learner", name)
    # Delta-AIC relative to best (lowest) AIC among identifiable models
    best_aic = np.nanmin(df["AIC"].values)
    df["dAIC_vs_best"] = df["AIC"] - best_aic
    # Delta-AIC of M4 vs M1 (the headline comparison)
    aic_m1 = df.loc[df.model.str.startswith("M1"), "AIC"].values[0]
    aic_m4 = df.loc[df.model.str.startswith("M4"), "AIC"].values[0]
    df["dAIC_M1_minus_M4"] = aic_m1 - aic_m4
    return df, g


def main():
    out_rows = []
    for name, fn in [("IQL", "equicorr_iql.csv"), ("VDN", "equicorr_vdn.csv")]:
        df, g = refit_learner(name, HERE / fn)
        print(f"\n=== {name} (signed equicorr, eps=0, 16 conditions) ===")
        print("Condition means (gap = -mean_reward):")
        print(g.to_string(index=False))
        print("\nModel comparison:")
        cols = ["model", "R2", "AIC", "BIC", "RMSE", "k_params",
                "dAIC_vs_best", "dAIC_M1_minus_M4"]
        print(df[cols].to_string(index=False))
        out_rows.append(df)

    allres = pd.concat(out_rows, ignore_index=True)
    out_csv = HERE / "functional_form_refit.csv"
    allres.to_csv(out_csv, index=False)
    print(f"\nSaved -> {out_csv}")


if __name__ == "__main__":
    main()
