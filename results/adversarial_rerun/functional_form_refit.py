#!/usr/bin/env python3
"""
functional_form_refit.py
========================
Re-estimate the H4 functional-form model comparison (M0--M4 of the consent-friction
functional) on the valid signed target-coordinate equicorrelation data, in place of the
discredited first-stage sign-blind n=125 factorial.

Background
----------
The paper's Table `tab:model-comparison` (M1 R^2=0.108, AIC=322.3; M4 R^2=0.753,
AIC=165.7; Delta-AIC=156.6) was computed on the FIRST-STAGE sign-blind factorial,
whose nominal alpha axis realizes corr = alpha^2/(alpha^2+(1-|alpha|)^2) -- a |alpha|-only
U-shape. Rejecting the monotone 1/(1+alpha) denominator THERE is circular: it was
guaranteed to misfit an even-in-alpha artifact. This script re-estimates the same
four candidate forms on the equicorrelation re-run, where signed cross-agent target-coordinate
correlation rho is realized directly (rho<0 is negative target dependence, not the |alpha| mirror).

Estimator classes and information-criterion convention
------------------------------------------------------
- DV: reward gap. Gap g = R* - mean_reward = -mean_reward (R*=0).
- The pre-stated H4 adequacy rule applies only to equal-condition OLS on the 16
  (rho, sigma) means per learner. This remains the headline estimator.
- Two sensitivity estimators retain the replication variance that the condition-mean
  fit discards: inverse-variance WLS on the same 16 means, using 1/Var(mean), and OLS
  on all 480 trained-replication rows under a working-independence likelihood. The latter
  rows retain matched problem blocks across learners and are a descriptive sensitivity, not
  480 unqualified independent experimental units. The WLS likelihood treats the estimated
  cell-mean variances as fixed precision weights, with a fail-closed finite-positive
  variance guard. AICc values are compared only among
  models fit by the same estimator to the same rows and weights; they are not compared
  across the three estimator classes. Replication-level R2 is likewise not evaluated
  against the condition-level H4 threshold.
- Raw statsmodels AIC/BIC are retained for audit, but the manuscript reports strict
  Burnham--Anderson small-sample AICc. Statsmodels counts only mean-model coefficients
  in AIC; strict AIC also counts the estimated residual variance, so
  AIC_strict = AIC_statsmodels + 2 and K_total = k_mean_params + 1.
  AICc = AIC_strict + 2*K_total*(K_total+1)/(n-K_total-1).
- RMSE is the unweighted sqrt(RSS/n), retained as a descriptive quantity. WLS also
  records its weighted RMSE separately.

Honest scope constraint
-----------------------
The equicorr data has epsilon=0 and only two sigma levels {0.6,1.0}. So this refit
cleanly re-tests the signed target-coordinate correlation (rho) and implemented reward-scale
coefficient (sigma) dependence of the functional form -- NOT the entropy term. It does not
directly manipulate objective- or reward-function alignment. Consequences for the four forms
at epsilon=0:
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


def load_replications(csv_path):
    """Return replication rows and their condition-level summaries."""
    d = pd.read_csv(csv_path)
    d["gap"] = -d["mean_reward"]                       # R* = 0 -> gap = -mean_reward
    g = (d.groupby(["rho", "sigma"], as_index=False)
           .agg(gap=("gap", "mean"),
                gap_var=("gap", "var"),
                eps=("epsilon", "first"),
                n_reps=("gap", "size")))
    g["gap_sem2"] = g["gap_var"] / g["n_reps"]
    if not np.isfinite(g["gap_sem2"]).all() or (g["gap_sem2"] <= 0).any():
        raise ValueError("condition-mean variances must be finite and positive for WLS")
    return d, g


def condition_means(csv_path):
    """Backward-compatible condition-mean accessor used by earlier audits."""
    _, g = load_replications(csv_path)
    return g


def fit_regression(y, X_no_const, label, estimator, weights=None):
    """Fit one model and return raw plus strict small-sample fit criteria."""
    X = sm.add_constant(X_no_const, has_constant="add")
    y_array = np.asarray(y, float)
    X_array = np.asarray(X, float)
    if weights is None:
        m = sm.OLS(y_array, X_array).fit()
        weight_array = np.ones(len(y_array), dtype=float)
    else:
        weight_array = np.asarray(weights, float)
        m = sm.WLS(y_array, X_array, weights=weight_array).fit()
    rss = float(np.sum(m.resid ** 2))
    n = len(y)
    rmse = float(np.sqrt(rss / n))                    # paper: sqrt(RSS/n)
    weighted_rmse = float(np.sqrt(np.average(m.resid ** 2, weights=weight_array)))
    k_mean_params = int(X.shape[1])                   # incl. intercept
    k_total = k_mean_params + 1                       # plus residual variance
    aic_statsmodels = float(m.aic)
    aic_strict = aic_statsmodels + 2.0
    aicc = aic_strict + 2.0 * k_total * (k_total + 1) / (n - k_total - 1)
    return {
        "model": label,
        "estimator": estimator,
        "n": n,
        "k_params": k_mean_params,                    # legacy alias
        "k_mean_params": k_mean_params,
        "k_total": k_total,
        "R2": float(m.rsquared),
        "AIC": aic_statsmodels,                       # legacy raw statsmodels field
        "BIC": float(m.bic),                          # legacy raw statsmodels field
        "AIC_statsmodels": aic_statsmodels,
        "BIC_statsmodels": float(m.bic),
        "AIC_strict": aic_strict,
        "AICc": aicc,
        "RMSE": rmse,
        "RMSE_weighted": weighted_rmse,
        "slope_or_coefs": m.params[1:].round(4).tolist(),
    }


def fit_ols(y, X_no_const, label):
    """Backward-compatible equal-weight OLS helper."""
    return fit_regression(y, X_no_const, label, "condition_ols_equal_weight")


def model_specs(rho, sig):
    """Return the identifiable model matrices plus the degenerate M3 marker."""
    return [
        ("M0: Reward scale sigma only", sig.reshape(-1, 1)),
        ("M1: Friction sigma/(1+rho)", (sig / (1.0 + rho)).reshape(-1, 1)),
        ("M2: Additive sigma-rho", (sig - rho).reshape(-1, 1)),
        ("M3: Multiplicative sigma*eps*(1-rho)", None),
        ("M4: Independent rho+sigma", np.column_stack([rho, sig])),
    ]


def fit_model_set(y, rho, sig, estimator, weights=None):
    """Fit M0--M4 for one estimator, then compute within-estimator deltas."""
    rows = []
    for label, matrix in model_specs(rho, sig):
        if matrix is None:
            rows.append({
                "model": label,
                "estimator": estimator,
                "n": len(y), "k_params": np.nan, "k_mean_params": np.nan,
                "k_total": np.nan, "R2": np.nan,
                "AIC": np.nan, "BIC": np.nan,
                "AIC_statsmodels": np.nan, "BIC_statsmodels": np.nan,
                "AIC_strict": np.nan, "AICc": np.nan, "RMSE": np.nan,
                "RMSE_weighted": np.nan,
                "slope_or_coefs": "NOT IDENTIFIABLE (composite==0 at eps=0)",
            })
        else:
            rows.append(fit_regression(y, matrix, label, estimator, weights=weights))

    df = pd.DataFrame(rows)
    best_aic = np.nanmin(df["AIC_statsmodels"].values)
    df["dAIC_vs_best"] = df["AIC_statsmodels"] - best_aic
    aic_m1 = df.loc[df.model.str.startswith("M1"), "AIC_statsmodels"].values[0]
    aic_m4 = df.loc[df.model.str.startswith("M4"), "AIC_statsmodels"].values[0]
    df["dAIC_M1_minus_M4"] = aic_m1 - aic_m4

    best_aicc = np.nanmin(df["AICc"].values)
    df["dAICc_vs_best"] = df["AICc"] - best_aicc
    aicc_m1 = df.loc[df.model.str.startswith("M1"), "AICc"].values[0]
    aicc_m4 = df.loc[df.model.str.startswith("M4"), "AICc"].values[0]
    aicc_m0 = df.loc[df.model.str.startswith("M0"), "AICc"].values[0]
    df["dAICc_M1_minus_M4"] = aicc_m1 - aicc_m4
    df["dAICc_M1_minus_M0"] = aicc_m1 - aicc_m0
    return df


def refit_learner(name, csv_path):
    d, g = load_replications(csv_path)
    assert (g["eps"] == 0).all(), "expected epsilon=0 throughout"
    rho, sig, y = g["rho"].values, g["sigma"].values, g["gap"].values

    headline = fit_model_set(y, rho, sig, "condition_ols_equal_weight")

    precision_weights = 1.0 / g["gap_sem2"].to_numpy()
    wls = fit_model_set(
        y, rho, sig, "condition_wls_inverse_mean_variance", weights=precision_weights
    )

    rep_y = d["gap"].to_numpy()
    rep_rho = d["rho"].to_numpy()
    rep_sig = d["sigma"].to_numpy()
    replication = fit_model_set(
        rep_y, rep_rho, rep_sig, "replication_ols_sensitivity"
    )

    robustness = pd.concat([headline, wls, replication], ignore_index=True)
    robustness.insert(0, "learner", name)
    robustness["likelihood_unit"] = np.where(
        robustness["estimator"].eq("replication_ols_sensitivity"),
        "trained-replication row (working-independence descriptive sensitivity)",
        "rho-by-sigma condition mean",
    )
    robustness["R2_scope"] = np.where(
        robustness["estimator"].eq("condition_ols_equal_weight"),
        "pre-stated H4 condition-level adequacy rule",
        "sensitivity only; not evaluated against H4 threshold",
    )
    robustness["condition_variance_ratio_max_min"] = (
        float(g["gap_var"].max() / g["gap_var"].min())
    )
    robustness["AICc_comparison_scope"] = (
        "within learner and estimator only; never across estimator classes"
    )
    robustness["precision_weight_guard"] = np.where(
        robustness["estimator"].eq("condition_wls_inverse_mean_variance"),
        "1/(sample variance/n_reps); require finite positive variance",
        "not applicable",
    )

    headline = robustness.loc[
        robustness["estimator"].eq("condition_ols_equal_weight")
    ].copy()
    legacy_columns = [
        "learner", "model", "n", "k_params", "k_mean_params", "k_total", "R2",
        "AIC", "BIC", "AIC_statsmodels", "BIC_statsmodels", "AIC_strict", "AICc",
        "RMSE", "slope_or_coefs", "dAIC_vs_best", "dAIC_M1_minus_M4",
        "dAICc_vs_best", "dAICc_M1_minus_M4", "dAICc_M1_minus_M0",
    ]
    return headline[legacy_columns], g, robustness


def main():
    out_rows = []
    robustness_rows = []
    for name, fn in [("IQL", "equicorr_iql.csv"), ("VDN", "equicorr_vdn.csv")]:
        df, g, robustness = refit_learner(name, HERE / fn)
        print(f"\n=== {name} (signed equicorr, eps=0, 16 conditions) ===")
        print("Condition means (gap = -mean_reward):")
        print(g.to_string(index=False))
        print("\nModel comparison:")
        cols = ["model", "R2", "AICc", "RMSE", "k_mean_params", "k_total",
                "dAICc_vs_best", "dAICc_M1_minus_M4", "dAICc_M1_minus_M0",
                "AIC_statsmodels", "BIC_statsmodels"]
        print(df[cols].to_string(index=False))
        out_rows.append(df)
        robustness_rows.append(robustness)

    allres = pd.concat(out_rows, ignore_index=True)
    out_csv = HERE / "functional_form_refit.csv"
    allres.to_csv(out_csv, index=False)
    print(f"\nSaved -> {out_csv}")

    robust = pd.concat(robustness_rows, ignore_index=True)
    robust_csv = HERE / "functional_form_refit_robustness.csv"
    robust.to_csv(robust_csv, index=False)
    summary_cols = [
        "learner", "estimator", "model", "n", "R2", "AICc",
        "dAICc_vs_best", "dAICc_M1_minus_M4", "dAICc_M1_minus_M0",
        "condition_variance_ratio_max_min",
    ]
    print("\n=== Estimator robustness (AICc compared within estimator only) ===")
    print(robust[summary_cols].to_string(index=False))
    print(f"\nSaved -> {robust_csv}")


if __name__ == "__main__":
    main()
