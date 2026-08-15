"""Analyse the reduced signed rho x observation-noise follow-up.

The primary estimand is the paired change in *clean frozen-policy* gap from
training at epsilon=0 to epsilon=1. Pairing holds preferences and initial network
weights fixed for each (mode,rho,sigma,replication) block.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epsilon0", nargs="+", required=True)
    ap.add_argument("--epsilon1", nargs="+", required=True)
    args = ap.parse_args()
    d0 = pd.concat([pd.read_csv(path) for path in args.epsilon0], ignore_index=True)
    d1 = pd.concat([pd.read_csv(path) for path in args.epsilon1], ignore_index=True)
    key = ["mode", "target_support", "rho", "sigma", "replication"]
    provenance = "latent_problem_sha256"
    if provenance not in d0.columns or provenance not in d1.columns:
        raise ValueError("epsilon follow-up lacks latent problem-instance hashes")
    keep = key + [provenance, "eval_reward_clean", "eval_reward_matched", "eval_reward_explore_001"]
    paired = d0[keep].merge(d1[keep], on=key, suffixes=("_eps0", "_eps1"), validate="one_to_one")
    expected = len(d1)
    if len(paired) != expected:
        raise ValueError("every epsilon=1 row must have exactly one epsilon=0 partner")
    if not paired[f"{provenance}_eps0"].equals(paired[f"{provenance}_eps1"]):
        raise AssertionError("epsilon grids are not paired on latent problem instances")
    for protocol in ("clean", "matched", "explore_001"):
        paired[f"gap_{protocol}_eps0"] = -paired[f"eval_reward_{protocol}_eps0"] / paired["sigma"]
        paired[f"gap_{protocol}_eps1"] = -paired[f"eval_reward_{protocol}_eps1"] / paired["sigma"]
        paired[f"noise_penalty_{protocol}"] = (
            paired[f"gap_{protocol}_eps1"] - paired[f"gap_{protocol}_eps0"]
        )

    interaction_rows = []
    overall_rows = []
    for (support, mode), group in paired.groupby(["target_support", "mode"]):
        for protocol in ("clean", "matched", "explore_001"):
            fit = ols(f"noise_penalty_{protocol} ~ rho + C(sigma)", data=group).fit(cov_type="HC3")
            ci = fit.conf_int().loc["rho"]
            categorical = ols(
                f"noise_penalty_{protocol} ~ C(rho) + C(sigma)", data=group
            ).fit(cov_type="HC3")
            rho_terms = [
                i for i, term in enumerate(categorical.params.index)
                if term.startswith("C(rho)[T.")
            ]
            restriction = np.zeros((len(rho_terms), len(categorical.params)))
            for row, col in enumerate(rho_terms):
                restriction[row, col] = 1.0
            categorical_test = categorical.wald_test(restriction, scalar=True)
            stake_fit = ols(
                f"noise_penalty_{protocol} ~ rho * C(sigma)", data=group
            ).fit(cov_type="HC3")
            stake_term = next(
                term for term in stake_fit.params.index
                if "rho" in term and "C(sigma)" in term and ":" in term
            )
            interaction_rows.append({
                "target_support": support, "mode": mode, "protocol": protocol,
                "rho_by_epsilon": fit.params["rho"], "ci_low": ci.iloc[0],
                "ci_high": ci.iloc[1], "p_HC3": fit.pvalues["rho"],
                "p_rho_categorical_HC3": float(categorical_test.pvalue),
                "rho_by_epsilon_slope_difference_sigma1_minus_point6": (
                    stake_fit.params[stake_term]
                ),
                "p_rho_by_epsilon_by_sigma_HC3": stake_fit.pvalues[stake_term],
                "n_pairs": len(group),
            })
            values = group[f"noise_penalty_{protocol}"]
            sem = stats.sem(values)
            crit = stats.t.ppf(0.975, len(values) - 1)
            overall_rows.append({
                "target_support": support, "mode": mode, "protocol": protocol,
                "mean_noise_penalty": values.mean(),
                "ci_low": values.mean() - crit * sem,
                "ci_high": values.mean() + crit * sem,
                "p_paired": stats.ttest_1samp(values, 0.0).pvalue,
                "n_pairs": len(values),
            })
    interactions = pd.DataFrame(interaction_rows)
    clean_mask = interactions["protocol"].eq("clean")
    interactions.loc[clean_mask, "p_Holm_linear_clean_modes"] = multipletests(
        interactions.loc[clean_mask, "p_HC3"], method="holm"
    )[1]
    interactions.loc[clean_mask, "p_Holm_categorical_clean_modes"] = multipletests(
        interactions.loc[clean_mask, "p_rho_categorical_HC3"], method="holm"
    )[1]
    interactions.to_csv(HERE / "signed_entropy_interactions.csv", index=False)
    overall = pd.DataFrame(overall_rows)
    overall_clean = overall["protocol"].eq("clean")
    overall.loc[overall_clean, "p_Holm_clean_modes"] = multipletests(
        overall.loc[overall_clean, "p_paired"], method="holm"
    )[1]
    overall.to_csv(HERE / "signed_entropy_overall.csv", index=False)

    cell_rows = []
    for (support, mode, rho), group in paired.groupby(["target_support", "mode", "rho"]):
        values = group["noise_penalty_clean"]
        test = stats.ttest_1samp(values, 0.0)
        cell_rows.append({
            "target_support": support, "mode": mode, "rho": rho,
            "mean_clean_frozen_noise_penalty": values.mean(),
            "t_paired": test.statistic, "p_raw": test.pvalue, "n_pairs": len(values),
        })
    cells = pd.DataFrame(cell_rows)
    cells["p_Holm_all_cells"] = multipletests(cells["p_raw"], method="holm")[1]
    cells.to_csv(HERE / "signed_entropy_within_rho.csv", index=False)

    conditional_rows = []
    for (support, mode, rho, sigma), group in paired.groupby(
        ["target_support", "mode", "rho", "sigma"]
    ):
        values = group["noise_penalty_clean"]
        test = stats.ttest_1samp(values, 0.0)
        conditional_rows.append({
            "target_support": support, "mode": mode, "rho": rho, "sigma": sigma,
            "mean_clean_frozen_noise_penalty": values.mean(),
            "t_paired": test.statistic, "p_raw": test.pvalue, "n_pairs": len(values),
        })
    conditional = pd.DataFrame(conditional_rows)
    conditional["p_Holm_all_rho_sigma_cells"] = multipletests(
        conditional["p_raw"], method="holm"
    )[1]
    conditional.to_csv(HERE / "signed_entropy_within_rho_sigma.csv", index=False)

    within_sigma_rows = []
    for (support, mode, sigma), group in paired.groupby(
        ["target_support", "mode", "sigma"]
    ):
        fit = ols("noise_penalty_clean ~ rho", data=group).fit(cov_type="HC3")
        ci = fit.conf_int().loc["rho"]
        within_sigma_rows.append({
            "target_support": support, "mode": mode, "sigma": sigma,
            "rho_by_epsilon": fit.params["rho"], "ci_low": ci.iloc[0],
            "ci_high": ci.iloc[1], "p_HC3": fit.pvalues["rho"], "n_pairs": len(group),
        })
    within_sigma = pd.DataFrame(within_sigma_rows)
    within_sigma["p_Holm_mode_by_sigma"] = multipletests(
        within_sigma["p_HC3"], method="holm"
    )[1]
    within_sigma.to_csv(HERE / "signed_entropy_within_sigma_slopes.csv", index=False)
    paired.to_csv(HERE / "signed_entropy_paired_blocks.csv", index=False)
    print(interactions.to_string(index=False))
    print(overall.to_string(index=False))
    print(cells.to_string(index=False))
    print(conditional.to_string(index=False))
    print(within_sigma.to_string(index=False))


if __name__ == "__main__":
    main()
