"""Regenerate the Reviewer3 follow-up estimands without touching legacy CSVs.

The follow-up record separates late-training reward from frozen held-out policy
evaluation, decomposes the frozen gap against an exact T=100 dynamic oracle, and
compares the legacy out-of-box target design with the feasible-centred control.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def robust_slope(data: pd.DataFrame, outcome: str) -> dict:
    fit = ols(f"{outcome} ~ rho", data=data).fit(cov_type="HC3")
    ci = fit.conf_int().loc["rho"]
    return {
        "outcome": outcome,
        "slope": fit.params["rho"],
        "ci_low": ci.iloc[0],
        "ci_high": ci.iloc[1],
        "p_HC3": fit.pvalues["rho"],
        "n": len(data),
    }


def load(paths: list[str]) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame()
    frames = [pd.read_csv(path) for path in paths]
    return pd.concat(frames, ignore_index=True)


def add_outcomes(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()
    d["gap_train"] = -d["mean_reward"]
    d["gap_eval"] = -d["eval_reward_clean"]
    d["gap_explore"] = -d["eval_reward_explore_001"]
    d["gap_train_norm"] = -d["mean_reward"] / d["sigma"]
    d["gap_eval_norm"] = -d["eval_reward_clean"] / d["sigma"]
    d["gap_explore_norm"] = -d["eval_reward_explore_001"] / d["sigma"]
    return d


def analyse_shared(d: pd.DataFrame) -> None:
    d = add_outcomes(d)
    if "latent_problem_sha256" not in d.columns:
        raise ValueError("shared follow-up lacks preference-instance provenance hashes")
    hash_counts = d.groupby(["rho", "sigma", "replication"])[
        "latent_problem_sha256"
    ].nunique()
    if hash_counts.max() != 1:
        raise AssertionError("shared modes/supports are not paired on latent problem instances")
    required = {
        "oracle_reward_continuous_static", "oracle_reward_reachable_static",
        "oracle_reward_dynamic_T100", "policy_residual_clean",
    }
    missing = required.difference(d.columns)
    if missing:
        raise ValueError(f"shared follow-up missing oracle columns: {sorted(missing)}")

    # Exact frozen-gap identity, all on the mean-agent and stake-normalized scale.
    d["floor_continuous_norm"] = -d["oracle_reward_continuous_static"] / d["sigma"]
    d["quantization_cost_norm"] = d["oracle_quantization_cost"] / d["sigma"]
    d["transient_cost_norm"] = d["oracle_transient_cost"] / d["sigma"]
    d["policy_residual_norm"] = d["policy_residual_clean"] / d["sigma"]
    d["exploration_cost_norm"] = d["exploration_cost_001"] / d["sigma"]
    d["window_minus_eval_norm"] = (
        d["eval_reward_clean"] - d["mean_reward"]
    ) / d["sigma"]
    d["oracle_identity_residual"] = d["gap_eval_norm"] - (
        d["floor_continuous_norm"] + d["quantization_cost_norm"]
        + d["transient_cost_norm"] + d["policy_residual_norm"]
    )
    if d["oracle_identity_residual"].abs().max() > 2e-5:
        raise AssertionError("dynamic-oracle frozen-gap identity failed")
    for component in (
        "oracle_quantization_cost", "oracle_transient_cost", "policy_residual_clean"
    ):
        if d[component].min() < -2e-5:
            raise AssertionError(f"negative oracle component beyond tolerance: {component}")

    slope_rows = []
    component_rows = []
    for (support, mode), group in d.groupby(["target_support", "mode"]):
        for outcome in ("gap_train_norm", "gap_eval_norm", "gap_explore_norm"):
            row = robust_slope(group, outcome)
            row.update(target_support=support, mode=mode)
            slope_rows.append(row)
        for component in (
            "floor_continuous_norm", "quantization_cost_norm", "transient_cost_norm",
            "policy_residual_norm", "exploration_cost_norm", "window_minus_eval_norm",
        ):
            row = robust_slope(group, component)
            row.update(target_support=support, mode=mode, component=component,
                       mean=group[component].mean())
            component_rows.append(row)

    pd.DataFrame(slope_rows).to_csv(HERE / "shared_outcome_slopes.csv", index=False)
    pd.DataFrame(component_rows).to_csv(HERE / "dynamic_oracle_components.csv", index=False)

    # Within-stake opposition vs indifference on the frozen outcome, corrected as
    # one family separately within each target-support design.
    contrast_rows = []
    for support, support_data in d.groupby("target_support"):
        local = []
        extreme = support_data.loc[support_data.rho < 0, "rho"].min()
        for (mode, sigma), group in support_data.groupby(["mode", "sigma"]):
            a = group.loc[group.rho == extreme, "gap_eval_norm"]
            b = group.loc[group.rho == 0, "gap_eval_norm"]
            t_stat, p_raw = stats.ttest_ind(a, b, equal_var=False)
            local.append({
                "target_support": support, "mode": mode, "sigma": sigma,
                "rho_opposition": extreme, "difference": a.mean() - b.mean(),
                "t_Welch": t_stat, "p_raw": p_raw,
                "n_opposition": len(a), "n_indifference": len(b),
            })
        adjusted = multipletests([row["p_raw"] for row in local], method="holm")[1]
        for row, p_holm in zip(local, adjusted):
            row["p_Holm_family4"] = p_holm
        contrast_rows.extend(local)
    pd.DataFrame(contrast_rows).to_csv(
        HERE / "frozen_within_stake_contrasts.csv", index=False
    )

    # The original planned family averaged the two stake cells with equal weight.
    # Retain that estimand explicitly, now on the clean frozen outcome, while the
    # table above exposes the conditional values the average can conceal.
    pooled_rows = []
    interaction_rows = []
    for (support, mode), group in d.groupby(["target_support", "mode"]):
        rho_lo, rho_hi = group["rho"].min(), group["rho"].max()
        neutral = group.loc[group.rho == 0, "gap_eval_norm"]
        for label, extreme in (("opposition", rho_lo), ("cooperation", rho_hi)):
            value = group.loc[group.rho == extreme, "gap_eval_norm"]
            test = stats.ttest_ind(value, neutral, equal_var=False)
            pooled_rows.append({
                "target_support": support, "mode": mode, "contrast": label,
                "rho_extreme": extreme, "stake_mixture": "equal weight on sigma=.6,1.0",
                "difference": value.mean() - neutral.mean(),
                "t_Welch": test.statistic, "p_raw": test.pvalue,
                "n_extreme": len(value), "n_indifference": len(neutral),
            })
        endpoint = group[group.rho.isin([rho_lo, 0.0])].copy()
        endpoint["opposition"] = (endpoint.rho == rho_lo).astype(int)
        for outcome, scale in (("gap_eval", "raw"), ("gap_eval_norm", "stake-normalized")):
            full_fit = ols(f"{outcome} ~ rho * sigma", data=group).fit(cov_type="HC3")
            full_ci = full_fit.conf_int().loc["rho:sigma"]
            endpoint_fit = ols(
                f"{outcome} ~ opposition * sigma", data=endpoint
            ).fit(cov_type="HC3")
            endpoint_ci = endpoint_fit.conf_int().loc["opposition:sigma"]
            interaction_rows.extend([
                {
                    "target_support": support, "mode": mode, "outcome_scale": scale,
                    "estimand": "full-grid rho x sigma", "estimate": full_fit.params["rho:sigma"],
                    "ci_low": full_ci.iloc[0], "ci_high": full_ci.iloc[1],
                    "p_HC3": full_fit.pvalues["rho:sigma"], "n": len(group),
                },
                {
                    "target_support": support, "mode": mode, "outcome_scale": scale,
                    "estimand": "opposition-vs-zero x sigma", "estimate": endpoint_fit.params["opposition:sigma"],
                    "ci_low": endpoint_ci.iloc[0], "ci_high": endpoint_ci.iloc[1],
                    "p_HC3": endpoint_fit.pvalues["opposition:sigma"], "n": len(endpoint),
                },
            ])
    pooled_frame = pd.DataFrame(pooled_rows)
    for support, idx in pooled_frame.groupby("target_support").groups.items():
        pooled_frame.loc[idx, "p_Holm_family4"] = multipletests(
            pooled_frame.loc[idx, "p_raw"], method="holm"
        )[1]
    pooled_frame.to_csv(HERE / "frozen_pooled_extreme_contrasts.csv", index=False)
    pd.DataFrame(interaction_rows).to_csv(
        HERE / "frozen_stake_interactions.csv", index=False
    )

    # Does moving every target into the state box change the frozen rho slope?
    support_rows = []
    paired_rows = []
    if d["target_support"].nunique() == 2:
        paired_modes = d.groupby("mode")["target_support"].nunique()
        for mode in paired_modes[paired_modes == 2].index:
            group = d[d["mode"] == mode]
            fit = ols("gap_eval_norm ~ rho * C(target_support)", data=group).fit(cov_type="HC3")
            term = next(k for k in fit.params.index if k.startswith("rho:C(target_support)"))
            ci = fit.conf_int().loc[term]
            support_rows.append({
                "mode": mode, "term": term, "estimate": fit.params[term],
                "ci_low": ci.iloc[0], "ci_high": ci.iloc[1], "p_HC3": fit.pvalues[term],
                "n": len(group),
            })
            wide = group.pivot(
                index=["rho", "sigma", "replication"],
                columns="target_support", values="gap_eval_norm",
            ).dropna()
            if set(wide.columns) != {"legacy-gaussian0", "feasible-centered"}:
                raise ValueError("target-support pairing is incomplete")
            paired = wide.reset_index()
            paired["support_change"] = (
                paired["feasible-centered"] - paired["legacy-gaussian0"]
            )
            pfit = ols("support_change ~ rho", data=paired).fit(cov_type="HC3")
            pci = pfit.conf_int().loc["rho"]
            paired_rows.append({
                "mode": mode,
                "estimand": "slope of paired feasible-minus-legacy frozen gap",
                "estimate": pfit.params["rho"], "ci_low": pci.iloc[0],
                "ci_high": pci.iloc[1], "p_HC3": pfit.pvalues["rho"],
                "n_pairs": len(paired),
            })
    support_frame = pd.DataFrame(support_rows)
    paired_frame = pd.DataFrame(paired_rows)
    if not support_frame.empty:
        support_frame["p_Holm_modes"] = multipletests(
            support_frame["p_HC3"], method="holm"
        )[1]
    if not paired_frame.empty:
        paired_frame["p_Holm_modes"] = multipletests(
            paired_frame["p_HC3"], method="holm"
        )[1]
    support_frame.to_csv(HERE / "target_support_interactions.csv", index=False)
    paired_frame.to_csv(
        HERE / "target_support_paired_slope_changes.csv", index=False
    )


def analyse_environment(d: pd.DataFrame, name: str) -> None:
    d = add_outcomes(d)
    if "latent_problem_sha256" not in d.columns:
        raise ValueError(f"{name} follow-up lacks preference-instance provenance hashes")
    hash_keys = ["rho", "sigma", "replication"]
    hash_counts = d.groupby(hash_keys)["latent_problem_sha256"].nunique()
    if hash_counts.max() != 1:
        raise AssertionError(f"{name} modes/supports/overlaps are not problem-instance paired")
    rows = []
    keys = ["target_support", "mode"] + (["w"] if "w" in d.columns else [])
    for key, group in d.groupby(keys):
        if not isinstance(key, tuple):
            key = (key,)
        labels = dict(zip(keys, key))
        for outcome in ("gap_train_norm", "gap_eval_norm", "gap_explore_norm"):
            row = robust_slope(group, outcome)
            row.update(labels)
            rows.append(row)
    pd.DataFrame(rows).to_csv(HERE / f"{name}_outcome_slopes.csv", index=False)

    support_tests = []
    paired_tests = []
    if d["target_support"].nunique() == 2:
        strata = ["mode"] + (["w"] if "w" in d.columns else [])
        for key, group in d.groupby(strata):
            if not isinstance(key, tuple):
                key = (key,)
            labels = dict(zip(strata, key))
            fit = ols("gap_eval_norm ~ rho * C(target_support)", data=group).fit(cov_type="HC3")
            term = next(k for k in fit.params.index if k.startswith("rho:C(target_support)"))
            ci = fit.conf_int().loc[term]
            support_tests.append({
                **labels, "estimate": fit.params[term], "ci_low": ci.iloc[0],
                "ci_high": ci.iloc[1], "p_HC3": fit.pvalues[term], "n": len(group),
            })
            pair_index = ["rho", "sigma", "replication"]
            wide = group.pivot(
                index=pair_index, columns="target_support", values="gap_eval_norm"
            ).dropna()
            if set(wide.columns) != {"legacy-gaussian0", "feasible-centered"}:
                raise ValueError(f"{name} target-support pairing is incomplete")
            paired = wide.reset_index()
            paired["support_change"] = (
                paired["feasible-centered"] - paired["legacy-gaussian0"]
            )
            pfit = ols("support_change ~ rho", data=paired).fit(cov_type="HC3")
            pci = pfit.conf_int().loc["rho"]
            paired_tests.append({
                **labels, "estimate": pfit.params["rho"], "ci_low": pci.iloc[0],
                "ci_high": pci.iloc[1], "p_HC3": pfit.pvalues["rho"],
                "n_pairs": len(paired),
            })
    support_frame = pd.DataFrame(support_tests)
    paired_frame = pd.DataFrame(paired_tests)
    if not support_frame.empty:
        support_frame["p_Holm_strata"] = multipletests(
            support_frame["p_HC3"], method="holm"
        )[1]
    if not paired_frame.empty:
        paired_frame["p_Holm_strata"] = multipletests(
            paired_frame["p_HC3"], method="holm"
        )[1]
    support_frame.to_csv(
        HERE / f"{name}_target_support_interactions.csv", index=False
    )
    paired_frame.to_csv(
        HERE / f"{name}_target_support_paired_slope_changes.csv", index=False
    )

    if "w" in d.columns:
        tests = []
        baseline_tests = []
        shape_tests = []
        for (support, mode), group in d.groupby(["target_support", "mode"]):
            fit_hc3 = ols("gap_eval_norm ~ rho * w", data=group).fit(cov_type="HC3")
            block = group[["rho", "sigma", "replication"]].astype(str).agg("|".join, axis=1)
            fit_cluster = ols("gap_eval_norm ~ rho * w", data=group).fit(
                cov_type="cluster", cov_kwds={"groups": block}
            )
            ci = fit_cluster.conf_int().loc["rho:w"]
            tests.append({
                "target_support": support, "mode": mode,
                "interaction": fit_cluster.params["rho:w"],
                "ci_low": ci.iloc[0], "ci_high": ci.iloc[1],
                "p_cluster_problem_block": fit_cluster.pvalues["rho:w"],
                "p_HC3_sensitivity": fit_hc3.pvalues["rho:w"],
                "n": len(group), "n_problem_blocks": block.nunique(),
            })
            neutral = group[group["rho"].eq(0)].copy()
            neutral_block = neutral[["sigma", "replication"]].astype(str).agg("|".join, axis=1)
            baseline_hc3 = ols("gap_eval_norm ~ w", data=neutral).fit(cov_type="HC3")
            baseline_cluster = ols("gap_eval_norm ~ w", data=neutral).fit(
                cov_type="cluster", cov_kwds={"groups": neutral_block}
            )
            baseline_ci = baseline_cluster.conf_int().loc["w"]
            baseline_tests.append({
                "target_support": support, "mode": mode,
                "baseline_slope_on_w": baseline_cluster.params["w"],
                "ci_low": baseline_ci.iloc[0], "ci_high": baseline_ci.iloc[1],
                "p_cluster_problem_block": baseline_cluster.pvalues["w"],
                "p_HC3_sensitivity": baseline_hc3.pvalues["w"],
                "n": len(neutral), "n_problem_blocks": neutral_block.nunique(),
            })
            categorical = ols(
                "gap_eval_norm ~ rho * C(w) + C(sigma)", data=group
            ).fit(cov_type="cluster", cov_kwds={"groups": block})
            terms = []
            for col, term in enumerate(categorical.params.index):
                if "rho" not in term or "C(w)" not in term or ":" not in term:
                    continue
                match = re.search(r"T\.([^\]]+)", term)
                if match:
                    terms.append((float(match.group(1)), col))
            terms.sort()
            if len(terms) != group["w"].nunique() - 1:
                raise AssertionError("could not identify all categorical rho-by-w terms")
            joint_r = np.zeros((len(terms), len(categorical.params)))
            for row, (_, col) in enumerate(terms):
                joint_r[row, col] = 1.0
            joint_test = categorical.wald_test(joint_r, scalar=True)
            ramp_r = np.zeros((len(terms) - 1, len(categorical.params)))
            first_w, first_col = terms[0]
            for row, (w_value, col) in enumerate(terms[1:]):
                ramp_r[row, first_col] = w_value
                ramp_r[row, col] = -first_w
            ramp_test = categorical.wald_test(ramp_r, scalar=True)
            shape_tests.append({
                "target_support": support, "mode": mode,
                "p_categorical_slope_modulation_cluster": float(joint_test.pvalue),
                "df_categorical": len(terms),
                "p_departure_from_linear_ramp_cluster": float(ramp_test.pvalue),
                "df_ramp_departure": len(terms) - 1,
                "n": len(group), "n_problem_blocks": block.nunique(),
            })
        pd.DataFrame(tests).to_csv(HERE / "partial_frozen_interactions.csv", index=False)
        pd.DataFrame(baseline_tests).to_csv(
            HERE / "partial_frozen_neutral_baselines.csv", index=False
        )
        pd.DataFrame(shape_tests).to_csv(
            HERE / "partial_frozen_shape_tests.csv", index=False
        )


def make_partial_figures(d: pd.DataFrame) -> None:
    """Render the paired frozen-policy partial-sharing figures of record."""
    d = add_outcomes(d)
    modes = list(d["mode"].drop_duplicates())
    colours = {"iql": "#7a1f3d", "vdn": "#235789"}

    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    for mode in modes:
        rows = []
        for w, group in d[d["mode"] == mode].groupby("w"):
            slope = robust_slope(group, "gap_eval_norm")
            rows.append({"w": w, **slope})
        frame = pd.DataFrame(rows).sort_values("w")
        ax.errorbar(
            frame["w"], frame["slope"],
            yerr=[frame["slope"] - frame["ci_low"], frame["ci_high"] - frame["slope"]],
            marker="o", capsize=4, linewidth=1.7, label=mode.upper(),
            color=colours.get(mode),
        )
    ax.axhline(0, color="0.45", linewidth=0.8, linestyle="--")
    ax.set(xlabel="Shared-resource fraction w", ylabel="Frozen normalized-gap slope on rho")
    ax.set_xticks(sorted(d["w"].unique()))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(HERE / "partial_frozen_slopes.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(modes), figsize=(10.2, 4.0), sharey=True, squeeze=False)
    for ax, mode in zip(axes[0], modes):
        subset = d[d["mode"] == mode]
        for w, group in subset.groupby("w"):
            summary = group.groupby("rho")["gap_eval_norm"].agg(["mean", "sem"]).reset_index()
            ax.errorbar(
                summary["rho"], summary["mean"], yerr=1.96 * summary["sem"],
                marker="o", capsize=2.5, linewidth=1.25, label=f"w={w:.2g}",
            )
        ax.axhline(0, color="0.6", linewidth=0.6)
        ax.set_title(mode.upper())
        ax.set_xlabel("Signed target correlation rho")
    axes[0, 0].set_ylabel("Frozen normalized gap")
    axes[0, -1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "partial_frozen_gap_by_overlap.pdf")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shared", nargs="*", default=[])
    ap.add_argument("--separable", nargs="*", default=[])
    ap.add_argument("--partial", nargs="*", default=[])
    args = ap.parse_args()

    shared = load(args.shared)
    separable = load(args.separable)
    partial = load(args.partial)
    if not shared.empty:
        analyse_shared(shared)
    if not separable.empty:
        analyse_environment(separable, "separable")
    if not partial.empty:
        analyse_environment(partial, "partial")
        make_partial_figures(partial)
    if shared.empty and separable.empty and partial.empty:
        raise SystemExit("provide at least one --shared/--separable/--partial CSV")


if __name__ == "__main__":
    main()
