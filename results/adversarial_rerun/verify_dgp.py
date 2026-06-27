"""GATE: empirically verify the redesigned DGP produces a SIGNED correlation curve.

Prints realised mean pairwise cross-agent target correlation vs the control knob
for (A) the equicorrelated continuum and (B) the two-team block, plus -- as a
sanity check -- the ORIGINAL buggy sampler to show it never goes negative.
Saves a CSV and a plot. We do not run any RL until this curve is confirmed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from results.adversarial_rerun.dgp import (  # noqa: E402
    sample_agent_params_equicorr,
    sample_agent_params_team,
    realised_mean_pairwise_corr,
)
from friction_marl.experiments.factorial_design import sample_agent_params as orig_sampler  # noqa: E402

N_AGENTS = 4
N_RESOURCES = 3
N_DRAWS = 20000
SIGMA = 0.6
SEED = 12345


def theoretical_orig(alpha):
    a2 = alpha * alpha
    return a2 / (a2 + (1 - abs(alpha)) ** 2) if (a2 + (1 - abs(alpha)) ** 2) > 0 else 0.0


def main():
    lines = []
    def p(s=""):
        print(s); lines.append(s)

    p("=" * 78)
    p("DGP VERIFICATION GATE  (n_agents=4, n_resources=3, draws=%d)" % N_DRAWS)
    p("=" * 78)

    rows = []

    # (A) equicorrelated continuum -----------------------------------------
    p("\n(A) EQUICORRELATED continuum  -- target rho is the knob; bound rho>=-1/3")
    p(f"{'rho_target':>10} {'pooled_corr':>12} {'per_draw_corr':>14}  verdict")
    rhos = [-0.33, -0.20, -0.10, 0.0, 0.20, 0.40, 0.60, 0.80]
    for rho in rhos:
        pooled, perdraw = realised_mean_pairwise_corr(
            sample_agent_params_equicorr, rho, SIGMA,
            N_AGENTS, N_RESOURCES, N_DRAWS, SEED)
        verdict = "NEGATIVE" if pooled < -0.02 else ("~zero" if abs(pooled) <= 0.02 else "positive")
        p(f"{rho:>10.2f} {pooled:>12.4f} {perdraw:>14.4f}  {verdict}")
        rows.append({"dgp": "equicorr", "control": rho, "target_rho": rho,
                     "pooled_corr": pooled, "per_draw_corr": perdraw})

    # (B) two-team block ----------------------------------------------------
    p("\n(B) TWO-TEAM block  -- opp is the knob; between-team corr = -opp^2,")
    p("    within-team corr = +opp^2, mean pairwise = -opp^2/3")
    p(f"{'opp':>10} {'mean_pooled':>12} {'within':>10} {'between':>10}")
    opps = [0.0, 0.3, 0.5, 0.7, 0.85, 1.0]
    for opp in opps:
        # mean pooled
        pooled, _ = realised_mean_pairwise_corr(
            sample_agent_params_team, opp, SIGMA,
            N_AGENTS, N_RESOURCES, N_DRAWS, SEED)
        # within / between decomposition (agents 0,1 = team A; 2,3 = team B)
        rng = np.random.default_rng(SEED + 7)
        comp = [[] for _ in range(N_AGENTS)]
        for _ in range(N_DRAWS):
            _, t = sample_agent_params_team(rng, N_AGENTS, N_RESOURCES, opp, SIGMA)
            for i in range(N_AGENTS):
                comp[i].append(t[i])
        comp = [np.concatenate(c) for c in comp]
        C = np.corrcoef(np.stack(comp))
        within = np.mean([C[0, 1], C[2, 3]])
        between = np.mean([C[0, 2], C[0, 3], C[1, 2], C[1, 3]])
        p(f"{opp:>10.2f} {pooled:>12.4f} {within:>10.4f} {between:>10.4f}")
        rows.append({"dgp": "team", "control": opp, "target_rho": -opp * opp / 3,
                     "pooled_corr": pooled, "within": within, "between": between})

    # (sanity) ORIGINAL buggy sampler --------------------------------------
    p("\n(sanity) ORIGINAL sampler  -- alpha<0 should STAY positive (the bug)")
    p(f"{'alpha':>10} {'pooled_corr':>12} {'theory|a|':>12}")
    alphas = [-0.8, -0.4, 0.0, 0.4, 0.8]
    for a in alphas:
        pooled, _ = realised_mean_pairwise_corr(
            orig_sampler, a, SIGMA, N_AGENTS, N_RESOURCES, N_DRAWS, SEED)
        p(f"{a:>10.2f} {pooled:>12.4f} {theoretical_orig(a):>12.4f}")
        rows.append({"dgp": "original", "control": a, "target_rho": theoretical_orig(a),
                     "pooled_corr": pooled})

    pd.DataFrame(rows).to_csv(HERE / "dgp_verification.csv", index=False)

    # ---- gate decision ----
    eq = [r for r in rows if r["dgp"] == "equicorr"]
    neg_ok = any(r["pooled_corr"] < -0.05 for r in eq if r["control"] < 0)
    pos_ok = any(r["pooled_corr"] > 0.5 for r in eq if r["control"] >= 0.6)
    zero_ok = any(abs(r["pooled_corr"]) < 0.02 for r in eq if r["control"] == 0.0)
    p("\n" + "=" * 78)
    p("GATE: equicorr spans NEGATIVE -> 0 -> POSITIVE ?")
    p(f"  negative arm realised (<-0.05) : {neg_ok}")
    p(f"  zero at rho=0    (|corr|<0.02) : {zero_ok}")
    p(f"  strong positive (>0.5 @ rho>=0.6): {pos_ok}")
    passed = neg_ok and pos_ok and zero_ok
    p(f"  >>> GATE {'PASSED' if passed else 'FAILED'} <<<")
    p("=" * 78)

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(7, 5))
    eqdf = pd.DataFrame([r for r in rows if r["dgp"] == "equicorr"])
    teamdf = pd.DataFrame([r for r in rows if r["dgp"] == "team"])
    origdf = pd.DataFrame([r for r in rows if r["dgp"] == "original"])
    ax.plot(eqdf["control"], eqdf["pooled_corr"], "o-", color="#7a1f2b",
            lw=2, label="(A) equicorr: realised corr vs rho_target")
    ax.plot(eqdf["control"], eqdf["control"], ":", color="#7a1f2b", alpha=0.5,
            label="ideal (corr = rho_target)")
    ax.plot(origdf["control"], origdf["pooled_corr"], "s--", color="grey",
            lw=2, label="ORIGINAL buggy (corr vs alpha) -- stays +")
    ax.axhline(0, color="k", lw=0.7)
    ax.axhline(-1 / 3, color="b", ls=":", lw=1, label="equicorr PSD bound -1/3")
    ax.set_xlabel("control knob (rho_target for A, alpha for original)")
    ax.set_ylabel("realised mean pairwise cross-agent target correlation")
    ax.set_title("DGP redesign: realised correlation is now genuinely SIGNED")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "dgp_verification.png", dpi=150)
    p(f"\nSaved dgp_verification.csv + .png")

    (HERE / "DGP_VERIFICATION.txt").write_text("\n".join(lines))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
