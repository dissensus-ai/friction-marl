"""Control check #1 applied to both adversarial DGPs: does realised correlation match nominal?

Motivation: the equicorr and team arms returned significant slopes of OPPOSITE sign on the column
both label `rho`. At most one of them manipulates signed cross-agent correlation. This script
settles it by measuring what each sampler actually realises.

Result (see TEAM_DGP_SIGN_BLIND_30JUL2026.md): equicorr realises nominal rho to 3dp; the team
sampler's covariance goes as opp**2, so its mean pairwise correlation is -opp**2/3 -- always
negative, symmetric in the sign of opp, and never equal to its label.

Run:  PYTHONPATH=<repo> python3 results/mode_comparison_postfix/verify_team_dgp.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "results" / "adversarial_rerun"))

from dgp import (  # noqa: E402
    realised_mean_pairwise_corr,
    sample_agent_params_equicorr,
    sample_agent_params_team,
)

RHOS = [-0.33, -0.2, -0.1, 0.0, 0.2, 0.4, 0.6, 0.8]
N_AGENTS, N_RESOURCES, SIGMA = 4, 3, 0.6


def main():
    print("Control check #1 — nominal vs realised mean pairwise correlation")
    print(f"{'nominal':>9}{'equicorr':>12}{'team':>12}{'-opp^2/3':>12}")
    for r in RHOS:
        e, _ = realised_mean_pairwise_corr(
            sample_agent_params_equicorr, r, SIGMA, N_AGENTS, N_RESOURCES, 4000, 7)
        t, _ = realised_mean_pairwise_corr(
            sample_agent_params_team, r, SIGMA, N_AGENTS, N_RESOURCES, 4000, 7)
        print(f"{r:>9.2f}{e:>12.3f}{t:>12.3f}{-(r * r) / 3:>12.3f}")

    print("\nSign test — does the team design distinguish -x from +x?")
    for x in [0.1, 0.2, 0.33]:
        a, _ = realised_mean_pairwise_corr(
            sample_agent_params_team, -x, SIGMA, N_AGENTS, N_RESOURCES, 8000, 11)
        b, _ = realised_mean_pairwise_corr(
            sample_agent_params_team, +x, SIGMA, N_AGENTS, N_RESOURCES, 8000, 11)
        print(f"  opp=-{x:<5} -> {a:+.4f}   opp=+{x:<5} -> {b:+.4f}   diff={a - b:+.5f}")


if __name__ == "__main__":
    main()
