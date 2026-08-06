"""How much of the nominal preference correlation the environment actually delivers.

Written 4 Aug 2026 in response to an external review.

The equicorrelated sampler draws targets in R^m with a specified cross-agent
correlation rho. But an agent's reward is a weighted quadratic loss on a state
confined to the box [0, C]^m, so the state it actually prefers is not tau_i --
it is clip(tau_i, 0, C), the box-constrained minimiser. Targets are marginally
N(0,1) while C = 10, so the upper bound never binds and the lower one binds
about half the time.

The consequence is that the axis the agents experience is not the axis the CSV
labels. This script measures the gap. It is the same class of check as
dgp_corr_verification (nominal vs realised correlation of tau) pushed one step
further: nominal -> realised -> *feasible*.

Run:  PYTHONPATH=<repo> python3 results/adversarial_rerun/feasible_correlation.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from results.adversarial_rerun.dgp import sample_agent_params_equicorr

N_RESOURCES, CAPACITY = 3, 10.0
# n = 4 is the main grid (PSD bound rho >= -1/(n-1) = -1/3). n = 2 is the
# strong-opposition control, the only design that reaches rho = -1 -- and the
# one the paper calls "the cleanest adversarial test the design admits", so it
# is exactly where the feasible-vs-latent gap needs a number rather than a
# presumption. Added 6 Aug 2026.
ARMS = {4: [-0.33, -0.2, -0.1, 0.0, 0.2, 0.4, 0.6, 0.8],
        2: [-1.0, -0.6, -0.33, 0.0, 0.4, 0.8]}
N_DRAWS = 20_000
SEED = 20260804


def pair_corr(stacked):
    """Pooled correlation over all unordered agent pairs and all resources."""
    x, y = [], []
    n = stacked.shape[1]
    for i in range(n):
        for k in range(i + 1, n):
            x.append(stacked[:, i, :].ravel())
            y.append(stacked[:, k, :].ravel())
    return float(np.corrcoef(np.concatenate(x), np.concatenate(y))[0, 1])


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for n_agents, rhos in ARMS.items():
      for rho in rhos:
        draws = np.stack([
            sample_agent_params_equicorr(rng, n_agents, N_RESOURCES, rho, 1.0)[1]
            for _ in range(N_DRAWS)
        ])
        latent = pair_corr(draws)
        feasible = pair_corr(np.clip(draws, 0.0, CAPACITY))
        rows.append({
            "n_agents": n_agents,
            "rho_nominal": rho,
            "corr_latent": round(latent, 4),
            "corr_feasible": round(feasible, 4),
            # Retention is only meaningful away from rho = 0, where both terms
            # are noise and their ratio is undefined in the limit.
            "retained": round(feasible / latent, 3) if abs(rho) > 1e-9 else None,
        })

    df = pd.DataFrame(rows)
    out = Path(__file__).resolve().parent / "feasible_correlation.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\n-> {out}")
    for n_agents in ARMS:
        d = df[df.n_agents == n_agents]
        neg = d[d.rho_nominal < 0].retained.astype(float).mean()
        pos = d[d.rho_nominal > 0].retained.astype(float).mean()
        print(f"\nn={n_agents}: mean retention  opposed {neg:.2f}   cooperative {pos:.2f}")
    strong = df[(df.n_agents == 2) & (df.rho_nominal == -1.0)].iloc[0]
    print(f"\nSTRONG-OPPOSITION CONTROL (n=2, nominal rho=-1): latent "
          f"{strong.corr_latent:+.4f} -> feasible {strong.corr_feasible:+.4f} "
          f"(retained {strong.retained:.2f}).")
    print("The box constraint attenuates the axis asymmetrically, and it bites HARDEST")
    print("at the design the paper calls its cleanest adversarial test.")


if __name__ == "__main__":
    main()
