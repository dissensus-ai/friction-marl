"""Verify the realised cross-agent target correlation matches the target rho for
the n=2 equicorrelated DGP used in Experiment 2 (the gate before trusting the
strong-opposition contrast). For n=2 the PSD bound is rho >= -1/(n-1) = -1, so
rho = -1 (perfect anti-correlation) is reachable -- the eigh-based square root is
exact at the boundary. Also re-confirms the n=4 separable run reuses the SAME
verified equicorr DGP.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from results.adversarial_rerun.dgp import (  # noqa: E402
    sample_agent_params_equicorr, realised_mean_pairwise_corr,
)

N_RESOURCES = 3
SIGMA = 1.0
N_DRAWS = 20000
SEED = 7


def main():
    print("=" * 72)
    print("DGP correlation gate -- realised pairwise corr vs target rho")
    print("=" * 72)

    rows = []
    # ---- n=2 (Experiment 2): full signed range incl. rho = -1 ----
    print("\n[n=2 equicorr]  PSD bound: rho >= -1  (rho=-1 reachable)")
    for rho in [-1.0, -0.6, -0.2, 0.0, 0.4, 0.8]:
        pooled, per_draw = realised_mean_pairwise_corr(
            sample_agent_params_equicorr, rho, SIGMA,
            n_agents=2, n_resources=N_RESOURCES, n_draws=N_DRAWS, seed=SEED)
        print(f"  rho_target={rho:+.2f}  ->  pooled={pooled:+.4f}   "
              f"per_draw(m={N_RESOURCES})={per_draw:+.4f}")
        rows.append({"n_agents": 2, "rho_target": rho,
                     "pooled_corr": pooled, "per_draw_corr": per_draw})

    # ---- n=4 (Experiment 1 separable reuses the SAME DGP): PSD bound -1/3 ----
    print("\n[n=4 equicorr]  PSD bound: rho >= -1/3  (same DGP as separable run)")
    for rho in [-0.33, -0.2, -0.1, 0.0, 0.2, 0.4, 0.6, 0.8]:
        pooled, per_draw = realised_mean_pairwise_corr(
            sample_agent_params_equicorr, rho, SIGMA,
            n_agents=4, n_resources=N_RESOURCES, n_draws=N_DRAWS, seed=SEED)
        print(f"  rho_target={rho:+.2f}  ->  pooled={pooled:+.4f}   "
              f"per_draw(m={N_RESOURCES})={per_draw:+.4f}")
        rows.append({"n_agents": 4, "rho_target": rho,
                     "pooled_corr": pooled, "per_draw_corr": per_draw})

    df = pd.DataFrame(rows)
    out = HERE / "dgp_corr_verification.csv"
    df.to_csv(out, index=False)

    # gate check
    n2 = df[df["n_agents"] == 2]
    max_err = float((n2["pooled_corr"] - n2["rho_target"]).abs().max())
    reaches_neg1 = float(n2[n2["rho_target"] == -1.0]["pooled_corr"].iloc[0])
    print("\n" + "-" * 72)
    print(f"n=2 max |pooled - target| = {max_err:.4f}")
    print(f"n=2 realised corr at rho=-1 = {reaches_neg1:+.4f}  "
          f"({'GATE PASSED: reaches genuine -1' if reaches_neg1 < -0.95 else 'CHECK'})")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
