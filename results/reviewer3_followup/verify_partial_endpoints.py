"""Gate: partial sharing must reduce exactly to its two endpoint environments."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from friction_marl.experiments.factorial_design import build_action_map
from results.adversarial_rerun.dgp import sample_agent_params_equicorr
import results.review_upgrade.b_vdn_runner as shared
from results.reviewer3_controls.separable_runner import run_condition_separable
from results.partial_sharing.partial_runner import run_condition_partial


def matrix(rows: list[dict], columns: list[str]) -> np.ndarray:
    return np.asarray([[row[column] for column in columns] for row in rows], dtype=float)


def main() -> None:
    device = torch.device("cpu")
    seed = 20260813
    n_reps, n_episodes, n_eval = 2, 5, 2
    rho, sigma, epsilon = -0.2, 0.6, 0.0
    action_map = torch.tensor(build_action_map(shared.N_RESOURCES), device=device)
    master = np.random.default_rng(seed)
    probe = torch.tensor(
        master.uniform(0.0, shared.CAPACITY, size=(128, shared.N_RESOURCES)).astype(np.float32),
        device=device,
    )
    condition_seed = int(master.integers(0, 2**62))
    common = [
        "mean_reward", "eval_reward_clean", "eval_reward_matched",
        "eval_reward_explore_001", "convergence_time",
        *[f"agent_{i}_reward" for i in range(shared.N_AGENTS)],
    ]

    shared.sample_agent_params = sample_agent_params_equicorr
    for mode in ("iql", "vdn"):
        shared_rows = shared.run_condition(
            rho, sigma, epsilon, mode, n_reps, n_episodes, device,
            condition_seed, action_map, probe, n_eval, epsilon,
        )
        partial_shared_rows = run_condition_partial(
            sample_agent_params_equicorr, shared.N_RESOURCES, rho, sigma,
            epsilon, mode, n_reps, n_episodes, device, condition_seed,
            action_map, probe, n_eval, epsilon,
        )
        if not np.allclose(
            matrix(shared_rows, common), matrix(partial_shared_rows, common),
            atol=1e-6, rtol=0.0,
        ):
            raise AssertionError(
                f"partial k=m does not reproduce shared-pool endpoint for {mode}"
            )

        separable_rows = run_condition_separable(
            sample_agent_params_equicorr, rho, sigma, epsilon, mode, n_reps,
            n_episodes, device, condition_seed, action_map, probe, n_eval,
            epsilon,
        )
        partial_separable_rows = run_condition_partial(
            sample_agent_params_equicorr, 0, rho, sigma, epsilon, mode,
            n_reps, n_episodes, device, condition_seed, action_map, probe,
            n_eval, epsilon,
        )
        if not np.allclose(
            matrix(separable_rows, common),
            matrix(partial_separable_rows, common),
            atol=1e-6, rtol=0.0,
        ):
            raise AssertionError(
                f"partial k=0 does not reproduce separable endpoint for {mode}"
            )

    print(
        "partial-sharing endpoints PASS for IQL and VDN: "
        "k=0 == separable; k=m == shared"
    )


if __name__ == "__main__":
    main()
