"""Add reproducible condition seeds and preference-instance hashes to follow-up CSVs."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from results.adversarial_rerun.dgp import (
    sample_agent_params_equicorr,
    sample_agent_params_equicorr_feasible,
)
from results.review_upgrade.b_vdn_runner import N_AGENTS, N_RESOURCES, problem_instance_hash


def augment(path: Path) -> None:
    data = pd.read_csv(path)
    required = {"rho", "sigma", "replication", "eval_seed", "target_support"}
    if missing := required.difference(data.columns):
        raise ValueError(f"{path} missing columns {sorted(missing)}")
    hashes: dict[tuple[float, float, int], str] = {}
    latent_hashes: dict[tuple[float, float, int], str] = {}
    seeds: dict[tuple[float, float], int] = {}
    for (rho, sigma, support), group in data.groupby(["rho", "sigma", "target_support"]):
        eval_seeds = group["eval_seed"].unique()
        if len(eval_seeds) != 1:
            raise ValueError(f"{path}: cell has multiple eval seeds")
        condition_seed = int(eval_seeds[0]) - 1_000_003
        seeds[(rho, sigma)] = condition_seed
        sampler = (
            sample_agent_params_equicorr_feasible
            if support == "feasible-centered"
            else sample_agent_params_equicorr
        )
        rng = np.random.default_rng(condition_seed)
        replications = sorted(group["replication"].astype(int).unique())
        if replications != list(range(len(replications))):
            raise ValueError(f"{path}: non-contiguous replication ids in cell")
        for replication in replications:
            weights, targets = sampler(rng, N_AGENTS, N_RESOURCES, rho, sigma)
            hashes[(rho, sigma, replication)] = problem_instance_hash(weights, targets)
            canonical_targets = targets - 5.0 if support == "feasible-centered" else targets
            latent_hashes[(rho, sigma, replication)] = problem_instance_hash(
                weights, canonical_targets
            )
    data["condition_seed"] = [seeds[(r.rho, r.sigma)] for r in data.itertuples()]
    data["problem_instance_sha256"] = [
        hashes[(r.rho, r.sigma, int(r.replication))] for r in data.itertuples()
    ]
    data["latent_problem_sha256"] = [
        latent_hashes[(r.rho, r.sigma, int(r.replication))] for r in data.itertuples()
    ]
    data.to_csv(path, index=False)
    print(f"augmented {len(data)} rows -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path)
    args = ap.parse_args()
    for path in args.paths:
        augment(path)


if __name__ == "__main__":
    main()
