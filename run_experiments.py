"""Reference CPU entry point for the original sign-blind IQL factorial.

This is a portable baseline/smoke-test runner.  It is not the generator for the
paper's post-fix GPU results or signed control battery; those commands live in
the corresponding ``results/*/FINDINGS.md`` files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from friction_marl.experiments.factorial_design import run_factorial_experiment
from friction_marl.experiments.analysis import run_analysis
from friction_marl.utils.visualization import (
    plot_heatmaps,
    plot_learning_curves,
    plot_model_comparison,
    plot_regression_diagnostics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the reference CPU sign-blind IQL factorial"
    )
    parser.add_argument("--n-agents", type=int, default=4)
    parser.add_argument("--n-resources", type=int, default=3)
    parser.add_argument("--n-replications", type=int, default=30)
    parser.add_argument("--n-episodes", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alphas", type=float, nargs="+", default=None)
    parser.add_argument("--sigmas", type=float, nargs="+", default=None)
    parser.add_argument("--epsilons", type=float, nargs="+", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    episode_df, replication_df, policy_vectors = run_factorial_experiment(
        n_agents=args.n_agents,
        n_resources=args.n_resources,
        n_replications=args.n_replications,
        n_episodes=args.n_episodes,
        seed=args.seed,
        output_dir=output_dir,
        alphas=args.alphas,
        sigmas=args.sigmas,
        epsilons=args.epsilons,
    )

    policy_path = output_dir / "policy_vectors.npz"
    np.savez(policy_path, **policy_vectors)

    metrics_df, regression_df = run_analysis(output_dir)

    analysis_dir = output_dir / "analysis"
    plot_heatmaps(metrics_df, analysis_dir)
    plot_learning_curves(episode_df, analysis_dir)
    plot_regression_diagnostics(metrics_df, analysis_dir)
    plot_model_comparison(regression_df, analysis_dir)


if __name__ == "__main__":
    main()
