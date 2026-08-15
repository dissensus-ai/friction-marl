#!/usr/bin/env python3
"""Analyze the frozen epsilon-zero, zero-bias residual diagnostic.

The outcome is mean reward (higher is better), matching the -0.166 CPU-minus-GPU
offset named in PROTOCOL.md.  Bootstrap resampling uses the 25 factorial
conditions as the sampling unit; replications are first averaged within cell.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent
ZERO_PATH = HERE / "replication_results.csv"
CPU_PATH = RESULTS / "full_factorial_postfix" / "replication_results.csv"
GPU_PATH = RESULTS / "gpu_factorial_postfix" / "replication_results.csv"
KEYS = ["alpha", "sigma", "epsilon", "replication"]
CELL_KEYS = ["alpha", "sigma"]
MARGIN = 0.10
N_BOOT = 50_000
BOOT_SEED = 20260808


def condition_bootstrap(values: np.ndarray) -> tuple[float, float]:
    """Percentile CI for a mean, resampling conditions with replacement."""
    rng = np.random.default_rng(BOOT_SEED)
    draws = rng.choice(values, size=(N_BOOT, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def eps0(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame.loc[np.isclose(frame["epsilon"], 0.0)].copy()


def main() -> None:
    if not ZERO_PATH.exists():
        raise SystemExit(f"No zero-bias results yet: {ZERO_PATH}")

    zero = eps0(ZERO_PATH)
    cpu = eps0(CPU_PATH)
    gpu = eps0(GPU_PATH)

    reps = sorted(int(x) for x in zero["replication"].unique())
    expected_reps = list(range(max(reps) + 1))
    if reps != expected_reps:
        raise ValueError(f"Non-contiguous replication stage: found {reps}")
    stage = len(reps)

    counts = zero.groupby(CELL_KEYS).size()
    if len(counts) != 25 or not (counts == stage).all():
        raise ValueError(
            f"Expected 25 complete cells x {stage} reps; got {counts.describe().to_dict()}"
        )

    gpu_cell = gpu.groupby(CELL_KEYS, as_index=False)["mean_reward"].mean().rename(
        columns={"mean_reward": "gpu_mean_reward"}
    )
    zero_cell = zero.groupby(CELL_KEYS, as_index=False)["mean_reward"].mean().rename(
        columns={"mean_reward": "zero_bias_mean_reward"}
    )
    cells = zero_cell.merge(gpu_cell, on=CELL_KEYS, validate="one_to_one")
    cells["zero_bias_minus_gpu"] = (
        cells["zero_bias_mean_reward"] - cells["gpu_mean_reward"]
    )

    matched_cpu = cpu.loc[cpu["replication"].isin(reps)]
    paired = zero.merge(
        matched_cpu[KEYS + ["mean_reward"]],
        on=KEYS,
        how="inner",
        validate="one_to_one",
        suffixes=("_zero_bias", "_default_bias"),
    )
    if len(paired) != len(zero):
        raise ValueError(f"Matched {len(paired)} of {len(zero)} zero-bias rows")
    paired["zero_minus_default"] = (
        paired["mean_reward_zero_bias"] - paired["mean_reward_default_bias"]
    )
    paired_cell = paired.groupby(CELL_KEYS, as_index=False)["zero_minus_default"].mean()

    default_cell = (
        matched_cpu.groupby(CELL_KEYS, as_index=False)["mean_reward"].mean()
        .rename(columns={"mean_reward": "default_cpu_mean_reward"})
        .merge(gpu_cell, on=CELL_KEYS, validate="one_to_one")
    )
    default_cell["default_cpu_minus_gpu"] = (
        default_cell["default_cpu_mean_reward"] - default_cell["gpu_mean_reward"]
    )

    primary = float(cells["zero_bias_minus_gpu"].mean())
    primary_ci = condition_bootstrap(cells["zero_bias_minus_gpu"].to_numpy())
    secondary = float(paired_cell["zero_minus_default"].mean())
    secondary_ci = condition_bootstrap(paired_cell["zero_minus_default"].to_numpy())
    matched_default_offset = float(default_cell["default_cpu_minus_gpu"].mean())

    moved_toward_gpu = abs(primary) < abs(matched_default_offset)
    primary_ci_contained = primary_ci[0] >= -MARGIN and primary_ci[1] <= MARGIN
    primary_ci_outside = primary_ci[1] < -MARGIN or primary_ci[0] > MARGIN
    # A paired initialization change with the same sign as the default CPU-GPU
    # offset pushes the CPU arm farther away.  No change is inconclusive rather
    # than a wrong-way result.
    moved_wrong_way = secondary * matched_default_offset > 0

    if primary_ci_contained and moved_toward_gpu:
        decision = "support"
    elif primary_ci_outside or moved_wrong_way:
        decision = "refute"
    else:
        decision = "extend"

    summary = {
        "stage_replications_per_condition": stage,
        "n_conditions": 25,
        "outcome": "mean_reward_higher_is_better",
        "equivalence_margin": MARGIN,
        "bootstrap_conditions": N_BOOT,
        "zero_bias_cpu_minus_gpu": {
            "estimate": primary,
            "ci95": list(primary_ci),
        },
        "zero_bias_minus_default_cpu_paired": {
            "estimate": secondary,
            "ci95": list(secondary_ci),
        },
        "matched_default_cpu_minus_gpu": matched_default_offset,
        "moved_toward_gpu": moved_toward_gpu,
        "moved_wrong_way": moved_wrong_way,
        "decision": decision,
    }

    analysis_dir = HERE / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    summary_text = json.dumps(summary, indent=2) + "\n"
    cells.to_csv(analysis_dir / "condition_offsets.csv", index=False)
    cells.to_csv(analysis_dir / f"condition_offsets_stage_{stage:02d}.csv", index=False)
    paired.to_csv(analysis_dir / "paired_replication_changes.csv", index=False)
    paired.to_csv(
        analysis_dir / f"paired_replication_changes_stage_{stage:02d}.csv", index=False
    )
    (analysis_dir / "summary.json").write_text(summary_text, encoding="utf-8")
    (analysis_dir / f"summary_stage_{stage:02d}.json").write_text(
        summary_text, encoding="utf-8"
    )
    print(summary_text, end="")


if __name__ == "__main__":
    main()
