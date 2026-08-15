"""Independent small-horizon checks for the exact shared-pool oracle."""
from __future__ import annotations

from itertools import product
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from friction_marl.utils.dynamic_oracle import shared_pool_oracle_components


def reward(state: float, weights: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(-weights * (state - targets) ** 2))


def brute_two_step(start: float, weights: np.ndarray, targets: np.ndarray) -> float:
    best = -np.inf
    for d1, d2 in product(range(-4, 5), repeat=2):
        s1 = np.clip(start + d1, 0.0, 10.0)
        s2 = np.clip(s1 + d2, 0.0, 10.0)
        best = max(best, (reward(s1, weights, targets) + reward(s2, weights, targets)) / 2)
    return best


def main() -> None:
    rng = np.random.default_rng(20260813)
    starts = [0.0, 1.0, 5.0, 10.0] + [float(rng.uniform(0.0, 10.0)) for _ in range(50)]
    for start in starts:
        weights = rng.uniform(0.2, 1.0, size=4)
        targets = rng.normal(1.0, 2.0, size=4)
        got = shared_pool_oracle_components(
            weights[None, :, None], targets[None, :, None],
            np.array([[[start]]]), horizon=2, capacity=10,
        )["dynamic_oracle"][0]
        want = brute_two_step(start, weights, targets)
        if not np.isclose(got, want, atol=1e-10):
            raise AssertionError(f"two-step oracle mismatch: {got} vs {want}")
    print(f"{len(starts)}/{len(starts)} independent two-step brute-force checks PASS")


if __name__ == "__main__":
    main()
