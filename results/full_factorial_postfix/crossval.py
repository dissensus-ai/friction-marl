"""GPU/CPU cross-validation, recomputed after the CPU arm's defects were fixed.

Written 6 Aug 2026. Source of record for every number in the manuscript's
cross-validation subsection.

BACKGROUND. The CPU arm carried three defects the GPU arm did not:
  1. action encoding -- build_action_map emits {-1,0,+1} but the env decoder
     expects MultiDiscrete {0,1,2} and subtracts one, giving deltas {-2,-1,0}.
     The shared resource state could only ever FALL.
  2. observation noise scaled by eps rather than sqrt(eps) (covariance eps^2 I
     where Definition 3.3 specifies eps I).
  3. replay stored a CLEAN next observation beside a noisy current one.
Only (1) is unique to the CPU path; (2) and (3) were fixed on the GPU side in
July and never propagated. All three are fixed as of 2c0bfd5 and the arm was
re-run in full (3,750 runs, 32.8 h).

WHAT THIS TESTS. The manuscript attributed a systematic one-directional offset
of 0.478 between the implementations to defect (1): an environment whose state
can only decrease sits further from every agent's target. That is a falsifiable
prediction -- fix the decoder and the offset should vanish. It does.

Run:  python3 results/full_factorial_postfix/crossval.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
K = ["alpha", "sigma", "epsilon"]


def condition_means(df, col="mean_reward", name="v"):
    return df.groupby(K, as_index=False)[col].mean().rename(columns={col: name})


def icc_3_1(a, b):
    """Two-way mixed, single measure, absolute agreement is NOT what we want
    here -- consistency (3,1) is, since a pure level shift is the thing under
    test and we report it separately as the offset."""
    x = np.vstack([a, b]).T
    n, k = x.shape
    gm = x.mean()
    ms_r = k * ((x.mean(1) - gm) ** 2).sum() / (n - 1)
    ms_e = ((x - x.mean(1, keepdims=True) - x.mean(0) + gm) ** 2).sum() / ((n - 1) * (k - 1))
    return float((ms_r - ms_e) / (ms_r + (k - 1) * ms_e))


def main():
    new = condition_means(
        pd.read_csv(REPO / "results/full_factorial_postfix/replication_results.csv"),
        name="cpu_new")
    gpu = condition_means(
        pd.read_csv(REPO / "results/gpu_factorial_postfix/replication_results.csv"),
        name="gpu")
    old = pd.read_csv(REPO / "results/full_factorial/analysis/metrics.csv")
    old["cpu_old"] = -old["reward_gap"]          # gap = -mean reward

    m = new.merge(gpu, on=K).merge(old[K + ["cpu_old"]], on=K)
    assert len(m) == 125, f"expected 125 matched conditions, got {len(m)}"

    print(f"matched conditions: {len(m)}\n")
    print(f"{'arm':30}{'offset':>9}{'Spearman':>10}{'Pearson':>9}{'ICC(3,1)':>10}")
    for label, col in [("OLD CPU (broken decoder)", "cpu_old"),
                       ("NEW CPU (all three fixed)", "cpu_new")]:
        print(f"{label:30}{(m[col] - m.gpu).mean():+9.3f}"
              f"{stats.spearmanr(m[col], m.gpu).statistic:10.3f}"
              f"{stats.pearsonr(m[col], m.gpu).statistic:9.3f}"
              f"{icc_3_1(m[col].values, m.gpu.values):10.3f}")

    d = m.cpu_new - m.gpu
    lo, hi = d.mean() - 1.96 * d.std(), d.mean() + 1.96 * d.std()
    inside = 100 * ((d > lo) & (d < hi)).mean()
    print(f"\nnew arm Bland-Altman: bias {d.mean():+.3f}, "
          f"limits of agreement [{lo:.3f}, {hi:.3f}], {inside:.1f}% inside")

    by_eps = m.groupby("epsilon").apply(lambda g: (g.cpu_new - g.gpu).mean(),
                                        include_groups=False)
    print("offset by epsilon: " + ", ".join(f"{k}={v:+.3f}" for k, v in by_eps.items()))

    out = REPO / "results/full_factorial_postfix/crossval_conditions.csv"
    m.to_csv(out, index=False)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
