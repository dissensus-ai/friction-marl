"""Did the 29-30 Jul corrections actually bite, and where?

Written 4 Aug 2026. The manuscript's Limitations section discloses two defects in
the factorial runner:
  (i)  observation noise scaled by epsilon instead of sqrt(epsilon), so realized
       covariance was eps^2 I where Definition 3.3 specifies eps I;
  (ii) the replay buffer stored a CLEAN next observation beside a noisy current
       one, so TD targets bootstrapped on unobserved state.

Both are gated on epsilon > 0. So if the corrections did what they should, the
conditions at epsilon = 0 should move only by run-to-run noise while the
epsilon > 0 conditions move more. This script tests that, and is the source of
record for the numbers quoted in the disclosure paragraph.

IMPORTANT SCOPE NOTE. The pre-fix run survives only as CONDITION MEANS
(results/gpu_factorial/analysis/condition_statistics.csv, n=30 per condition);
its per-replication CSV was never committed. The comparison below is therefore
condition-level on both sides, and is a paired comparison across the 125 matched
(alpha, sigma, epsilon) cells. A per-replication version is not recoverable.

Run:  python3 results/gpu_factorial_postfix/fix_impact.py
"""
from pathlib import Path

import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
PRE = REPO / "results/gpu_factorial/analysis/condition_statistics.csv"
POST = REPO / "results/gpu_factorial_postfix/replication_results.csv"
KEYS = ["alpha", "sigma", "epsilon"]


def main():
    pre = pd.read_csv(PRE)[KEYS + ["reward_mean"]].rename(
        columns={"reward_mean": "pre"})
    post = (pd.read_csv(POST)
            .groupby(KEYS, as_index=False)["mean_reward"].mean()
            .rename(columns={"mean_reward": "post"}))

    df = pre.merge(post, on=KEYS, validate="one_to_one")
    assert len(df) == 125, f"expected 125 matched conditions, got {len(df)}"
    df["abs_delta"] = (df.post - df.pre).abs()

    zero = df[df.epsilon == 0].abs_delta
    pos = df[df.epsilon > 0].abs_delta

    # Welch (unequal n and variance) plus a rank test, since |delta| is bounded
    # below at zero and right-skewed.
    t_p = stats.ttest_ind(pos, zero, equal_var=False).pvalue
    u_p = stats.mannwhitneyu(pos, zero, alternative="greater").pvalue

    print(f"matched conditions:            {len(df)}")
    print(f"mean |change|, epsilon = 0:    {zero.mean():.4f}  (n={len(zero)}, "
          f"median {zero.median():.4f})")
    print(f"mean |change|, epsilon > 0:    {pos.mean():.4f}  (n={len(pos)}, "
          f"median {pos.median():.4f})")
    print(f"ratio:                         {pos.mean()/zero.mean():.2f}x")
    print(f"Welch t-test  (pos > zero):    p = {t_p:.4g}")
    print(f"Mann-Whitney  (pos > zero):    p = {u_p:.4g}")
    print()
    print("By epsilon level:")
    for e, g in df.groupby("epsilon"):
        print(f"  eps={e:<5} mean |change| = {g.abs_delta.mean():.4f}  (n={len(g)})")

    out = REPO / "results/gpu_factorial_postfix/fix_impact.csv"
    df.to_csv(out, index=False)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
