"""Isolated recompute of the Appendix A.1 policy-vs-reward stability figures
(paper Sec. "Dynamic equilibria", `sec:dynamic-eq`).

Why this exists
---------------
The A.1 headline numbers are produced by `dynamics_analysis.py`, but that script is a
~550-line figure generator (loads matplotlib, renders ~30 PNG/PDF panels) and the two
stability fractions are not separated out. This file isolates *only* the A.1 provenance:
the exact computations, no plotting, with a self-check against the reported values.

The three logic blocks below are lifted verbatim (parameters and all) from
`dynamics_analysis.py`:
  * policy convergence  -> `compute_convergence_time` (delta_pi = 1e-3, L2), MAX_CONV = 1200
  * reward convergence  -> reward-plateau within 10% of final, tick grid arange(20)*50+25
  * late-training std   -> std of last-250-episode window, grouped by alpha

Reported (paper):  policy 0.85% (32/3750) | reward 99.3% | late-std ~0.04-0.06,
alpha=0 marginally higher (0.056 vs 0.043-0.050).

Data inputs (per-condition production data of the 5x5x5 GPU factorial):
    results/gpu_factorial/replication_results.csv   (has convergence_time, policy_key)
    results/gpu_factorial/learning_curves.npz        (per-run windowed reward curve)
NOTE: both match `.gitignore` patterns (`results/**/replication_results.csv`, `*.npz`)
and are NOT committed -- they are the Zenodo artifact bundle
(concept DOI 10.5281/zenodo.20636467). Fetch them into results/gpu_factorial/ before
running, or point --base at a local copy.

Run:  python results/gpu_factorial/analysis/a1_stability_recompute.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# --- parameters, identical to dynamics_analysis.py -------------------------------
N_EPISODES = 1000
LC_WINDOW = 50
N_WINDOWS = N_EPISODES // LC_WINDOW          # 20
EPISODE_TICKS = np.arange(N_WINDOWS) * LC_WINDOW + LC_WINDOW // 2   # [25, 75, ..., 975]
MAX_CONV = 1200.0                            # censoring bound for convergence_time
DELTA_PI = 1e-3                              # policy-stability tolerance (documented)
REWARD_TOL = 0.10                            # reward-plateau tolerance (10% of final)
LATE_EPISODES = 250


def policy_convergence_rate(df: pd.DataFrame):
    conv = df["convergence_time"] < MAX_CONV
    return float(conv.mean()), int(conv.sum()), len(df)


def reward_convergence_rate(df: pd.DataFrame, lc) -> float:
    flags = []
    for _, row in df.iterrows():
        key = row["policy_key"]
        if key not in lc:
            continue
        curve = lc[key]
        final_val = curve[-3:].mean()
        if abs(final_val) < 1e-8:
            flags.append(True)
            continue
        tol = REWARD_TOL * abs(final_val)
        conv_ep = N_EPISODES
        for w in range(len(curve)):
            if all(abs(curve[w2] - final_val) <= tol for w2 in range(w, len(curve))):
                conv_ep = EPISODE_TICKS[w]
                break
        flags.append(conv_ep < N_EPISODES)
    return float(np.mean(flags))


def late_training_std(df: pd.DataFrame, lc) -> pd.Series:
    nlast = max(1, int(N_WINDOWS * LATE_EPISODES / N_EPISODES))
    rows = []
    for _, row in df.iterrows():
        key = row["policy_key"]
        if key in lc:
            rows.append({"alpha": row["alpha"],
                         "late_std": float(np.std(lc[key][-nlast:]))})
    return pd.DataFrame(rows).groupby("alpha")["late_std"].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path,
                    default=Path(__file__).resolve().parents[1])  # results/gpu_factorial
    args = ap.parse_args()
    df = pd.read_csv(args.base / "replication_results.csv")
    lc = np.load(args.base / "learning_curves.npz")

    p_rate, p_n, n = policy_convergence_rate(df)
    r_rate = reward_convergence_rate(df, lc)
    lstd = late_training_std(df, lc)

    print(f"replications: {n}")
    print(f"policy  convergence: {100*p_rate:.3f}%  ({p_n}/{n})   [paper 0.85%, 32 runs]")
    print(f"reward  convergence: {100*r_rate:.3f}%              [paper 99.3%]")
    print("late-training std by alpha (paper ~0.04-0.06, alpha=0 = 0.056):")
    for a, v in lstd.items():
        print(f"    alpha={a:+.1f}: {v:.3f}")

    ok = (p_n == 32 and abs(p_rate - 0.0085) < 5e-4 and abs(r_rate - 0.993) < 5e-3
          and abs(lstd.get(0.0, np.nan) - 0.056) < 5e-3)
    print("\nA.1 figures reproduce the paper." if ok
          else "\n*** MISMATCH vs paper A.1 figures ***")


if __name__ == "__main__":
    main()
