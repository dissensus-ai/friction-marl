"""rliable robustness report for the IQL / VDN / MAPPO comparison (ADDITIVE).

Aggregates the per-replication episode-reward CSVs the runners emit into the
stratified-bootstrap estimators recommended by Agarwal et al. (2021), "Deep RL at
the Edge of the Statistical Precipice" (NeurIPS): interquartile mean (IQM) with
stratified-bootstrap confidence intervals, performance profiles (score
distributions), and pairwise probability-of-improvement. These are reported
INSTEAD of a single mean +/- SEM so the algorithm comparison is robust to
per-run outliers and heavy tails.

INPUT
-----
Any mix of the runner CSVs (each row = one replication of one condition), passed
either as explicit files or discovered by glob under --input-dir. The algorithm
label is taken from the `mode` column when present, else inferred from the file
name suffix (`*_iql.csv` / `*_vdn.csv` / `*_mappo.csv`). The per-run score is the
`--metric` column (default `mean_reward`, higher = better); a "task" is one design
cell (the tuple of whichever of alpha/rho/sigma/epsilon/overlap_k are present), and
"runs" are the replications within a task. Only tasks present for EVERY algorithm
are kept, so the score matrices are aligned and rectangular.

USAGE
-----
    python -m results.rliable_report.rliable_report \
        --inputs results/review_upgrade/vdn_vs_iql.csv \
        --out-dir results/rliable_report

    python -m results.rliable_report.rliable_report \
        --input-dir results/adversarial_rerun --glob 'equicorr_*.csv' \
        --out-dir results/rliable_report/equicorr
"""
from __future__ import annotations

import argparse
import glob as globmod
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Columns that are NEVER part of a task key.
_NON_DESIGN = {
    "mode", "replication", "mean_reward", "convergence_time", "eval_gap",
    "policy_key", "env", "dgp", "n_agents", "w",
}
_DESIGN_ORDER = ["alpha", "rho", "overlap_k", "sigma", "epsilon"]


def _infer_mode(df: pd.DataFrame, path: str) -> pd.Series:
    if "mode" in df.columns:
        return df["mode"].astype(str)
    base = os.path.basename(path).lower()
    for m in ("mappo", "vdn", "iql"):
        if base.endswith(f"_{m}.csv") or f"_{m}" in base:
            return pd.Series([m] * len(df), index=df.index)
    return pd.Series(["unknown"] * len(df), index=df.index)


def _design_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in _DESIGN_ORDER if c in df.columns]
    # any leftover non-agent, non-reserved numeric column also counts as design
    for c in df.columns:
        if c in cols or c in _NON_DESIGN:
            continue
        if c.startswith("agent_") and c.endswith("_reward"):
            continue
        if c not in cols:
            cols.append(c)
    return cols


def load_scores(paths: list[str], metric: str) -> dict[str, np.ndarray]:
    """Build {algo: (n_runs, n_tasks)} score matrices aligned on the shared tasks."""
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        if metric not in df.columns:
            raise SystemExit(f"metric '{metric}' not in {p} (have {list(df.columns)})")
        df = df.copy()
        df["__algo__"] = _infer_mode(df, p)
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)

    design = _design_cols(full)
    full["__task__"] = full[design].astype(str).agg("|".join, axis=1)

    algos = sorted(a for a in full["__algo__"].unique() if a != "unknown")
    if not algos:
        raise SystemExit("no algorithm labels found (need a 'mode' column or *_<mode>.csv names)")

    # tasks common to every algorithm
    task_sets = [set(full.loc[full["__algo__"] == a, "__task__"]) for a in algos]
    common = sorted(set.intersection(*task_sets)) if task_sets else []
    if not common:
        raise SystemExit("no tasks are shared across all algorithms; cannot align matrices")

    score_dict: dict[str, np.ndarray] = {}
    for a in algos:
        sub = full[full["__algo__"] == a]
        cols = []
        n_runs = None
        for task in common:
            vals = sub.loc[sub["__task__"] == task, metric].to_numpy(dtype=float)
            if n_runs is None:
                n_runs = len(vals)
            cols.append(vals[:n_runs] if len(vals) >= n_runs else
                        np.pad(vals, (0, n_runs - len(vals)), constant_values=np.nan))
        score_dict[a] = np.column_stack(cols)  # (n_runs, n_tasks)
    return score_dict


def run_report(score_dict: dict[str, np.ndarray], out_dir: Path,
               reps: int = 2000, n_thresholds: int = 41) -> pd.DataFrame:
    from rliable import library as rly
    from rliable import metrics

    out_dir.mkdir(parents=True, exist_ok=True)
    algos = list(score_dict)

    # --- IQM (+ mean, median, optimality gap) with stratified-bootstrap CIs ---
    def agg(x):
        return np.array([
            metrics.aggregate_median(x),
            metrics.aggregate_iqm(x),
            metrics.aggregate_mean(x),
        ])

    point, cis = rly.get_interval_estimates(score_dict, agg, reps=reps)
    names = ["median", "iqm", "mean"]
    rows = []
    for a in algos:
        for i, nm in enumerate(names):
            rows.append({
                "algorithm": a, "estimator": nm,
                "point": float(point[a][i]),
                "ci_low": float(cis[a][0, i]), "ci_high": float(cis[a][1, i]),
            })
    agg_df = pd.DataFrame(rows)
    agg_df.to_csv(out_dir / "aggregate_estimates.csv", index=False)

    # --- performance profiles (score distributions) ---
    allscores = np.concatenate([v[np.isfinite(v)] for v in score_dict.values()])
    lo, hi = np.percentile(allscores, [1, 99])
    thresholds = np.linspace(lo, hi, n_thresholds)
    score_dist, dist_cis = rly.create_performance_profile(score_dict, thresholds, reps=reps)
    prof_rows = []
    for a in algos:
        for j, tau in enumerate(thresholds):
            prof_rows.append({
                "algorithm": a, "tau": float(tau),
                "fraction_runs_above": float(score_dist[a][j]),
                "ci_low": float(dist_cis[a][0, j]), "ci_high": float(dist_cis[a][1, j]),
            })
    pd.DataFrame(prof_rows).to_csv(out_dir / "performance_profiles.csv", index=False)

    # --- probability of improvement (all ordered pairs) ---
    pairs = {f"{x},{y}": (score_dict[x], score_dict[y])
             for x in algos for y in algos if x != y}
    poi, poi_cis = rly.get_interval_estimates(pairs, metrics.probability_of_improvement, reps=reps)
    poi_rows = []
    for key in pairs:
        x, y = key.split(",")
        poi_rows.append({
            "algorithm_x": x, "algorithm_y": y,
            "prob_x_beats_y": float(np.asarray(poi[key]).ravel()[0]),
            "ci_low": float(np.asarray(poi_cis[key]).ravel()[0]),
            "ci_high": float(np.asarray(poi_cis[key]).ravel()[-1]),
        })
    pd.DataFrame(poi_rows).to_csv(out_dir / "probability_of_improvement.csv", index=False)

    # optional plots (skipped silently if matplotlib is unavailable)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from rliable import plot_utils
        fig, ax = plt.subplots(figsize=(7, 5))
        plot_utils.plot_performance_profiles(
            score_dist, thresholds, performance_profile_cis=dist_cis,
            xlabel=r"episode reward $(\tau)$", ax=ax)
        fig.tight_layout()
        fig.savefig(out_dir / "performance_profiles.png", dpi=150)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"[rliable_report] plot skipped: {exc}")

    return agg_df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="*", default=[], help="explicit CSV files")
    ap.add_argument("--input-dir", type=str, default=None, help="directory to glob")
    ap.add_argument("--glob", type=str, default="*_iql.csv *_vdn.csv *_mappo.csv",
                    help="space-separated glob patterns under --input-dir")
    ap.add_argument("--metric", type=str, default="mean_reward",
                    help="per-run score column (higher = better)")
    ap.add_argument("--out-dir", type=str, default="results/rliable_report")
    ap.add_argument("--reps", type=int, default=2000, help="bootstrap resamples")
    args = ap.parse_args()

    paths = list(args.inputs)
    if args.input_dir:
        for pat in args.glob.split():
            paths.extend(sorted(globmod.glob(os.path.join(args.input_dir, pat))))
    paths = sorted(set(paths))
    if not paths:
        raise SystemExit("no input CSVs (pass --inputs and/or --input-dir/--glob)")

    print(f"[rliable_report] inputs: {paths}")
    score_dict = load_scores(paths, args.metric)
    for a, m in score_dict.items():
        print(f"  {a}: score matrix {m.shape} (runs x tasks)")

    try:
        agg_df = run_report(score_dict, Path(args.out_dir), reps=args.reps)
    except ImportError:
        raise SystemExit(
            "rliable is not installed. `pip install rliable` (already added to "
            "requirements.txt) and re-run.")
    print(f"[rliable_report] wrote estimates to {args.out_dir}")
    print(agg_df.to_string(index=False))


if __name__ == "__main__":
    main()
