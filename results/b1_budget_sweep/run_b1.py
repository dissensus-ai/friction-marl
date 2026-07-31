"""B1: is MAPPO undertrained relative to IQL's ~25x optimiser-step budget?

The 29-30 Jul session answered this (plateau by 250 episodes, indistinguishable from
IQL at both cells) but its CSVs lived only in an ephemeral scratchpad and the exact
invocation was not recorded -- so the measurement was unreproducible and its p-values
cannot be cited. This is the committed replacement: same design, fresh seeds, CSV in
the repo. Numbers are expected to move within sampling noise relative to the session
record; the session table is superseded by this file's output.

Design: cells (alpha=0.0, sigma=0.6) easy and (alpha=-0.8, sigma=1.0) adversarial
(alpha enters the b_vdn_runner DGP as |alpha| -- the magnitude family, matching the
original run's cell labels); mappo at 250/500/1000 episodes, iql at 1000 as anchor;
eps=0; 30 reps. DV of record: held-out eval reward = -eval_gap (b_vdn_runner stores
eval_gap as the NEGATED eval reward).

Run:  PYTHONPATH=<repo> python3 results/b1_budget_sweep/run_b1.py --device 0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

import results.review_upgrade.b_vdn_runner as vr  # noqa: E402
from friction_marl.experiments.factorial_design import build_action_map  # noqa: E402

CELLS = [
    {"label": "easy", "alpha": 0.0, "sigma": 0.6},
    {"label": "adversarial", "alpha": -0.8, "sigma": 1.0},
]
JOBS = [("mappo", 250), ("mappo", 500), ("mappo", 1000), ("iql", 1000)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--n-reps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1031)
    ap.add_argument("--out", type=str, default=str(HERE / "b1_budget_sweep.csv"))
    args = ap.parse_args()

    device = torch.device("cpu") if args.device == "cpu" else torch.device(f"cuda:{int(args.device)}")
    action_map_gpu = torch.tensor(build_action_map(vr.N_RESOURCES), device=device)
    master = np.random.default_rng(args.seed)
    probe = master.uniform(0.0, vr.CAPACITY, size=(128, vr.N_RESOURCES)).astype(np.float32)
    probe_gpu = torch.tensor(probe, device=device)

    # Seed per (cell, mode, episodes) so any single run is independently regenerable.
    seeds = {(c["label"], m, e): int(master.integers(0, 2**62)) for c in CELLS for (m, e) in JOBS}

    rows_all, t0 = [], time.time()
    for cell in CELLS:
        for mode, n_ep in JOBS:
            tc = time.time()
            rows = vr.run_condition(cell["alpha"], cell["sigma"], 0.0, mode, args.n_reps,
                                    n_ep, device, seeds[(cell["label"], mode, n_ep)],
                                    action_map_gpu, probe_gpu)
            for r in rows:
                r["cell"] = cell["label"]
                r["n_episodes"] = n_ep
                r["eval_reward"] = -r["eval_gap"]
            rows_all.extend(rows)
            pd.DataFrame(rows_all).to_csv(args.out, index=False)
            ev = -np.mean([r["eval_gap"] for r in rows])
            print(f"  {cell['label']:<12} {mode:<6} ep={n_ep:<5} eval_reward={ev:+.4f}  "
                  f"{time.time()-tc:.1f}s", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    df = pd.DataFrame(rows_all)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min -> {args.out}")

    from scipy import stats
    print("\nWelch tests on eval_reward (mirrors the session-record table):")
    for label in ("easy", "adversarial"):
        d = df[df.cell == label]
        m = {e: d[(d["mode"] == "mappo") & (d.n_episodes == e)]["eval_reward"].values for e in (250, 500, 1000)}
        iql = d[(d["mode"] == "iql") & (d.n_episodes == 1000)]["eval_reward"].values
        for a, b, name in [(m[250], m[500], "mappo 250 vs 500"),
                           (m[500], m[1000], "mappo 500 vs 1000"),
                           (m[1000], iql, "mappo 1000 vs iql 1000")]:
            t, p = stats.ttest_ind(a, b, equal_var=False)
            print(f"  {label:<12} {name:<24} t={t:+.2f} p={p:.3f}")


if __name__ == "__main__":
    main()
