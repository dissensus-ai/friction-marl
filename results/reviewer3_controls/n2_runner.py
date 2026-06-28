"""EXPERIMENT 2 -- n=2 CLEAN STRONG-OPPOSITION (reviewer3 Critical #2 + Major #4).

Reviewer concern: (a) the two-team n=4 design confounds strong BETWEEN-team
opposition with strong WITHIN-team cooperation; (b) the equicorrelated n=4 sweep is
PSD-capped at rho = -1/(n-1) = -1/3, so it can never reach genuine strong opposition.

CLEAN FIX: with n=2 agents there is exactly ONE pair, so the equicorrelated rho
reaches FULL -1 with NO within-team-cooperation confound. We run the SAME shared-pool
main env (`b_vdn_runner.run_condition`, UNCHANGED) but with N_AGENTS=2, sweeping the
signed correlation rho in {-1.0, -0.6, -0.2, 0.0, +0.4, +0.8}. This is the genuinely
strong-adversarial test the n=4 design could not give.

We monkeypatch ONLY (i) N_AGENTS -> 2 and (ii) the preference sampler -> the signed
equicorrelated DGP. The env dynamics, network, hyperparameters, losses and seeding
are the verbatim shared-pool runner.

Output CSV uses a `rho` column + `env='shared'`, `n_agents=2`.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

import results.review_upgrade.b_vdn_runner as vr  # noqa: E402
from results.adversarial_rerun.dgp import sample_agent_params_equicorr  # noqa: E402
from friction_marl.experiments.factorial_design import build_action_map  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["iql", "vdn"])
    ap.add_argument("--rhos", nargs="+", type=float,
                    default=[-1.0, -0.6, -0.2, 0.0, 0.4, 0.8])
    ap.add_argument("--sigmas", nargs="+", type=float, default=[0.6, 1.0])
    ap.add_argument("--epsilon", type=float, default=0.0)
    ap.add_argument("--n-reps", type=int, default=30)
    ap.add_argument("--n-episodes", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    # --- monkeypatch: n=2 + signed equicorr sampler. Everything else verbatim. ---
    vr.N_AGENTS = 2
    vr.sample_agent_params = lambda rng, A, M, ctrl, sigma: \
        sample_agent_params_equicorr(rng, A, M, ctrl, sigma)

    if args.device == "cpu":
        device = torch.device("cpu"); dev_name = "CPU"
    else:
        device = torch.device(f"cuda:{int(args.device)}")
        dev_name = torch.cuda.get_device_name(int(args.device))

    action_map = build_action_map(vr.N_RESOURCES)
    action_map_gpu = torch.tensor(action_map, device=device)
    master = np.random.default_rng(args.seed)
    probe = master.uniform(0.0, vr.CAPACITY, size=(128, vr.N_RESOURCES)).astype(np.float32)
    probe_gpu = torch.tensor(probe, device=device)

    cond_keys = [(a, s) for a in args.rhos for s in args.sigmas]
    cond_seeds = {k: int(master.integers(0, 2**62)) for k in cond_keys}

    all_rows: List[dict] = []
    out_path = Path(args.out)
    t0 = time.time()
    total = len(args.modes) * len(cond_keys)
    done = 0
    print(f"=== n=2 CLEAN STRONG-OPPOSITION (shared pool) on {device} ({dev_name}) ===")
    print(f"modes={args.modes} rhos={args.rhos} sigmas={args.sigmas} "
          f"eps={args.epsilon} reps={args.n_reps} eps_len={args.n_episodes}  N_AGENTS={vr.N_AGENTS}")
    for mode in args.modes:
        for (rho, s) in cond_keys:
            tc = time.time()
            rows = vr.run_condition(rho, s, args.epsilon, mode, args.n_reps,
                                    args.n_episodes, device, cond_seeds[(rho, s)],
                                    action_map_gpu, probe_gpu)
            for r in rows:
                r["rho"] = r.pop("alpha")
                r["env"] = "shared"
                r["n_agents"] = 2
            all_rows.extend(rows)
            done += 1
            pd.DataFrame(all_rows).to_csv(out_path, index=False)
            print(f"  [{done}/{total}] {mode} rho={rho:+.2f} s={s:.1f}  "
                  f"gap={-np.mean([r['mean_reward'] for r in rows]):.4f}  "
                  f"{time.time()-tc:.1f}s", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    print(f"\nDone in {(time.time()-t0)/60:.1f} min -> {out_path}")


if __name__ == "__main__":
    main()
