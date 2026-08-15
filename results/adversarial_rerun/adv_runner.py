"""STEP 3: targeted re-run on the REDESIGNED (signed) DGP.

Reuses the existing b_vdn_runner machinery (BatchedQNet / JointReplayBuffer /
run_condition / matched per-condition seeds) UNCHANGED -- we only monkeypatch the
preference sampler so the control knob is now the genuinely-signed cross-agent
correlation rho (equicorr) or team-opposition strength opp (team).

Because cond_seeds and the probe are drawn from the master RNG in a fixed order
that depends only on the declared seed grid, --sigmas and --seed (not on --modes), running IQL
in one process and VDN in another with identical --rhos/--sigmas/--seed yields
matched preference draws. ``--seed-grid-rhos`` can preserve those full-grid seeds in a
reduced follow-up. Runs made after deterministic PyTorch seeding was added
also match initial weights; the archived June CSVs predate that guarantee.

Output CSV uses a `rho` column (the signed control) in place of `alpha`.
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
from results.adversarial_rerun.dgp import (  # noqa: E402
    sample_agent_params_equicorr,
    sample_agent_params_equicorr_feasible,
    sample_agent_params_team,
)
from friction_marl.experiments.factorial_design import build_action_map  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dgp", choices=["equicorr", "team"], default="equicorr")
    ap.add_argument("--modes", nargs="+", default=["iql", "vdn", "mappo"])
    # One flag, TWO meanings -- this is the trap. For --dgp equicorr the control IS the
    # signed pairwise correlation rho in [-1/(n-1), 1]. For --dgp team it is the opposition
    # STRENGTH opp in [0, 1], which enters squared (within-team corr +opp^2, between-team
    # -opp^2), so its sign is meaningless and the equicorr default silently duplicates its
    # negative half while never reaching opp=1 where the design's value lies. Leaving the
    # default unset per-dgp below, and validating the range, so that cannot happen again.
    ap.add_argument("--rhos", "--opps", dest="rhos", nargs="+", type=float, default=None,
                    help="equicorr: signed rho in [-1/(n-1), 1]. team: opposition strength "
                         "opp in [0, 1] (sign has NO effect; it enters as opp^2).")
    ap.add_argument(
        "--seed-grid-rhos", nargs="+", type=float, default=None,
        help="optional superset used only to assign stable condition seeds; every --rhos "
             "value must occur in it. This allows a reduced follow-up to pair exactly with "
             "a previously run full grid.",
    )
    ap.add_argument("--sigmas", nargs="+", type=float, default=[0.6, 1.0])
    ap.add_argument("--epsilon", type=float, default=0.0)
    ap.add_argument("--n-reps", type=int, default=30)
    ap.add_argument("--n-episodes", type=int, default=1000)
    ap.add_argument("--n-eval-episodes", type=int, default=100)
    ap.add_argument(
        "--target-support",
        choices=["legacy-gaussian0", "feasible-centered"],
        default="legacy-gaussian0",
        help="legacy targets are N(0,1); feasible-centered translates the same signed "
             "Gaussian design to the box centre and rejects any out-of-box draw",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="0")   # "cpu" or GPU index
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    # --- swap sampler: control knob (passed in place of 'alpha') is rho or opp ---
    if args.dgp == "equicorr":
        sampler = (sample_agent_params_equicorr_feasible
                   if args.target_support == "feasible-centered"
                   else sample_agent_params_equicorr)
        ctrl_name = "rho"
        if args.rhos is None:
            args.rhos = [-0.33, -0.2, -0.1, 0.0, 0.2, 0.4, 0.6, 0.8]
        # Mirror of the team-branch guard: below the PSD bound -1/(n-1) the sampler
        # silently clips realised correlation to the bound while the CSV keeps the
        # nominal label -- the same nominal-vs-realised trap, other branch.
        lo = -1.0 / (vr.N_AGENTS - 1)
        bad = [v for v in args.rhos if v < lo - 1e-9 or v > 1.0]
        if bad:
            ap.error(
                f"--dgp equicorr takes rho in [{lo:.4f}, 1] (PSD bound -1/(n-1) at "
                f"n={vr.N_AGENTS}), got {bad}. Values below the bound are silently "
                "clipped to it while keeping their nominal label, producing flat "
                "duplicated cells that bias any slope regressed on the label."
            )
    else:
        if args.target_support != "legacy-gaussian0":
            ap.error("--target-support feasible-centered is currently defined only for equicorr")
        sampler = sample_agent_params_team
        ctrl_name = "opp"
        if args.rhos is None:
            args.rhos = [0.0, 0.25, 0.5, 0.75, 1.0]
        bad = [v for v in args.rhos if v < 0.0 or v > 1.0]
        if bad:
            ap.error(
                f"--dgp team takes opposition STRENGTH opp in [0,1], got {bad}. "
                "opp enters as opp^2, so negative values silently duplicate their positive "
                "mirror (mean pairwise corr = -opp^2/3 either way) and produce a spurious "
                "slope if regressed on. Pass e.g. --opps 0 0.25 0.5 0.75 1.0."
            )
    vr.sample_agent_params = lambda rng, A, M, ctrl, sigma: sampler(rng, A, M, ctrl, sigma)

    if args.device == "cpu":
        device = torch.device("cpu")
        dev_name = "CPU"
    else:
        device = torch.device(f"cuda:{int(args.device)}")
        dev_name = torch.cuda.get_device_name(int(args.device))

    action_map = build_action_map(vr.N_RESOURCES)
    action_map_gpu = torch.tensor(action_map, device=device)
    master = np.random.default_rng(args.seed)
    probe = master.uniform(0.0, vr.CAPACITY, size=(128, vr.N_RESOURCES)).astype(np.float32)
    probe_gpu = torch.tensor(probe, device=device)

    # SAME order as b_vdn_runner.main => matched seeds across processes/modes. A declared
    # seed-grid superset lets reduced follow-ups retain the full-grid seed for each cell.
    cond_keys = [(a, s) for a in args.rhos for s in args.sigmas]
    seed_rhos = args.seed_grid_rhos if args.seed_grid_rhos is not None else args.rhos
    missing_seed_keys = [rho for rho in args.rhos if rho not in seed_rhos]
    if missing_seed_keys:
        ap.error(f"--seed-grid-rhos omits selected values: {missing_seed_keys}")
    seed_keys = [(a, s) for a in seed_rhos for s in args.sigmas]
    seed_schedule = {k: int(master.integers(0, 2**62)) for k in seed_keys}
    cond_seeds = {k: seed_schedule[k] for k in cond_keys}

    all_rows: List[dict] = []
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    total = len(args.modes) * len(cond_keys)
    done = 0
    print(f"=== ADVERSARIAL RE-RUN [{args.dgp}] on {device} ({dev_name}) ===")
    print(f"modes={args.modes} rhos={args.rhos} sigmas={args.sigmas} "
          f"eps={args.epsilon} reps={args.n_reps} eps_len={args.n_episodes}")
    for mode in args.modes:
        for (rho, s) in cond_keys:
            tc = time.time()
            rows = vr.run_condition(rho, s, args.epsilon, mode, args.n_reps,
                                    args.n_episodes, device, cond_seeds[(rho, s)],
                                    action_map_gpu, probe_gpu,
                                    n_eval_episodes=args.n_eval_episodes,
                                    eval_eps=args.epsilon)
            # relabel alpha -> the control's REAL name: rho (signed) or opp (strength).
            # Naming the team control "rho" is what invited regressing a squared quantity
            # as if it were signed.
            for r in rows:
                r[ctrl_name] = r.pop("alpha")
                r["dgp"] = args.dgp
                r["target_support"] = args.target_support
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
