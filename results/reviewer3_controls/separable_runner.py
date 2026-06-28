"""EXPERIMENT 1 -- SEPARABLE-RESOURCE CONTROL (reviewer3 Critical #1).

Reviewer concern: the signed-correlation law (gap tracks cross-agent preference
correlation rho) might be SHARED-STATE CONTENTION -- 4 agents physically fighting
over one common resource pool -- rather than a property of preference correlation
per se.

This builds a SEPARABLE-resource variant of the env where EACH AGENT acts on its
OWN independent resource pool: its actions update ONLY its own state; there is no
shared/aggregated state; reward = its own weighted quadratic loss vs its own target.
The SAME signed equicorrelated preference DGP is used (so cross-agent target
correlation rho is identical to the shared-pool run); only the env COUPLING is
removed.

The ONLY differences from `b_vdn_runner.run_condition` (the shared-pool main env)
are the four marked env-dynamics lines:
  * state is per-agent (R, A, M)            (vs shared (R, M))
  * each agent observes its OWN state        (vs the common state)
  * new_state = state + own delta            (vs state + SUM of all agents' deltas)
  * diff = own next-state - own target       (vs shared next-state - own target)
Everything else -- network, optimizer, hyperparameters, eps schedule, replay
buffer, per-condition seeds, IQL/VDN losses, policy probe -- is IMPORTED VERBATIM
from b_vdn_runner so any shape difference is attributable to the env coupling alone.

Matched-seed property is preserved (cond_seeds drawn from the master RNG in fixed
(rho x sigma) order, independent of mode), so IQL on one GPU and VDN on another use
identical preference draws & inits and are mergeable.

Output CSV uses a `rho` column (signed control) + `env='separable'`.
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

# import the EXACT machinery / hyperparameters used by the shared-pool runner
from results.review_upgrade.b_vdn_runner import (  # noqa: E402
    BatchedQNet, JointReplayBuffer,
    N_AGENTS, N_RESOURCES, HIDDEN_DIM, LR, GAMMA,
    EPS_START, EPS_END, EPS_DECAY, BUFFER_CAPACITY, BATCH_SIZE, TAU,
    CAPACITY, EPISODE_LENGTH, POLICY_SAMPLE_INTERVAL,
)
from friction_marl.utils.metrics import compute_convergence_time  # noqa: E402


def run_condition_separable(sampler, rho, sigma, cond_eps, mode, n_reps, n_episodes,
                            device, seed, action_map_gpu, probe_gpu):
    """SEPARABLE-resource analogue of b_vdn_runner.run_condition.

    Per-agent independent resource pools: agent i's action moves ONLY pool i, and
    agent i is rewarded by pool i vs target i. The preference sampler is the signed
    equicorrelated DGP (rho = cross-agent target correlation), identical to shared.
    """
    rng = np.random.default_rng(seed)
    # Seed the global torch RNG from the same per-condition seed so network init,
    # epsilon-greedy exploration, and replay sampling are reproducible from this commit
    # forward (archived CSVs predate this fix = a single non-regenerable realization).
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    R, A = n_reps, N_AGENTS
    N = R * A
    obs_dim = N_RESOURCES
    act_dim = action_map_gpu.shape[0]

    q_net = BatchedQNet(N, obs_dim, act_dim, HIDDEN_DIM, device)
    target_net = BatchedQNet(N, obs_dim, act_dim, HIDDEN_DIM, device)
    target_net.copy_from(q_net)
    optimizer = torch.optim.Adam(q_net.params(), lr=LR)
    buf = JointReplayBuffer(R, BUFFER_CAPACITY, obs_dim, device)

    weights_np = np.zeros((R, A, N_RESOURCES), dtype=np.float32)
    targets_np = np.zeros((R, A, N_RESOURCES), dtype=np.float32)
    for r in range(R):
        w, t = sampler(rng, A, N_RESOURCES, rho, sigma)
        weights_np[r] = w
        targets_np[r] = t
    weights_t = torch.tensor(weights_np, device=device)
    targets_t = torch.tensor(targets_np, device=device)

    noise_gen = torch.Generator(device=device)
    noise_gen.manual_seed(int(rng.integers(0, 2**62)))
    action_map_delta = action_map_gpu.float()
    probe = probe_gpu.unsqueeze(0).expand(N, -1, -1)

    last100 = torch.zeros(R, 100, A, device=device)
    policy_history = [[[] for _ in range(A)] for _ in range(R)]
    global_step = 0
    can_train = False

    for ep in range(n_episodes):
        # ENV-DIFF 1: per-agent independent initial states (R, A, M)
        states = torch.rand(R, A, N_RESOURCES, device=device) * CAPACITY
        ep_rewards = torch.zeros(R, A, device=device)

        for step in range(EPISODE_LENGTH):
            global_step += 1
            # ENV-DIFF 2: each agent observes its OWN pool (already (R,A,M))
            obs_all = states
            if cond_eps > 0:
                noise = torch.randn(R, A, N_RESOURCES, device=device, generator=noise_gen) * cond_eps
                obs_in = obs_all + noise
            else:
                obs_in = obs_all
            obs_flat = obs_in.reshape(N, obs_dim)

            eps = EPS_START + min(global_step / EPS_DECAY, 1.0) * (EPS_END - EPS_START)
            with torch.no_grad():
                q_vals = q_net.forward(obs_flat.unsqueeze(1)).squeeze(1)
            greedy = q_vals.argmax(dim=1)
            rand_mask = torch.rand(N, device=device) < eps
            rand_act = torch.randint(0, act_dim, (N,), device=device)
            act_idx = torch.where(rand_mask, rand_act, greedy)  # (N,)

            deltas = action_map_delta[act_idx].reshape(R, A, N_RESOURCES)
            # ENV-DIFF 3: each agent's delta updates ONLY its own pool (NO sum over agents)
            new_states = (states + deltas).clamp(0.0, CAPACITY)  # (R, A, M)

            # ENV-DIFF 4: own next-state vs own target (no shared state)
            diff = new_states - targets_t                        # (R, A, M)
            rewards = (weights_t * -(diff ** 2)).sum(dim=2)       # (R, A)

            nobs = new_states                                     # each agent's OWN next obs
            done_val = 1.0 if step == EPISODE_LENGTH - 1 else 0.0

            buf.push(obs_in, act_idx.reshape(R, A), rewards,
                     nobs, torch.full((R,), done_val, device=device))

            ep_rewards += rewards
            states = new_states

            if not can_train:
                can_train = buf.can_sample(BATCH_SIZE)
            if can_train:
                b_obs, b_act, b_rew, b_nobs, b_done = buf.sample(BATCH_SIZE)
                B = BATCH_SIZE
                o = b_obs.permute(0, 2, 1, 3).reshape(N, B, obs_dim)
                no = b_nobs.permute(0, 2, 1, 3).reshape(N, B, obs_dim)
                a = b_act.permute(0, 2, 1).reshape(N, B)
                r = b_rew.permute(0, 2, 1).reshape(N, B)

                q_all = q_net.forward(o)
                q_taken = q_all.gather(2, a.unsqueeze(2)).squeeze(2)
                with torch.no_grad():
                    nq = target_net.forward(no).max(dim=2)[0]

                if mode == "iql":
                    with torch.no_grad():
                        done_net = b_done.unsqueeze(1).expand(R, A, B).reshape(N, B)
                        tgt = r + GAMMA * (1.0 - done_net) * nq
                    loss = ((q_taken - tgt) ** 2).mean()
                elif mode == "vdn":
                    q_taken_g = q_taken.reshape(R, A, B)
                    nq_g = nq.reshape(R, A, B)
                    r_g = r.reshape(R, A, B)
                    q_tot = q_taken_g.sum(dim=1)
                    with torch.no_grad():
                        nq_sum = nq_g.sum(dim=1)
                        r_team = r_g.sum(dim=1)
                        y = r_team + GAMMA * (1.0 - b_done) * nq_sum
                    loss = ((q_tot - y) ** 2).mean()
                else:
                    raise ValueError(mode)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                target_net.soft_update_from(q_net, TAU)

        if ep >= n_episodes - 100:
            last100[:, ep - (n_episodes - 100)] = ep_rewards / EPISODE_LENGTH

        if ep % POLICY_SAMPLE_INTERVAL == 0 or ep == n_episodes - 1:
            with torch.no_grad():
                pv = q_net.forward(probe).argmax(dim=2).cpu().numpy()
            for r in range(R):
                for ag in range(A):
                    idx = r * A + ag
                    dist = np.bincount(pv[idx], minlength=act_dim).astype(np.float32) / 128.0
                    dist = (1 - eps) * dist + eps * (1.0 / act_dim)
                    policy_history[r][ag].append(dist)

    last100_cpu = last100.cpu().numpy()
    rows = []
    for r in range(R):
        last_rew = last100_cpu[r].mean(axis=0)
        conv = [compute_convergence_time(policy_history[r][ag]) * POLICY_SAMPLE_INTERVAL
                for ag in range(A)]
        rows.append({
            "mode": mode, "rho": rho, "sigma": sigma, "epsilon": cond_eps,
            "env": "separable", "replication": r,
            **{f"agent_{i}_reward": float(last_rew[i]) for i in range(A)},
            "mean_reward": float(last_rew.mean()),
            "convergence_time": float(np.mean(conv)),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["iql", "vdn"])
    ap.add_argument("--rhos", nargs="+", type=float,
                    default=[-0.33, -0.2, -0.1, 0.0, 0.2, 0.4, 0.6, 0.8])
    ap.add_argument("--sigmas", nargs="+", type=float, default=[0.6, 1.0])
    ap.add_argument("--epsilon", type=float, default=0.0)
    ap.add_argument("--n-reps", type=int, default=30)
    ap.add_argument("--n-episodes", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    if args.device == "cpu":
        device = torch.device("cpu"); dev_name = "CPU"
    else:
        device = torch.device(f"cuda:{int(args.device)}")
        dev_name = torch.cuda.get_device_name(int(args.device))

    action_map = build_action_map(N_RESOURCES)
    action_map_gpu = torch.tensor(action_map, device=device)
    master = np.random.default_rng(args.seed)
    probe = master.uniform(0.0, CAPACITY, size=(128, N_RESOURCES)).astype(np.float32)
    probe_gpu = torch.tensor(probe, device=device)

    # SAME draw order as b_vdn_runner / adv_runner => matched seeds across modes/GPUs.
    cond_keys = [(a, s) for a in args.rhos for s in args.sigmas]
    cond_seeds = {k: int(master.integers(0, 2**62)) for k in cond_keys}

    all_rows: List[dict] = []
    out_path = Path(args.out)
    t0 = time.time()
    total = len(args.modes) * len(cond_keys)
    done = 0
    print(f"=== SEPARABLE-RESOURCE CONTROL on {device} ({dev_name}) ===")
    print(f"modes={args.modes} rhos={args.rhos} sigmas={args.sigmas} "
          f"eps={args.epsilon} reps={args.n_reps} eps_len={args.n_episodes}")
    for mode in args.modes:
        for (rho, s) in cond_keys:
            tc = time.time()
            rows = run_condition_separable(
                sample_agent_params_equicorr, rho, s, args.epsilon, mode,
                args.n_reps, args.n_episodes, device, cond_seeds[(rho, s)],
                action_map_gpu, probe_gpu)
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
