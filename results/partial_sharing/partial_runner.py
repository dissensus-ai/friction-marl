"""PARTIAL-SHARING CONTROL -- the mechanistic bridge reviewer #5 asked for.

WHY THIS EXISTS
---------------
Two endpoints are already on disk:

  * w=1 (FULLY SHARED, full contention): `adversarial_rerun/equicorr_{iql,vdn}.csv`
    -- 4 agents share ONE common resource pool; the alignment gap tracks signed rho
    with slope ~= -0.755 (IQL) / -0.838 (VDN).
  * w=0 (FULLY SEPARABLE, no contention): `reviewer3_controls/separable_{iql,vdn}.csv`
    -- each agent has its OWN independent pool; the gap-vs-rho curve is FLAT
    (slope ~= -0.029 / -0.053).

Reviewer #5's request: parametrically vary the DEGREE of shared-state contention
between these extremes and show the alignment-gap slope SCALES with that degree.
If contention DEGREE mediates the alignment effect, |slope| should rise
monotonically from ~0 at w=0 to ~0.8 at w=1.

THE OVERLAP CONSTRUCTION  (fraction-of-resources-shared)
--------------------------------------------------------
With m = N_RESOURCES = 3 resources we make k of the m resources a COMMON POOL
(all agents contend on them, summed dynamics) and the remaining m-k resources
PRIVATE (per-agent, separable dynamics). The overlap level is

    w = k / m   in {0, 1/3, 2/3, 1}   for k in {0,1,2,3}.

This is the cleanest principled interpolation because it reduces EXACTLY to the two
existing runners at the endpoints (same code paths, same reward semantics):

  * k=0  (w=0): every resource is private -> identical to `separable_runner`.
  * k=m  (w=1): every resource is the common pool -> identical to the shared-pool
                main env (`b_vdn_runner.run_condition`).

State is carried as a per-agent tensor (R, A, m). The first k columns are the
SHARED columns: they are initialised to a single common value per (rep,column),
and at every step the common pool accumulates the SUM of all agents' deltas on
those columns, then is broadcast back to every agent (so those columns stay
identical across agents at all times -- a genuine shared pool). The remaining m-k
PRIVATE columns evolve per-agent (own delta only), exactly as in the separable env.

  ENV-DIFF vs separable: shared columns use the summed common-pool update.
  ENV-DIFF vs shared    : private columns use the per-agent update.

Reward is UNCHANGED: agent i's per-agent weighted quadratic loss of its OWN
post-state row vs its OWN target, summed over all m resources. Only the SUBSTRATE
on which actions land (shared vs private) changes with k; preferences/targets are
the SAME signed equicorrelated DGP (rho = cross-agent target correlation) used by
both endpoint runs, so the preference structure is held fixed across overlap levels.

Everything else -- BatchedQNet, JointReplayBuffer, optimizer, hyperparameters,
eps schedule, IQL/VDN losses, policy probe, per-condition matched seeds -- is
IMPORTED VERBATIM from `b_vdn_runner`, so any change in the gap-vs-rho slope is
attributable to the overlap degree alone.

Matched-seed property: cond_seeds are drawn from the master RNG in fixed
(overlap x rho x sigma) order, independent of --mode, so IQL (GPU0) and VDN (GPU1)
see identical preference draws & inits for the same (w,rho,sigma) and are mergeable.

Output CSV columns: mode, overlap_k, w, rho, sigma, epsilon, env='partial',
replication, agent_*_reward, mean_reward, convergence_time.
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


def run_condition_partial(sampler, overlap_k, rho, sigma, cond_eps, mode, n_reps,
                          n_episodes, device, seed, action_map_gpu, probe_gpu):
    """PARTIAL-SHARING analogue of b_vdn_runner.run_condition.

    `overlap_k` in {0,...,N_RESOURCES}: number of SHARED (common-pool) resources;
    the first `overlap_k` columns are shared (summed dynamics), the rest private
    (per-agent dynamics). overlap_k=0 -> separable; overlap_k=N_RESOURCES -> shared.
    Preference sampler is the signed equicorr DGP (rho = cross-agent target corr).
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
    M = N_RESOURCES
    obs_dim = M
    act_dim = action_map_gpu.shape[0]
    k = int(overlap_k)

    q_net = BatchedQNet(N, obs_dim, act_dim, HIDDEN_DIM, device)
    target_net = BatchedQNet(N, obs_dim, act_dim, HIDDEN_DIM, device)
    target_net.copy_from(q_net)
    optimizer = torch.optim.Adam(q_net.params(), lr=LR)
    buf = JointReplayBuffer(R, BUFFER_CAPACITY, obs_dim, device)

    weights_np = np.zeros((R, A, M), dtype=np.float32)
    targets_np = np.zeros((R, A, M), dtype=np.float32)
    for r in range(R):
        w, t = sampler(rng, A, M, rho, sigma)
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
        # Per-agent initial states (R, A, M). SHARED columns get ONE common value
        # per (rep, column) broadcast across agents (a single shared pool); PRIVATE
        # columns are per-agent. (k=M => fully shared init == main env;
        #                          k=0 => fully per-agent init == separable env.)
        states = torch.rand(R, A, M, device=device) * CAPACITY
        if k > 0:
            common_init = torch.rand(R, 1, k, device=device) * CAPACITY
            states[:, :, :k] = common_init.expand(R, A, k)
        ep_rewards = torch.zeros(R, A, device=device)

        for step in range(EPISODE_LENGTH):
            global_step += 1
            obs_all = states                                   # (R,A,M) own row
            if cond_eps > 0:
                noise = torch.randn(R, A, M, device=device, generator=noise_gen) * cond_eps
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

            deltas = action_map_delta[act_idx].reshape(R, A, M)
            # PRIVATE columns: per-agent own delta (separable dynamics).
            new_states = states + deltas                        # (R,A,M) tentative
            # SHARED columns: common pool accumulates SUM of all agents' deltas,
            # then broadcast back to every agent (shared-pool dynamics).
            if k > 0:
                shared_current = states[:, 0, :k]               # (R,k) common value
                shared_delta_sum = deltas[:, :, :k].sum(dim=1)  # (R,k) sum over agents
                pool_next = shared_current + shared_delta_sum   # (R,k)
                new_states[:, :, :k] = pool_next.unsqueeze(1).expand(R, A, k)
            new_states = new_states.clamp(0.0, CAPACITY)        # (R,A,M)

            # Reward: own post-state row vs own target (unchanged across overlap).
            diff = new_states - targets_t                       # (R,A,M)
            rewards = (weights_t * -(diff ** 2)).sum(dim=2)     # (R,A)

            nobs = new_states                                   # each agent's OWN next row
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
            "mode": mode, "overlap_k": k, "w": k / M, "rho": rho, "sigma": sigma,
            "epsilon": cond_eps, "env": "partial", "replication": r,
            **{f"agent_{i}_reward": float(last_rew[i]) for i in range(A)},
            "mean_reward": float(last_rew.mean()),
            "convergence_time": float(np.mean(conv)),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["iql", "vdn"])
    ap.add_argument("--overlaps", nargs="+", type=int, default=[0, 1, 2, 3],
                    help="number of SHARED resources k (0..N_RESOURCES); w=k/N_RESOURCES")
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

    # Fixed (overlap x rho x sigma) draw order, independent of --mode => matched
    # preference draws & inits for the same (w,rho,sigma) across IQL/VDN processes.
    cond_keys = [(k, a, s) for k in args.overlaps for a in args.rhos for s in args.sigmas]
    cond_seeds = {key: int(master.integers(0, 2**62)) for key in cond_keys}

    all_rows: List[dict] = []
    out_path = Path(args.out)
    t0 = time.time()
    total = len(args.modes) * len(cond_keys)
    done = 0
    print(f"=== PARTIAL-SHARING CONTROL on {device} ({dev_name}) ===")
    print(f"modes={args.modes} overlaps(k)={args.overlaps} (w={[k/N_RESOURCES for k in args.overlaps]}) "
          f"rhos={args.rhos} sigmas={args.sigmas} eps={args.epsilon} "
          f"reps={args.n_reps} eps_len={args.n_episodes}")
    for mode in args.modes:
        for (k, rho, s) in cond_keys:
            tc = time.time()
            rows = run_condition_partial(
                sample_agent_params_equicorr, k, rho, s, args.epsilon, mode,
                args.n_reps, args.n_episodes, device, cond_seeds[(k, rho, s)],
                action_map_gpu, probe_gpu)
            all_rows.extend(rows)
            done += 1
            pd.DataFrame(all_rows).to_csv(out_path, index=False)
            print(f"  [{done}/{total}] {mode} k={k}(w={k/N_RESOURCES:.2f}) rho={rho:+.2f} s={s:.1f}  "
                  f"gap={-np.mean([r['mean_reward'] for r in rows]):.4f}  "
                  f"{time.time()-tc:.1f}s", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    print(f"\nDone in {(time.time()-t0)/60:.1f} min -> {out_path}")


if __name__ == "__main__":
    main()
