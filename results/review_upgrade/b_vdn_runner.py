"""Deliverable B: centralized value-decomposition (VDN) vs independent IQL.

Tests whether the U-shape over alpha (structured preferences -- either aligned OR
opposed -- coordinate better than unstructured alpha=0) is IQL-specific, or survives
under a CENTRALIZED-TRAINING learner.

We use VDN (Sunehag et al. 2018): Q_tot = sum_i Q_i(o_i, a_i), trained against the
TEAM reward r_team = sum_i r_i with a joint TD target
    y = r_team + gamma * sum_i max_a' Q_i(o_i', a')
Centralized training (shared team reward + additive joint value couples the agents),
decentralized execution (each agent still acts by argmax of its OWN Q).

The runner supports --mode {iql,vdn}. EVERYTHING ELSE is identical (env, reward,
preference sampling, network, optimizer, hyperparameters, per-condition seeds), so
any difference in the alpha-shape is attributable to the algorithm alone. IQL-mode
here reproduces the original independent-learner result as a matched control.

Targeted design (not the full 125x30): alpha sweep x {low, mid, high} stakes, eps=0.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

from friction_marl.experiments.factorial_design import (
    build_action_map,
    sample_agent_params,
)
from friction_marl.utils.metrics import compute_convergence_time

# --- hyperparameters: identical to run_gpu.py ---
N_AGENTS = 4
N_RESOURCES = 3
HIDDEN_DIM = 64
LR = 1e-3
GAMMA = 0.99
EPS_START = 0.1
EPS_END = 0.01
EPS_DECAY = 5000
BUFFER_CAPACITY = 100_000
BATCH_SIZE = 64
TAU = 0.01
CAPACITY = 10.0
EPISODE_LENGTH = 100
POLICY_SAMPLE_INTERVAL = 200
LEARNING_CURVE_WINDOW = 50


class BatchedQNet:
    """N independent 3-layer MLPs as batched tensors (one MLP per (rep,agent))."""

    def __init__(self, n, obs_dim, act_dim, hidden, device):
        self.n = n
        self.device = device
        k1 = (1 / obs_dim) ** 0.5
        k2 = (1 / hidden) ** 0.5
        self.w1 = torch.empty(n, obs_dim, hidden, device=device).uniform_(-k1, k1).requires_grad_(True)
        self.b1 = torch.zeros(n, 1, hidden, device=device).requires_grad_(True)
        self.w2 = torch.empty(n, hidden, hidden, device=device).uniform_(-k2, k2).requires_grad_(True)
        self.b2 = torch.zeros(n, 1, hidden, device=device).requires_grad_(True)
        self.w3 = torch.empty(n, hidden, act_dim, device=device).uniform_(-k2, k2).requires_grad_(True)
        self.b3 = torch.zeros(n, 1, act_dim, device=device).requires_grad_(True)

    def forward(self, x):
        h = torch.relu(torch.bmm(x, self.w1) + self.b1)
        h = torch.relu(torch.bmm(h, self.w2) + self.b2)
        return torch.bmm(h, self.w3) + self.b3

    def params(self):
        return [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]

    def copy_from(self, src):
        with torch.no_grad():
            for s, d in zip(src.params(), self.params()):
                d.copy_(s)

    def soft_update_from(self, src, tau=TAU):
        with torch.no_grad():
            for s, d in zip(src.params(), self.params()):
                d.mul_(1 - tau).add_(s, alpha=tau)


class JointReplayBuffer:
    """Stores joint transitions per replication: (R, cap, A, ...). Works for both modes."""

    def __init__(self, n_reps, cap, obs_dim, device):
        self.R = n_reps
        self.A = N_AGENTS
        self.cap = cap
        self.device = device
        self.obs = torch.zeros(n_reps, cap, self.A, obs_dim, device=device)
        self.act = torch.zeros(n_reps, cap, self.A, dtype=torch.long, device=device)
        self.rew = torch.zeros(n_reps, cap, self.A, device=device)
        self.nobs = torch.zeros(n_reps, cap, self.A, obs_dim, device=device)
        self.done = torch.zeros(n_reps, cap, device=device)
        self.ptr = 0
        self.size = 0

    def push(self, obs, act, rew, nobs, done):
        i = self.ptr
        self.obs[:, i] = obs
        self.act[:, i] = act
        self.rew[:, i] = rew
        self.nobs[:, i] = nobs
        self.done[:, i] = done
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, B):
        idx = torch.randint(0, self.size, (self.R, B), device=self.device)
        ar = torch.arange(self.R, device=self.device).unsqueeze(1)
        return (self.obs[ar, idx], self.act[ar, idx], self.rew[ar, idx],
                self.nobs[ar, idx], self.done[ar, idx])

    def can_sample(self, B):
        return self.size >= B


def run_condition(alpha, sigma, cond_eps, mode, n_reps, n_episodes, device, seed,
                  action_map_gpu, probe_gpu):
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
        w, t = sample_agent_params(rng, A, N_RESOURCES, alpha, sigma)
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
        states = torch.rand(R, N_RESOURCES, device=device) * CAPACITY
        ep_rewards = torch.zeros(R, A, device=device)

        for step in range(EPISODE_LENGTH):
            global_step += 1
            obs_all = states.unsqueeze(1).expand(-1, A, -1)  # (R,A,M)
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
            new_states = (states + deltas.sum(dim=1)).clamp(0.0, CAPACITY)

            diff = new_states.unsqueeze(1) - targets_t  # (R,A,M)
            rewards = (weights_t * -(diff ** 2)).sum(dim=2)  # (R,A)

            nobs = new_states.unsqueeze(1).expand(-1, A, -1)  # clean next obs (R,A,M)
            done_val = 1.0 if step == EPISODE_LENGTH - 1 else 0.0

            buf.push(obs_in, act_idx.reshape(R, A), rewards,
                     nobs, torch.full((R,), done_val, device=device))

            ep_rewards += rewards
            states = new_states

            if not can_train:
                can_train = buf.can_sample(BATCH_SIZE)
            if can_train:
                b_obs, b_act, b_rew, b_nobs, b_done = buf.sample(BATCH_SIZE)  # (R,B,A,..)
                B = BATCH_SIZE
                # reshape to per-net (N,B,obs)
                o = b_obs.permute(0, 2, 1, 3).reshape(N, B, obs_dim)
                no = b_nobs.permute(0, 2, 1, 3).reshape(N, B, obs_dim)
                a = b_act.permute(0, 2, 1).reshape(N, B)          # (N,B)
                r = b_rew.permute(0, 2, 1).reshape(N, B)          # (N,B)

                q_all = q_net.forward(o)                          # (N,B,act)
                q_taken = q_all.gather(2, a.unsqueeze(2)).squeeze(2)  # (N,B)
                with torch.no_grad():
                    nq = target_net.forward(no).max(dim=2)[0]     # (N,B)

                if mode == "iql":
                    with torch.no_grad():
                        # done is per-replication; broadcast to each agent's net.
                        done_net = b_done.unsqueeze(1).expand(R, A, B).reshape(N, B)
                        tgt = r + GAMMA * (1.0 - done_net) * nq
                    loss = ((q_taken - tgt) ** 2).mean()
                elif mode == "vdn":
                    # group nets back to (R,A,B); sum over agents -> Q_tot (R,B)
                    q_taken_g = q_taken.reshape(R, A, B)
                    nq_g = nq.reshape(R, A, B)
                    r_g = r.reshape(R, A, B)
                    q_tot = q_taken_g.sum(dim=1)                  # (R,B)
                    with torch.no_grad():
                        nq_sum = nq_g.sum(dim=1)                  # (R,B)
                        r_team = r_g.sum(dim=1)                   # (R,B)
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
                pv = q_net.forward(probe).argmax(dim=2).cpu().numpy()  # (N,128)
            for r in range(R):
                for ag in range(A):
                    idx = r * A + ag
                    dist = np.bincount(pv[idx], minlength=act_dim).astype(np.float32) / 128.0
                    dist = (1 - eps) * dist + eps * (1.0 / act_dim)
                    policy_history[r][ag].append(dist)

    last100_cpu = last100.cpu().numpy()
    rows = []
    for r in range(R):
        last_rew = last100_cpu[r].mean(axis=0)  # (A,)
        conv = [compute_convergence_time(policy_history[r][ag]) * POLICY_SAMPLE_INTERVAL
                for ag in range(A)]
        rows.append({
            "mode": mode, "alpha": alpha, "sigma": sigma, "epsilon": cond_eps,
            "replication": r,
            **{f"agent_{i}_reward": float(last_rew[i]) for i in range(A)},
            "mean_reward": float(last_rew.mean()),
            "convergence_time": float(np.mean(conv)),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["iql", "vdn"])
    ap.add_argument("--alphas", nargs="+", type=float, default=[-0.8, -0.4, 0.0, 0.4, 0.8])
    ap.add_argument("--sigmas", nargs="+", type=float, default=[0.2, 0.6, 1.0])
    ap.add_argument("--epsilon", type=float, default=0.0)
    ap.add_argument("--n-reps", type=int, default=30)
    ap.add_argument("--n-episodes", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", type=str, default=str(Path(__file__).resolve().parent / "vdn_vs_iql.csv"))
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.device}")
    action_map = build_action_map(N_RESOURCES)
    action_map_gpu = torch.tensor(action_map, device=device)
    master = np.random.default_rng(args.seed)
    probe = master.uniform(0.0, CAPACITY, size=(128, N_RESOURCES)).astype(np.float32)
    probe_gpu = torch.tensor(probe, device=device)

    # Same per-condition seed for BOTH modes => matched preference draws & init.
    cond_keys = [(a, s) for a in args.alphas for s in args.sigmas]
    cond_seeds = {k: int(master.integers(0, 2**62)) for k in cond_keys}

    all_rows: List[dict] = []
    out_path = Path(args.out)
    t0 = time.time()
    total = len(args.modes) * len(cond_keys)
    done = 0
    print(f"=== VDN vs IQL targeted run: {device} ({torch.cuda.get_device_name(args.device)}) ===")
    print(f"modes={args.modes} alphas={args.alphas} sigmas={args.sigmas} "
          f"eps={args.epsilon} reps={args.n_reps} eps_len={args.n_episodes}")
    for mode in args.modes:
        for (a, s) in cond_keys:
            tc = time.time()
            rows = run_condition(a, s, args.epsilon, mode, args.n_reps, args.n_episodes,
                                 device, cond_seeds[(a, s)], action_map_gpu, probe_gpu)
            all_rows.extend(rows)
            done += 1
            pd.DataFrame(all_rows).to_csv(out_path, index=False)
            print(f"  [{done}/{total}] mode={mode} a={a:+.1f} s={s:.1f}  "
                  f"gap={-np.mean([r['mean_reward'] for r in rows]):.4f}  "
                  f"{time.time()-tc:.1f}s")
            torch.cuda.empty_cache()
    print(f"\nDone in {(time.time()-t0)/60:.1f} min -> {out_path}")


if __name__ == "__main__":
    main()
