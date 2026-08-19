"""HISTORICAL RUNNER — LEGACY SIGN-BLIND ALPHA SAMPLER.

Outputs are retained as failure provenance; alpha labels here do not establish delivered negative
target correlation or a genuine-opposition effect. See results/README.md.

Deliverable B: centralized value-decomposition (VDN) vs independent IQL vs MAPPO.

Tests whether the U-shape over alpha (structured preferences -- either aligned OR
opposed -- coordinate better than unstructured alpha=0) is IQL-specific, or survives
under a CENTRALIZED-TRAINING learner.

We use VDN (Sunehag et al. 2018): Q_tot = sum_i Q_i(o_i, a_i), trained against the
TEAM reward r_team = sum_i r_i with a joint TD target
    y = r_team + gamma * sum_i max_a' Q_i(o_i', a')
Centralized training (shared team reward + additive joint value couples the agents),
decentralized execution (each agent still acts by argmax of its OWN Q).

We ALSO add MAPPO (Yu et al. 2022): an on-policy actor-critic with per-agent
categorical policies and a CENTRALIZED critic V(s) over the shared state, trained
with GAE(lambda) on the team return and PPO-clip. Environment, preference sampling,
observation-noise convention and condition-seed schedule match the off-policy branch.
The training objective does not: IQL uses individual rewards, whereas VDN and MAPPO
use team return. Mode differences therefore compare learner/objective bundles.

The runner supports --mode {iql,vdn,mappo}. IQL-mode here reproduces the
original independent-learner result as a matched control.

Targeted design (not the full 125x30): alpha sweep x {low, mid, high} stakes, eps=0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

from friction_marl.experiments.factorial_design import (
    build_action_map,
    sample_agent_params,
)
from friction_marl.utils.dynamic_oracle import shared_pool_oracle_components
from friction_marl.utils.metrics import compute_convergence_time


def problem_instance_hash(weights: np.ndarray, targets: np.ndarray) -> str:
    """Stable digest of the exact float32 preference instance given to one run."""
    payload = np.asarray(weights, dtype="<f4").tobytes(order="C")
    payload += np.asarray(targets, dtype="<f4").tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()

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


def _run_condition_offpolicy(alpha, sigma, cond_eps, mode, n_reps, n_episodes, device, seed,
                             action_map_gpu, probe_gpu, n_eval_episodes=100, eval_eps=0.0):
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

    n_tail = min(100, n_episodes)
    last100 = torch.zeros(R, n_tail, A, device=device)
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
                # obs noise ~ N(0, eps*I): std = sqrt(eps) so covariance = eps*I (eps=0 stays 0)
                noise = torch.randn(R, A, N_RESOURCES, device=device, generator=noise_gen) * (cond_eps ** 0.5)
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

            # Store next_obs with FRESH matched i.i.d. noise (same std sqrt(eps)) so TD
            # targets don't bootstrap on a privileged denoised state (obs noise ~ N(0, eps*I)).
            nobs_clean = new_states.unsqueeze(1).expand(-1, A, -1)  # (R,A,M)
            if cond_eps > 0:
                nobs = nobs_clean + torch.randn(R, A, N_RESOURCES, device=device,
                                                generator=noise_gen) * (cond_eps ** 0.5)
            else:
                nobs = nobs_clean
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

        if ep >= n_episodes - n_tail:
            last100[:, ep - (n_episodes - n_tail)] = ep_rewards / EPISODE_LENGTH

        if ep % POLICY_SAMPLE_INTERVAL == 0 or ep == n_episodes - 1:
            with torch.no_grad():
                pv = q_net.forward(probe).argmax(dim=2).cpu().numpy()  # (N,128)
            for r in range(R):
                for ag in range(A):
                    idx = r * A + ag
                    dist = np.bincount(pv[idx], minlength=act_dim).astype(np.float32) / 128.0
                    dist = (1 - eps) * dist + eps * (1.0 / act_dim)
                    policy_history[r][ag].append(dist)

    # --- frozen held-out eval (additive; no optimizer.step / buf.push / soft_update) ---
    # NumPy generates the reset bank so the exact same starts can be reconstructed
    # offline and passed to the finite-horizon dynamic oracle.  This also makes the
    # held-out protocol device-independent when IQL/VDN are split across GPUs.
    eval_seed = int(seed + 1_000_003)
    eval_rng = np.random.default_rng(eval_seed)
    eval_initial_np = eval_rng.uniform(
        0.0, CAPACITY, size=(n_eval_episodes, R, N_RESOURCES)
    ).astype(np.float32)
    eval_initial = torch.tensor(eval_initial_np, device=device)

    def _reset(gen):
        return torch.rand(R, N_RESOURCES, device=device, generator=gen) * CAPACITY

    def _observe(st):
        return st.unsqueeze(1).expand(-1, A, -1)

    def _step(st, deltas):
        ns = (st + deltas.sum(dim=1)).clamp(0.0, CAPACITY)
        d = ns.unsqueeze(1) - targets_t
        return ns, (weights_t * -(d ** 2)).sum(dim=2)

    eval_clean = evaluate_frozen(
        q_net, _reset, _observe, _step, action_map_delta,
        R, A, obs_dim, act_dim, device, eval_seed,
        n_eval_episodes, 0.0, initial_states=eval_initial,
    )
    eval_matched = evaluate_frozen(
        q_net, _reset, _observe, _step, action_map_delta,
        R, A, obs_dim, act_dim, device, eval_seed + 1,
        n_eval_episodes, cond_eps, initial_states=eval_initial,
    )
    eval_explore = evaluate_frozen(
        q_net, _reset, _observe, _step, action_map_delta,
        R, A, obs_dim, act_dim, device, eval_seed + 2,
        n_eval_episodes, cond_eps, initial_states=eval_initial,
        action_epsilon=EPS_END,
    )
    oracle = shared_pool_oracle_components(
        weights_np, targets_np, eval_initial_np,
        horizon=EPISODE_LENGTH, capacity=int(CAPACITY),
    )

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
            # Keep the legacy aliases, but make the evaluation condition explicit.
            "eval_reward": float(eval_clean[r].mean()),
            "eval_gap": float(-eval_clean[r].mean()),
            "eval_reward_clean": float(eval_clean[r].mean()),
            "eval_reward_matched": float(eval_matched[r].mean()),
            "eval_reward_explore_001": float(eval_explore[r].mean()),
            "eval_seed": eval_seed,
            "condition_seed": int(seed),
            "problem_instance_sha256": problem_instance_hash(weights_np[r], targets_np[r]),
            "n_eval_episodes": n_eval_episodes,
            "oracle_reward_continuous_static": float(oracle["continuous_static"][r]),
            "oracle_reward_reachable_static": float(oracle["reachable_static"][r]),
            "oracle_reward_dynamic_T100": float(oracle["dynamic_oracle"][r]),
            "oracle_quantization_cost": float(oracle["quantization_cost"][r]),
            "oracle_transient_cost": float(oracle["transient_cost"][r]),
            "policy_residual_clean": float(
                oracle["dynamic_oracle"][r] - eval_clean[r].mean()
            ),
            "observation_noise_cost": float(
                eval_clean[r].mean() - eval_matched[r].mean()
            ),
            "exploration_cost_001": float(
                eval_matched[r].mean() - eval_explore[r].mean()
            ),
            "training_window_minus_frozen_explore": float(
                last_rew.mean() - eval_explore[r].mean()
            ),
        })
    return rows


# ===========================================================================
# MAPPO (on-policy, centralized critic) + shared frozen-eval helper.
# Purely ADDITIVE: iql/vdn above are unchanged (byte-for-byte apart from the
# obs-noise-units and replay-leak bug fixes and the additive eval_gap column).
# ===========================================================================
@dataclass
class MAPPOConfig:
    clip: float = 0.2
    gae_lambda: float = 0.95
    ppo_epochs: int = 4
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    rollout_len: int = EPISODE_LENGTH
    normalize_returns: bool = True


def compute_gae(rewards, values, dones, gamma, lam, last_value):
    """GAE(lambda). rewards/values/dones: (T, R). last_value: (R,).

    Returns (advantages, returns), each (T, R). done[t]=1 => no bootstrap past t
    (matches the off-policy branch, which zeros the TD target's bootstrap at the
    terminal step), so all three modes treat the episode boundary identically.
    """
    T, R = rewards.shape
    adv = torch.zeros(T, R, device=rewards.device)
    lastgae = torch.zeros(R, device=rewards.device)
    for t in range(T - 1, -1, -1):
        nonterminal = 1.0 - dones[t]
        nextval = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * nextval * nonterminal - values[t]
        lastgae = delta + gamma * lam * nonterminal * lastgae
        adv[t] = lastgae
    return adv, adv + values


def evaluate_frozen(net, reset_fn, observe_fn, step_fn, action_map_delta,
                    R, A, obs_dim, act_dim, device, eval_seed,
                    n_eval_episodes=100, eval_eps=0.0, *, initial_states=None,
                    action_epsilon=0.0):
    """Frozen greedy rollout on HELD-OUT initial states.

    No optimizer.step / buf.push / soft_update. Actions are greedy unless
    ``action_epsilon`` requests a frozen epsilon-greedy diagnostic. Observation noise
    is off by default; ``eval_eps>0`` re-injects N(0, eval_eps*I).  ``initial_states``
    optionally supplies a device-independent held-out reset bank with shape (E,R,...)
    so a dynamic oracle can be scored on precisely the same starts. Returns per-rep
    mean evaluation reward, shape (R,A).
    """
    N = R * A
    gen = torch.Generator(device=device)
    gen.manual_seed(int(eval_seed))
    if n_eval_episodes <= 0:
        return np.full((R, A), np.nan)
    if initial_states is not None and len(initial_states) != n_eval_episodes:
        raise ValueError("initial_states episode count does not match n_eval_episodes")
    total = torch.zeros(R, A, device=device)
    for episode in range(n_eval_episodes):
        states = (initial_states[episode].clone()
                  if initial_states is not None else reset_fn(gen))
        ep = torch.zeros(R, A, device=device)
        for _t in range(EPISODE_LENGTH):
            obs = observe_fn(states)  # (R,A,M)
            if eval_eps > 0:
                obs = obs + torch.randn(R, A, N_RESOURCES, device=device,
                                        generator=gen) * (eval_eps ** 0.5)
            obs_flat = obs.reshape(N, obs_dim)
            with torch.no_grad():
                q = net.forward(obs_flat.unsqueeze(1)).squeeze(1)  # (N, act)
            act = q.argmax(dim=1)
            if action_epsilon > 0:
                random_mask = torch.rand(N, device=device, generator=gen) < action_epsilon
                random_action = torch.randint(
                    0, act_dim, (N,), device=device, generator=gen
                )
                act = torch.where(random_mask, random_action, act)
            deltas = action_map_delta[act].reshape(R, A, N_RESOURCES)
            states, rewards = step_fn(states, deltas)
            ep += rewards
        total += ep / EPISODE_LENGTH
    return (total / n_eval_episodes).cpu().numpy()  # (R,A)


def _mappo_train(actor, critic, optimizer, train_reset_fn, eval_reset_fn, observe_fn,
                 step_fn, global_state_fn, action_map_delta, probe, R, A, obs_dim,
                 act_dim, n_episodes, device, noise_gen, noise_std, cfg,
                 eval_seed, n_eval_episodes, eval_eps):
    """Generic on-policy MAPPO trainer shared by the shared-pool, separable and
    partial-sharing envs. Env dynamics enter ONLY through the closures
    (train_reset_fn/eval_reset_fn/observe_fn/step_fn/global_state_fn), so the RL
    core (GAE + PPO-clip + value loss + entropy) is written and tested once.

    NO replay (on-policy): each episode is one rollout of cfg.rollout_len steps,
    then cfg.ppo_epochs of full-batch PPO. Returns a list of per-rep dicts with keys
    replication / agent_i_reward / mean_reward / convergence_time / eval_gap; callers
    merge in the condition-label keys (mode/alpha-or-rho/sigma/epsilon/env/...).
    """
    N = R * A
    T = cfg.rollout_len
    n_tail = min(100, n_episodes)
    last100 = torch.zeros(R, n_tail, A, device=device)
    policy_history = [[[] for _ in range(A)] for _ in range(R)]

    for ep in range(n_episodes):
        states = train_reset_fn()
        ep_rewards = torch.zeros(R, A, device=device)

        buf_obs = torch.zeros(T, N, obs_dim, device=device)
        buf_act = torch.zeros(T, N, dtype=torch.long, device=device)
        buf_logp = torch.zeros(T, N, device=device)
        buf_state = torch.zeros(T, R, global_state_fn(states).shape[1], device=device)
        buf_val = torch.zeros(T, R, device=device)
        buf_rteam = torch.zeros(T, R, device=device)
        buf_done = torch.zeros(T, R, device=device)

        for step in range(T):
            gstate = global_state_fn(states)  # (R, gdim)
            obs_all = observe_fn(states)       # (R,A,M)
            if noise_std > 0:
                obs_in = obs_all + torch.randn(R, A, N_RESOURCES, device=device,
                                               generator=noise_gen) * noise_std
            else:
                obs_in = obs_all
            obs_flat = obs_in.reshape(N, obs_dim)

            with torch.no_grad():
                logits = actor.forward(obs_flat.unsqueeze(1)).squeeze(1)  # (N, act)
                dist = torch.distributions.Categorical(logits=logits)
                act_idx = dist.sample()                                   # (N,)
                logp = dist.log_prob(act_idx)                             # (N,)
                v = critic.forward(gstate.unsqueeze(1)).reshape(R)        # (R,)

            deltas = action_map_delta[act_idx].reshape(R, A, N_RESOURCES)
            new_states, rewards = step_fn(states, deltas)                 # (R,A)
            done_val = 1.0 if step == T - 1 else 0.0

            buf_obs[step] = obs_flat
            buf_act[step] = act_idx
            buf_logp[step] = logp
            buf_state[step] = gstate
            buf_val[step] = v
            buf_rteam[step] = rewards.sum(dim=1)                          # team reward (R,)
            buf_done[step] = done_val

            ep_rewards += rewards
            states = new_states

        with torch.no_grad():
            last_v = critic.forward(global_state_fn(states).unsqueeze(1)).reshape(R)
        adv, ret = compute_gae(buf_rteam, buf_val, buf_done, GAMMA, cfg.gae_lambda, last_v)

        # each agent in a replication shares that replication's TEAM advantage
        if cfg.normalize_returns:
            # Normalise WITHIN each replication, over its own T timesteps only, so that
            # replication r's policy gradient depends solely on r's own returns.
            # A global reduction over the flattened (T, R*A) tensor rescales every seed
            # by the other replications' statistics and adds avoidable cross-block
            # dependence. Keep normalization within each replication instead.
            # adv is (T, R); reducing over dim=0 keeps each replication separate.
            adv = (adv - adv.mean(dim=0, keepdim=True)) / (adv.std(dim=0, keepdim=True) + 1e-8)
        adv_N = adv.unsqueeze(2).expand(T, R, A).reshape(T, N)

        obs_NT = buf_obs.permute(1, 0, 2)     # (N, T, obs)
        act_NT = buf_act.permute(1, 0)        # (N, T)
        oldlogp_NT = buf_logp.permute(1, 0)   # (N, T)
        adv_NT = adv_N.permute(1, 0)          # (N, T)
        state_RT = buf_state.permute(1, 0, 2)  # (R, T, gdim)
        ret_RT = ret.permute(1, 0)            # (R, T)

        for _ in range(cfg.ppo_epochs):
            logits = actor.forward(obs_NT)     # (N, T, act)
            dist = torch.distributions.Categorical(logits=logits)
            new_logp = dist.log_prob(act_NT)   # (N, T)
            entropy = dist.entropy().mean()
            ratio = torch.exp(new_logp - oldlogp_NT)
            surr1 = ratio * adv_NT
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip) * adv_NT
            policy_loss = -torch.min(surr1, surr2).mean()
            v_pred = critic.forward(state_RT).squeeze(-1)  # (R, T)
            value_loss = ((v_pred - ret_RT) ** 2).mean()
            loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if ep >= n_episodes - n_tail:
            last100[:, ep - (n_episodes - n_tail)] = ep_rewards / EPISODE_LENGTH

        if ep % POLICY_SAMPLE_INTERVAL == 0 or ep == n_episodes - 1:
            with torch.no_grad():
                pv = actor.forward(probe).argmax(dim=2).cpu().numpy()  # (N,128)
            for r in range(R):
                for ag in range(A):
                    idx = r * A + ag
                    dist_h = np.bincount(pv[idx], minlength=act_dim).astype(np.float32) / 128.0
                    policy_history[r][ag].append(dist_h)

    eval_rew = evaluate_frozen(actor, eval_reset_fn, observe_fn, step_fn, action_map_delta,
                               R, A, obs_dim, act_dim, device, eval_seed,
                               n_eval_episodes, eval_eps)  # (R,A)

    last100_cpu = last100.cpu().numpy()
    per_rep = []
    for r in range(R):
        last_rew = last100_cpu[r].mean(axis=0)  # (A,)
        conv = [compute_convergence_time(policy_history[r][ag]) * POLICY_SAMPLE_INTERVAL
                for ag in range(A)]
        per_rep.append({
            "replication": r,
            **{f"agent_{i}_reward": float(last_rew[i]) for i in range(A)},
            "mean_reward": float(last_rew.mean()),
            "convergence_time": float(np.mean(conv)),
            # eval_reward is the unambiguous name; eval_gap (its negation) is kept for
            # backward compatibility with committed CSVs. See results/DATA_DICTIONARY.md.
            "eval_reward": float(eval_rew[r].mean()),
            "eval_gap": float(-eval_rew[r].mean()),
        })
    return per_rep


def run_condition_mappo(alpha, sigma, cond_eps, n_reps, n_episodes, device, seed,
                        action_map_gpu, probe_gpu, n_eval_episodes=100, eval_eps=0.0):
    """MAPPO on the SHARED-POOL main env. Actor = BatchedQNet (obs -> 27 logits,
    Categorical); centralized critic = BatchedQNet(n=R, obs_dim=M, out_dim=1) over the
    shared state s (R, M). Matched condition seeds and the same sample_agent_params
    draw order align preference draws across modes; this is not an algorithm-only
    comparison because the learner and training objective change together.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    R, A = n_reps, N_AGENTS
    N = R * A
    obs_dim = N_RESOURCES
    act_dim = action_map_gpu.shape[0]

    actor = BatchedQNet(N, obs_dim, act_dim, HIDDEN_DIM, device)     # per-agent policy
    critic = BatchedQNet(R, obs_dim, 1, HIDDEN_DIM, device)          # centralized V(s)
    optimizer = torch.optim.Adam(actor.params() + critic.params(), lr=LR)

    # SAME preference-draw order as _run_condition_offpolicy (cross-mode comparability).
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
    noise_std = cond_eps ** 0.5  # obs noise ~ N(0, eps*I): std=sqrt(eps) (eps=0 -> 0)

    def train_reset():
        return torch.rand(R, N_RESOURCES, device=device) * CAPACITY

    def eval_reset(gen):
        return torch.rand(R, N_RESOURCES, device=device, generator=gen) * CAPACITY

    def observe(st):
        return st.unsqueeze(1).expand(-1, A, -1)  # (R,A,M) shared state broadcast

    def step_fn(st, deltas):
        ns = (st + deltas.sum(dim=1)).clamp(0.0, CAPACITY)
        d = ns.unsqueeze(1) - targets_t
        return ns, (weights_t * -(d ** 2)).sum(dim=2)

    def global_state(st):
        return st  # shared state (R, M) IS the centralized-critic input

    per_rep = _mappo_train(actor, critic, optimizer, train_reset, eval_reset, observe,
                           step_fn, global_state, action_map_delta, probe, R, A, obs_dim,
                           act_dim, n_episodes, device, noise_gen, noise_std, MAPPOConfig(),
                           seed + 1_000_003, n_eval_episodes, eval_eps)
    for row in per_rep:
        row.update({"mode": "mappo", "alpha": alpha, "sigma": sigma, "epsilon": cond_eps})
    return per_rep


def run_condition(alpha, sigma, cond_eps, mode, n_reps, n_episodes, device, seed,
                  action_map_gpu, probe_gpu, n_eval_episodes=100, eval_eps=0.0):
    """Dispatcher: on-policy MAPPO for mode=='mappo', else off-policy IQL/VDN. Same
    signature / env / per-condition seed for every mode so the three are matched and
    mergeable. Downstream runners (adv_runner / n2_runner) call this verbatim.
    """
    if mode == "mappo":
        return run_condition_mappo(alpha, sigma, cond_eps, n_reps, n_episodes, device,
                                   seed, action_map_gpu, probe_gpu, n_eval_episodes, eval_eps)
    return _run_condition_offpolicy(alpha, sigma, cond_eps, mode, n_reps, n_episodes,
                                    device, seed, action_map_gpu, probe_gpu,
                                    n_eval_episodes, eval_eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["iql", "vdn", "mappo"])
    ap.add_argument("--alphas", nargs="+", type=float, default=[-0.8, -0.4, 0.0, 0.4, 0.8])
    ap.add_argument("--sigmas", nargs="+", type=float, default=[0.2, 0.6, 1.0])
    ap.add_argument("--epsilon", type=float, default=0.0)
    ap.add_argument("--n-reps", type=int, default=30)
    ap.add_argument("--n-episodes", type=int, default=1000)
    ap.add_argument("--n-eval-episodes", type=int, default=100,
                    help="held-out frozen-eval episodes per condition (0 disables)")
    ap.add_argument("--eval-eps", type=float, default=0.0,
                    help="obs-noise variance during frozen eval (0=off; std=sqrt(eval_eps))")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="0", help='CUDA index or "cpu"')
    ap.add_argument("--out", type=str, default=str(Path(__file__).resolve().parent / "vdn_vs_iql.csv"))
    args = ap.parse_args()

    if args.device == "cpu":
        device = torch.device("cpu")
        dev_name = "CPU"
    else:
        device = torch.device(f"cuda:{int(args.device)}")
        dev_name = torch.cuda.get_device_name(int(args.device))
    action_map = build_action_map(N_RESOURCES)
    action_map_gpu = torch.tensor(action_map, device=device)
    master = np.random.default_rng(args.seed)
    probe = master.uniform(0.0, CAPACITY, size=(128, N_RESOURCES)).astype(np.float32)
    probe_gpu = torch.tensor(probe, device=device)

    # Same per-condition seed for ALL modes => matched preference draws & init.
    cond_keys = [(a, s) for a in args.alphas for s in args.sigmas]
    cond_seeds = {k: int(master.integers(0, 2**62)) for k in cond_keys}

    all_rows: List[dict] = []
    out_path = Path(args.out)
    t0 = time.time()
    total = len(args.modes) * len(cond_keys)
    done = 0
    print(f"=== IQL vs VDN vs MAPPO targeted run: {device} ({dev_name}) ===")
    print(f"modes={args.modes} alphas={args.alphas} sigmas={args.sigmas} "
          f"eps={args.epsilon} reps={args.n_reps} eps_len={args.n_episodes} "
          f"eval_n={args.n_eval_episodes} eval_eps={args.eval_eps}")
    for mode in args.modes:
        for (a, s) in cond_keys:
            tc = time.time()
            rows = run_condition(a, s, args.epsilon, mode, args.n_reps, args.n_episodes,
                                 device, cond_seeds[(a, s)], action_map_gpu, probe_gpu,
                                 n_eval_episodes=args.n_eval_episodes, eval_eps=args.eval_eps)
            all_rows.extend(rows)
            done += 1
            pd.DataFrame(all_rows).to_csv(out_path, index=False)
            print(f"  [{done}/{total}] mode={mode} a={a:+.1f} s={s:.1f}  "
                  f"gap={-np.mean([r['mean_reward'] for r in rows]):.4f}  "
                  f"eval_gap={np.mean([r['eval_gap'] for r in rows]):.4f}  "
                  f"{time.time()-tc:.1f}s")
            if device.type == "cuda":
                torch.cuda.empty_cache()
    print(f"\nDone in {(time.time()-t0)/60:.1f} min -> {out_path}")


if __name__ == "__main__":
    main()
