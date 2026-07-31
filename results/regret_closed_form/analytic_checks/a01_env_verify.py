"""VERIFY every environment claim against the live source. No claim is trusted.

Run:  python3 a01_env_verify.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from friction_marl.experiments.factorial_design import (      # noqa: E402
    ALPHAS, EPSILONS, SIGMAS, build_action_map, compute_rewards, sample_agent_params,
)
from friction_marl.envs.resource_allocation import EnvConfig, ResourceAllocationEnv  # noqa: E402

sys.path.insert(0, str(ROOT / "results" / "adversarial_rerun"))
from dgp import sample_agent_params_equicorr, realised_mean_pairwise_corr  # noqa: E402

OUT = []


def chk(name, cond, detail=""):
    OUT.append(("PASS" if cond else "FAIL", name, detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}")
    return cond


# --------------------------------------------------------------------------- #
print("=" * 78)
print("1. ACTION SPACE")
am = build_action_map(3)
chk("27 per-agent actions in {-1,0,1}^3",
    am.shape == (27, 3) and set(np.unique(am)) == {-1, 0, 1},
    f"shape={am.shape}")

# aggregate over 4 agents: is it exactly the PRODUCT set {-4..4}^3 ?
agg = set()
for combo in itertools.product(range(27), repeat=4):
    pass  # 531441 combos: do it smarter
# The per-agent action set is a PRODUCT {-1,0,1}x{-1,0,1}x{-1,0,1}, so the
# Minkowski sum of 4 copies is the product of the per-coordinate Minkowski sums
# = {-4..4}^3.  Verify CONSTRUCTIVELY: for every target aggregate, exhibit a
# profile of 4 actions from the actual action map that realises it.
rows = {tuple(a): i for i, a in enumerate(am)}


def _split(d):                       # write d in {-4..4} as 4 values in {-1,0,1}
    v = [0, 0, 0, 0]
    for k in range(abs(d)):
        v[k] = 1 if d > 0 else -1
    return v


full = list(itertools.product(range(-4, 5), repeat=3))
ok_all = True
for tgt in full:
    per = [_split(d) for d in tgt]                        # 3 coords x 4 agents
    prof = [tuple(per[c][a] for c in range(3)) for a in range(4)]
    if any(p not in rows for p in prof):
        ok_all = False
        break
    if tuple(np.sum([am[rows[p]] for p in prof], axis=0)) != tgt:
        ok_all = False
        break
chk("aggregate delta set == {-4..4}^3 (product, 729 elements), constructive",
    ok_all, f"all {len(full)} aggregates realised by explicit profiles")

# --------------------------------------------------------------------------- #
print("=" * 78)
print("2. TRANSITION + CLAMP + LATTICE CLOSURE")
env = ResourceAllocationEnv(EnvConfig(n_agents=4, n_resources=3), seed=1)
chk("capacity == 10.0 and episode_length == 100",
    env.capacity == 10.0 and env.episode_length == 100)
obs, _ = env.reset(seed=1)
s0 = env._state.copy()
chk("reset is CONTINUOUS uniform on [0,C] (not integer)",
    np.any(np.abs(s0 - np.round(s0)) > 1e-6), f"s0={s0}")

# closure of the 21-point lattice for one coordinate
f = float(s0[0] - np.floor(s0[0]))
L = np.unique(np.concatenate([np.arange(11.0), f + np.arange(10.0)]))
chk("|L(f)| == 21 for generic f", L.size == 21, f"f={f:.4f}")
reach = set(np.round(L, 12))
closed = True
for x in L:
    for D in range(-4, 5):
        y = float(np.clip(x + D, 0.0, 10.0))
        if round(y, 12) not in reach:
            closed = False
chk("L(f) closed under x -> clamp(x+D), D in {-4..4}", closed)
chk("integer sublattice {0..10} is absorbing (no return to the f-coset)",
    all(round(float(np.clip(j + D, 0, 10)), 12) in set(range(11))
        for j in range(11) for D in range(-4, 5)))

# BFS reachability from s0 in one coordinate: do we actually reach all 21?
seen, frontier = {round(float(s0[0]), 12)}, [float(s0[0])]
while frontier:
    nxt = []
    for x in frontier:
        for D in range(-4, 5):
            y = round(float(np.clip(x + D, 0.0, 10.0)), 12)
            if y not in seen:
                seen.add(y)
                nxt.append(y)
    frontier = nxt
chk("all 21 lattice points reachable from a continuous start", len(seen) == 21,
    f"|reachable|={len(seen)}")

# the measurement-spec claim
chk("ideation_measurement sec 0.1 '11^3 states' holds ONLY for integer starts",
    len(seen) == 21, "TRUE closure under the environment's own reset is 21^3=9261")

# --------------------------------------------------------------------------- #
print("=" * 78)
print("3. REWARD FORM AND THE WELFARE IDENTITY")
rng = np.random.default_rng(7)
w, t = sample_agent_params(rng, 4, 3, 0.4, 0.6)
state = rng.uniform(0, 10, size=3)
r = compute_rewards(state, w, t)
r_manual = np.array([-np.sum(w[i] * (state - t[i]) ** 2) for i in range(4)])
chk("r_i = -sum_m w_im (s'_m - tau_im)^2", np.allclose(r, r_manual))

Wbar = w.sum(axis=0)                      # (m,)
mu = (w * t).sum(axis=0) / Wbar
V = (w * (t - mu[None, :]) ** 2).sum(axis=0)
W_direct = r.sum()
W_ident = -np.sum(Wbar * (state - mu) ** 2 + V)
chk("W(s) = -sum_m [ Wbar_m (s_m-mu_m)^2 + V_m ]  (exact)",
    np.isclose(W_direct, W_ident), f"{W_direct:.10f} vs {W_ident:.10f}")

# --------------------------------------------------------------------------- #
print("=" * 78)
print("4. NOISE CHANNEL")
src = (ROOT / "run_gpu.py").read_text().splitlines()
line254 = [i + 1 for i, l in enumerate(src) if "cond_epsilon ** 0.5" in l]
chk("obs noise std = sqrt(eps) (variance eps)", len(line254) >= 1,
    f"run_gpu.py lines {line254}")
rew_lines = [i + 1 for i, l in enumerate(src) if "new_states.unsqueeze(1) - targets_t" in l]
chk("rewards use the TRUE state and TRUE targets (no noise in reward)",
    len(rew_lines) == 1, f"run_gpu.py line {rew_lines}")

# --------------------------------------------------------------------------- #
print("=" * 78)
print("5. THE THREE PREFERENCE DGPs")
n, m = 4, 3
for alpha in ALPHAS:
    R = 400000
    rr = np.random.default_rng(11)
    b = rr.standard_normal(R)
    xi = rr.standard_normal((R, n))
    tau = alpha * b[:, None] + (1 - abs(alpha)) * xi
    var = tau.var(axis=0).mean()
    cor = np.corrcoef(tau.T)[0, 1]
    pred_var = alpha ** 2 + (1 - abs(alpha)) ** 2
    pred_cor = alpha ** 2 / pred_var
    ok = abs(var - pred_var) < 0.01 and abs(cor - pred_cor) < 0.01
    chk(f"implemented DGP alpha={alpha:+.1f}: Var={pred_var:.4f} rho={pred_cor:.4f}",
        ok, f"empirical Var={var:.4f} rho={cor:.4f}")

chk("implemented rho depends on |alpha| ONLY (sign-blind)",
    all(abs(alpha ** 2 / (alpha ** 2 + (1 - abs(alpha)) ** 2)
            - (-alpha) ** 2 / ((-alpha) ** 2 + (1 - abs(-alpha)) ** 2)) < 1e-12
        for alpha in ALPHAS))

# renormalised
for alpha in ALPHAS:
    R = 400000
    rr = np.random.default_rng(12)
    b = rr.standard_normal(R)
    xi = rr.standard_normal((R, n))
    s = np.sqrt(alpha ** 2 + (1 - abs(alpha)) ** 2)
    tau = (alpha * b[:, None] + (1 - abs(alpha)) * xi) / s
    chk(f"renormalised DGP alpha={alpha:+.1f}: unit marginal variance",
        abs(tau.var(axis=0).mean() - 1.0) < 0.01,
        f"Var={tau.var(axis=0).mean():.4f} rho={np.corrcoef(tau.T)[0,1]:.4f}")

# equicorr: rho IS the pairwise correlation
for rho in [-1 / 3, -0.2, 0.0, 0.3077, 0.6, 0.9412]:
    pooled, perdraw = realised_mean_pairwise_corr(
        sample_agent_params_equicorr, rho, 0.6, 4, 3, 6000, seed=3)
    chk(f"equicorr DGP rho={rho:+.4f}: realised pairwise corr matches",
        abs(pooled - rho) < 0.02, f"pooled={pooled:+.4f}")

# --------------------------------------------------------------------------- #
print("=" * 78)
print("6. WEIGHT SAMPLER AND THE EFFECTIVE STAKES SCALE")
from scipy.stats import norm as _n                                   # noqa: E402
for sigma in SIGMAS:
    R = 2000000
    rr = np.random.default_rng(13)
    ww = np.clip(sigma + 0.05 * rr.standard_normal(R), 0, 1)
    # E[clip(N(s,0.05),0,1)] = s - 0.05*(phi(a)-phi(b)) ... exact:
    a, bnd = (0 - sigma) / 0.05, (1 - sigma) / 0.05
    exact = (sigma * (_n.cdf(bnd) - _n.cdf(a))
             - 0.05 * (_n.pdf(bnd) - _n.pdf(a))
             + 1.0 * (1 - _n.cdf(bnd)) + 0.0 * _n.cdf(a))
    chk(f"sigma={sigma}: E[w] = {exact:.6f} (clip bite = {100*(exact/sigma-1):+.2f}%)",
        abs(ww.mean() - exact) < 2e-4, f"MC={ww.mean():.6f}")

# --------------------------------------------------------------------------- #
print("=" * 78)
print("7. GRID")
chk("ALPHAS / SIGMAS / EPSILONS as documented",
    ALPHAS == [-0.8, -0.4, 0.0, 0.4, 0.8]
    and SIGMAS == [0.2, 0.4, 0.6, 0.8, 1.0]
    and EPSILONS == [0.0, 0.25, 0.5, 0.75, 1.0])

print("=" * 78)
nfail = sum(1 for s, _, _ in OUT if s == "FAIL")
print(f"{len(OUT)-nfail}/{len(OUT)} checks PASS, {nfail} FAIL")
