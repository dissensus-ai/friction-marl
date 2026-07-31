"""Structural theorems behind the closed form, each checked numerically.

T1  SEPARABILITY: the planner's, the naive-myopic and the stage-Nash problems all
    factorise EXACTLY into 3 independent scalar problems.
T2  WELFARE ALGEBRA + CANCELLATION of the contention floor V_m in G.
T3  DEADZONE-SIGN law: the noiseless naive-myopic aggregate response is
        D(x) = sum_i sgn(tau_i - x) 1{|tau_i - x| >= 1/2}
T4  NASH CHARACTERISATION: a profile with post-state y is a pure Nash equilibrium
    of the interior stage game iff  |D - H(y) + L(y)| <= M(y),  where
        H(y)=#{tau_i > y+1/2}, L(y)=#{tau_i < y-1/2}, M(y)=n-H-L.
T5  G >= 0 POINTWISE (so F = max(G,0) = G and B = 0 at the ENSEMBLE level).
T6  QUANTIZATION CANCELS at zero preference dispersion under stage-Nash
    (no irreducible baseline) but NOT under naive-myopic (over-correction floor).
T7  My 21-state chain reproduces a brute-force simulation of the FULL 3-resource
    27-action environment.
"""
from __future__ import annotations

import itertools
import numpy as np

from a00_lib import (CAP, DELTAS, N_AGENTS, TIDX, T_EPISODE, agent_probs,
                     aggregate_pmf, delegated_occupancy, draw_problem, gap_batch,
                     planner_occupancy, start_index, state_values,
                     transition_matrix)

rng = np.random.default_rng(20260730)
OUT = []


def chk(name, cond, detail=""):
    OUT.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}")


# --------------------------------------------------------------------------- #
print("=" * 78)
print("T3 / T1  DEADZONE-SIGN LAW and SEPARABILITY, vs brute force over 27 actions")


def brute_naive_action(shat_i, tau_i):
    """argmin over the actual 27 delta vectors of the agent's own next-step loss,
    others treated as static, clamp applied. shat_i, tau_i: (3,)."""
    best, bd = None, np.inf
    for d in itertools.product([-1, 0, 1], repeat=3):
        y = np.clip(shat_i + np.array(d, float), 0.0, CAP)
        loss = np.sum((y - tau_i) ** 2)          # weights are a positive scalar per
        if loss < bd - 1e-12:                    # coordinate -> cannot move the argmin
            bd, best = loss, np.array(d, float)
    return best


ok = True
for _ in range(4000):
    s = rng.uniform(1.5, 8.5, size=3)            # interior, clamp inactive
    tau = rng.normal(0, 1.5, size=3) + s
    b = brute_naive_action(s, tau)
    a = np.clip(np.round(tau - s), -1, 1)        # deadzone-sign, per coordinate
    if not np.allclose(a, b):
        ok = False
chk("T3 interior naive BR == per-coordinate clip(round(tau-shat),-1,1)", ok)

ok2, ndiff = True, 0
for _ in range(4000):
    s = rng.uniform(0, CAP, size=3)              # includes the clamp region
    tau = rng.normal(0, 1.0, size=3)
    b = brute_naive_action(s, tau)
    a = np.clip(np.round(tau - s), -1, 1)
    y_b = np.clip(s + b, 0, CAP)
    y_a = np.clip(s + a, 0, CAP)
    if not np.allclose(np.sum((y_b - tau) ** 2), np.sum((y_a - tau) ** 2)):
        ok2 = False
        ndiff += 1
chk("T3 clamp-UNAWARE deadzone rule differs from the true argmax near the wall",
    not ok2, f"differs on {100*ndiff/4000:.1f}% of draws -> must use clamp-aware")

# validate the threshold representation of the clamp-AWARE rule
from a00_lib import thresholds, _delta_star                            # noqa: E402
bad = 0
for _ in range(300):
    tau1 = rng.normal(0, 1.5, size=(1, 1))
    for eps in (0.0, 0.25, 1.0):
        tA, tB = thresholds(tau1, eps)
        us = np.linspace(-8, 18, 2001).reshape(1, -1)
        got = _delta_star(us, tau1[:, :1], eps)[0]
        pred = np.where(us[0] < tA[0, 0], 1, np.where(us[0] > tB[0, 0], -1, 0))
        bad += int(np.sum(got != pred) > 3)     # allow <=3 grid cells at a switch
chk("T3 clamp-aware BR is a two-threshold monotone rule in u = shat + c",
    bad == 0, f"{bad} violations over 900 (tau, eps) cases")

# separability of the 27-action argmin: does it decompose coordinatewise?
ok3 = True
for _ in range(2000):
    s = rng.uniform(0, CAP, size=3)
    tau = rng.normal(0, 2, size=3)
    b = brute_naive_action(s, tau)
    per = np.array([brute_naive_action(np.array([s[c], 5.0, 5.0]),
                                       np.array([tau[c], 5.0, 5.0]))[0]
                    for c in range(3)])
    y_b = np.clip(s + b, 0, CAP)
    y_p = np.clip(s + per, 0, CAP)
    if not np.isclose(np.sum((y_b - tau) ** 2), np.sum((y_p - tau) ** 2)):
        ok3 = False
chk("T1 the 27-action argmin factorises into 3 scalar argmins", ok3)

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T4  NASH CHARACTERISATION vs exhaustive 3^4 = 81 profile enumeration")


def nash_agg_bruteforce(x, tau, cap_on=False):
    """All pure-strategy Nash aggregates of the one-shot scalar stage game."""
    out = set()
    for prof in itertools.product([-1, 0, 1], repeat=len(tau)):
        D = sum(prof)
        y = np.clip(x + D, 0, CAP) if cap_on else x + D
        good = True
        for i, di in enumerate(prof):
            rest = D - di
            best = min((np.clip(x + rest + d, 0, CAP) if cap_on else x + rest + d)
                       for d in (-1, 0, 1))  # placeholder, replaced below
            vals = []
            for d in (-1, 0, 1):
                yy = np.clip(x + rest + d, 0, CAP) if cap_on else x + rest + d
                vals.append((yy - tau[i]) ** 2)
            if (y - tau[i]) ** 2 > min(vals) + 1e-12:
                good = False
                break
        if good:
            out.add(D)
    return out


def nash_agg_formula(x, tau):
    """|D - H(y) + L(y)| <= M(y) with y = x + D."""
    out = set()
    n = len(tau)
    for D in range(-n, n + 1):
        y = x + D
        H = int(np.sum(tau > y + 0.5))
        L = int(np.sum(tau < y - 0.5))
        M = n - H - L
        if abs(D - (H - L)) <= M:
            out.add(D)
    return out


ok, nonempty, sizes = True, 0, []
for _ in range(6000):
    x = rng.uniform(2, 8)
    tau = rng.normal(x, 1.2, size=4)
    a, b = nash_agg_bruteforce(x, tau, cap_on=False), nash_agg_formula(x, tau)
    if a != b:
        ok = False
    if a:
        nonempty += 1
    sizes.append(len(a))
chk("T4 characterisation == exhaustive enumeration (interior)", ok)
chk("T4 a pure Nash equilibrium always exists", nonempty == 6000,
    f"mean #distinct Nash aggregates = {np.mean(sizes):.2f}")

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T2 / T5 / T7  chain vs a brute-force simulator of the FULL environment")


def brute_env_rollout(tau3, w3, s0, eps, T=T_EPISODE, seed=0, n_mc=400):
    """Simulate the ACTUAL environment: 3 resources, 4 agents, 27 actions,
    each agent taking the TRUE argmax of its own expected reward (clamp-aware,
    flat prior), computed by brute force over all 27 delta vectors."""
    from a00_lib import expected_clamped_loss
    r = np.random.default_rng(seed)
    acts = np.array(list(itertools.product([-1.0, 0.0, 1.0], repeat=3)))   # (27,3)
    tot = 0.0
    for _ in range(n_mc):
        s = s0.copy()
        acc = 0.0
        for _t in range(T):
            shat = np.repeat(s[None, :], 4, axis=0)
            if eps > 0:
                shat = shat + np.sqrt(eps) * r.standard_normal((4, 3))
            d = np.empty((4, 3))
            for i in range(4):
                # expected loss of each of the 27 vectors = sum over resources
                L = np.zeros(27)
                for c in range(3):
                    dint = np.clip(np.round(tau3[i, c] - shat[i, c]), -1, 1)
                    L += w3[i, c] * (expected_clamped_loss(
                        shat[i, c] + acts[:, c], tau3[i, c], eps)
                        + 1e-9 * np.abs(acts[:, c] - dint))
                d[i] = acts[np.argmin(L)]
            s = np.clip(s + d.sum(axis=0), 0.0, CAP)
            acc += -np.sum(w3 * (s[None, :] - tau3) ** 2)
        tot += acc / T
    return tot / n_mc


maxerr = 0.0
for trial in range(6):
    eps = [0.0, 0.0, 0.25, 0.5, 1.0, 0.75][trial]
    tau3 = rng.normal(0, 1, size=(4, 3))
    w3 = np.clip(0.6 + 0.05 * rng.standard_normal((4, 3)), 0, 1)
    s0 = rng.uniform(0, CAP, size=3)
    sim = brute_env_rollout(tau3, w3, s0, eps, n_mc=(1 if eps == 0 else 500),
                            seed=trial)
    # chain: 3 independent scalar problems
    tot = 0.0
    for m in range(3):
        g = gap_batch(tau3[:, m][None, :], w3[:, m][None, :],
                      np.array([s0[m]]), eps, protocol="naive")
        occ, xv = g["occ_d"][0], g["xval"][0]
        Wb = g["Wbar"][0]
        mu = g["mu"][0]
        V = float(np.sum(w3[:, m] * (tau3[:, m] - mu) ** 2))
        tot += -(Wb * np.sum(occ * (xv - mu) ** 2) + V)
    err = abs(sim - tot) / max(1.0, abs(tot))
    maxerr = max(maxerr, err)
    print(f"    eps={eps:4.2f}  brute-force sim={sim:9.4f}   exact chain={tot:9.4f}"
          f"   rel.err={err:.2e}")
chk("T7 exact 21-state chain reproduces the full-environment simulator",
    maxerr < 6e-3, f"max rel err {maxerr:.2e} (MC error of the simulator)")

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T4b  Nash aggregate UNIQUENESS with the clamp, on the actual lattice")
from a00_lib import nash_pmf_wrap                                       # noqa: E402
t, w, x0 = draw_problem(rng, 1500, "lam", 1.0, 0.6)
t = t + rng.normal(0, 1, size=(1500, 1))
xv = state_values(x0 - np.floor(x0))
mu = (w * t).sum(1) / w.sum(1)
_, isna = nash_pmf_wrap(t, xv, mu)
cnt = isna.sum(axis=2)
chk("T4b a pure Nash aggregate exists at every reachable state", cnt.min() >= 1)
chk("T4b the Nash aggregate is a.s. UNIQUE (clamp included)", cnt.max() == 1,
    f"max #Nash aggregates = {cnt.max()}, mean = {cnt.mean():.4f}")

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T5  IS G >= 0 POINTWISE?")
for proto, epss in (("naive", (0.0, 0.25, 1.0)), ("sne", (0.25, 1.0)),
                    ("nash", (0.0,))):
    worst = np.inf
    for eps in epss:
        t, w, x0 = draw_problem(rng, 3000, "lam", 1.0, 0.6)
        t = t + rng.normal(0, 1, size=(3000, 1))
        g = gap_batch(t, w, x0, eps, protocol=proto)
        worst = min(worst, g["G"].min())
    chk(f"T5 [{proto}] G >= 0 for every draw", worst > -1e-9, f"min G = {worst:.3e}")

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T6  ZERO-DISPERSION LIMIT (eps=0): is there an irreducible baseline?")
print("    lambda   E[G] naive   E[G] Nash    P(G>0) naive   P(G>0) Nash")
for lam in (0.0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0):
    t, w, x0 = draw_problem(rng, 6000, "lam", lam, 0.6)
    t = t + rng.normal(0, 1, size=(6000, 1))     # common level; dispersion = lam
    gn = gap_batch(t, w, x0, 0.0, protocol="naive")
    ge = gap_batch(t, w, x0, 0.0, protocol="nash")
    print(f"    {lam:5.2f}   {gn['G'].mean():10.4f}  {ge['G'].mean():10.5f}"
          f"    {(gn['G']>1e-9).mean():10.3f}  {(ge['G']>1e-9).mean():11.3f}")

# explicit worked example of naive over-correction at PERFECT alignment
x, tau = 3.4, 5.0
traj = [x]
for _ in range(6):
    d = sum(int(np.clip(np.round(tau - traj[-1]), -1, 1)) for _ in range(4))
    traj.append(float(np.clip(traj[-1] + d, 0, CAP)))
print(f"\n    naive, 4 IDENTICAL targets tau=5, x0=3.4 -> {[round(v,2) for v in traj]}"
      f"   (2-cycle, amplitude 4)")
print(f"    stage-Nash aggregates at x=3.4: "
      f"{sorted(nash_agg_formula(3.4, np.full(4, 5.0)))}"
      f"  -> post-state 5.4 = the nearest lattice point (quantization floor, exact)")

print("=" * 78)
print(f"{sum(OUT)}/{len(OUT)} structural checks PASS")
