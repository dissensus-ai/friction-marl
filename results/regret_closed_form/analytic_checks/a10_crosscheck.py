"""Cross-check the planner benchmark against the numbers already in the corpus,
and quantify the LEARNED-agent vs EXACT-best-response divergence.

ideation_measurement.md sec 0.1 reports, for seed-42 condition draws:
   alpha=-0.8 sigma=0.6 : V*_p = -0.767   eval_reward_clean = -1.309  regret 0.542
   alpha= 0.0 sigma=0.6 : V*_p = -1.014   eval_reward_clean = -2.287  regret 1.273
   alpha= 0.8 sigma=1.0 : V*_p = -1.158   eval_reward_clean = -2.287  regret 1.129
"""
import numpy as np
from a00_lib import (CAP, N_AGENTS, N_RES, planner_occupancy, start_index,
                     state_values, targets_implemented, gap_batch)

R = 40000
print("=" * 76)
print("X1  PLANNER per-step welfare V*_p (mean over the preference DGP)")
print("     alpha  sigma |  V*_p exact (this memo)  |  spec sec 0.1 value")
spec = {(-0.8, 0.6): -0.767, (0.0, 0.6): -1.014, (0.8, 1.0): -1.158}
for (a, s), v in spec.items():
    rng = np.random.default_rng(20260730)
    tot = 0.0
    for m in range(N_RES):
        t = targets_implemented(rng.standard_normal(R), rng.standard_normal((R, N_AGENTS)), a)
        w = np.clip(s + 0.05 * rng.standard_normal((R, N_AGENTS)), 0, 1)
        x0 = rng.uniform(0, CAP, R)
        Wb = w.sum(1); mu = (w * t).sum(1) / Wb
        V = (w * (t - mu[:, None]) ** 2).sum(1)
        xv = state_values(x0 - np.floor(x0)); si = start_index(x0)
        occ = planner_occupancy(xv, mu, Wb, start_idx=si)
        tot += (-(Wb * (occ * (xv - mu[:, None]) ** 2).sum(1) + V)).mean()
    print(f"     {a:+.1f}   {s:.1f}  |     {tot:8.4f}            |  {v:8.3f}")
print("  (the spec's numbers are for ONE seed-42 condition draw of 30 replications;")
print("   these are population means over the same DGP, so agreement is in level")
print("   and ordering, not to 3 d.p.)")

print("=" * 76)
print("X2  EXACT-best-response delegates vs the TRAINED Q-learners")
print("    trained eval_reward_clean (spec sec 0.1) vs exact-BR realised welfare")
for (a, s), _ in spec.items():
    rng = np.random.default_rng(20260730)
    tot = {"naive": 0.0, "nash": 0.0}
    for m in range(N_RES):
        t = targets_implemented(rng.standard_normal(R), rng.standard_normal((R, N_AGENTS)), a)
        w = np.clip(s + 0.05 * rng.standard_normal((R, N_AGENTS)), 0, 1)
        x0 = rng.uniform(0, CAP, R)
        Wb = w.sum(1); mu = (w * t).sum(1) / Wb
        V = (w * (t - mu[:, None]) ** 2).sum(1)
        for proto in ("naive", "nash"):
            g = gap_batch(t, w, x0, 0.0, protocol=proto)
            occ = g["occ_d"]; xv = g["xval"]
            tot[proto] += (-(Wb * (occ * (xv - mu[:, None]) ** 2).sum(1) + V)).mean()
    print(f"     alpha={a:+.1f} sigma={s:.1f}: naive-myopic W = {tot['naive']:8.4f}"
          f"   stage-Nash(best) W = {tot['nash']:8.4f}"
          f"   trained IQL eval = {[-1.309,-2.287,-2.287][list(spec).index((a,s))]:7.3f}")
