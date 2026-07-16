# A1 — Feasibility vs learning decomposition of the cooperative reward-gap decline

**Date:** 2026-06-27
**Reviewer finding addressed (Critical):** *"confounded coordination-friction metric"* — the
cooperative decline in the reward gap may be pure FEASIBILITY (the achievable joint
optimum rises toward 0 as targets align), not reduced learning friction.

**Verdict: FLAT — the cooperative dividend is *entirely* feasibility, and the
learning-controlled component moves the WRONG way.** The reviewer's critique is
confirmed, and is in fact stronger than argued.

---

## 1. Reward definition (exact, from source)

Per-agent, per-step reward (`b_vdn_runner.py:184-185`), shared resource state
`s ∈ [0, CAPACITY]^M`, `CAPACITY = 10`, `M = 3` resources, `A = 4` agents:

```
r_i(s) = - Σ_j  w_ij · (s_j − τ_ij)²
```

- `s` is the **single shared** resource-level vector (every agent observes/affects the
  same `s`; `obs_all = states.unsqueeze(1).expand(-1,A,-1)`), evolved by summed ±1 actions
  and **clipped to the box** `[0, 10]^3` (`new_states.clamp(0, CAPACITY)`).
- `τ_ij` = agent *i*'s ideal level for resource *j* (equicorr DGP, unit marginal variance,
  cross-agent corr = ρ).
- `w_ij` = stake/weight ~ `clip(N(σ, 0.05), 0, 1)`.
- The DV of record is the **mean per-agent reward** `R̄`; the gap of record is
  `ΔR = R* − R̄` with `R* = 0` (each agent's unconstrained ideal → reward 0).

## 2. Decomposition

```
ΔR  =  (R* − R^joint)        +   (R^joint − R̄)
       └─ F: feasibility floor┘    └─ L: learning shortfall ┘
```

- **R^joint** = best achievable *mean* per-agent reward at a **single shared** static
  state in the box: `R^joint = (1/A)·max_{s∈[0,10]^3} Σ_i r_i(s)`. The summed reward
  separates by resource, so the optimum is the closed-form **stake-weighted average of
  targets, clipped to the box**: `s*_j = clip( Σ_i w_ij τ_ij / Σ_i w_ij , 0, 10 )`.
  Verified against `scipy.optimize` (L-BFGS-B): max |closed-form − numeric| = **3.8e-11**.
- **F = −R^joint ≥ 0**: unavoidable loss from forcing all agents onto one shared state
  (plus box clipping). Depends ONLY on preferences → identical for IQL and VDN (same seeds).
- **L = R^joint − R̄ ≥ 0**: what learning controls — how much reward the learners leave on
  the table relative to the best achievable shared state. `R̄ ≤ R^joint` always (time-avg of
  per-state mean reward ≤ its max); confirmed: **min L = +0.289 (IQL), +0.157 (VDN)** across
  all 480 reps/learner.

**Provenance / exactness.** Targets and weights were regenerated from the *exact* seeded
DGP path of `adv_runner.py` (master seed 42 → probe draw → 16 per-condition seeds →
per-rep `standard_normal` targets then `normal(σ,0.05)` weights), so per-rep `F` is the
feasibility floor of the *same* preference draws the RL trained on. Internal checks:
(i) decomposition `F + L = ΔR` holds to 1e-16; (ii) the regenerated `ΔR` per-condition means
reproduce the run logs to 4 dp (e.g. IQL ρ=−0.33,σ=0.6 → 2.0410; VDN ρ=+0.8,σ=1.0 → 2.3304);
(iii) the regenerated **gap_norm slope reproduces the slope of record exactly: −0.7552 (IQL,
record −0.755), −0.8380 (VDN, record −0.838).**

All regressions use named statsmodels attributes (`.params`, `.pvalues`, `.rsquared`).
Environment: clean uv venv, Python 3.13.11 (numpy 2.5.0, pandas 3.0.3, scipy 1.18.0,
statsmodels 0.14.6).

## 3. Results — L (learning shortfall) vs ρ, per learner

Replication-level (n = 480 = 8 ρ × 2 σ × 30 reps), DV normalized by σ to match the slope of
record (`ΔR/σ ~ ρ`). Component slopes add to the gap slope by construction.

| learner | term | slope (·/σ ~ ρ) | p | R² | share of ΔR-slope |
|---|---|---|---|---|---|
| IQL | ΔR (gap, of record) | **−0.7552** | 1.7e-04 | 0.029 | 100% |
| IQL | F (feasibility) | **−1.1061** | 9.3e-13 | 0.101 | **+146.5%** |
| IQL | L (learning) | **+0.3509** | 1.0e-06 | 0.049 | **−46.5%** |
| VDN | ΔR (gap, of record) | **−0.8380** | 7.3e-07 | 0.050 | 100% |
| VDN | F (feasibility) | **−1.1061** | 9.3e-13 | 0.101 | **+132.0%** |
| VDN | L (learning) | **+0.2680** | 5.7e-09 | 0.069 | **−32.0%** |

Raw-gap units (no σ-normalization) tell the same story: feasibility share 139.9% (IQL) /
128.6% (VDN); learning share −39.9% / −28.6% (L_raw slopes +0.269 p=4.7e-5, +0.210 p=1.9e-7).
Condition-level (16 cells) is identical in sign and significance: L_norm slope +0.351
(IQL, p=2.5e-4, R²=0.63) / +0.268 (VDN, p=3.5e-3, R²=0.47); F_norm −1.106 (p=1.2e-6, R²=0.82).

**The learning shortfall L significantly INCREASES with cooperation (positive slope, p ≤ 1e-6
both learners). It does not decline — it gets worse.** The entire (>100%) decline in the
gap is the feasibility floor collapsing; the learning channel partially offsets it.

## 4. Why the feasibility floor falls — it is shared-state conflict, not box clipping

Splitting `F = box-clip + conflict` (box-clip = single-agent box-constrained optimum loss,
present even with zero conflict; conflict = extra loss from one shared state). Exact to 1e-16:

| component (·/σ ~ ρ) | slope | p | R² | mean | reading |
|---|---|---|---|---|---|
| box-clip floor | −0.0107 | 0.94 | 0.000 | 1.407 | **flat in ρ** (unit-var targets vs [0,10] box) |
| shared-state conflict | −1.0954 | 2.8e-45 | 0.341 | 0.917 | **collapses with ρ** (1.41 → 0.21 over ρ −0.33→0.8) |

So the feasibility decline is *precisely* the reviewer's mechanism: as targets align, the
shared-state conflict vanishes (the achievable joint optimum rises toward 0). Box clipping is
a fixed offset and contributes nothing to the ρ-trend.

## 5. LaTeX-ready table

```latex
\begin{table}[t]
\centering
\caption{Feasibility--learning decomposition of the cooperative reward-gap decline.
Reward gap $\Delta R = R^* - \bar R$ ($R^*{=}0$) splits as $\Delta R = (R^*-R^{\mathrm{joint}})
+ (R^{\mathrm{joint}}-\bar R) = F + L$, where $R^{\mathrm{joint}}$ is the best achievable
mean reward at a single shared state in the box. Slopes are of the $\sigma$-normalized
component on $\rho$ (replication level, $n{=}480$); component slopes sum to the gap slope.}
\label{tab:rjoint-decomp}
\begin{tabular}{llrrr}
\toprule
Learner & Component & Slope ($/\sigma\sim\rho$) & $p$ & Share of $\Delta R$ slope \\
\midrule
\multirow{3}{*}{IQL}
 & $\Delta R$ (gap of record)      & $-0.755$ & $1.7\times10^{-4}$ & $100\%$ \\
 & $F$ (feasibility floor)         & $-1.106$ & $9.3\times10^{-13}$ & $+147\%$ \\
 & $L$ (learning shortfall)        & $+0.351$ & $1.0\times10^{-6}$ & $-47\%$ \\
\midrule
\multirow{3}{*}{VDN}
 & $\Delta R$ (gap of record)      & $-0.838$ & $7.3\times10^{-7}$ & $100\%$ \\
 & $F$ (feasibility floor)         & $-1.106$ & $9.3\times10^{-13}$ & $+132\%$ \\
 & $L$ (learning shortfall)        & $+0.268$ & $5.7\times10^{-9}$ & $-32\%$ \\
\bottomrule
\end{tabular}
\end{table}
```

## 6. Verdict (plain, 3 lines)

1. The cooperative decline in the reward gap (slope −0.755 IQL / −0.838 VDN) is **entirely a
   feasibility effect**: the achievable joint optimum rises toward 0 as targets align (F slope
   −1.106, p≈1e-12), accounting for >100% of the decline; this is shared-state conflict, not box clipping.
2. The learning-controlled part **moves the wrong way** — the learning shortfall L
   *significantly increases* with cooperation (+0.351 IQL p=1e-6, +0.268 VDN p=6e-9): the
   learners convert the easier, more-aligned problem into reward *less* efficiently, not more.
3. **No genuine learning-friction reduction survives.** The empirical "cooperation reduces
   coordination friction" claim is a metric confound and must be downgraded; the cooperative
   dividend is achievability, and if anything the learning channel argues against the claim.

**Artifacts:** `rjoint_decomposition_summary.csv`, `rjoint_conditions_{iql,vdn}.csv` (this dir).
**Recompute script (committed):** `rjoint_recompute.py` (this dir). Replays the exact
seeded DGP path of `results/adversarial_rerun/adv_runner.py` (imports the real
`sample_agent_params_equicorr`), computes the closed-form `R^joint`/`F`/`L` per
replication, runs the replication-level regressions, includes the box-clip/conflict
floor split, and verifies its output against the two committed CSVs (reproduces the
summary slopes to <1e-6 relative error and the per-condition means to ~1e-15). Run with
`python results/review2_jun27/rjoint_recompute.py`; add `--write` to regenerate the CSVs.
The original recompute lived in an un-committed `scratchpad/` dir (now lost); this script
supersedes it.
