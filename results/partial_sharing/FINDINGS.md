# Partial-Sharing Control: does the alignment-gap slope scale with the DEGREE of shared-state contention?

**Date:** 2026-06-27
**Scope:** The mechanistic test reviewer #5 asked for. Two endpoints were already on
disk — *fully separable* (no contention) is FLAT, *fully shared* (full contention)
is SLOPED. This experiment parametrically varies the **degree** of shared-state
contention between those extremes and asks whether the alignment-gap slope scales
(monotonically) with that degree. A monotone, significant trend is the mechanistic
evidence that **contention degree mediates the alignment effect** (rather than the
slope being an all-or-nothing property of "some shared state").
**New dir:** `results/partial_sharing/` (this dir). **The paper was NOT modified.**
Compute: both AMD GPUs under ROCm 6.3, torch 2.9.1, py3.14. 30 reps × 1000 episodes
per condition, ε=0 — the same harness/rigor as the prior `adversarial_rerun/` and
`reviewer3_controls/` controls.

---

## HEADLINE

**The alignment-gap slope scales monotonically with the overlap degree, for BOTH
learners, and the scaling is statistically significant.** As the fraction of
resources that are common-pool (contended) rises from w=0 → 1, the slope of the
normalized reward gap vs signed preference correlation ρ falls from ≈0 (flat) to
≈−0.75 (steep):

| overlap w (k/3) | IQL slope (p) | VDN slope (p) |
|---|---|---|
| 0.00 (k=0, separable) | −0.022 (0.90) | −0.066 (0.70) |
| 0.33 (k=1) | −0.210 (0.29) | −0.244 (0.21) |
| 0.67 (k=2) | **−0.602 (9.2e-4)** | **−0.614 (2.0e-4)** |
| 1.00 (k=3, fully shared) | **−0.741 (8.5e-4)** | **−0.755 (5.5e-5)** |

- **Perfectly monotone:** Spearman(w, signed slope) = **−1.000** and
  Spearman(w, |slope|) = **+1.000** for both learners (4 levels each).
- **Significant mediation (interaction OLS `gap_norm ~ ρ * w`, w continuous, n=1920
  per mode):** the ρ:w interaction is **−0.765 [95% CI −1.276, −0.254], p=3.4e-3
  (IQL)** and **−0.731 [−1.203, −0.259], p=2.4e-3 (VDN)** — the per-ρ slope becomes
  significantly more negative as overlap rises, with implied w=1 slopes (−0.78 both)
  matching the directly-fitted endpoint.
- **Endpoint sanity gate PASSED:** the new runner's own w=0 and w=1 reproduce the
  pre-existing `separable_*` (flat) and `equicorr_*` (sloped) reference runs
  (see table below), so the middle levels are trustworthy.

**Mechanistic reading the bridge supports:** the signed-correlation friction law is
not switched on by the mere existence of any shared state — it is **graded in the
amount of shared-state contention.** Each additional common-pool resource that
forces agents to reconcile their targets in one state adds roughly proportional
slope. This is the dose-response curve that turns the binary "shared vs separable"
result of `reviewer3_controls/` into a continuous mediation claim.

**Honest nuance (does NOT overturn the headline):** the trend is monotone but
**not linear** — it is concave / mildly threshold-shaped. One shared resource out
of three (w=0.33) is still essentially flat and **not individually significant**
(IQL p=0.29, VDN p=0.21); the steepening only becomes individually significant once
a **majority** of resources are shared (k≥2). So the cleanest statement is
"slope increases monotonically with contention degree and the overall scaling is
significant," not "slope is a linear function of w." The aggregate interaction test
is significant precisely because the high-overlap levels carry the effect.

---

## The overlap construction (documented)

With m = `N_RESOURCES` = 3 resources, overlap level **w = k/m** makes **k of the m
resources a COMMON POOL** (all 4 agents act on it; their deltas are SUMMED into one
shared state, contention) and the remaining **m−k resources PRIVATE** (each agent
acts only on its own copy, separable). Levels **w ∈ {0, 1/3, 2/3, 1}** for k=0,1,2,3.

This *fraction-of-resources-shared* design was chosen over a continuous
state-mixing because it reduces **exactly** (bit-for-bit, verified) to the two
existing runners at the endpoints — same reward semantics, same code paths:

- **k=0 (w=0):** every resource private → identical to `reviewer3_controls/separable_runner.py`.
- **k=m (w=1):** every resource common-pool → identical to the shared-pool main env
  `review_upgrade/b_vdn_runner.run_condition`.

**Dynamics.** State is carried per-agent as (R, A, m). The first k (SHARED) columns
are initialised to a single common value per (rep, column) and, each step, the
common pool accumulates the **sum of all agents' deltas** on those columns and is
broadcast back to every agent (so shared columns stay identical across agents — a
genuine shared pool). The remaining m−k (PRIVATE) columns evolve per-agent (own
delta only). **Reward is unchanged** at every w: agent i's per-agent weighted
quadratic loss of its own post-state row vs its own target, summed over all m
resources. Only the *substrate* (shared vs private) on which actions land changes
with k.

**Endpoint-equivalence unit test (deterministic):** with identical
states/deltas/targets/weights, the partial update at k=3 equals the shared-env
update (max abs diff in both next-state and reward = **0.0**) and at k=0 equals the
separable-env update (max abs diff = **0.0**); shared columns verified to stay
identical across agents for k=1,2 (across-agent std = 0).

**Preferences held fixed across overlap.** All levels use the **same signed
equicorrelated DGP** `adversarial_rerun/dgp.sample_agent_params_equicorr` (ρ =
cross-agent target correlation, unit marginal target variance for every ρ, weights
w~N(σ,0.05) clipped). Only the env coupling varies, so any change in the gap-vs-ρ
slope is attributable to the overlap degree alone.

**Reused machinery.** `BatchedQNet`, `JointReplayBuffer`, optimizer,
hyperparameters, ε-schedule, IQL/VDN losses, policy probe, matched per-condition
seeds — **imported verbatim** from `b_vdn_runner` (exactly as `separable_runner.py`
does). Grid: ρ ∈ {−0.33,−0.20,−0.10,0,0.20,0.40,0.60,0.80}, σ ∈ {0.6,1.0}, ε=0,
30 reps, 1000 episodes — identical to the two endpoint runs. Matched seeds: cond
seeds drawn from the master RNG in fixed order independent of `--mode`, so IQL
(GPU0) and VDN (GPU1) saw identical preference draws & inits per (w,ρ,σ).

**Stats.** All test statistics via **named scipy/statsmodels attributes**
(`linregress.slope/.stderr/.pvalue/.rvalue`, `spearmanr.statistic/.pvalue`,
statsmodels `.params/.pvalues/.conf_int`) — no positional tuple unpacking.

---

## SANITY GATE — endpoints reproduce the pre-existing reference runs

The new runner's own endpoints were fit independently and compared to the frozen
reference CSVs (`separable_*` for w=0, `equicorr_*` for w=1):

| mode | endpoint | new slope (p) | reference slope | 95% CIs overlap | verdict |
|---|---|---|---|---|---|
| IQL | w=0 separable | −0.022 (0.90) | −0.029 | yes | FLAT, match ✓ |
| IQL | w=1 shared | −0.741 (8.5e-4) | −0.755 | yes | SLOPED, match ✓ |
| VDN | w=0 separable | −0.066 (0.70) | −0.053 | yes | FLAT, match ✓ |
| VDN | w=1 shared | −0.755 (5.5e-5) | −0.838 | yes | SLOPED, match ✓ |

**GATE PASSED** — both endpoints reproduce (flat at w=0, steep and significant at
w=1, CIs overlapping the references). The interpolation is therefore anchored to
the known results and the middle levels can be trusted.

---

## Normalized gap ΔR/σ vs ρ, by overlap (pooled over σ)

```
 [IQL]  w \ rho :  -0.33  -0.20  -0.10  +0.00  +0.20  +0.40  +0.60  +0.80
        w=0.00 :   3.90   3.95   3.94   4.10   3.48   3.98   3.87   3.97   (flat)
        w=0.33 :   4.14   4.33   3.90   4.18   3.96   4.05   3.88   4.00   (flat)
        w=0.67 :   3.85   3.91   3.94   3.82   3.52   3.33   3.36   3.38   (sloping)
        w=1.00 :   4.29   4.56   3.93   4.24   4.11   3.70   3.67   3.60   (sloped)

 [VDN]  w \ rho :  -0.33  -0.20  -0.10  +0.00  +0.20  +0.40  +0.60  +0.80
        w=0.00 :   4.02   4.12   4.11   4.21   3.65   4.10   3.97   4.08   (flat)
        w=0.33 :   4.09   4.29   3.80   4.12   3.91   3.95   3.80   3.91   (flat)
        w=0.67 :   3.52   3.62   3.63   3.56   3.29   3.03   3.09   3.01   (sloping)
        w=1.00 :   3.49   3.80   3.27   3.49   3.40   3.03   2.91   2.79   (sloped)
```

The cooperation dividend (gap drop at ρ>0) is absent at w=0/0.33 and emerges at
w=0.67/1.00 — see `gap_vs_rho_by_overlap.png`.

---

## Trend / mediation tests (the headline statistics)

**Per-level slopes** (`slope_vs_overlap.csv`): table above; both learners give a
**perfectly monotone** decrease (Spearman(w, slope) = −1.000, p=0.000;
Spearman(w, |slope|) = +1.000, p=0.000). |slope| is monotonically non-decreasing in
w for both.

**Interaction OLS** `gap_norm ~ ρ * w` per mode (`interaction_test.csv`, n=1920):

| mode | slope @ w=0 (coef ρ) | ρ:w interaction [95% CI] | p | implied slope @ w=1 |
|---|---|---|---|---|
| IQL | −0.011 | **−0.765** [−1.276, −0.254] | **3.4e-3** | −0.776 |
| VDN | −0.054 | **−0.731** [−1.203, −0.259] | **2.4e-3** | −0.786 |

The ρ:w coefficient is the change in the per-ρ slope per unit of overlap; it is
significantly **negative** for both learners → **contention degree mediates the
alignment effect**. The implied w=1 slopes (−0.78) match the directly-fitted
endpoints (−0.74 / −0.76), confirming internal consistency.

---

## Plots

- **`slope_vs_overlap.png`/`.pdf`** — the headline: slope d(ΔR/σ)/dρ vs overlap w,
  both learners, with 95% slope CIs; the pre-existing separable (w=0) and shared
  (w=1) reference slopes overlaid as open markers (they sit on the partial-run
  endpoints). Monotone descent from ≈0 to ≈−0.75.
- **`gap_vs_rho_by_overlap.png`/`.pdf`** — gap-vs-ρ curves colored by w (per mode):
  flat at w≤0.33, progressively steeper at w≥0.67.

---

## Limitations / honest caveats

1. **Monotone but nonlinear.** The slope is concave / threshold-shaped in w: the
   w=0.33 (1 of 3 shared) level is still flat and not individually significant, and
   most of the slope appears for k≥2. The claim supported is "slope increases
   monotonically with contention degree, overall scaling significant," **not** a
   linear dose-response. With only 4 overlap levels the functional form between them
   is under-resolved; a finer w-grid (more resources m so k/m has finer steps, or a
   continuous state-mixing parameter) would localise the threshold.
2. **Granularity of w is set by m=3.** With 3 resources only 4 overlap levels exist.
   The construction generalises (raise m for finer w), but that was not run here.
3. **Spearman p=0 is mechanical** (4 perfectly-ordered points). The load-bearing
   significance is the n=1920 interaction test, not the rank correlation.
4. **Same env family** (ResourceAllocation and its partial-overlap variant), n=4
   agents, ε=0, σ ∈ {0.6,1.0}, equicorrelated signed DGP — identical scope to the
   prior controls; other env families / agent counts are future work.
5. **R² of the per-ρ slope is small** (≤0.03) even at w=1 — consistent with the
   prior runs: ρ shifts the *mean* gap modestly against high per-replication
   variance. The effect is a reliable shift in the mean (significant slope), not a
   tight per-run predictor; inherited from the underlying finding.
6. **Compute provenance / interruption.** The first single-process full-grid run
   (`--overlaps 0 1 2 3`) was killed by an infrastructure wall-clock cap at ~59 min,
   after completing overlaps k=0,1,2 (IQL) / k=0,1 (VDN). The incomplete overlaps
   were **re-run per-overlap** (`--overlaps 2 3`, same seed/grid, identical args
   across modes) under `nohup` and **merged** (`merge_resume.py`): k=0,1 from the
   full-grid run, k=2,3 from the resume run. Consequence: k=0,1 used full-grid cond
   seeds and k=2,3 used per-overlap cond seeds. Within **every** overlap the IQL/VDN
   draws are matched (same args), and overlaps are never seed-compared to each other,
   so the per-(mode,w) slope analysis is unaffected. Each merged CSV is the full
   4 overlap × 16 cond × 30 rep = 1920 rows (integrity-checked in the merge).

---

## Files

| File | Contents |
|---|---|
| `partial_runner.py` | The partial-sharing runner (overlap k of m resources shared; reduces exactly to separable at k=0 and to the shared main env at k=m; reuses `b_vdn_runner` machinery verbatim) |
| `partial_iql.csv` / `partial_vdn.csv` | **Canonical merged** raw per-replication results (1920 rows each: 4 overlaps × 8 ρ × 2 σ × 30 reps) |
| `merge_resume.py` | Merges full-grid k=0,1 with per-overlap-resume k=2,3 (+integrity asserts) |
| `_orig_partial_{iql,vdn}.csv` | Full-grid run before interruption — provenance |
| `partial_{iql,vdn}_resume.csv` | Per-overlap resume run (k=2,3) — provenance |
| `analyze_partial.py` | Clean analysis (named-attribute scipy/statsmodels; sanity gate; per-w slopes; Spearman; interaction OLS; plots) |
| `ANALYSIS.txt` | Full printed analysis log |
| `slope_vs_overlap.csv` | Per-(mode,w) slope, SE, 95% CI, p, R² |
| `interaction_test.csv` | ρ:w mediation coefficient + CI + p per mode |
| `sanity_gate.csv` | Endpoint reproduction (new vs reference slopes) |
| `gap_vs_rho_by_overlap.{png,pdf}` | Gap-vs-ρ curves colored by overlap w |
| `slope_vs_overlap.{png,pdf}` | Headline: slope vs overlap w, with reference endpoints |

**Reproduce:**
`python results/partial_sharing/partial_runner.py --modes iql --overlaps 0 1 2 3 --device 0 --seed 42 --out results/partial_sharing/partial_iql.csv`
(and `--modes vdn --device 1 ...`), then
`python results/partial_sharing/analyze_partial.py`.
(If interrupted, finish missing overlaps per-overlap and run `merge_resume.py` first.)
