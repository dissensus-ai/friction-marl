# Partial-Sharing Control: does the alignment-gap slope scale with the DEGREE of shared-state contention?

**Original run:** 2026-06-27
**Result of record updated:** 2026-08-13
**Scope:** The mechanistic test reviewer #5 asked for. Two endpoints were already on
disk — *fully separable* (no contention) is FLAT, *fully shared* (full contention)
is SLOPED. This experiment parametrically varies the **degree** of shared-state
contention between those extremes and asks whether the alignment-gap slope scales
(monotonically) with that degree. A monotone, significant trend is evidence that
**contention degree modulates the alignment effect** (rather than the
slope being an all-or-nothing property of "some shared state").
**New dir:** `results/partial_sharing/` (this dir). The June control run did not
modify the paper; the 8--10 August 2026 interpretation and robust-inference updates
are now incorporated in DAI-2606 and in this file.
Compute: both AMD GPUs under ROCm 6.3, torch 2.9.1, py3.14. 30 reps × 1000 episodes
per condition, ε=0 — the same harness/rigor as the prior `adversarial_rerun/` and
`reviewer3_controls/` controls.

---

## Result of record: paired frozen held-out follow-up

Reviewer3 correctly noted that the June outcome mixed learned-policy quality with continued
updates and exploration, and that its overlap levels did not share identical problem draws. The
13 August follow-up fixes both defects. Policies are frozen and scored on 100 held-out resets;
preference instances and network initializations are paired across overlap; an exact endpoint gate
reuses the standalone separable (`k=0`) and shared (`k=3`) runs and trains only `k=1,2`.

| overlap w (k/3) | IQL frozen slope [95% CI] (p) | VDN frozen slope [95% CI] (p) |
|---|---|---|
| 0.00 (k=0) | -0.219 [-0.563, +0.126] (.213) | -0.125 [-0.496, +0.246] (.509) |
| 0.33 (k=1) | -0.566 [-0.956, -0.176] (.0045) | -0.594 [-0.949, -0.240] (.0010) |
| 0.67 (k=2) | -0.807 [-1.297, -0.318] (.0012) | -0.895 [-1.263, -0.528] (1.8e-6) |
| 1.00 (k=3) | -1.156 [-1.695, -0.618] (2.6e-5) | -1.162 [-1.515, -0.808] (1.2e-10) |

The primary problem-block-clustered `rho x w` interaction is **-0.916** [−1.304,
−0.528], p=3.7e-6 (IQL) and **−1.023** [−1.260, −0.787], p=2.0e-17 (VDN).
Categorical slope modulation is detected (p=3.6e-6 / 3.1e-16); departure from a
linear ramp is not (p=.890 / .191). Thus every partially shared arm now has a detected negative
slope and the old “majority-shared threshold” suggestion is withdrawn. The evidence supports
graded modulation; four levels still do not prove exact linearity.

The neutral (`rho=0`) baseline does not rise with sharing: clustered slope −0.390 IQL and −0.849
VDN. That is opposite the proposed additive difficulty confound, although a difficulty-by-rho
interaction remains inseparable from contention in this design. The result identifies a scope
condition in these environment variants, not a unique mechanism.

Current data and regenerators:
`results/reviewer3_followup/partial_legacy_paired.csv`,
`partial_outcome_slopes.csv`, `partial_frozen_interactions.csv`,
`partial_frozen_shape_tests.csv`, and `analyze_followup.py`.

## Historical June analysis — provenance only, superseded for inference

The material below preserves the independently drawn, late-training June analysis so the change
is auditable. Its numerical estimates are not the current result and should not be quoted as such.

### Historical headline

**The alignment-gap slope scales monotonically with the overlap degree, for BOTH
learners, and the scaling is statistically significant.** As the fraction of
resources that are common-pool (contended) rises from w=0 → 1, the slope of the
normalized reward gap vs signed preference correlation ρ falls from ≈0 (flat) to
≈−0.75 (steep):

| overlap w (k/3) | IQL slope (p) | VDN slope (p) |
|---|---|---|
| 0.00 (k=0, separable) | −0.022 (0.91) | −0.066 (0.73) |
| 0.33 (k=1) | −0.210 (0.35) | −0.244 (0.27) |
| 0.67 (k=2) | **−0.602 (3.4e-3)** | **−0.614 (7.8e-4)** |
| 1.00 (k=3, fully shared) | **−0.741 (2.9e-3)** | **−0.755 (2.8e-4)** |

- **Perfectly monotone:** Spearman(w, signed slope) = **−1.000** and
  Spearman(w, |slope|) = **+1.000** for both learners (4 levels each).
- **Significant scaling (interaction OLS `gap_norm ~ ρ * w`, w continuous, HC3
  uncertainty, n=1920 per mode):** the ρ:w interaction is **−0.765 [95% CI
  −1.348, −0.183], p=0.010 (IQL)** and **−0.731 [−1.260, −0.203], p=0.0067
  (VDN)** — the per-ρ slope becomes
  significantly more negative as overlap rises, with implied w=1 slopes (−0.78 both)
  matching the directly-fitted endpoint.
- **Endpoint sanity gate PASSED:** the new runner's own w=0 and w=1 reproduce the
  pre-existing `separable_*` (flat) and `equicorr_*` (sloped) reference runs
  (see table below), so the middle levels are trustworthy.

**Mechanistic reading the bridge supports:** the signed-correlation gap relation is
not switched on by the mere existence of any shared state — it is **graded in the
amount of shared-state contention.** The ordered point estimates and significant
ρ-by-w interaction turn the binary "shared vs separable" result of
`reviewer3_controls/` into a graded dose-response claim. Because overlap also changes
credit-assignment difficulty, this experiment identifies an interaction, not a
causal mediation effect.

**Honest nuance (does NOT overturn the headline):** one shared resource out of three
(w=0.33) is descriptively close to flat and not individually significant (HC3:
IQL p=0.35, VDN p=0.27); the slopes are individually significant at k≥2. That
pattern does not establish a majority-sharing threshold. Under HC3, the four-level
categorical slope deviations are not jointly significant at 0.05 (F(3,1912)=2.43,
p=0.063 IQL; F=2.56, p=0.053 VDN), and a direct slope-shape test cannot distinguish
the four estimates from a linear ramp (F(2,1912)=0.12, p=0.890 IQL; F=0.12,
p=0.887 VDN). The categorical full-surface improvement is driven by nonlinear
baseline shifts. The supported statement is graded modulation under the linear
interaction, alongside a monotone point-estimate ordering; threshold/curvature
remains unresolved.

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
30 reps, 1000 episodes — identical to the two endpoint runs. The condition-seed
schedule is fixed independently of `--mode`, so IQL (GPU0) and VDN (GPU1) used
matched preference draws per (w,ρ,σ). The archived runs predate deterministic
PyTorch seeding and do not support a claim of identical network initializations.

**Stats.** Per-overlap and continuous-interaction coefficients use statsmodels OLS
with HC3 robust uncertainty (`.params/.bse/.pvalues/.conf_int`); the four-level
Spearman p-value is the exact two-sided permutation value. HC3 Wald restrictions
provide the confirmatory categorical slope and shape tests; classical nested F tests
and AIC are retained as descriptive audit outputs. No positional result unpacking is used.

---

## SANITY GATE — endpoints reproduce the pre-existing reference runs

The new runner's own endpoints were fit independently and compared to the frozen
reference CSVs (`separable_*` for w=0, `equicorr_*` for w=1):

| mode | endpoint | new slope (p) | reference slope | 95% CIs overlap | verdict |
|---|---|---|---|---|---|
| IQL | w=0 separable | −0.022 (0.91) | −0.029 | yes | FLAT, match ✓ |
| IQL | w=1 shared | −0.741 (2.9e-3) | −0.755 | yes | SLOPED, match ✓ |
| VDN | w=0 separable | −0.066 (0.73) | −0.053 | yes | FLAT, match ✓ |
| VDN | w=1 shared | −0.755 (2.8e-4) | −0.838 | yes | SLOPED, match ✓ |

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

## Trend and interaction tests (the headline statistics)

**Per-level slopes** (`slope_vs_overlap.csv`): table above; both learners give a
**perfectly monotone** decrease (Spearman(w, slope) = −1.000;
Spearman(w, |slope|) = +1.000). With four levels the exact two-sided permutation
p-value is 0.083, so the rank statistic is descriptive; |slope| is monotonically
non-decreasing in w for both.

**Interaction OLS with HC3 uncertainty** `gap_norm ~ ρ * w` per mode
(`interaction_test.csv`, n=1920):

| mode | slope @ w=0 (coef ρ) | ρ:w interaction [95% CI] | p | implied slope @ w=1 |
|---|---|---|---|---|
| IQL | −0.011 | **−0.765** [−1.348, −0.183] | **0.010** | −0.776 |
| VDN | −0.054 | **−0.731** [−1.260, −0.203] | **0.0067** | −0.786 |

The ρ:w coefficient is the change in the per-ρ slope per unit of overlap; it is
significantly **negative** for both learners, so the slope changes with contention
degree. The implied w=1 slopes (−0.78) match the directly-fitted endpoints
(−0.74 / −0.76), confirming internal consistency. The full categorical surface
beats the linear surface, but this is driven by nonlinear baseline shifts: the
slope-shape test itself is F(2,1912)=0.15, p=0.86 for both learners.

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

1. **Monotone; shape unresolved.** The w=0.33 (1 of 3 shared) level is descriptively
   flat and not individually significant, while the k≥2 estimates are significant.
   The four slopes nevertheless do not depart detectably from a linear ramp under HC3
   (F(2,1912)=0.12, p=0.890 IQL; F=0.12, p=0.887 VDN). A finer w-grid (more resources m so
   k/m has finer steps, or a continuous state-mixing parameter) would be needed to
   distinguish a linear relation from a threshold.
2. **Granularity of w is set by m=3.** With 3 resources only 4 overlap levels exist.
   The construction generalises (raise m for finer w), but that was not run here.
3. **The four-point rank order is descriptive.** Perfect ordering gives
   Spearman = ±1, but its exact two-sided permutation p-value is 0.083. The
   load-bearing inference is the n=1920 HC3 linear interaction; categorical HC3
   slope/shape tests bound what can be claimed about curvature, and the rank
   correlation remains descriptive.
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
| `analyze_partial.py` | Clean analysis (sanity gate; HC3 per-w slopes; exact Spearman; interaction and surface-shape tests; plots) |
| `ANALYSIS.txt` | Full printed analysis log |
| `slope_vs_overlap.csv` | Per-(mode,w) slope, SE, 95% CI, p, R² |
| `interaction_test.csv` | ρ:w interaction coefficient + HC3 SE, CI, and p per mode |
| `surface_shape_tests.csv` | Categorical-vs-linear surface, slope-difference, slope-shape, and intercept-shape tests |
| `sanity_gate.csv` | Endpoint reproduction (new vs reference slopes) |
| `gap_vs_rho_by_overlap.{png,pdf}` | Gap-vs-ρ curves colored by overlap w |
| `slope_vs_overlap.{png,pdf}` | Headline: slope vs overlap w, with reference endpoints |

**Reproduce:**
`python results/partial_sharing/partial_runner.py --modes iql --overlaps 0 1 2 3 --device 0 --seed 42 --out results/partial_sharing/partial_iql.csv`
(and `--modes vdn --device 1 ...`), then
`python results/partial_sharing/analyze_partial.py`.
(If interrupted, finish missing overlaps per-overlap and run `merge_resume.py` first.)
