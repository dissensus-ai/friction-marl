# Reviewer3 Controls: Two Targeted Re-runs on the Signed-Correlation Finding

**Date:** 2026-06-27
**Scope:** Two control experiments that settle reviewer3's two CRITICAL concerns
about the paper's signed cross-agent-correlation finding (coordination gap =
-mean reward TRACKS signed preference correlation rho: cooperation rho>0 lowers
friction, opposition rho<0 ~ indifference rho=0). Same harness/DGP/style as the
prior `results/adversarial_rerun/`.
**New dir:** `results/reviewer3_controls/` (this dir). **Nothing else was modified;
the paper was not touched.** Compute: both AMD GPUs under ROCm 6.3, torch
2.9.1, py3.14. 30 reps x 1000 episodes per condition (mirrors the prior runs).

---

## HEADLINE VERDICTS

**Experiment 1 (separable resources, Critical #1): the signed-correlation law
FLATTENS.** With each agent on its OWN independent resource pool, the gap-vs-rho
curve goes flat (IQL slope -0.029, p=0.87; VDN slope -0.053, p=0.76; one-way
ANOVA across rho n.s. for both), whereas the shared-pool law is strong and
significant (IQL slope -0.755, p=2e-4; VDN -0.838, p=7e-7). **The friction is
SHARED-STATE CONTENTION, not preference correlation per se. The central claim is
therefore BOUNDED to common-pool / shared-state environments.** This is exactly
the bound the reviewer asked us to find, and we report it plainly.

**Experiment 2 (n=2 clean strong opposition, Critical #2 + Major #4): at genuine,
unconfounded rho=-1, opposition does NOT beat indifference.** With n=2 the single
pair reaches FULL rho=-1 (verified -1.0000) with no within-team-cooperation
confound and no -1/3 PSD cap. Opposition **TIES** indifference for VDN
(diff +0.35, Welch p=0.37, MWU p=0.85) and is **directionally WORSE but only
fragilely so** for IQL (diff +1.27, Welch p=0.041 *pooled-only*; MWU p=0.24;
per-sigma p=0.24/0.09 -- not significant within either sigma). In NO case does
opposition beat indifference. **Cooperation robustly beats indifference in both
learners** (IQL -1.08, p=0.004; VDN -0.68, p=0.037). This confirms the n=4
monotone reading -- only alignment helps -- at honest, unconfounded rho=-1.

Both controls **strengthen the paper's honest core** (alignment is the structural
factor) while **bounding it** (the law lives in shared-state friction).

---

## Setup common to both experiments

- **DGP:** reused verbatim from `results/adversarial_rerun/dgp.py` -- the signed
  equicorrelated sampler `sample_agent_params_equicorr` (per-resource targets
  ~ N(0, Sigma(rho)), Sigma(rho)=(1-rho)I+rho 11^T, eigh-based PSD sqrt, unit
  marginal target variance for every rho). Weights `w ~ N(sigma,0.05)` clipped to
  [0,1] unchanged (sigma = preference intensity / stakes).
- **Learners:** IQL (independent) + VDN (centralized value decomposition), the
  exact `results/review_upgrade/b_vdn_runner.py` machinery (BatchedQNet,
  JointReplayBuffer, matched per-condition seeds, eps schedule, hyperparameters).
- **Matched seeds:** cond-seeds drawn from the master RNG in fixed (rho x sigma)
  order, independent of mode -> IQL (GPU0) and VDN (GPU1) saw identical preference
  draws & inits, so the halves are mergeable and any mode difference is the
  algorithm alone.
- **Stats:** all test statistics extracted via the **named scipy attributes**
  (`.statistic` / `.pvalue`, plus `linregress.slope/.rvalue/.pvalue`,
  `f_oneway`) -- no positional tuple unpacking (the smell flagged in the prior
  `adversarial_rerun/analyze.py` is not inherited). Welch 95% CIs use the
  Welch-Satterthwaite dof.
- **DGP gate (`verify_corr.py`, `dgp_corr_verification.csv`, 20k draws):**
  n=2 realised pooled corr tracks target to |err| <= 0.002 and **reaches
  rho=-1.0000 exactly** at the boundary; n=4 (separable's DGP) tracks identically.
  GATE PASSED.

---

## EXPERIMENT 1 -- SEPARABLE-RESOURCE CONTROL (Critical #1)

**Question.** Is the gap-tracks-rho law a property of preference correlation, or
just SHARED-STATE CONTENTION (n agents fighting over one common pool)?

**Design.** A SEPARABLE variant of the env (`separable_runner.py`,
`run_condition_separable`): each agent acts on its OWN independent resource pool
(its delta updates only its own state -- NO sum into a shared state), and is
rewarded by its OWN pool vs its OWN target. The ONLY changes from the shared-pool
main env are 4 marked env-dynamics lines; network/optimizer/hyperparameters/eps
schedule/replay/seeds/IQL-VDN losses/probe are imported verbatim from
`b_vdn_runner`. SAME signed equicorr DGP, SAME rho grid as the shared run
(rho in {-0.33,-0.20,-0.10,0,0.20,0.40,0.60,0.80}, sigma in {0.6,1.0}, 30 reps,
1000 ep). Overlaid against the shared-pool `adversarial_rerun/equicorr_{iql,vdn}.csv`.

**Result -- normalized gap dR/sigma vs rho (pooled over sigma):**

| arch/mode | -0.33 | -0.20 | -0.10 | 0.00 | +0.20 | +0.40 | +0.60 | +0.80 |
|---|---|---|---|---|---|---|---|---|
| shared / IQL  | 3.86 | 4.07 | 4.08 | 4.02 | 3.65 | 3.39 | 3.34 | **3.31** |
| shared / VDN  | 3.20 | 3.36 | 3.38 | 3.35 | 3.02 | 2.76 | 2.69 | **2.43** |
| separable/IQL | 3.87 | 3.97 | 3.94 | 4.09 | 3.52 | 3.97 | 3.85 | 3.96 |
| separable/VDN | 4.05 | 4.13 | 4.09 | 4.22 | 3.66 | 4.15 | 3.97 | 4.11 |

**Tests (normalized gap):**

| mode | arch | slope(gap~rho) | slope p | R^2 | ANOVA F (k=8) | ANOVA p | coop(+0.8)-indiff | p | opp(-0.33)-indiff | p |
|---|---|---|---|---|---|---|---|---|---|---|
| IQL | shared    | **-0.755** | 1.7e-4 | 0.029 | 2.50 | 1.6e-2 | **-0.707** | 0.048 | -0.154 | 0.60 |
| IQL | separable | -0.029 | 0.87 | 0.000 | 0.90 | 0.51 | -0.131 | 0.67 | -0.224 | 0.24 |
| VDN | shared    | **-0.838** | 7.3e-7 | 0.050 | 4.16 | 1.9e-4 | **-0.924** | 0.0018 | -0.159 | 0.54 |
| VDN | separable | -0.053 | 0.76 | 0.000 | 0.89 | 0.52 | -0.117 | 0.71 | -0.173 | 0.38 |

- Shared-vs-separable **gap-vs-rho shape correlation = +0.22 (IQL) / +0.15 (VDN)**
  -- the curves are essentially unrelated; the monotone alignment ramp is gone.
- Separable raw gap is flat over rho at BOTH sigmas (sigma=0.6: ~2.2-2.6;
  sigma=1.0: ~3.5-4.0) and scales mechanically with sigma -> the flatness is not a
  normalization artifact.
- Note the separable gap sits at ~the shared *indifference* level (IQL ~3.9 vs
  shared rho=0 = 4.0); the shared run's cooperation dividend (the drop below that
  baseline) is precisely what vanishes once the pool is not shared.

**VERDICT (Exp1): the signed-correlation law is driven by SHARED-STATE CONTENTION.**
In separable resources cooperation no longer lowers the gap (and opposition no
longer raises it): the curve is flat. **The paper's central claim must be bounded
to common-pool / shared-state friction** -- when agents are not physically forced
to reconcile their targets in one shared state, preference correlation has no
detectable effect on coordination. Plot: `exp1_shared_vs_separable.png`.
(Mechanistically intuitive: with independent pools each agent steers its own state
to its own target regardless of what the others want; rho cannot bind.)

---

## EXPERIMENT 2 -- n=2 CLEAN STRONG-OPPOSITION (Critical #2 + Major #4)

**Question.** The n=4 team DGP confounds strong between-team OPPOSITION with strong
within-team COOPERATION, and equicorrelated n=4 is PSD-capped at rho=-1/3 -- so the
paper never had a clean strong-adversarial point. With n=2 there is exactly ONE
pair, so equicorrelated rho reaches FULL -1 with NO confound. At genuine rho=-1,
does opposition BEAT, TIE, or LOSE to indifference?

**Design.** `n2_runner.py`: the SAME shared-pool main env (`b_vdn_runner.run_condition`,
UNCHANGED), monkeypatching only N_AGENTS=2 and the signed equicorr sampler.
rho in {-1.0,-0.6,-0.2,0,0.4,0.8}, sigma in {0.6,1.0}, IQL+VDN, 30 reps, 1000 ep.
Realised pairwise corr verified (rho=-1 -> -1.0000).

**Result -- normalized gap dR/sigma vs rho (pooled over sigma):**

| mode | -1.00 | -0.60 | -0.20 | 0.00 | +0.40 | +0.80 |
|---|---|---|---|---|---|---|
| IQL | 5.31 | **5.39** | 4.93 | 4.04 | 4.04 | **2.96** |
| VDN | 3.62 | **3.97** | 3.81 | 3.28 | 3.47 | **2.60** |

**Headline contrasts (gap_norm, n=60 each = 30 reps x 2 sigma):**

| mode | opp(rho=-1) | indiff(rho=0) | coop(rho=+0.8) | OPP-INDIFF [95% CI] | Welch t, p | MWU p | COOP-INDIFF, p |
|---|---|---|---|---|---|---|---|
| IQL | 5.313 (sd 4.27) | 4.043 (sd 2.08) | 2.960 | **+1.270** [+0.05, +2.49] | +2.07, **0.041** | 0.24 | -1.083, **0.0045** |
| VDN | 3.624 (sd 2.50) | 3.279 (sd 1.60) | 2.602 | +0.345 [-0.42, +1.11] | +0.90, 0.37 | 0.85 | -0.677, **0.037** |

**VERDICT (Exp2): at genuine unconfounded rho=-1, opposition does NOT beat
indifference.**
- **VDN: opposition TIES indifference** (diff +0.35, Welch p=0.37, MWU p=0.85).
- **IQL: opposition is directionally WORSE, but the significance is FRAGILE** --
  the Welch p=0.041 is *pooled-only*; the rank-based MWU is n.s. (p=0.24), and
  neither sigma alone is significant (sigma=0.6 p=0.24, sigma=1.0 p=0.09). The
  opposition arm has a heavy right tail (sd 4.27 vs 2.08 at indifference): some
  replications coordinate badly under strong opposition, inflating the mean more
  than the median. Honest read: **opposition <= indifference, never >.**
- **Cooperation robustly beats indifference** for both learners (IQL p=0.0045,
  VDN p=0.037).
- Minor wrinkle: the single worst point is rho=-0.6 (not -1) for both learners --
  the opposition arm plateaus/rises rather than peaking exactly at -1, so it is
  "monotone-ish, opposition-region elevated" rather than strictly monotone. The
  direction (opposition region > indifference > cooperation) is unambiguous.

This is the clean strong-adversarial test the n=4 design could not give, and it
confirms the prior honest conclusion: **only alignment helps; genuine opposition
(even at rho=-1) is at best tied with, and for independent learners weakly worse
than, indifference.** Plot: `exp2_n2_gap_vs_rho.png`.

---

## Limitations / honest caveats

1. **Exp1 bounds the claim, by design.** The flat separable result is the correct,
   intended outcome of the control: it tells us the friction is shared-state
   contention. The paper should state the signed-correlation law as a property of
   *common-pool* coordination, not of preference correlation in the abstract.
2. **Exp2 IQL significance is fragile.** The "opposition worse" effect for IQL is
   Welch-pooled-significant only (p=0.041), MWU-n.s. and per-sigma-n.s., driven by
   a heavy-tailed opposition distribution. We therefore claim only "opposition does
   not beat indifference" (robust) -- not "opposition is significantly worse."
3. **Separable absolute level is set by a clamping floor.** With pool states in
   [0,capacity] and targets ~ N(0,1) (so ~half are negative and unreachable), the
   separable gap has an irreducible floor; this floor is rho-invariant (marginal
   target variance is held at 1 for every rho), which is exactly why the curve is
   flat. The flatness is about the SHAPE over rho, not the level.
4. **n=2 vs n=4 absolute gaps are not directly comparable** (different agent
   counts / pool pressure); Exp2 contrasts are all WITHIN the n=2 sweep.
5. **sigma resolution** = 2 levels (as in the prior runs); not the focus.
6. **Single env family** (ResourceAllocationEnv and its separable twin). The two
   controls probe the two structural axes the reviewer named (shared-vs-separable
   state, confound-free strong opposition); other env families are future work.

---

## Files

| File | Contents |
|---|---|
| `separable_runner.py` | Exp1 runner -- separable-resource env (reuses b_vdn_runner machinery; 4 marked env-dynamics changes) |
| `n2_runner.py` | Exp2 runner -- shared-pool main env, monkeypatched to N_AGENTS=2 + signed equicorr |
| `verify_corr.py` / `dgp_corr_verification.csv` | DGP gate: realised corr vs target (n=2 reaches -1.0000) |
| `separable_{iql,vdn}.csv` | Exp1 raw per-replication results (30 reps x 16 conds each) |
| `n2_{iql,vdn}.csv` | Exp2 raw per-replication results (30 reps x 12 conds each) |
| `analyze_controls.py` | Clean analysis (named-attribute scipy stats; Welch CIs) |
| `ANALYSIS.txt` | Full printed analysis log |
| `exp1_separable_headline.csv` / `exp1_shared_vs_separable.png` | Exp1 verdict table + overlay plot |
| `exp2_n2_headline.csv` / `exp2_n2_gap_vs_rho.png` | Exp2 verdict table + gap-vs-rho plot |

**Reproduce:**
`python results/reviewer3_controls/verify_corr.py` ->
`separable_runner.py --modes iql --device 0 ...` + `--modes vdn --device 1 ...`
(matched seeds) -> `n2_runner.py --modes iql --device 0 ...` + `--modes vdn --device 1 ...`
-> `python results/reviewer3_controls/analyze_controls.py`.
