# Adversarial Re-run: Does Structured Opposition Beat Indifference?

**Date:** 2026-06-27
**Scope:** Redesign the friction-MARL preference DGP so the cross-agent correlation
genuinely spans cooperative -> neutral -> adversarial, re-run IQL + VDN on a targeted
GPU-accelerated design, and test whether genuine opposition coordinates better than
indifference (a real U-shape) or not.
**Outputs:** `results/adversarial_rerun/` (this dir). The prior paper-backing run in
`results/review_upgrade/` was **not modified**.

---

## VERDICT (headline)

**No. Structured opposition does NOT beat indifference.**

- With a genuinely *signed* preference correlation, the reward gap is **monotone in rho**:
  cooperation (rho->+1) is best, and the gap rises as rho falls. Only **alignment** beats
  indifference.
- Equicorrelated opposition (capped at rho = -1/3 for n=4) is **statistically tied with
  indifference** -- there is no symmetric "adversarial arm." (IQL: neutral-adversarial
  = +0.15, p~0.60; VDN: +0.16, p~0.54.)
- **Strong** opposition (two-team DGP, between-team corr down to -1.0) is **worse**, not
  better, than indifference -- monotone-worsening for IQL (gap 3.43 -> 4.54 as opposition
  strength 0->1; trend slope +0.98, p_trend = 0.048; strongest opposition is the worst
  case), flat-but-elevated for VDN (indifference remains the lowest gap).
- The previously reported **|alpha|-symmetric U-shape ("structure of either sign helps")
  was an artifact** of the buggy shared-base DGP, in which alpha<0 was in fact *positively*
  correlated (~+0.94, identical to alpha>0). Once the sign is real, the symmetry collapses:
  the signed-linear term explains **99% (IQL) / 95% (VDN)** of the rho-variance (old design:
  0% was sign). VDN preserves this shape (curve corr 0.976) and uniformly lowers the gap.

This **vindicates the honest reading** of the paper: the only thing that genuinely helps
coordination is preference *alignment*; "opposition coordinates well" was an
inference-of-record bug, not a real phenomenon.

---

## 1. The DGP bug and the fix (verified)

### The bug
Original `factorial_design.sample_agent_params`:

    tau_i = alpha*b + (1-|alpha|)*xi_i        (b shared across all agents)

For n agents the population cross-agent correlation is

    rho(alpha) = alpha^2 / (alpha^2 + (1-|alpha|)^2)

-- a function of **|alpha| only**, always >= 0. At alpha = +-0.8, rho ~ **0.94 for both
signs**. So alpha<0 was never adversarial: every agent shared the same base -b and stayed
strongly *positively* correlated. The only unstructured point was alpha=0 (rho=0), which
is why it looked "worst" -- a |alpha|-symmetry artifact, not a U over genuine alignment.

Empirically confirmed (`verify_dgp.py`, `dgp_verification.csv`):

| alpha (original) | realised corr | theory alpha^2/(alpha^2+(1-|alpha|)^2) |
|---|---|---|
| -0.80 | **+0.941** | 0.941 |
| -0.40 | +0.307 | 0.308 |
|  0.00 | -0.002 | 0.000 |
| +0.40 | +0.304 | 0.308 |
| +0.80 | +0.941 | 0.941 |

alpha = -0.8 and alpha = +0.8 are *identically* positively correlated. The "adversarial"
arm did not exist.

### The fix -- two principled designs
The control knob is now the cross-agent correlation **rho directly**, and it is genuinely
signed. (Hard constraint: an equicorrelated matrix for n=4 is PSD only for
rho >= -1/(n-1) = **-1/3**.)

**(A) Equicorrelated continuum [primary].** Per resource, draw the n-agent target vector
from N(0, Sigma(rho)), Sigma(rho) = (1-rho)I + rho*11^T, via an eigh-based PSD square root
(stable at the -1/3 boundary). Marginal target variance is held at 1 for every rho (the
original's target variance drifted with alpha), so reward magnitudes are comparable across
the sweep and to the original rho=0 reference. Clean signed continuum; adversarial capped
at -1/3.

**(B) Two-team block [robustness, strong opposition].** 4 agents -> 2 teams of 2; team A
shares +base, team B shares -base, strength `opp` in [0,1]. Within-team corr = +opp^2,
between-team corr = **-opp^2** (reaches -1.0 at opp=1 -> genuine strong opposition, beyond
the equicorrelated bound). Mean pairwise = -opp^2/3.

In both, weights `w ~ N(sigma, 0.05)` clipped to [0,1] are **unchanged** -- sigma keeps its
"preference intensity / stakes" meaning. The IQL/VDN runners are reused verbatim; only
the sampler is swapped (monkeypatch).

### Verified realised correlation range (the gate)
`verify_dgp.py` (20,000 draws) -- the curve genuinely goes negative:

| rho_target (equicorr) | -0.33 | -0.20 | -0.10 | 0.00 | 0.20 | 0.40 | 0.60 | 0.80 |
|---|---|---|---|---|---|---|---|---|
| realised pooled corr | **-0.330** | **-0.200** | **-0.100** | -0.000 | 0.200 | 0.400 | 0.600 | 0.800 |

Equicorr tracks the target essentially exactly. Two-team: between-team corr 0 -> **-1.00**
at opp=1 (within-team 0 -> +1.00, mean pairwise 0 -> -1/3). **GATE PASSED**
(spans negative -> 0 -> positive). Plot: `dgp_verification.png`.

---

## 2. GPU vs CPU timing (measured, not assumed)

Nets are tiny (3->64->64->27) but the runner batches **all (rep x agent) nets + the whole
env loop** into single tensor ops, so the GPU still wins. Benchmark (`bench.py`, 30 reps x
100 ep, 3 conditions, extrapolated to the full 1000-ep run):

| Device | per-cond (100 ep) | full-run est (1000 ep) | speedup vs CPU |
|---|---|---|---|
| **GPU0 -- RX 7900 XTX** | **6.5 s** | **~65 s** | **2.65x** |
| GPU1 -- RX 7800 XT | 8.3 s | ~83 s | 2.09x |
| CPU -- 9900X (1 proc, torch threads) | 17.4 s | ~174 s | 1.0x |

ROCm 6.3 / torch 2.9.1; 3 HIP devices visible (0,1 = GPUs; 2 = CPU-as-HIP). **Fastest
path: the two GPUs in parallel.** Because cond-seeds and the probe are drawn in a fixed
order depending only on (rho x sigma, seed) -- not on mode -- IQL on GPU0 and VDN on GPU1
use matched preference draws, so the halves share the intended condition schedule and are
mergeable. The archived runs predate deterministic PyTorch seeding, so network initializations
are not claimed to be identical. Full equicorr run (32 conds x 30 reps x 1000 ep) wall-clock ~ **22 min**
across both GPUs (~70 s/cond IQL, ~80 s/cond VDN).

---

## 3. Targeted design

| | levels | values |
|---|---|---|
| rho (equicorr) | 8 | -0.33, -0.20, -0.10, 0.00, 0.20, 0.40, 0.60, 0.80 |
| opp (team) | 6 | 0.0, 0.3, 0.5, 0.7, 0.85, 1.0  (between-corr 0 -> -1.0) |
| sigma | 2 / 1 | equicorr {0.6, 1.0}; team {1.0} |
| modes | 2 | IQL, VDN (matched seeds) |
| reps x episodes | -- | 30 x 1000 (mirrors the prior review_upgrade run) |

epsilon (obs noise) = 0 throughout. The post-fix clean-evaluation analysis detects a
small first-stage epsilon penalty and bounds its pooled full-range contrast to 21.5%
of the stakes anchor. The later feasible-centred signed crossing in
`results/reviewer3_followup/` finds a larger positive mean penalty and a positive
`rho x epsilon` interaction, opposite the canonical ratio; that follow-up supersedes any attempt
to transport the first-stage bound across support or preference mixtures. Entropy remains fixed
at zero in the present archived signed run itself.

---

## 4. Results

### 4A. Equicorrelated continuum (primary) -- `gap_vs_rho_equicorr.png`

Normalized gap dR/sigma, mean over reps (pooled over sigma):

| rho | -0.33 | -0.20 | -0.10 | 0.00 | +0.20 | +0.40 | +0.60 | +0.80 |
|---|---|---|---|---|---|---|---|---|
| **IQL** | 3.86 | 4.07 | 4.08 | 4.02 | 3.65 | 3.39 | 3.34 | **3.31** |
| **VDN** | 3.20 | 3.36 | 3.38 | 3.36 | 3.02 | 2.76 | 2.69 | **2.43** |

- **Cooperative (rho=+0.8) is best** for both learners (lowest gap).
- The curve is a **ramp / half-U**: a high plateau for rho <= 0, declining monotonically
  for rho > 0. Worst point rho = **-0.10** for both (the indifference / mild-opposition
  plateau is statistically flat).
- **Opposition does not beat indifference.** Neutral (rho=0) vs most-adversarial (rho=-0.33):
  IQL +0.15 (Welch p=0.60, MWU p=0.50); VDN +0.16 (p=0.54, MWU p=0.46). Tied.
- **Alignment does beat indifference.** Neutral vs cooperative (rho=+0.8): IQL -0.71
  (p=0.048); VDN -0.92 (p=0.0018).

### 4B. Two-team strong opposition (robustness) -- `gap_vs_opposition_team.png`

Raw gap (sigma=1.0), between-team corr = -opp^2:

| opp | 0.00 | 0.30 | 0.50 | 0.70 | 0.85 | 1.00 |
|---|---|---|---|---|---|---|
| between-corr | 0.00 | -0.09 | -0.25 | -0.49 | -0.72 | **-1.00** |
| **IQL** gap | 3.43 | 3.94 | 3.86 | 3.85 | 4.38 | **4.54** |
| **VDN** gap | 2.84 | 3.37 | 3.14 | 3.01 | 3.22 | 3.03 |

- **IQL: opposition is monotonically worst.** gap rises with opposition strength (trend
  slope +0.98, p_trend = 0.048); strongest opposition (between-corr -1.0) is the worst
  case (4.54 vs indifference 3.43, delta +1.11; opp=0-vs-1 Welch p=0.08 -- marginal under
  high sigma=1 variance, but the full-sweep trend is significant and the direction
  unambiguous).
- **VDN: indifference is best** (opp=0 -> 2.84, the minimum); opposition lifts the gap to
  ~3.0-3.4 but with no significant trend (slope +0.10, p=0.78) -- centralized team-reward
  training largely absorbs the conflict but never makes opposition *better* than
  indifference.

Strong opposition therefore **does not** recover a U-shape; it confirms opposition <=
indifference, and (for independent learners) strictly worse.

### 4C. eta^2 and the sign question

Replication-level eta^2 (gap ~ C(rho)*C(sigma)):

| effect | IQL raw | IQL norm | VDN raw | VDN norm |
|---|---|---|---|---|
| C(rho) | 0.029 | 0.036 | 0.047 | 0.058 |
| C(sigma) | 0.235 | 0.000 | 0.225 | 0.000 |
| C(rho):C(sigma) | 0.015 | 0.014 | 0.017 | 0.014 |

- **sigma dominates the RAW gap (eta^2~0.22-0.24) but essentially collapses after sigma-normalization (eta^2=0.0002-0.0004)**
  -- the "stakes dominate" effect is mechanical (degree-1 weight scaling), exactly as the
  prior review_upgrade analysis found. The structural (learned) effect of preference
  correlation, rho, is real but modest (eta^2 ~ 0.04-0.06 normalized).
- **Sign now matters.** On normalized gap: signed-linear(rho) R^2 = 0.029 (IQL) / 0.050
  (VDN) vs quadratic(rho+rho^2) R^2 = 0.029 / 0.053 -- the **signed-linear term explains
  99% / 95%** of the rho-variance, with a **negative slope (-0.76 / -0.84)**: more
  correlation -> smaller gap, monotone. This is the exact opposite of the original design,
  where 0% of the alpha-variance was sign and the relationship was |alpha|-symmetric. The
  DGP fix flips the structure from symmetric to genuinely signed.
- Direct sign check at |rho|=0.2: gap_norm(-0.2)=4.07 vs gap_norm(+0.2)=3.65 (IQL),
  3.36 vs 3.02 (VDN) -- cooperation beats opposition at equal magnitude (directional;
  single-pair t~1.5, p~0.14, with the full-sweep regression carrying the significance).

### 4D. VDN

- **VDN preserves the IQL shape** -- gap-vs-rho curve correlation = **0.976**.
- VDN **uniformly lowers** the gap vs IQL at every rho by ~0.63-0.88 (all p < 0.05); it
  does *not* selectively rescue the neutral/adversarial region and does not manufacture a
  U. Centralized value decomposition helps coordination everywhere by a roughly constant
  amount, leaving the monotone alignment-helps shape intact.

---

## 5. Limitations / honest caveats

1. **Equicorrelated adversarial is capped at rho = -1/3** (PSD bound for n=4). Strong
   opposition is only reachable via the block/team structure, which is *not*
   equicorrelated. Both designs agree (opposition <= indifference), but the cleanest
   continuous arm cannot go past -1/3 by construction.
2. **Single environment** -- a shared common-pool resource (`ResourceAllocationEnv`: all
   agents' +-1 actions sum into one shared state). "Opposition hurts" is intuitive here
   because anti-correlated targets make agents physically fight over the same pool. A
   *separable* environment (independent resource pools per agent) could behave
   differently; the present result is specific to shared-state friction.
3. **sigma resolution.** 2 sigma levels (equicorr) / 1 (team) -- sigma was not the focus
   and eta^2(sigma) is not finely resolved, but the normalization argument (sigma effect
   is mechanical) reproduces the prior finding and is robust to that.
4. **Modest effect size + variance.** eta^2(rho) ~ 0.04-0.06; per-rep gap variance at
   sigma=1 is large, so the single-pair tests (e.g. IQL opp=0 vs opp=1, p=0.08) are
   marginal. The *direction* is consistent across both DGPs, both learners, and the
   full-sweep trend tests (p_trend=0.048) -- but tighter CIs would need more reps.
5. **Compute.** Runs used both AMD GPUs under ROCm 6.3; py3.13. No instability observed.

---

## 6. Files

| File | Contents |
|---|---|
| `dgp.py` | Redesigned signed samplers (equicorr + team) + correlation verifier |
| `verify_dgp.py` / `dgp_verification.{csv,png,txt}` | The verified correlation gate |
| `bench.py` | GPU-vs-CPU timing |
| `adv_runner.py` | Targeted runner (reuses `b_vdn_runner`; rho/opp in place of alpha) |
| `equicorr_{iql,vdn}.csv`, `team_{iql,vdn}.csv` | Raw per-replication results |
| `analyze.py` / `ANALYSIS_equicorr.txt` / `gap_vs_rho_equicorr.{png,pdf}` | Primary analysis |
| `analyze_team.py` / `ANALYSIS_team.txt` / `gap_vs_opposition_team.png` | Strong-opposition robustness |
| `headline_{equicorr,team}.csv`, `centralization_equicorr.csv` | Verdict tables |

**Reproduce:** `python results/adversarial_rerun/verify_dgp.py` -> `bench.py` ->
`adv_runner.py --dgp equicorr ...` (x2, devices 0/1) -> `analyze.py --csvs equicorr_iql.csv
equicorr_vdn.csv --tag equicorr`; then `--dgp team` -> `analyze_team.py`.
