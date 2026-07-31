# The exact delegation-gap surface G(σ, α, ε) on the friction-MARL environment

**Numericist's report.** 30 July 2026. Independent of the analyst; this surface is
the ground truth their closed form gets tested against.

Everything here is computed by code in this directory, seeded, and saved. Nothing
is quoted from any prior run, memo or paper. Every environment fact was read back
out of source and asserted (`validate_lattice.py::v1_constants`) before use.

| file | what |
|---|---|
| `core.py` | primitives: lattice, planner, naive-myopic (exact), stage-Nash (MC), chains |
| `validate_lattice.py` → `VALIDATION.json` | V1–V8, all pass; nothing downstream is trusted without them |
| `run_grid.py` → `G_grid_*_{cesaro,T100}.csv`, `diagnostics_*.csv` | the G surface |
| `channels.py` → `CHANNELS.csv` | variance / bias / floor decomposition |
| `floor_closed_form.py` → `FLOOR_CLOSED_FORM.csv` | the quantization floor, closed form + validation |
| `noise_channel.py` → `NOISE_CHANNEL.csv` | the noise-response variance channel |
| `planner_transient.py` → `PLANNER_TRANSIENT.csv` | exact T=100 planner benchmark |
| `nash_diagnostics.py` → `NASH_NE_CENSUS.csv`, `NASH_BELIEF_WEDGE.csv` | equilibrium existence/uniqueness/selection |
| `expansion.py` → `P1/P2/P3_*.csv`, `EXPANSION.json` | the local expansion about (α,ε)=(0,0) |
| `fit_forms.py` → `FORM_FITS.csv` | candidate-form R²/AICc |
| `eps_cusp.py`, `nash_eps0plus.py`, `nash_selection_sensitivity.py` | the ε→0+ probes and the equilibrium-selection check |
| `independent_crosscheck.py` → `CROSSCHECK.json` | **second implementation** of the headline claims, written from scratch without importing `core.py` (§5) |
| `crosscheck_planner_incidence.py` → `CROSSCHECK_PLANNER_INCIDENCE.json` | planner magnitudes vs the spec; the stakes-weighted incidence object |
| `crosscheck_spec_seeds.py` → `CROSSCHECK_SPEC_SEEDS.json` | replay of `run_gpu.py`'s seed-42 condition draws (§0.3) |

(`analytic_checks/` in this directory is the *analyst's* scratch, written by a
different agent. Nothing in this report comes from it. Scripts `a01`–`a09` have
outputs on disk; `a10_crosscheck.py` has none and was never run.)

---

## 0. Environment, verified — and one correction to the gate's premise

Read back from source and asserted, not assumed (V1):

* `resource_allocation.py:57-60` reset ~ Uniform[0, C]³, C = 10.0; `:84-86` step is
  `clip(s + Σᵢ δᵢ, 0, C)`; `:17` episode length 100.
* `factorial_design.py:55-60` δᵢ ∈ {−1,0,+1}³; `:63-69` rᵢ = Σₘ wᵢₘ·(−(s′ₘ−τᵢₘ)²);
  `:50-51` w ~ N(σ, 0.05) clipped to [0,1]; `:42-48` τᵢ = αb + (1−|α|)ξᵢ.
* `run_gpu.py:250-256` obsᵢ = s + N(0, εI) (std √ε); `:279-281` rewards use the TRUE
  state and TRUE (w, τ). n = 4, m = 3.

### 0.1 The "1331 states × 729 actions" claim (`ideation_measurement.md` §0.1)

**Correct for integer start states; wrong for the environment as implemented — and
the correction makes the computation *bigger*, not approximate.**

Let f = frac(x₀) for one coordinate. The exact closure of the coordinate dynamics is

```
L(f) = {f, f+1, …, f+9}  ∪  {0, 1, …, 10}          → 21 points (11 if f = 0)
```

The integer sublattice is entered *only* through the clamp (from x ≤ 4 a −4 move
lands exactly on 0; from x ≥ 6 a +4 move lands exactly on 10), and once entered it
is closed. Verified three ways in `validate_lattice.py::v3_lattice`: by
`lattice_transition_index` (which raises if any successor is off-lattice), over 200
random offsets, and by brute-force forward exploration from a random start
(|reachable| = 21, exactly).

So the exact counterfactual MDP is **21³ = 9261 states × 9³ = 729 aggregate actions**
almost surely, and 1331 × 729 only on the measure-zero event f = 0. **There is no
discretisation error**: the lattice is the exact reachable set, not a grid
approximation of a continuum. The state is continuous, but the *reachable* state
set from any given reset is finite and this is it.

### 0.2 Exact backward induction, and separability

`validate_lattice.py::v4_v5` runs undiscounted finite-horizon (T = 100) backward
induction on the FULL 3-D lattice (9261 × 729 per step) and on the per-resource
problems separately. They agree to **3.98 × 10⁻¹³**. The separability (fact S1:
additive reward, componentwise transition, product action set, isotropic noise)
therefore holds exactly, and all grid work is done per resource, which is what makes
1000 preference draws × 168 cells × two delegation concepts affordable.

The planner's long-run value equals its static best lattice point (V5): from any
start it reaches the optimum in ≤ 3 of 100 steps and stays. The *transient* is not
free, though — `PLANNER_TRANSIENT.csv` puts it at **0.535 welfare units per step
averaged over the grid** (exact T = 100 backward induction from the actual x₀ vs the
static optimum). That is why the Cesàro design is primary here and the T = 100
tables are secondary.

### 0.3 ⚠️ The spec's "verified magnitudes" for V\*_p do not reproduce, and one of them is above the achievable ceiling

`ideation_measurement.md` §0.1 tabulates three planner values as *verified*, on
"seed-42-reproducible condition seeds". I replayed that seed construction exactly
(`crosscheck_spec_seeds.py`: `run_gpu.py:456-460` master RNG → 128 probe states →
125 condition seeds in `generate_conditions()` order, then
`factorial_design.sample_agent_params` called 30× on the condition RNG — verified
to be the *only* consumer of that RNG, and `run_gpu.py:31` imports the same
sampler). On those exact draws:

| α | σ | spec V\*_p | exact static optimum, per agent | **ceiling** = −Σₘ Vₘ/n |
|---|---|---|---|---|
| −0.8 | 0.6 | −0.767 | −0.598 | −0.046 |
| 0.0 | 0.6 | **−1.014** | **−1.488** | **−1.266** |
| 0.8 | 1.0 | −1.158 | −0.883 | −0.096 |

The **ceiling** column is the supremum of per-agent per-step welfare over *all*
states, continuous or not: W(s) ≤ −Σₘ Vₘ for every s, because Vₘ is the weighted
variance of the targets about their own weighted mean. No policy, planner or
otherwise, can beat it.

**The α = 0 entry violates it.** The spec claims −1.014 where the best attainable
value on those draws is −1.266 — it asserts a planner welfare 0.25 units/agent
better than the theoretical maximum. This is not a sampling accident of my replay:
under the DGP the ceiling over 30-replication blocks is **−1.349 ± 0.116**, and
P(ceiling > −1.014) = **0.001**. The other two entries are off by +28% and +31%
in the opposite direction (the exact optimum is *better* than the spec claims),
so there is no single units convention — per-agent vs ensemble is a factor of 4,
and no factor reconciles all three rows.

**Consequence.** The §0.1 magnitudes table should be withdrawn, and with it §0.2's
"49.9% variance reduction" claim, which is computed from those V\* values. **The
qualitative claim §0.1 exists to support — that the benchmark is exactly
computable rather than approximable — is nevertheless correct**, and §0.2's
*mechanism* (subtracting a benchmark built from the same preference draw removes
the parameter-draw variance component) is sound; only the arithmetic is wrong. My
own planner numbers throughout this report are computed independently and are not
affected.

---

## 1. Deliverable 1 — E[G] as an explicit function of (σ, α, ε)

### 1.1 The exact reduction (this is the closed form, and it is exact)

With Wbarₘ = Σᵢ wᵢₘ, s*ₘ = Σᵢ wᵢₘτᵢₘ / Wbarₘ, and Vₘ = Σᵢ wᵢₘ(τᵢₘ − s*ₘ)²,

```
W(s) = − Σₘ [ Wbarₘ (sₘ − s*ₘ)²  +  Vₘ ]
```

Vₘ is common to the benchmark and to the realisation, so **it cancels identically in
G = W* − W**. Therefore

```
        G  =  Σₘ  Wbarₘ · [  E_π (xₘ − s*ₘ)²   −   min_{x ∈ L(fₘ)} (x − s*ₘ)²  ]
                            └── delegated tracking ──┘  └── QUANTIZATION FLOOR ──┘
```

G is a **pure tracking-error difference**. Three consequences, all exact:

1. **G ≥ 0 identically — as a theorem, not an empirical finding.** The delegated
   behaviour is a feasible policy for the principal, and the principal's benchmark is
   the optimum over a superset, so W(realised) ≤ W*(planner) always. Confirmed
   numerically: over all 168 cells × 1000 draws × both belief models,
   `B_mean = max(−G,0)` is **0 to machine precision** (max 8.3 × 10⁻¹⁸).
   **The "delegation surplus" B can only be non-zero at the ENSEMBLE level if the
   principal benchmark is boundedly rational — it is not a property of the
   environment.** Per *principal*, however, B is common: Gᵢ < 0 for at least one
   principal in up to **63.5%** of draws under naive delegation and **90.2%** under stage-Nash
   (`P_any_Gi_neg`), because the ensemble-optimal lattice point is not principal
   i's optimum.
2. **Exact degree-1 homogeneity in σ, up to the sampler's own dispersion.** The
   delegated policy is *exactly independent of w* (fact S4: agent i maximises
   −wᵢₘ(post − τᵢₘ)² with wᵢₘ > 0, and a positive scalar cannot change an argmax), so
   σ enters G only through Wbarₘ (multiplicative) and s*ₘ (a weighted mean). With
   w ≡ σ this gives **E[G] = 4σ · E[Δ(τ, f, ε)]** exactly, with no σ–α or σ–ε
   interaction whatsoever. With the implemented sampler w ~ N(σ, 0.05)∧[0,1],
   G(σ=1.0)/G(σ=0.2)/5 = **0.953 – 0.984** across all 56 (family, control, ε) cells:
   the ≈ 2–5% deficit is the clip at 1.0 (E[min(N(1,0.05),1)] = 1 − 0.05/√(2π) =
   0.980) plus the weighting of s*.
3. **The "irreducible σ/2 baseline" cannot survive.** The quantization floor is
   large — mean **1.88 welfare units** over the grid, and up to 6.04 — and it is
   *exactly* the kind of "loss present even in paradise" that the σ/2 term was
   supposed to name. It cancels out of G by construction. The gate's §0.3 argument is
   confirmed by direct computation, not by algebra alone.

### 1.2 The quantization floor, in closed form (`floor_closed_form.py`)

The floor is `Σₘ Wbarₘ · min_{x∈L(fₘ)}(x − s*ₘ)²`. Two regimes:

* **Interior, generic.** L(f) is two interleaved unit grids offset by f ~ U(0,1). s*
  falls in a gap of length f w.p. f and of length 1−f w.p. 1−f; inside a gap of
  length g the distance to the nearer endpoint is U[0, g/2], so E[d²|f] =
  (f³+(1−f)³)/12 and

  ```
  E_f[ d² ]  =  1/24  =  0.0416667
  ```

  Monte-Carlo check (4 × 10⁶ draws): **0.041672**. ✔
* **Boundary.** s* ≤ 0 is absorbed by the lattice point 0 *exactly*, contributing
  s*². Under every DGP here s* is symmetric about 0, so this branch has probability
  ½ and contributes Var(s*)/2. Since τ ~ N(0,1)-scale and the state lives on [0,10],
  **about half of all resource-optima sit at the lower boundary** — a structural
  feature of this environment that the corpus does not mention.

Combining:

```
E[floor] ≈ 3 · E[Wbar_m] · ( Var(s*)/2 + 1/24 ),      Var(s*) closed form:
   implemented    τᵢ = αb + (1−|α|)ξᵢ                Var(s*) = α² + (1−|α|)²/4
   renormalised   ÷ √(α²+(1−|α|)²)                   Var(s*) = (α²+(1−|α|)²/4)/(α²+(1−|α|)²)
   equicorrelated pairwise corr ρ                    Var(s*) = (1+3ρ)/4
```

The theoretical Var(s*) matches the empirical to < 2% at every grid point. The floor
formula is accurate to **4.4 – 16%** wherever Var(s*) ≳ 0.25 (it over-predicts,
because for small positive s* the nearest lattice point is the boundary 0 rather than
a generic interior point). It fails badly at the PSD boundary ρ = −1/3, where
Var(s*) = (1+3ρ)/4 = **0** exactly — the four targets sum to a constant, the ensemble
optimum is deterministic and lands on the lattice point 0, and the measured floor
collapses to 0.03 against a predicted 0.10–0.50. That is the formula behaving
correctly at a degenerate point, and it is worth knowing: **the equicorrelated
adversarial arm is the one place in the whole design where the principal has no
quantization loss at all.**

### 1.3 The noise-response variance channel, derived (`noise_channel.py`)

For the naive-myopic rule the decision is a two-threshold step function of the private
signal (monotonicity in ŝ is *asserted*, not assumed — `naive_thresholds` raises
otherwise), so **exactly**

```
pᵢ⁺(x) = Φ((θ1ᵢ − x)/√ε),   pᵢ⁻(x) = 1 − Φ((θ2ᵢ − x)/√ε),
mD(x) = Σᵢ (pᵢ⁺ − pᵢ⁻),     vD(x) = Σᵢ ( pᵢ⁺ + pᵢ⁻ − (pᵢ⁺ − pᵢ⁻)² )
```

with θ1ᵢ ≈ τᵢ − ½, θ2ᵢ ≈ τᵢ + ½ away from the clamp. Because the signals are
independent across agents, the aggregate-move law is the exact 4-fold convolution and
the whole delegated chain is computed **without any Monte Carlo over the noise**.

The natural next step — linearise the drift and read off an AR(1) stationary
variance, Var ≈ vD(x̄)/(κ(2−κ)) with κ = −mD′(x̄) = Σᵢ[φ((θ1ᵢ−x̄)/√ε)+φ((θ2ᵢ−x̄)/√ε)]/√ε
— **does not work here, and the reason is the finding.** Measured at the chain's own
modal state:

| DGP | ε=0.01 | ε=0.05 | ε=0.25 | ε=0.5 | ε=1.0 |
|---|---|---|---|---|---|
| median κ (implemented α=0) | 4.73 | 3.56 | 2.70 | 2.42 | 2.15 |
| fraction of draws with κ ≥ 2 | 0.78 | 0.83 | 0.82 | 0.77 | 0.66 |

κ ≥ 2 means |1−κ| ≥ 1: the linearised delegated dynamics is **at or past the
flip-bifurcation boundary**. Four independent agents each correcting a unit-width
deadband produce an aggregate feedback gain of 2–5 against a step size of 1. The
delegated failure mode in this environment is therefore **over-correction / herding
overshoot**, not variance injection, and no local reduction of the variance channel
is valid — the exact chain is required. The AR(1) ratio has median 0.07 – 2.7×10¹⁰
with a usable-denominator fraction of only 16 – 41%.

The same table contains the mechanism for the ε result in §2: **noise monotonically
reduces κ** (4.73 → 2.15 as ε goes 0.01 → 1.0). Observation noise de-synchronises the
herd and moves the system back toward stability.

### 1.4 The two channels (`CHANNELS.csv`)

```
G = Σₘ Wbarₘ Var_π(xₘ)                                     ← VARIANCE channel
  + Σₘ Wbarₘ [ (E_π xₘ − s*ₘ)² − (xlatₘ − s*ₘ)² ]           ← BIAS channel
```

Averaged over each DGP family (all σ, all controls):

| family | ε | G | variance ch. | bias ch. | floor (cancels) | noise-induced Δvariance |
|---|---|---|---|---|---|---|
| implemented | 0.00 | 4.62 | 3.78 | 0.84 | 1.61 | 0.000 |
| implemented | 0.25 | 4.24 | 4.00 | 0.25 | 1.61 | 0.214 |
| implemented | 0.50 | 4.40 | 4.16 | 0.24 | 1.61 | 0.384 |
| implemented | 1.00 | 4.94 | 4.54 | 0.40 | 1.61 | 0.758 |
| equicorr | 0.00 | 3.72 | 3.10 | 0.62 | 1.59 | 0.000 |
| equicorr | 1.00 | 4.61 | 4.16 | 0.46 | 1.59 | 1.058 |
| renormalised | 0.00 | 5.18 | 4.37 | 0.81 | 2.38 | 0.000 |
| renormalised | 1.00 | 5.23 | 4.87 | 0.36 | 2.38 | 0.500 |

Two opposing ε-channels, both real:

* the **noise-response variance channel is increasing in ε** (≈ 0.75–1.06 × ε at
  moderate ε; Θ(√ε) as ε → 0, see §2.2);
* the **overshoot/bias channel is sharply decreasing in ε** (0.84 → 0.25 at
  ε = 0.25 for the implemented DGP): noise breaks the deterministic limit cycle.

Their sum is **non-monotone in ε with an interior minimum near ε ≈ 0.25**. At high
|α| the second channel dominates and G *falls* monotonically in ε; at α = 0 the first
dominates beyond ε ≈ 0.25 and G rises. This is the whole ε story, and no
single-signed ∂F/∂ε can represent it.

### 1.5 The G surface (naive-myopic, certainty-equivalent beliefs, Cesàro)

σ = 0.6 slice; full grid in `G_grid_naive_ce_cesaro.csv` (1000 preference draws
per cell, common random numbers across all cells, per-cell SE ≤ 0.41, typically
0.02–0.15).

| DGP / control | ε=0 | ε=0.25 | ε=0.5 | ε=1.0 |
|---|---|---|---|---|
| implemented α=−0.8 | 6.984 | 6.388 | 6.000 | 5.888 |
| implemented α=−0.4 | 3.551 | 3.091 | 3.574 | 4.497 |
| implemented α=0 | 2.243 | 2.431 | 3.050 | 4.129 |
| implemented α=+0.4 | 3.473 | 3.066 | 3.562 | 4.494 |
| implemented α=+0.8 | 7.123 | 6.482 | 6.078 | 5.949 |
| renormalised α=−0.8 | 7.939 | 7.290 | 6.758 | 6.375 |
| renormalised α=0 | 2.243 | 2.431 | 3.050 | 4.129 |
| renormalised α=+0.8 | 8.105 | 7.405 | 6.858 | 6.455 |
| equicorr ρ=−0.33 | 0.826 | 1.448 | 2.305 | 3.680 |
| equicorr ρ=0 | 2.282 | 2.454 | 3.050 | 4.109 |
| equicorr ρ=+0.4 | 4.486 | 4.113 | 4.323 | 4.919 |
| equicorr ρ=+0.8 | 7.440 | 6.343 | 6.028 | 5.952 |

### 1.6 Per-principal incidence

`Σᵢ Gᵢ = G` holds to 1.4 × 10⁻¹⁴ (arithmetic check on every cell). Reported per
cell in the grid CSVs: `G_i_mean` for i = 0..3, `Gi_min_mean`, `Gi_max_mean`,
`P_any_Gi_neg`. Headline: the gap is **materially unequally incident** — over the
grid the best-off principal averages Gᵢ = 0.81 and the worst-off 1.50, a 1.8×
spread from a *symmetric* DGP, and at least one principal strictly *gains* from delegation relative to the
ensemble-optimal plan in a mean 15.9% of draws (max 63.5%) under naive delegation
and a mean 56.0% (max 90.2%) under stage-Nash.

The **stakes-weighted incidence object** Σᵢ sᵢGᵢ/Σᵢ sᵢ (sᵢ = Σₘ wᵢₘ) is now measured
rather than asserted (`crosscheck_planner_incidence.py::x8`, independent
implementation, 3000 draws):

| cell | plain meanᵢ Gᵢ | stakes-weighted | difference | corr(sᵢ, Gᵢ) | CV(sᵢ) |
|---|---|---|---|---|---|
| implemented α=0, σ=0.6, ε=0.25 | 0.6144 | 0.6165 | **+0.35%** | +0.087 | 0.048 |
| implemented α=0.8, σ=0.2, ε=0 | 0.5818 | 0.5908 | **+1.55%** | +0.123 | 0.145 |

The re-weighting is **positive but ≤ 1.6%**. The sign is not an accident: Gᵢ is
linear in wᵢ, so a principal with larger stakes mechanically bears a larger gap,
giving corr(sᵢ, Gᵢ) ≈ +0.09 to +0.12. The magnitude is small only because the
weight sampler's dispersion (0.05) is small relative to its mean — CV(sᵢ) rises
from 0.048 at σ = 0.6 to 0.145 at σ = 0.2, and the effect rises with it. **A stakes
design that varied sᵢ across principals would make this object bite; this DGP does
not, so it is a footnote here and would not be under Form B/C stakes.**

---

## 2. Deliverable 2 — the expansion about (α, ε) = (0, 0) at fixed σ

The canonical object is **F = σ(1+ε)/(1+α)**. At the expansion point its
first-order content is

```
F(σ,0,0) = σ ;   ∂F/∂σ = 1 (degree-1 homogeneous) ;
∂F/∂ε|₀ = σ = F(σ,0,0) ;   ∂F/∂α|₀ = −σ = −F(σ,0,0).
```

So in units of G(σ,0,0) the canonical form predicts **ε-slope = +1 and α-slope =
−1**. Each is tested separately.

### 2.1 σ: degree-1 homogeneity — MATCHES, exactly, and for the wrong reason

`P1_sigma_homogeneity.csv`, 2000 draws, σ from 0.05 to 1.2:

| σ | 0.05 | 0.10 | 0.20 | 0.40 | 0.60 | 0.80 | 1.00 | 1.20 |
|---|---|---|---|---|---|---|---|---|
| G/σ, implemented α=0, ε=0 | 4.756 | 3.970 | **3.763** | **3.713** | **3.704** | **3.702** | 3.627 | 3.084 |
| G/σ, implemented α=0.8, ε=0 | 13.224 | 12.243 | **12.170** | **12.163** | **12.161** | **12.160** | 11.920 | 10.131 |

On the plateau **σ ∈ [0.2, 0.8] the homogeneity is exact to four significant
figures**. It breaks below (the clip at 0 truncates w ~ N(σ,0.05) once σ ≲ 2×0.05)
and above (the clip at 1 saturates Wbar; G(1.2) ≈ G(1.0)). Over the full range
including the clipped tails the log–log slope is 0.91–0.96.

**But this is a units identity, not a friction result.** Fact S4 proves the
delegated policy is exactly independent of w, so σ can only enter G as the
multiplicative reward scale. Any DV in reward units is degree-1 homogeneous in σ
by construction. The canonical form's σ factor is correct and carries no
information about delegation.

### 2.2 ε: NO first-order term exists, and the leading term has the wrong sign

`EPS_CUSP.csv` / `EPS_CUSP_FITS.csv`, 8000 draws, paired, σ = 0.6, ε from 10⁻¹²
to 10⁻¹, thresholds computed once (CE thresholds are ε-free) so every point is
the *same* preference draw evaluated at a different ε.

**(a) There is a knife-edge at ε = 0 exactly.** The lower clamp x = 0 is
simultaneously a lattice point and a decision threshold: for any agent with
τᵢ ∈ (−½, 0) both δ = −1 and δ = 0 land the post-state on 0, the clamped losses
tie *exactly*, and the tie rule decides. Since s*ₘ < 0 for ≈ half of all
resources, the chain spends a lot of time on that boundary. Consequently

| DGP | G at ε = 0 (tie rule T1) | G at ε → 0+ | jump | % |
|---|---|---|---|---|
| implemented α=0 | 2.3269 | 1.8829 | +0.4440 | **+23.6%** |
| implemented α=0.8 | 7.1846 | 7.1543 | +0.0304 | +0.42% |
| equicorr ρ=−0.33 | 0.8445 | 0.6292 | +0.2153 | **+34.2%** |
| equicorr ρ=+0.8 | 7.2098 | 6.8813 | +0.3286 | +4.8% |

E[G](0) ≠ lim_{ε→0+} E[G](ε). This is a property of the environment (a clamp that
coincides with a lattice point), not of the code, and it means **the ε = 0 column
of every table in this report is convention-dependent by up to a third.** Every
ε > 0 number is unaffected. All expansion work below uses the ε → 0+ base.

**(b) The limit is a SQUARE-ROOT CUSP, so no first-order expansion in ε exists.**
For a fixed preference draw every decision probability is 0 or 1 up to
exp(−d²/2ε), d = distance from the state to the nearest threshold, so E[G] is flat
to all orders at 0+ for a.e. draw. Averaged over the DGP, d has a bounded density
with p(0) > 0, and E_τ[Φ(−d/√ε)] ≈ p(0)√ε/√(2π): a Θ(√ε) cusp.

Measured (log–log slope of |E[G](ε) − E[G](0+)| over ε ∈ [10⁻⁸, 10⁻³]):
**0.391, 0.503, 0.428, 0.591** for the four DGPs — mean 0.478 against a √ε
prediction of 0.5 and a linear prediction of 1.0.

The sharpest falsification of a linear term is the difference quotient. If
E[G] had a first-order term c·ε, then ΔG/ε would converge to c. Measured:

| DGP | ΔG/ε at ε=10⁻⁶ | at ε=10⁻⁴ | ratio | √ε prediction | linear prediction |
|---|---|---|---|---|---|
| implemented α=0 | 3936.5 | 401.5 | **9.80** | 10 | 1 |
| implemented α=0.8 | 15561.7 | 1052.6 | **14.8** | 10 | 1 |
| equicorr ρ=−0.33 | 1517.6 | 128.0 | **11.86** | 10 | 1 |
| equicorr ρ=+0.8 | 4681.4 | 924.4 | **5.06** | 10 | 1 |

The quotient diverges as ε → 0. **∂E[G]/∂ε does not exist at ε = 0+; it is +∞.**
The canonical form's finite slope σ is not an approximation to it.

**(c) Beyond the cusp, the ε effect is non-monotone and frequently NEGATIVE.**
`P2_eps_expansion.csv` (σ = 0.6, base ε = 0):

| DGP | ΔG at ε=0.01 | ΔG at ε=0.25 | ΔG at ε=1.0 |
|---|---|---|---|
| implemented α=0 | −0.217 | +0.174 | +1.850 |
| implemented α=0.4 | −0.427 | −0.392 | +1.007 |
| implemented α=0.8 | +0.870 | **−0.668** | **−1.245** |
| equicorr ρ=−0.33 | −0.094 | +0.675 | +2.912 |
| equicorr ρ=0 | −0.267 | +0.123 | +1.791 |
| equicorr ρ=+0.8 | −0.086 | **−1.073** | **−1.447** |

∂G/∂ε changes sign *within* the design: negative at every ε for the aligned
arms, positive at large ε for the unaligned ones. Under the exact-Bayes belief
model it is **negative everywhere**: G falls monotonically in ε in all 12
control × family cells (e.g. implemented α = 0, σ = 0.6: 2.243 → 0.902 → 0.788 →
0.739; equicorr ρ = −0.33: 0.826 → 0.084 → 0.036 → **0.010**, a 99% reduction).
Mechanism, measured in §1.3: noise cuts the aggregate feedback gain κ from ~4.7 to
~2.15 and so de-synchronises the herd that causes the overshoot.

**Verdict on the ε factor: fails on order (no linear term; the true behaviour is
√ε), fails on sign (negative over most of the domain), fails on monotonicity.**

### 2.3 α: the first-order coefficient is exactly 0 under the implemented and
renormalised DGPs, and POSITIVE under a genuinely signed correlation

**(a) E[G](α) = E[G](−α) exactly — a distributional identity, not an estimate.**
τᵢ(α; b, ξ) = αb + (1−|α|)ξᵢ and τᵢ(−α; −b, ξ) = −α(−b) + (1−|α|)ξᵢ are the *same
array*: `EXPANSION.json::P3_alpha_pm_symmetry` reports `max_abs_tau_diff = 0.0`
for α ∈ {0.05, 0.2, 0.4, 0.8} under **both** the implemented and the renormalised
DGP. Since b ~ N(0,I) is symmetric, the law of (τ₁…τ₄) is identical at ±α, hence
so is the law of G. **No signed α effect can exist under either DGP. The α knob is
a |α| knob, and the renormalisation does not fix that** — it fixes the marginal
variance, not the sign-blindness.

**(b) The first-order coefficient is zero, not merely sign-blind.** The
manipulated correlation is ρ(α) = α²/(α² + (1−|α|)²) = α² + O(α³), i.e. *quadratic*
in α. So even the |α|-kink one would expect is absent near 0. Measured
(`P3_alpha_axis.csv`, 2000 draws, σ = 0.6, OLS slope over α ∈ [−0.1, 0.1]):

| DGP | ε = 0 | ε = 0.5 | canonical prediction |
|---|---|---|---|
| implemented | **−0.083** | +0.186 | −G(0) = −2.22 |
| renormalised | **+0.100** | +0.175 | −G(0) = −2.22 |
| equicorrelated (slope in ρ) | **+5.71** | +2.83 | −G(0) = −2.35 |

For the implemented and renormalised DGPs the measured slope is within ±0.19 of
zero against a canonical prediction of −2.22: **a 12–27× magnitude error with no
reliable sign.** The G(α) profile is flat on [−0.1, +0.1] (2.187–2.233, i.e.
within one MC SE of 0.066) and then rises symmetrically on both wings — G at
|α| = 0.8 is **3.3×** its value at α = 0.

**(c) Under the equicorrelated DGP, where ρ IS the pairwise correlation and is
genuinely signed, the slope is large, smooth, and POSITIVE.** G rises
monotonically from 0.793 at ρ = −1/3 (the PSD boundary) to 8.304 at ρ = 0.95, a
10.5× increase. The canonical form asserts ∂F/∂α < 0. **The measured sign is the
opposite, on the only axis in the design where the sign of alignment is even
well-defined.**

The mechanism is §1.3's: four independent agents each correcting a unit deadband
give an aggregate feedback gain κ ≈ 2–5. When their targets agree they all push
the same way and the state overshoots by up to 4 units per step; when their
targets disagree the pushes cancel and the state tracks. **Alignment is what
creates the overshoot.**

⚠️ This sign is a property of the *delegation concept*, not only of the
environment — see §3.2. Under stage-Nash the ordering reverses.

### 2.4 Direct answer to the deliverable-2 question

> Is σ(1+ε)/(1+α) — up to affine reparameterisation — the first-order expansion of
> E[G] about (α, ε) = (0,0) at fixed σ?

**No, and not up to any affine reparameterisation.**

| factor | canonical | exact G | match? |
|---|---|---|---|
| degree-1 σ homogeneity | yes | yes, to 4 s.f. on σ ∈ [0.2, 0.8] | ✅ (a units identity: the delegated policy is provably σ-free) |
| ε slope at 0 | +σ, finite, linear | does not exist; Θ(√ε) cusp, right-derivative +∞ | ❌ order |
| ε sign | strictly + | −0.09 to −0.43 at ε=0.01 for 5 of 6 DGPs; monotonically negative under Bayes beliefs | ❌ sign |
| α slope at 0 (implemented) | −σ = −2.22 | −0.083 ± 0.09 (statistically 0; the knob's ρ is O(α²)) | ❌ magnitude |
| α slope at 0 (renormalised) | −σ = −2.22 | +0.100 | ❌ magnitude and sign |
| α slope at 0 (equicorr, ρ signed) | −σ = −2.35 | **+5.71** | ❌ sign |
| α shape | smooth 1/(1+α), signed | exactly a function of \|α\| (a proved identity) for both α-DGPs; strictly increasing | ❌ shape |
| separability σ·g(ε)·h(α) | assumed | ∂G/∂ε changes sign with \|α\| — strong ε×α interaction | ❌ |
| \|α\| shape | — | rises on both wings, flat bottom | — |
| (1−\|α\|)² shape | — | wrong direction (this shape *decreases* in \|α\|) | ❌ |

An affine map A + B·σ(1+ε)/(1+α) cannot repair any of this: it cannot manufacture
a √ε cusp from a linear term, it cannot make a smooth function of α equal a
function of |α|, and it needs B > 0 to keep the σ scale positive while needing
B < 0 to get the ε and α signs right.

---

## 3. Both delegation concepts, and the one place the canonical sign survives

### 3.1 Stage-Nash is well-posed here — measured, not assumed

`stage_nash_pmf` solves an *interim* equilibrium: agent i sees ŝᵢ = x + N(0,ε)
privately, takes the others' realised equilibrium deltas as known, and
best-responds under its own belief. Gauss-Seidel iterated best response from the
all-zero profile, agent order 0..3, tie rule T1, cycle cut at 100 sweeps.

* **Convergence:** over the entire 168-cell grid (1000 draws × 21 states × 512
  signal draws × 3 resources per cell), the non-converged fraction is **0.0** and
  the maximum sweeps used is **4**.
* **Existence and uniqueness** (`NASH_NE_CENSUS.csv`, exhaustive census over all
  3⁴ = 81 per-resource profiles): a pure equilibrium **always exists**
  (`frac_zero = 0.000` in all 12 probed conditions); it is unique in
  **69–87%** of (state, signal) cells, with multiplicity 13–31%. Multiplicity
  *rises* with alignment (0.169 at α = 0 → 0.312 at α = 0.8) — aligned targets
  turn the stage game into a coordination game — and *falls* with ε.
* **Selection sensitivity** (`NASH_SELECTION_SENSITIVITY.csv`): starting the BR
  dynamic at the naive-myopic profile instead of the all-zero profile gives
  **bit-identical** results; starting from uniformly random profiles (3 seeds,
  same signal draws) moves E[G] by **−1.7% to +5.3%**, median 0.5%. Equilibrium
  selection is a second-order perturbation, not a driver of any result here.
* **The ε = 0 knife-edge of §2.2(a) is a naive-rule artefact.** Under stage-Nash
  the jump is **0.00%** (implemented α=0), **0.64%** (implemented α=0.8),
  **0.00%** (equicorr ρ=−1/3), **0.00%** (equicorr ρ=+0.8) — `NASH_EPS0PLUS.csv`.
  Equilibrium agents already anticipate the others' contribution, so the boundary
  tie is strictly broken rather than left to convention. (Under the naive rule it
  is not: an agent that ties between −1 and 0 at x = 0 can flip the sign of the
  aggregate when a third agent is pushing +1.)

### 3.2 The two concepts disagree about the SIGN of the alignment effect

σ = 0.6 slice, Cesàro, `G_grid_nash_ce_cesaro.csv`:

| DGP / control | ε=0 | ε=0.25 | ε=0.5 | ε=1.0 |
|---|---|---|---|---|
| implemented α=−0.8 | 0.103 | 0.304 | 0.393 | 0.658 |
| implemented α=0 | **1.146** | 0.403 | 0.514 | 0.784 |
| implemented α=+0.8 | 0.102 | 0.305 | 0.399 | 0.664 |
| equicorr ρ=−0.33 | **1.952** | 0.190 | 0.361 | 0.718 |
| equicorr ρ=0 | 1.117 | 0.384 | 0.495 | 0.770 |
| equicorr ρ=+0.4 | 0.613 | 0.374 | 0.477 | 0.766 |
| equicorr ρ=+0.8 | 0.241 | 0.335 | 0.438 | 0.726 |

Side by side at ε = 0, σ = 0.6:

| | naive-myopic | stage-Nash |
|---|---|---|
| implemented α = 0 | 2.243 | 1.146 |
| implemented α = 0.8 | 7.123 | 0.102 |
| ∂G/∂\|α\| | **+ (3.2× rise)** | **− (11× fall)** |
| equicorr ρ = −1/3 → +0.8 | 0.826 → 7.440 (**+**) | 1.952 → 0.241 (**−**) |

**The sign of the alignment effect is not a property of the environment. It is a
property of the delegation solution concept.** Uncoordinated best responders
herd: when their targets agree they all push the same way and the state
overshoots by up to 4 units against a step size of 1 (fact §1.3: aggregate
feedback gain κ ≈ 2–5). Equilibrium responders internalise each other's moves, so
alignment lets exactly one of them act and the friction collapses.

This is the single most consequential finding for the corpus: **∂F/∂α < 0 is
recoverable, but only under a delegation concept the MARL experiments do not
implement.** `run_gpu.py` trains four *independent* Q-learners (IQL) on their own
observations with no joint-action conditioning — that is the uncoordinated family,
the one whose sign is positive.

Note also the collapse of the α effect once ε > 0 under stage-Nash: at ε = 0.25
the entire α range spans 0.30–0.40. Whatever "alignment effect" exists in this
environment lives almost entirely at ε = 0.

### 3.3 The belief model is a first-order design choice, not a detail

Naive-myopic, exactly Bayesian agents (diffuse prior, posterior N(ŝ, ε), exact
truncated-normal loss) vs certainty-equivalent agents, same draws, full grid:

| ε | mean ΔG (Bayes − CE) | min | max |
|---|---|---|---|
| 0.00 | 0.000 | 0.000 | 0.000 |
| 0.25 | −1.693 | −3.167 | −0.452 |
| 0.50 | −2.464 | −4.419 | −0.753 |
| 1.00 | −3.467 | −6.133 | −1.120 |

Median relative change **−52.8%**, extreme **−99.8%**. Agents that integrate over
their own uncertainty (rather than plugging in a point estimate) hedge toward
δ = 0 near the deadband, which removes most of the overshoot. Under Bayes beliefs
∂G/∂ε is **negative in every one of the 12 control × family combinations**.

Any closed form for G that does not name a belief model is under-specified by a
factor of two.

---

## 4. Deliverable 3 — VERDICT INPUT

### 4.1 Candidate-form fits (`FORM_FITS.csv`)

OLS on cell means; per-cell MC SE is two orders below the between-cell spread.
AICc with k = coefficients + 1. Δ AICc against the best model in each subset.

**The canonical form fitted as a single scaled regressor (G ≈ c·σ(1+ε)/(1+α))
has NEGATIVE R² — i.e. it fits worse than a constant — in 16 of the 18 subsets**
(3 DGP families × 3 rule/belief arms × 2 designs; the pooled `impl+renorm`
subsets are excluded from the count as they are not independent):

| subset (Cesàro) | R² of c·σ(1+ε)/(1+α) | R² of A + B·σ(1+ε)/(1+α) | best model | its R² |
|---|---|---|---|---|
| naive_ce \| implemented | **−0.363** | 0.324 | σ(c₀+c₁\|α\|+c₂α²+c₃ε+c₄ε²+c₅\|α\|ε) | 0.996 |
| naive_ce \| renormalised | **−0.343** | 0.314 | same | 0.997 |
| naive_ce \| equicorr | **−0.290** | 0.073 | σ(c₀+c₁/(1+ρ)+c₂ε), c₁ = **−7.02** | 0.882 |
| naive_bayes \| implemented | **−0.184** | 0.121 | quadratic | 0.989 |
| naive_bayes \| equicorr | **−0.389** | 0.030 | quadratic | 0.860 |
| nash_ce \| implemented | **−0.600** | 0.080 | quadratic | 0.872 |
| nash_ce \| renormalised | **−0.758** | 0.063 | quadratic | 0.883 |
| nash_ce \| equicorr | +0.161 | 0.236 | quadratic | 0.635 |

The two subsets where the canonical form is not worse than a constant are
`nash_ce_cesaro | equicorr` (R² = 0.161) and `nash_ce_T100 | equicorr`
(R² = 0.297) — precisely the one combination (equilibrium delegation, genuinely
signed correlation) where §3.2 shows ∂G/∂ρ < 0. Even there R² ≤ 0.30. The worst
fits are `nash_ce | implemented/renormalised`, at R² = **−0.60 to −1.07**.

Note that `M6_sigma_x_inv1plusa` — the model that *contains* a 1/(1+α) term with a
free coefficient — wins on AICc for the naive equicorr family, but its fitted
coefficient is **−7.02**: the data want that term with the *opposite* sign to the
canonical form.

### 4.2 Free-exponent test: G ≈ c·σ^p·(1+ε)^q·(1+α)^r

The canonical form asserts (p, q, r) = (1, 1, −1). Fitted:

| subset (Cesàro) | p (σ) | q (ε) | r (α) |
|---|---|---|---|
| naive_ce \| implemented | **0.986** | +0.243 | −0.114 |
| naive_ce \| renormalised | **0.986** | +0.169 | −0.121 |
| naive_ce \| equicorr | **0.983** | +0.720 | **+1.285** |
| naive_bayes \| implemented | **0.981** | **−1.794** | −0.215 |
| naive_bayes \| equicorr | 0.886 | **−2.571** | **+3.897** |
| nash_ce \| implemented | **0.969** | +1.287 | +0.105 |
| nash_ce \| equicorr | **0.955** | +0.193 | −0.325 |
| **canonical** | **1** | **1** | **−1** |

p is confirmed (0.955–0.986, and exactly 1 on the σ ∈ [0.2, 0.8] plateau where the
weight clips do not bite). q ranges over **[−2.57, +1.29]** with a sign that flips
with the belief model. r ranges over **[−0.33, +3.90]** and is never within a
factor of 3 of −1.

### 4.3 The verdict

> Is the canonical F a linearisation of G under stated restrictions (ADDITION), or
> is it not any limit of G (RETRACTION)?

**RETRACTION, with a precisely delineated survivor set. The canonical F is not a
first-order expansion of G about (α, ε) = (0, 0) in this environment, and no
affine reparameterisation makes it one.** Two of its three factors fail, and they
fail on order and sign, not on magnitude.

**Survives — put these in the scope-conditioned rewrite:**

| claim | status | why |
|---|---|---|
| **F ≥ 0** | ✅ **stronger than claimed — it is a theorem** | The delegated behaviour is a feasible policy for the principal and the benchmark is the optimum over a superset, so W(realised) ≤ W*. Confirmed: B = max(−G,0) ≤ 1.0 × 10⁻¹⁶ over 336 cells × 1000 draws × 2 concepts. `G₊ = G` here, so the truncation in F := G₊ costs nothing at the ENSEMBLE level. **Corollary the gate's "unresolved binary" needs: negative ensemble friction is impossible against an optimal principal — the "capable delegate beats a suboptimal principal" novelty strip requires a *boundedly rational* principal benchmark and cannot be obtained from this one.** |
| **Delegation surplus B, per principal** | ✅ **real and large** | Gᵢ < 0 for at least one principal in 0%–63.5% of draws under naive delegation (mean 15.9%) and 5.0%–90.2% under stage-Nash (mean 56.0%). B is a *per-principal* object, not an ensemble one. |
| **∂F/∂σ > 0, degree-1 homogeneous** | ✅ exact | G/σ constant to 4 s.f. on σ ∈ [0.2, 0.8]. But it is a **units identity** — the delegated policy is provably σ-free (fact S4) — so it carries no delegation content and should be stated as a normalisation, not a result. |
| **F = 0 ⟺ σ = 0** | ✅ **a.s. true here, for a different reason than the paper gives** | G = Σₘ Wbarₘ·Δₘ with Δₘ ≥ 0; Wbar ∝ σ; and Δ > 0 in every one of the 336 000 cell-draws computed. So σ = 0 ⇒ F = 0 and (in this environment) σ > 0 ⇒ F > 0. This survives — but as a quantization/coordination fact about the {−1,0,1} action set and the unit deadband, **not** as a consequence of (1+ε)/(1+α) > 0. The proof at `02_axiom.tex:179–183` still has to be replaced. |
| **∂ρ_S/∂F < 0 ⇒ selection pressure; the legitimacy carve** | ✅ | Depends only on F ≥ 0 and comparability, both of which hold. |

**Dies — no scope condition rescues these:**

| claim | why |
|---|---|
| **∂F/∂ε > 0 (Prop. Entropy Effect)** | There is no first-order ε term to have a sign: the expansion is Θ(√ε), right-derivative +∞ (measured log–log exponent 0.478 vs 0.5 predicted, 1.0 canonical; the difference quotient ΔG/ε *diverges* by the ε^(−1/2) law, ratio 9.8–14.8 per 100× in ε against a linear prediction of 1). Beyond the cusp the sign is negative for 5 of 6 DGPs at ε = 0.01, and monotonically negative at every ε under Bayesian beliefs. The Morris–Shin/Prat counterexample is not an exotic corner here; **it is the modal case.** |
| **∂F/∂α < 0 (Prop. Alignment Effect)** | Under both the implemented and the renormalised DGP, E[G](α) = E[G](−α) is an **exact distributional identity** (`max_abs_tau_diff = 0.0`), so no signed α effect can exist; and because the manipulated correlation is O(α²) the first-order coefficient is **0**, measured at −0.083 ± 0.09 against a canonical −2.22. Under the equicorrelated DGP, where ρ *is* the signed pairwise correlation, the slope is **+5.71** — the opposite sign, 2.4× the canonical magnitude. The sign is only recoverable under stage-Nash delegation, which the MARL experiments do not implement. |
| **Multiplicative separability σ·g(ε)·h(α)** | ∂G/∂ε changes sign as \|α\| rises (implemented, ε=0→1: +1.850 at α=0, −1.245 at α=0.8). A strong, sign-flipping ε×α interaction is present in every family. The saturated model that includes an \|α\|·ε term reaches R² = 0.99+; every separable model is ≥ 36 AICc units behind. |
| **The irreducible σ/2 baseline** | The quantization floor is the object that plays this role, it is large (mean 1.88, up to 6.04 welfare units), it is in closed form (§1.2), and it **cancels identically** in any difference-against-benchmark. Confirmed by direct computation. |
| **Prop. Modular Governance Fragility; additive Aggregation** | Both are algebraic consequences of the functional form that just failed. |
| **The "12/12 Lean mapping"** | Unaffected in truth-value, but three theorems now map to propositions that are being withdrawn. |

**One important limit on the scope of this verdict.** All of the above is *this
environment*: 4 agents, 3 resources, a {−1,0,+1} action set against a unit
deadband, targets on an N(0,1) scale inside a [0,10] box so that half the
resource-optima sit on the lower clamp. The failure mode it produces —
over-correction with aggregate feedback gain κ ≈ 2–5 — is what drives both the
wrong-sign α result and the negative ε result. A different environment could give
a different answer. What cannot be argued around is that **this is the environment
the corpus's own experiments run on**, and on it the canonical functional form
fits worse than a constant.

### 4.4 What the analyst's formula must reproduce to be believed

Ranked by how cheaply it falsifies a wrong closed form:

1. **G ≥ 0 identically** and G = Σₘ Wbarₘ[E_π(x−s*)² − min_L(x−s*)²] exactly.
2. **Exact degree-1 σ homogeneity** on σ ∈ [0.2, 0.8]; break above 0.8 via the
   clip at 1 (G(1.2) ≈ G(1.0)).
3. **Θ(√ε) cusp** at ε → 0+, and a **knife-edge discontinuity at ε = 0 exactly**
   worth +0.42% to +34.2% depending on DGP, which vanishes under stage-Nash.
4. **E[G](α) = E[G](−α)** exactly, and zero first-order α coefficient, under both
   α-DGPs.
5. **∂E[G]/∂ρ > 0** under equicorrelated targets with naive delegation, and
   **< 0** under stage-Nash.
6. **The quantization floor** E[d²] = 1/24 in the interior plus Var(s*)/2 from the
   lower clamp, with Var(s*) = α²+(1−|α|)²/4 (implemented) or (1+3ρ)/4
   (equicorrelated) — and its exact cancellation from G.
7. **κ ≈ 2–5**: any derivation that assumes a contracting delegated dynamic is
   modelling a different system.

### 4.5 Declared assumptions and known limitations

* **B1** (belief): agents hold a diffuse prior and posterior N(ŝ, ε); they know
  the clamp, not the lattice. CE is the ε-plug-in variant. Both are computed on
  the full grid for the naive rule; stage-Nash runs CE on the full grid with a
  Bayes wedge measured on a subgrid (`NASH_BELIEF_WEDGE.csv`, −99% to +34%).
* **N1** (equilibrium): interim equilibrium, Gauss-Seidel BR from the all-zero
  profile, agent order 0..3. Existence always, uniqueness 69–87%, selection
  sensitivity ≤ 5.3%.
* **T1** (ties): lexicographic — clamped loss, then unclamped squared error, then
  smallest |δ|, then smallest δ. Necessary: the naive decision rule is
  **non-monotone in ŝ** under a naive tie rule (the assertion in
  `naive_thresholds` fires), because at ŝ ≲ −1 all three moves clamp to the same
  post-state.
* **Truncation**: Cesàro burn-in 2⁹ = 512 steps by repeated squaring, then a
  512-step Cesàro average (correct for the periodic ε = 0 chains). The T = 100
  design uses the static planner optimum as its benchmark, which *overstates* the
  gap by the planner's own transient, measured at 0.535/step
  (`PLANNER_TRANSIENT.csv`). Use the Cesàro tables.
* **Monte Carlo**: 1000 preference draws per grid cell (2000 for the σ sweep and the
  α axis, 8000 for the ε cusp, 20 000 for the floor closed form), common random
  numbers across every cell.
  Per-cell SE 0.002–0.41, typically 0.02–0.15; the stage-Nash arm adds 512 signal
  draws per (state, resource, cell), whose effect on E[G] is bounded by the
  selection-sensitivity numbers above.
* **Not done**: n > 4 agents; m ≠ 3; a stakes design that varies sᵢ *across*
  principals (which is what would make the stakes-weighted incidence object bite);
  learned policies (deliberately — the point of this exercise is to remove them);
  the two-team block DGP from `dgp.py:88`.

---

## 5. Independent replication of the headline claims

Everything above rests on `core.py`. To keep a single implementation error from
propagating into the verdict, the load-bearing claims were **recomputed by a second
implementation written from scratch** (`independent_crosscheck.py`,
`crosscheck_planner_incidence.py`), which does not import `core.py` and shares no
code with it — different lattice construction, different tie-break coding,
different chain estimator (brute-force forward simulation instead of the exact
Cesàro chain algebra).

| check | claim | independent result | verdict |
|---|---|---|---|
| **X1** | reachable set is 21 points/coordinate (11 iff frac = 0) | brute-force forward closure: 11 at f = 0, exactly 21 for all 299 random offsets | ✅ **1331 × 729 is the measure-zero case; 9261 × 729 a.s.** |
| **X2** | E[G](α) = E[G](−α) is exact | max\|τ(α;b,ξ) − τ(−α;−b,ξ)\| = **0.0** for both α-DGPs, all α, 40 seeds | ✅ the α knob is a \|α\| knob |
| **X3** | W(s) = −Σₘ[Wbarₘ(sₘ−s\*ₘ)² + Vₘ] | max abs err **1.7 × 10⁻¹³** over 400 draws | ✅ Vₘ cancels in G |
| **X4** | G ≥ 0 per draw, and the reduction formula | brute-force forward sim (T = 4000, burn 1000): min G over draws = **+1.1 × 10⁻²**, frac(G < 0) = **0.000** | ✅ |
| **X5** | interior quantization floor E_f[d²] = 1/24 | MC (4 × 10⁶): **0.041653** vs 0.0416667 | ✅ |
| **X6** | Var(s\*) closed forms, all three DGPs | worst relative error **0.51%** across 10 (family, control) points; equicorr ρ = −1/3 gives exactly 0 | ✅ incl. the degenerate point |
| **X8** | the G surface itself | implemented α=0, σ=0.6, ε=0.25: **2.457** (3000 draws) vs `core.py`'s **2.431** — a **1.1%** gap between wholly independent implementations. α=0.8, σ=0.2, ε=0: **2.327** vs 2.374 scaled (2.0%) | ✅ |

The one check that *failed* is X7/X9 — and it failed against the **spec**, not
against this report: `ideation_measurement.md` §0.1's planner magnitudes are not
reproducible and one of them exceeds the analytic ceiling (§0.3).

**What this does not cover.** The second implementation replicates the naive-myopic
arm, the planner, the floor and the DGP identities. It does **not** independently
reimplement stage-Nash (the iterated-best-response solver exists only in
`core.py`); that arm's credibility rests instead on the internal checks in §3.1 —
zero non-convergence over the full grid, max 4 sweeps, exhaustive 81-profile
existence census, bit-identical results from a different BR starting profile, and
≤ 5.3% movement from random starts. A reader who wants the §3.2 sign reversal
independently confirmed should treat that as the open item.


