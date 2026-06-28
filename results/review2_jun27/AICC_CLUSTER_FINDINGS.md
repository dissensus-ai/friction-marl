# A3 — Two small statistical corrections (AICc + cluster-robust SE)

**Date:** 2026-06-27
**Scope:** Friction-MARL companion paper, adversarial re-run analyses.
**Environment:** uv venv, CPython 3.13.11; numpy 2.5.0, pandas 3.0.3, scipy 1.18.0,
statsmodels 0.14.6. (System python 3.14 segfaults pandas — not used.)
**Inputs (committed CSVs + seeded DGP only; no RL re-run):**
`results/adversarial_rerun/equicorr_{iql,vdn}.csv`,
`results/adversarial_rerun/functional_form_refit.{py,csv}`, and `dgp.py` /
`adv_runner.py` / `review_upgrade/b_vdn_runner.py` for the seeded weight reconstruction.
**Recompute script:** `results/review2_jun27/aicc_cluster_recompute.py`.
All scipy/statsmodels reads use named attributes (`.aic`, `.llf`, `.params`,
`.pvalues`, `.bse`, `.rsquared`) — no positional tuple indexing.

All numbers below are this RA's own recompute.

---

## A3(i) — AICc for the signed functional-form model comparison

### Setup
`functional_form_refit.py` fits the four candidate friction forms on the **signed
equicorrelation** condition means (`gap = -mean_reward`, averaged over 30 reps),
giving **n = 16** conditions per learner (8 ρ × 2 σ; ε = 0 throughout):

| Model | form (at ε=0) | mean params *k* (regressors incl. intercept) | identifiable? |
|---|---|---|---|
| M1 Friction | `σ/(1+ρ)` + intercept | 2 | yes |
| M2 Additive | `σ−ρ` + intercept | 2 | yes |
| M3 Multiplicative | `σ·ε·(1−ρ)` ≡ 0 at ε=0 | — | **degenerate (not identifiable)** |
| M4 Independent | `intercept + g_ρ·ρ + g_σ·σ` | 3 | yes |

The script uses **plain AIC** (`statsmodels OLS.aic`). With n = 16 and the
Burnham & Anderson total parameter count *K* = 3–4 (so n/K ≈ 4–5 ≪ 40), B&A require
the small-sample-corrected **AICc**:

```
AICc = AIC + 2K(K+1) / (n − K − 1)
```

### What *k* counts, and why σ² cancels in ΔAICc
- `statsmodels OLS.aic` counts only the **mean parameters** = regressors **including the
  intercept** (empirically verified: `k_used = (aic + 2·llf)/2` returns exactly 2, 2, 3
  for M1, M2, M4 — the residual variance σ² is **not** separately counted).
- Under the strict Burnham & Anderson convention σ² **is** an estimated parameter, so
  **K = k + 1** (M1/M2: K=3; M4: K=4 — matching the n/K ≈ 4–5 framing). Switching to this
  convention adds a **constant +2** to every model's base AIC
  (`AIC_BA = −2·llf + 2K = AIC_sm + 2`). That +2 is identical for M1, M2 and M4 and
  therefore **cancels in every ΔAICc**. The *only* residual effect of the σ²-counting
  choice is through the **nonlinear** correction `2K(K+1)/(n−K−1)`, which differs across
  models by < 1 AICc unit and cannot change the ranking. Concretely:

  ```
  ΔAICc(M4−M1) = ΔAIC_sm(M4−M1) + [ 2K_M4(K_M4+1)/(n−K_M4−1) − 2K_M1(K_M1+1)/(n−K_M1−1) ]
  ```

### Correction terms (n = 16)
| convention | K(M1)=K(M2) | corr | K(M4) | corr |
|---|---|---|---|---|
| **I — B&A strict (K = k+1, σ² counted)** | 3 | 2·3·4/12 = **2.0000** | 4 | 2·4·5/11 = **3.6364** |
| II — k only (statsmodels base) | 2 | 2·2·3/13 = **0.9231** | 3 | 2·3·4/12 = **2.0000** |

### Results — IQL (n=16)
| model | AIC_sm | AICc (conv. I) | AICc (conv. II) |
|---|---|---|---|
| M1 Friction `σ/(1+ρ)` | 25.840 | 29.840 | 26.763 |
| M2 Additive `σ−ρ` | 32.584 | 36.584 | 33.507 |
| M3 Multiplicative | degenerate (ε=0) | — | — |
| **M4 Independent** | **1.744** | **7.381** | **3.744** |

- **ΔAICc(M4−M1) = −22.46** (conv. I) / **−23.02** (conv. II). Plain ΔAIC = −24.10.
- ΔAICc(M4−M2) = −29.20 (conv. I).

### Results — VDN (n=16)
| model | AIC_sm | AICc (conv. I) | AICc (conv. II) |
|---|---|---|---|
| M1 Friction `σ/(1+ρ)` | 18.408 | 22.408 | 19.331 |
| M2 Additive `σ−ρ` | 25.098 | 29.098 | 26.021 |
| M3 Multiplicative | degenerate (ε=0) | — | — |
| **M4 Independent** | **−0.933** | **4.704** | **1.067** |

- **ΔAICc(M4−M1) = −17.70** (conv. I) / **−18.26** (conv. II). Plain ΔAIC = −19.34.
- ΔAICc(M4−M2) = −24.39 (conv. I).

### Verdict
**M4 ≻ M1 still holds, overwhelmingly.** ΔAICc(M4−M1) ≈ **−22.5 (IQL)** and **−17.7
(VDN)** under the strict B&A convention (≈ −23.0 / −18.3 under the k-only convention).
Both are far beyond the B&A threshold |Δ| > 10 ("essentially no support" for the inferior
model). The finite-sample penalty correctly charges M4 its extra parameter — it shrinks the
gap by only ~1.6 AICc units (the difference of the two correction terms, 3.636 − 2.000) and
comes nowhere near flipping the ranking. The independent additive form (M4) remains decisively
preferred over the friction form (M1) on the genuinely-signed data, exactly as on plain AIC.

### LaTeX-ready
> Because $n=16$ with $K=3$–$4$ ($n/K\approx4$–$5\ll40$), we report the small-sample
> corrected $\mathrm{AICc}=\mathrm{AIC}+2K(K+1)/(n-K-1)$ (Burnham \& Anderson). Counting
> the residual variance as an estimated parameter adds a constant $+2$ to every model's
> base AIC, which cancels in any $\Delta\mathrm{AICc}$; only the nonlinear correction term
> differs across models. The independent additive form (M4) remains strongly preferred:
> $\Delta\mathrm{AICc}(\text{M4}-\text{M1})=-22.5$ (IQL) and $-17.7$ (VDN), both far below
> the $\Delta>10$ ``no-support'' threshold, so the AIC$\to$AICc correction leaves the
> functional-form conclusion unchanged.

---

## A3(ii) — Cluster-robust SE for the n = 1920 ρ×w interaction

### Setup
Model **`gap/σ ~ ρ * w`** (ρ and w both continuous), fit per learner at the **agent level**:
each of the 480 replication-rows (8 ρ × 2 σ × 30 seeds) contributes its 4 agent rewards,
so **n = 1920**. `gap = −reward`, `gap_norm = gap/σ`. The per-agent **stakes weight
`w` (≈ σ) is reconstructed from the seeded DGP** (`w_i ~ N(σ,0.05)` clipped to [0,1],
averaged over the 3 resources), replaying the exact RNG draw order
(`master = default_rng(42)` → 128×3 probe draw → 16 per-condition seeds → per-rep
`standard_normal` targets then `normal` weights). **Validation:** the reconstructed
condition-mean gaps reproduce the run log to 4 dp (e.g. ρ=−0.33,σ=0.6 → 2.0410 vs log
2.041; ρ=0.8,σ=1.0 → 3.2057 vs 3.206), confirming the row/seed/agent alignment.

The naive fit treats all 1920 rows as i.i.d., ignoring that the 4 agents within a run share
one environment/episode and that the 30 seeds sit inside each (ρ,σ) condition cell — i.e.
the genuine independent unit is the **seeded run**, not the agent-row.

### Results — interaction coefficient and p (term `ρ:w`)
| learner | coef(ρ:w) | naive i.i.d. SE / p | cluster by **condition cell (ρ,σ)**, G=16 | cluster by **seed (ρ,σ,rep)**, G=480 |
|---|---|---|---|---|
| **IQL** | **−1.849** | SE 0.802, **p = 2.12e-2** | SE 0.753, **p = 1.40e-2** | SE 1.147, **p = 1.07e-1** |
| **VDN** | **−1.615** | SE 0.695, **p = 2.02e-2** | SE 0.692, **p = 1.96e-2** | SE 0.940, **p = 8.57e-2** |

(Coefficient is identical across covariance choices — only the SE/p change.)

### Comparison to the quoted naive p (3.4e-3 / 2.4e-3)
My naive p on the **σ-normalized** DV is **2.1e-2 (IQL) / 2.0e-2 (VDN)** — an order of
magnitude weaker than the quoted 3.4e-3 / 2.4e-3. The quoted figures match the
**un-normalized raw-gap** spec (`gap ~ ρ*w`), where I recover naive **p = 6.7e-4 (IQL) /
1.7e-4 (VDN)** — i.e. the same "highly significant" regime the quote refers to. Either DV
tells the same correction story (below). I report the σ-normalized spec as the headline
because the paper's structural claim is about `gap/σ` (the mechanical stakes effect removed).

Raw-gap spec for the record:
| learner | coef | naive p | cluster G=16 | cluster G=480 |
|---|---|---|---|---|
| IQL | −2.234 | 6.71e-4 | 8.67e-6 | 1.69e-2 |
| VDN | −2.150 | 1.73e-4 | 1.04e-6 | 5.02e-3 |

### Interpretation (important — the clustering grain matters)
1. **Clustering by condition cell (ρ,σ) as literally requested gives only G = 16 clusters**,
   well below the rule-of-thumb (G ≳ 30–50) for trustworthy cluster-robust asymptotics, and
   **ρ is constant within each cell** (a pure cluster-level regressor). With so few clusters
   the sandwich SE is **downward-biased**: on the normalized DV it barely moves the p
   (0.014 / 0.020 ≈ naive), and on the raw DV it pathologically **shrinks** it to 8.7e-6 /
   1.0e-6. G=16 cell-clustering is therefore **not reliable** here.
2. **The statistically correct independent unit is the seeded run** (each replication =
   one independent RNG seed; its 4 agent rows are not independent — that is the actual
   pseudoreplication inflating n from 480 to 1920). **Clustering by seed (G = 480)** is
   well-powered and pushes the σ-normalized interaction to **p ≈ 0.11 (IQL) / 0.086 (VDN)
   — no longer significant at 0.05**. (On the raw DV, seed-clustering still moves p from
   ~1e-4 to 5e-3–1.7e-2, i.e. 1–2 orders weaker.)
3. **Conclusion:** the apparent significance of the ρ×w interaction is largely a
   pseudoreplication artifact. On `gap/σ` (mechanical stakes effect removed), once
   observations are clustered at the genuine experimental unit the interaction is **at best
   marginal** (p ≈ 0.09–0.11). This is consistent with `FINDINGS.md`'s own caveat that the
   ρ direction is real but single-comparison/interaction significance is marginal
   (e.g. the |ρ|=0.2 sign test p ≈ 0.14).

### LaTeX-ready
> We re-fit the agent-level interaction $\text{gap}/\sigma \sim \rho * w$ ($n=1920$;
> $w\!\sim\!N(\sigma,0.05)$ the per-agent stakes weight) with cluster-robust standard
> errors. Clustering by condition cell $(\rho,\sigma)$ yields only $G=16$ clusters with
> $\rho$ constant within cluster, so the small-$G$ sandwich is downward-biased and barely
> alters the naive $p$ ($0.014$/$0.020$ for IQL/VDN). Clustering instead at the genuine
> independent unit — the seeded run ($G=480$), which removes the pseudoreplication of the
> four agents sharing one episode — the interaction coefficient ($-1.85$ IQL, $-1.62$ VDN)
> drops to $p=0.11$ (IQL) and $p=0.086$ (VDN), i.e. **not** significant at the $5\%$ level,
> versus the naive $p=0.021$/$0.020$. The $\rho\times w$ interaction is thus directional but
> not robust to proper clustering.

---

## Caveats
- **AICc:** the headline ΔAICc uses the strict Burnham & Anderson count (K = mean params +
  σ²; K = 3–4), matching the cited reference; the k-only (statsmodels) count gives the same
  conclusion (ΔAICc shifts < 1 unit). M3 is undefined at ε=0 (composite ≡ 0) and is excluded,
  as in the original script.
- **ρ×w naive p mismatch:** my recompute on the σ-normalized DV (≈0.02) does not reproduce
  the quoted 3.4e-3 / 2.4e-3; those correspond to the **un-normalized raw-gap** spec
  (my recompute 6.7e-4 / 1.7e-4). Both are reported; the correction's direction is identical.
- **w is reconstructed, not stored.** The CSVs do not contain the weights; `w` is regenerated
  from the seeded DGP (draw order verified against the runner and cross-checked via exact
  gap reproduction). If the original interaction used `w = σ` (the level) instead of the
  per-agent draw, the numbers are essentially identical (w–σ correlation 0.992; e.g. naive
  p 0.0246/0.0207 vs 0.0212/0.0202).
- **G=16 cell-clustering is reported because the task requested it**, but it is statistically
  unreliable (too few clusters, ρ constant within cluster); the seed-level (G=480) clustering
  is the defensible correction and is what the verdict rests on.
