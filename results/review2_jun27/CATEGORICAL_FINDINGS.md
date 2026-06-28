# A2 — Categorical-w refit (misspecified continuous interaction) + baseline-difficulty confound

**Date:** 2026-06-27 · **Reviewer items:** Major "misspecified continuous interaction", Major "confounded difficulty in partial-sharing"
**Data:** `results/partial_sharing/partial_{iql,vdn}.csv` (committed CSVs; no RL re-run).
**Design:** overlap `w ∈ {0, 1/3, 2/3, 1}` (= `k/3` shared resources), signed `ρ ∈ {−0.33,−0.20,−0.10,0,0.20,0.40,0.60,0.80}`, `σ ∈ {0.6,1.0}`, 30 seeds → **n = 1920 per learner**, 30/ cell, no missing rows.
**DV:** `gap_norm = −mean_reward / σ` (matches `analyze_partial.load`). **Predictor:** signed `ρ`. Pooled over `σ`.
**Env:** Python 3.13.11, pandas 3.0.3, scipy 1.18.0, statsmodels 0.14.6. All scipy/statsmodels results read via **named attributes** (`.slope/.pvalue/.params/.pvalues/.conf_int/.effect/.sd`; `anova_lm` columns `F`,`Pr(>F)`,`df_diff`). Per-w slopes from the pooled `C(w)*ρ` model use **HC3** robust covariance; cross-checked against independent per-w OLS (point slopes identical to ≥6 dp).
**Reproduce:** `scratchpad/a2_categorical.py` (main), `a2_refine.py` (contrasts + slope-shape test), `a2_decomp.py` (intercept-vs-slope decomposition). Outputs: `a2_per_w_slopes.csv`, `a2_baseline_by_w.csv`, `a2_model_comparisons.csv`, `a2_console.txt`.

---

## (i) Misspecification fix — `gap/σ ~ C(w) * ρ`, per-w ρ-slopes

### Per-w ρ-slope table (categorical model, HC3 95% CI)

| learner | w | k | ρ-slope | 95% CI | p | per-w R² |
|---|---|---|---|---|---|---|
| IQL | 0 | 0 | −0.022 | [−0.396, +0.353] | 0.909 | 0.000 |
| IQL | 1/3 | 1 | −0.210 | [−0.653, +0.234] | 0.354 | 0.002 |
| IQL | 2/3 | 2 | **−0.602** | [−1.004, −0.199] | **0.0034** | 0.023 |
| IQL | 1 | 3 | **−0.741** | [−1.230, −0.253] | **0.0029** | 0.023 |
| VDN | 0 | 0 | −0.066 | [−0.447, +0.315] | 0.734 | 0.000 |
| VDN | 1/3 | 1 | −0.244 | [−0.672, +0.185] | 0.266 | 0.003 |
| VDN | 2/3 | 2 | **−0.614** | [−0.973, −0.256] | **0.00078** | 0.029 |
| VDN | 1 | 3 | **−0.755** | [−1.163, −0.348] | **0.00028** | 0.033 |

(Independent per-w OLS gives identical point slopes; its CIs are marginally tighter — e.g. IQL w=1 OLS [−1.176,−0.307], p=8.5e-4. Same qualitative pattern.)

### Threshold shape, quantified
- Signed slopes ramp monotonically with overlap: IQL `−0.022 → −0.210 → −0.602 → −0.741`; VDN `−0.066 → −0.244 → −0.614 → −0.755`. **Spearman(w, slope) = −1.000 (p<0.001)** for both learners — perfect monotonicity.
- **Significance pattern is "flat-then-engaging":** the slope is **null and non-significant at w∈{0, 1/3}** (all p>0.25) and **significantly negative at w∈{2/3, 1}** (p≤0.003). The largest single increment is the engagement step w=1/3→2/3 (Δ|slope| = +0.39 IQL / +0.37 VDN), vs +0.19/+0.18 for 0→1/3 and +0.14/+0.14 for 2/3→1.
- Mean |slope|: low-overlap (w≤1/3) = 0.116 (IQL) / 0.155 (VDN); high-overlap (w≥2/3) = 0.672 / 0.685 — a ~4–6× increase.

### The single linear ρ×w coefficient is a misspecified summary — but precisely diagnosed
The pooled continuous model `gap/σ ~ ρ * w` reports one interaction coefficient: **−0.765 (IQL, p=0.0034)** / **−0.731 (VDN, p=0.0024)**. Model comparisons (nested F via `anova_lm`):

| comparison (nested F) | IQL | VDN | meaning |
|---|---|---|---|
| (A) `C(w)*ρ` vs `C(w)+ρ` | F(3,1912)=3.00, p=0.030 | F(3,1912)=3.19, p=0.023 | **per-w slopes DO differ** (interaction is real) |
| (B) `C(w)*ρ` vs `ρ*w` (linear) | F(4,1912)=5.04, p=4.8e-4 | F(4,1912)=3.85, p=4.0e-3 | the linear `ρ*w` surface is **inadequate** (categorical AIC-preferred: ΔAIC +12.2 / +7.4) |
| ↳ intercept-shape only (`ρ*w` → `C(w)+ρ+ρ:w`) | F(2)=9.95, **p=5.0e-5** | F(2)=7.57, **p=5.3e-4** | the **baseline is strongly non-linear in w** |
| ↳ slope-shape only (`C(w)+ρ+ρ:w` → `C(w)*ρ`) | F(2)=0.15, p=0.86 | F(2)=0.15, p=0.86 | the **slope's w-dependence is NOT distinguishable from linear** |

**Plain statement.** Refit with `w` categorical, the accurate description of the dose–response is the per-w slope table above: misalignment sensitivity is statistically null at low overlap (w≤1/3) and significantly negative at high overlap (w≥2/3), rising monotonically. The single pooled linear ρ×w coefficient (−0.77/−0.73) is a misspecified summary of the **joint** `gap/σ`-surface (the categorical model is AIC-preferred and beats it at p<0.005), and reading it as "the" slope at any given w is wrong — it averages a null low-w regime with a strong high-w regime.

**Honest caveat (do not overclaim a slope "threshold").** Decomposing *why* the linear model fails: the inadequacy lives **entirely in the intercept/baseline**, which is non-linear in w (intercept-shape p≈5e-5/5e-4). The **slope's** own dependence on w is **not** statistically separable from a linear ramp (slope-shape F p=0.86 for both; AIC even slightly prefers the linear-slope form, Δ≈−3.7). With only four overlap levels there is insufficient power to resolve a sharp slope threshold from a smooth linear ramp. So the supportable claims are: (a) the slope is monotone in w (Spearman −1.0), (b) it is null at w≤1/3 and significant at w≥2/3 by the per-w tests — descriptively the "flat-then-engaging" reading — and (c) the categorical per-w slopes, not the single linear interaction term, are the correct description of the surface. A statistically *established* threshold curvature in the slope is **not** claimed.

---

## (ii) Baseline-difficulty confound — does the ρ=0 level rise with w?

The confound hypothesis: more shared resources raise multi-agent credit-assignment difficulty independent of ρ, which would inflate the baseline gap (the `C(w)` intercept / ρ=0 cell mean) and could masquerade as a steeper dose–response.

### Baseline (ρ=0) `gap_norm` by w

| learner | w | k | intercept (model) | 95% CI | ρ=0 cell mean | 95% CI (raw, n=60) |
|---|---|---|---|---|---|---|
| IQL | 0 | 0 | 3.904 | [3.795, 4.012] | 4.102 | [3.784, 4.421] |
| IQL | 1/3 | 1 | 4.091 | [3.950, 4.232] | 4.180 | [3.873, 4.487] |
| IQL | 2/3 | 2 | 3.742 | [3.607, 3.877] | 3.824 | [3.492, 4.157] |
| IQL | 1 | 3 | 4.139 | [3.968, 4.310] | 4.243 | [3.889, 4.598] |
| VDN | 0 | 0 | 4.044 | [3.930, 4.157] | 4.209 | [3.894, 4.525] |
| VDN | 1/3 | 1 | 4.027 | [3.890, 4.164] | 4.120 | [3.827, 4.414] |
| VDN | 2/3 | 2 | 3.449 | [3.325, 3.573] | 3.562 | [3.252, 3.871] |
| VDN | 1 | 3 | 3.402 | [3.256, 3.549] | 3.494 | [3.207, 3.781] |

**Baseline trend (regress ρ=0 obs on w):**
- **IQL:** slope **+0.020** [95% CI −0.416, +0.456], **p=0.93**; Spearman ρ=+0.002, p=0.98 → **flat**. No difficulty level effect.
- **VDN:** slope **−0.811** [95% CI −1.209, −0.413], **p=8.1e-5**; Spearman ρ=−0.253, p=7.6e-5 → baseline **DECLINES** with w (the *opposite* of the confound's prediction; with neutral preferences, value-decomposition VDN benefits from shared structure).

**The predicted confound does not occur in either learner.** The reviewer's premise — that overlap raises the baseline gap — is not supported: IQL baseline is flat, VDN baseline falls.

### Insulation of the dose–response (slope) claim
The dose–response claim is about the ρ-**slope** (`∂gap/∂ρ` at fixed w), not the level. A pure additive baseline shift across w changes only the intercept and contributes **exactly 0** to the within-w slope by construction. Quantitatively, the slope steepens from w=0 to w=1 by **Δslope = −0.719 (IQL)** and **−0.689 (VDN)**, while a pure level shift predicts Δslope = 0 — so **0%** of the steepening is attributable to a baseline level shift. Empirically this is reinforced: across w the baseline and the slope move **independently** (IQL: baseline flat, slope steepens) or in **opposite** directions (VDN: baseline falls while slope steepens) — neither pattern is what a difficulty-level confound would generate. For scale, the ρ-driven swing at w=1 (slope × ρ-range 1.13) is ≈ −0.84 (IQL) / −0.85 (VDN), comparable to or larger than any baseline movement.

### Honest 2–3 sentence assessment
The additive difficulty-**level** confound the reviewer raises explains essentially none of the slope steepening: slopes are partial derivatives at fixed w and are invariant to additive level shifts, and the baseline does not even rise with w here (flat for IQL, declining for VDN). The one piece this design genuinely cannot rule out is a difficulty×ρ **interaction** — i.e. harder credit assignment at high w *amplifying* ρ-sensitivity — because increasing w moves contention-alignment and credit-assignment difficulty together; that amplification is observationally entangled with genuine overlap-mediation of alignment. But since baseline difficulty does not increase with w in this DGP, there is weak empirical footing for a large difficulty×ρ term, so the dose–response is well (though not perfectly) insulated.

---

## LaTeX-ready rows

### Per-w ρ-slope table (booktabs)
```latex
% gap/sigma ~ C(w)*rho ; per-w rho-slope, HC3 95% CI ; n=1920/learner, 30 seeds
\begin{tabular}{llcccc}
\toprule
Learner & $w$ ($k$) & $\rho$-slope & 95\% CI & $p$ & per-$w$ $R^2$ \\
\midrule
IQL & $0$ ($0$)   & $-0.022$ & $[-0.396,\,+0.353]$ & $0.909$  & $0.000$ \\
IQL & $1/3$ ($1$) & $-0.210$ & $[-0.653,\,+0.234]$ & $0.354$  & $0.002$ \\
IQL & $2/3$ ($2$) & $-0.602$ & $[-1.004,\,-0.199]$ & $0.0034$ & $0.023$ \\
IQL & $1$ ($3$)   & $-0.741$ & $[-1.230,\,-0.253]$ & $0.0029$ & $0.023$ \\
\midrule
VDN & $0$ ($0$)   & $-0.066$ & $[-0.447,\,+0.315]$ & $0.734$   & $0.000$ \\
VDN & $1/3$ ($1$) & $-0.244$ & $[-0.672,\,+0.185]$ & $0.266$   & $0.003$ \\
VDN & $2/3$ ($2$) & $-0.614$ & $[-0.973,\,-0.256]$ & $0.00078$ & $0.029$ \\
VDN & $1$ ($3$)   & $-0.755$ & $[-1.163,\,-0.348]$ & $0.00028$ & $0.033$ \\
\bottomrule
\end{tabular}
```

### Categorical-vs-linear model comparison + baseline trend
```latex
% Spearman(w,slope)=-1.000 (p<0.001) both learners
\begin{tabular}{lcc}
\toprule
Test & IQL & VDN \\
\midrule
Single linear $\rho\times w$ coef                      & $-0.765$ ($p{=}0.0034$) & $-0.731$ ($p{=}0.0024$) \\
(A) Slopes differ: $C(w){*}\rho$ vs $C(w){+}\rho$      & $F_{3,1912}{=}3.00,\,p{=}0.030$ & $F_{3,1912}{=}3.19,\,p{=}0.023$ \\
(B) Linear inadequate: $C(w){*}\rho$ vs $\rho{*}w$     & $F_{4,1912}{=}5.04,\,p{=}4.8\mathrm{e}{-4}$ & $F_{4,1912}{=}3.85,\,p{=}4.0\mathrm{e}{-3}$ \\
\quad $\hookrightarrow$ intercept-shape only          & $F_{2}{=}9.95,\,p{=}5.0\mathrm{e}{-5}$ & $F_{2}{=}7.57,\,p{=}5.3\mathrm{e}{-4}$ \\
\quad $\hookrightarrow$ slope-shape only              & $F_{2}{=}0.15,\,p{=}0.86$ & $F_{2}{=}0.15,\,p{=}0.86$ \\
AIC ($\rho{*}w$ / $C(w){*}\rho$)                       & $7246.2 / 7234.0$ & $6940.7 / 6933.3$ \\
\midrule
Baseline ($\rho{=}0$) trend vs $w$                     & $+0.020,\,p{=}0.93$ (flat) & $-0.811,\,p{=}8.1\mathrm{e}{-5}$ (declines) \\
\bottomrule
\end{tabular}
```

### Baseline-by-w rows
```latex
% baseline = gap/sigma at rho=0 (n=60/cell); model intercept [HC3] / raw cell-mean
IQL & $0$   & $3.904$ & $[3.795,4.012]$ & $4.102$ & $[3.784,4.421]$ \\
IQL & $1/3$ & $4.091$ & $[3.950,4.232]$ & $4.180$ & $[3.873,4.487]$ \\
IQL & $2/3$ & $3.742$ & $[3.607,3.877]$ & $3.824$ & $[3.492,4.157]$ \\
IQL & $1$   & $4.139$ & $[3.968,4.310]$ & $4.243$ & $[3.889,4.598]$ \\
VDN & $0$   & $4.044$ & $[3.930,4.157]$ & $4.209$ & $[3.894,4.525]$ \\
VDN & $1/3$ & $4.027$ & $[3.890,4.164]$ & $4.120$ & $[3.827,4.414]$ \\
VDN & $2/3$ & $3.449$ & $[3.325,3.573]$ & $3.562$ & $[3.252,3.871]$ \\
VDN & $1$   & $3.402$ & $[3.256,3.549]$ & $3.494$ & $[3.207,3.781]$ \\
```

---

## Bottom line
1. **Misspecification (i):** Replace the single linear ρ×w coefficient with the categorical per-w slopes. Slopes are monotone in w (Spearman −1.0), null at w≤1/3, significantly negative (~−0.6 to −0.76) at w≥2/3. The linear `ρ*w` model is AIC-rejected for the joint surface (p<0.005), but the rejection is driven by the **baseline** being non-linear in w (p≈5e-5), **not** by slope curvature (slope-shape p=0.86). So: report categorical per-w slopes as the accurate description; describe the dose–response as "null at low overlap, engaging at high overlap"; do **not** claim a statistically established slope threshold (4 levels, underpowered for curvature).
2. **Difficulty confound (ii):** The baseline (ρ=0) does **not** rise with w — flat for IQL (p=0.93), declining for VDN (p=8e-5). The slope is invariant to additive level shifts, so **0%** of the −0.72/−0.69 slope steepening is explained by a baseline level effect. Residual non-identifiability is limited to a difficulty×ρ interaction (which this design cannot isolate), but with no baseline difficulty increase there is little empirical basis for it.
