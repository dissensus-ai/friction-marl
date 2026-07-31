# Referee verdict on the closed-form delegation gap $G(\sigma,\alpha,\varepsilon)$

**Adversarial referee report, 30 July 2026.** Commissioned to attack the analyst's
`DERIVATION.md` and the numericist's `NUMERIC_REPORT.md`, compare them cell by cell, and
render the addition-vs-retraction verdict for the survivor list at
`axiom-of-consent/sections/01_introduction.tex:94`.

All referee computation is in `results/regret_closed_form/referee/` (`r1`–`r7`), written
from scratch. It imports neither `core.py` (numericist) nor `analytic_checks/a00_lib.py`
(analyst), and shares no code with `independent_crosscheck.py`. Where I quote a number as
"referee", it came out of a brute-force forward simulation of the actual environment or an
exhaustive enumeration, not out of either builder's chain algebra.

---

## 0. Verdict, in one page

**RETRACTION.** The canonical $F=\sigma(1+\varepsilon)/(1+\alpha)$ is not the first-order
expansion of $\mathbb{E}[G]$ about $(\alpha,\varepsilon)=(0,0)$ under any of the three
DGPs, and no affine reparameterisation makes it one. I reach the same headline as both
builders — **but through a materially different chain of reasoning, and I overturn the
central mechanistic claim they agree on.**

Three things I confirmed, against both builders:

1. **The environment facts.** 21-point reachable lattice per coordinate (11 iff
   $\mathrm{frac}(x_0)=0$); $9261\times729$, not $1331\times729$. Verified by my own
   brute-force closure: size 21 for all 300 random $x_0$, 11 at $x_0=3.0$.
2. **The $G$ surface.** Eleven cells of `G_grid_naive_ce_cesaro.csv` reproduced by
   full-3-D forward simulation, ratios **0.995–1.079** (§3.1). The numericist's naive arm
   is sound.
3. **Negative $R^2$ for the canonical form.** Fitted as a single scaled regressor on the
   168-cell grid I get $R^2=-0.401$ (naive-CE), $-0.216$ (naive-Bayes), $-0.594$
   (Nash-CE). It fits worse than a constant. Independently reproduced.

Four things I found that **neither** builder has right:

| # | finding | consequence |
|---|---|---|
| **R-1** | The two builders' $\varepsilon$ ladders **contradict each other in sign** and neither says why. It is not a bug: the analyst's ladder is **Bayes** beliefs, the numericist's is **certainty-equivalent**. My brute force confirms both. §3.2 | The whole "$\partial F/\partial\varepsilon$" question is **belief-model-conditional**, and the two reports read as if it were not. |
| **R-2** | The corpus's **own trained IQL agents do not sit on the naive-myopic branch.** Their alignment profile is $G(\lvert\alpha\rvert{=}0.8)/G(0) = \mathbf{1.06}$ — flat — against naive's $2.84$ and stage-Nash-BR's $0.44$. §5 | Both builders' pivotal argument ("IQL is the uncoordinated family, so the naive sign is the one that counts") is **false**. Neither exact concept describes the corpus's agents; the alignment effect on the corpus's own experiment is **indistinguishable from zero**. |
| **R-3** | The analyst's proposed replacement — $\tilde\Psi=\kappa_c\lambda^2+a_0$ with $\kappa_c(0)=+0.0752$ — is a **2-parameter fit to 3 points, all at $\lambda$ the environment cannot reach.** On the reachable design the coefficient is **negative** at every $\varepsilon$ ($-0.83,-0.67,-0.51,-0.31$; $R^2\ge0.98$). §4.3 | The replacement form is right in *structure* and wrong in *sign* on the region that matters. |
| **R-4** | Stage-Nash at $\varepsilon=0$ is a correspondence with **mean 4.9–8.1 pure equilibria per state** (max 50), and $\mathbb{E}[G]$ spans **0.025 → 149** across selection rules. My enumeration reproduces the analyst's bracket *and* the numericist's BR-dynamic point. §3.4 | The numericist's "selection sensitivity ≤ 5.3%, a second-order perturbation" is **false as stated**. And the sign of $\partial G/\partial\lvert\alpha\rvert$ under "stage-Nash" **itself flips with the selection rule** — a scope condition neither builder states. |

The verdict is RETRACTION for reasons **1 and 3 below**, which survive every belief model,
every delegation concept and every DGP; reason 2 collapses on inspection.

1. **Order.** There is no first-order term in $\varepsilon$ to have a sign. The approach to
   $\varepsilon\to0^+$ is a $\Theta(\sqrt\varepsilon)$ cusp with right-derivative $+\infty$
   — confirmed by my own ladder under CE (§3.2) and present in the analyst's Bayes ladder
   too. A finite slope $\sigma$ is not an approximation to $+\infty$.
2. ~~Sign of $\varepsilon$~~ — **does not survive scrutiny as a refutation.** At the
   expansion point ($\alpha=0$) under CE beliefs the exact $G$ tracks
   $G(0^+)\cdot(1+\varepsilon)$ to **3.3–9.7%** over $\varepsilon\in[0.25,1]$; the fitted
   exponent is $q=1.12$ against a canonical $1$ (§6.1). The canonical $\varepsilon$ factor
   is *locally right at the expansion point under one belief model.* It dies elsewhere, not
   here.
3. **Shape and identification of $\alpha$.** $\mathbb{E}[G](\alpha)=\mathbb{E}[G](-\alpha)$
   is an exact distributional identity under both $\alpha$-DGPs (§4.1), so no signed
   $\alpha$ effect exists there; on the one genuinely signed axis (equicorrelated $\rho$)
   the sign is **concept-dependent, selection-dependent, and zero at every $\varepsilon>0$**
   (§4.4); and on the corpus's own agents it is **flat** (§5). "$\partial F/\partial\alpha<0$"
   is not merely wrong here — it is **not identified** by this design.

Plus the structural killer neither builder foregrounds: **separability fails at the
expansion point itself.** The fitted $\varepsilon$ exponent conditional on $\alpha$ is
$q=+1.12$ at $\alpha=0$ (CE), $q=-0.17$ at $\alpha=0.8$ (CE), $q=-0.41$ at $\alpha=0$
(Bayes). A product form $\sigma\,g(\varepsilon)\,h(\alpha)$ cannot have an
$\varepsilon$-exponent that changes sign with $\alpha$.

---

## 1. What each builder got right, and what each got wrong

| claim | analyst | numericist | referee adjudication |
|---|---|---|---|
| lattice is 21 points/coord, $9261\times729$ | ✅ | ✅ | **both right** (r-lattice: 21/21 for 300 random $x_0$) |
| welfare identity, $V_m$ cancels | ✅ | ✅ | **both right**, verified to $10^{-13}$ |
| $G\ge0$ is a theorem | ✅ | ✅ | **both right, and for the right reason** — but see §2.1 on what the benchmark presupposes |
| quantization floor interior constant $1/24$ | ✅ | ✅ | **both right** (referee MC 0.041668) |
| floor $\approx \operatorname{Var}(\mu)/2 + c$ | $c=1/48$ | $c=1/24$ | **analyst right, numericist wrong** (§3.3): at $v=1$, exact 0.520982; $v/2+1/48$ errs $-0.03\%$, $v/2+1/24$ errs $+3.97\%$ |
| $\sigma_{\rm eff}$ normalisation | derived | not used | **analyst right and it matters**: median cross-$\sigma$ spread 0.61% with $\sigma_{\rm eff}$ vs 2.60% with plain $\sigma$ |
| weight-dispersion correction $+4.69\%$ at $\sigma=0.2$ | claimed & "confirmed" | — | **analyst wrong**: measured residual excess is $+1.46\%$ on the cell they quote and $+0.43\%$ median over the grid (§3.5) |
| $\lambda$-collapse across DGPs | "an identity the DGPs satisfy" | not tested | **true but vacuous** (§4.2): all three DGPs are literally the same Gaussian family $N(0,v_\tau\Sigma(\rho))$. Verified. There is nothing to satisfy. |
| $\varepsilon$ ladder sign | decreasing | increasing | **neither is wrong; they are different belief models** and neither says so at the point of use (§3.2) |
| "CE beliefs are a clamp artefact" | claimed | — | **analyst's "CE" is not CE** — it is a *clamp-unaware* agent (`clamp_aware=False`), which is not the numericist's CE and contradicts the analyst's own declared B1 (§3.2) |
| stage-Nash selection sensitivity | correspondence, 0.27→140 | "≤5.3%, second-order" | **analyst right, numericist wrong** — but the analyst's own conclusion needs a further scope condition (§3.4) |
| $\kappa\approx2$–5, past the flip boundary | ✅ (2.3–2.9) | ✅ (2.2–4.7) | **both right** (referee 2.30–3.06 at $\lambda=1$); the numericist's 4.73 at $\varepsilon=0.01$ is an evaluation-point artefact, not a disagreement |
| trained IQL sits on the naive branch | ✅ claimed | ✅ claimed | **both wrong** (§5) |
| $\kappa_c=+0.0752$ "predicted, not fitted" | headline | — | **analyst wrong on all three words** (§4.3) |
| spec §0.1 planner magnitudes are unreproducible | — | ✅ flagged | **numericist right**: independent ceiling check gives 30-rep block mean $-1.3431\pm0.1213$, $P(>-1.014)=0.0000$ |
| canonical form has negative $R^2$ | ✅ | ✅ | **both right**, independently reproduced |

---

## 2. Audit of the analytic derivation

### 2.1 The benchmark — is it begging the question?

**Yes, in one specific and consequential way, and both builders under-state it.**

The principal-executed benchmark is *full-information, joint-control* optimisation over
$D_m\in\{-4..4\}$. Two things follow *by construction*, not by discovery:

* **$G\ge0$ is a tautology.** The delegated behaviour is a feasible policy for the
  benchmark; the benchmark optimises over a superset. Both builders state this. It is
  correct — I verified that the aggregate action set is exactly the product
  $\{-4..4\}^3$, so the planner's feasible set really does contain every delegated
  trajectory, and hence $B\equiv0$ at the ensemble level. But this means the
  **"delegation surplus" strip is excluded by the choice of benchmark, not refuted by the
  environment.** Any benchmark with a boundedly-rational or noisy principal re-opens it.
* **The entire $\varepsilon$ channel is loaded onto the delegated side.** The principal
  observes $s$ exactly; the delegates do not. So $G(\varepsilon)$ is
  $W^\ast(\text{noise-free}) - W(\varepsilon)$, a difference between a constant and a
  decreasing function. **This is the single most question-begging feature of the setup**
  and neither report says so. The corpus's Entropy Effect is a claim about the *gap*; if
  the principal's own decision rule is subject to the same $\varepsilon$ (which is what
  "the outcome the principal's own decision rule would have selected" most naturally
  means when the principal is an agent in the world), then $W^\ast$ falls with
  $\varepsilon$ too and the gap can shrink, grow or stay flat. **The decision page must
  fix this: it currently says "full-information joint control" in one clause and defines
  $G$ against "the principal's own decision rule" in another. These are different
  objects.** See §8, item D-1.

Beyond that, the benchmark is correctly implemented on both sides. The analyst runs a
$T{=}100$ backward induction *from the same $x_0$*, so the planner's travelling transient
is inside the benchmark and cancels; the numericist uses the static lattice optimum, which
is the Cesàro limit of the same planner. That is why the analyst's $T{=}100$ numbers line
up with the numericist's **Cesàro** column and sit $\approx0.50$ below their $T{=}100$
column. The reconciliation is correct and I verified the arithmetic: the numericist
measures the planner transient at $0.535$/step and the delegated chain's own transient is
$\approx0.035$/step, so the two designs differ by $\approx0.50$, as observed.

### 2.2 Is stage-Nash well-defined given noisy observations?

**The analyst's information structure is coherent but is not the only reading, and their
own two implementations of it are not the same object.**

The declared concept (`DERIVATION` §2.2) is an *interim* equilibrium: agent $i$ observes
$\hat s_i$ privately, **treats the others' realised equilibrium deltas as known**, and
best-responds under its own belief. That is internally consistent — it is a Bayesian game
in which each agent's conjecture happens to be pointwise correct — but it is a strong
assumption: the agent is uncertain about the *state* and certain about the *others'
actions*, when the others' actions are themselves functions of their own noisy signals.
The numericist implements exactly this (Gauss–Seidel over realised deltas at each signal
draw). **This is not "each agent maximises own expected $r_i$ given its noisy obs, others'
deltas at equilibrium" in the game-theoretically standard sense**, which would have agent
$i$ integrate over the *distribution* of $\delta_{-i}$ induced by the others' signals.

The analyst's second implementation, SNE, does the latter in a degenerate way
($c_i=\mathbb{E}[D_{-i}\mid x]$, a certainty-equivalent conjecture) — and it **fails to
converge at $\varepsilon\le10^{-2}$** (residual 1.6–2.7, their §11 N1). The analyst is
right that this is the coordination game asserting itself rather than a solver defect. But
the consequence is stated too weakly: the two "stage-Nash" objects in the two reports are
**different equilibrium concepts**, they agree only where both are well-posed
($\varepsilon\gtrsim0.25$), and at $\varepsilon=0$ — where the entire alignment effect
lives — the object is a *correspondence*, not a function (§3.4).

**Referee position:** stage-Nash is well-defined at $\varepsilon=0$ only as a
correspondence, and for $\varepsilon>0$ only under a declared and non-standard information
structure. No claim conditioned on "stage-Nash" is admissible without naming both.

### 2.3 Lemma 6.1 (belief irrelevance off the clamp)

**Correct, and it is the best thing in either report.** Off the clamp,
$\mathbb{E}[(s+c_i+d-\tau_i)^2\mid\hat s_i]=(\hat s_i+c_i+d-\tau_i)^2+\varepsilon$, and
$\varepsilon$ is additive and $d$-free, so the Bayes argmin and the plug-in argmin are the
same map — a unit deadband centred on $\tau_i$. I verified the algebra symbolically and the
consequence numerically (the analyst's own `a11b` L2: every Bayes/CE disagreement lies
within a small multiple of the noise SD of a wall).

**But the use they put it to is wrong.** §9.3 concludes that "the entire belief-model wedge
… collapses to ≤0.11% the moment the clamp stops binding" and hence that the CE arm's
positive $\varepsilon$ slope "is a clamp artefact". Two objections:

* The two rules compared in `a11b` differ **only** in their treatment of the clamp
  (`clamp_aware=True` vs `False`). That they agree when the clamp is slack is a
  definition, not a finding. It carries no information about the belief model.
* The clamp **is in the environment**. Half of all resource optima sit on the lower wall
  (verified: $\mu$ is symmetric about 0 under every DGP, so $P(\mu<0)=1/2$). Showing that
  a phenomenon vanishes when you translate the targets +5 into the middle of the box shows
  that it is a boundary phenomenon *of this environment*, not that it is an artefact.

### 2.4 The channel decomposition and the floor

The three-channel identity
$G/\bar W=\operatorname{Var}_\pi(x)+b^2+2qb-\operatorname{Var}_{\rm plan}(x)$ is algebra and
is correct. The observation that **the quantization floor $\bar Wq^2$ cancels identically**
is correct and is the cleanest single kill on the "irreducible $\sigma/2$ baseline".

The floor constant: **the analyst is right and the numericist is wrong.** $1/24$ is the
*interior* constant; the interior branch carries only the mass $\mu>0$, i.e. probability
$1/2$. Referee MC, $8\times10^6$ draws per row, exact nearest-lattice-point search:

| $\operatorname{Var}\mu$ | exact | $v/2+1/48$ (analyst) | err | $v/2+1/24$ (numericist) | err |
|---|---|---|---|---|---|
| 0.0625 | 0.046108 | 0.052083 | $+12.96\%$ | 0.072917 | $+58.14\%$ |
| 0.25 | 0.145651 | 0.145833 | $\mathbf{+0.13\%}$ | 0.166667 | $+14.43\%$ |
| 1.0 | 0.520982 | 0.520833 | $\mathbf{-0.03\%}$ | 0.541667 | $+3.97\%$ |
| 1.3333 | 0.686849 | 0.687483 | $\mathbf{+0.09\%}$ | 0.708317 | $+3.13\%$ |

Interior residual measures $0.0201$–$0.0209$ against $1/48=0.020833$. The analyst's
small-$v$ branch $v-\tfrac43\sqrt{2/\pi}\,v^{3/2}$ is accurate to $0.5\%$ for
$v\le0.0625$ and diverges above $v\approx0.1$, exactly as they claim. Interior-only check
($\mu\sim U(2,8)$): $0.041668\pm0.000017$ vs $1/24=0.041667$. ✅

Note this dispute has **no verdict consequence** — the floor cancels in $G$ — but it is the
one place where the analyst caught a real error in the numericist's algebra, and the
numericist had noticed the symptom ("it over-predicts by 4.4–16%") while mis-diagnosing
the cause.

---

## 3. Audit of the numericist's code, with independent recomputation

I read `core.py`, `run_grid.py`, `validate_lattice.py`, `eps_cusp.py` and
`nash_diagnostics.py` in full, then recomputed rather than trusted.

### 3.1 Eleven cells recomputed by brute-force forward simulation

`referee/r2_bruteforce.py` draws preferences, resets $x_0\sim U[0,10]^3$, gives each agent
$\hat s_i=s+N(0,\varepsilon I)$, has it argmin its own clamped expected loss over
$\{-1,0,1\}^3$ with the others treated as static, steps
$s'=\mathrm{clip}(s+\sum\delta_i)$, pays $W(s')$, and time-averages over $T=2200$ with 400
burn-in. Benchmark = the static ensemble-optimal point of the reachable lattice. 1500
draws, entirely different random numbers from `run_grid.py`.

| family | control | $\sigma$ | $\varepsilon$ | referee | `core.py` Cesàro | ratio |
|---|---|---|---|---|---|---|
| implemented | 0.0 | 0.6 | 0 | $2.4202\pm0.0779$ | $2.2430\pm0.0888$ | 1.079 |
| implemented | 0.0 | 0.6 | 0.25 | $2.4804\pm0.0530$ | $2.4313\pm0.0613$ | 1.020 |
| implemented | 0.0 | 0.6 | 1.0 | $4.1417\pm0.0414$ | $4.1291\pm0.0496$ | 1.003 |
| implemented | 0.8 | 0.6 | 0 | $7.1404\pm0.1992$ | $7.1227\pm0.2476$ | 1.003 |
| implemented | −0.8 | 0.6 | 0.25 | $6.3570\pm0.1436$ | $6.3883\pm0.1780$ | 0.995 |
| renormalised | 0.8 | 0.6 | 0.25 | $7.5005\pm0.1591$ | $7.4050\pm0.1961$ | 1.013 |
| equicorr | −0.33 | 0.6 | 0 | $0.8479\pm0.0288$ | $0.8259\pm0.0351$ | 1.027 |
| equicorr | 0.8 | 0.6 | 1.0 | $5.9845\pm0.0905$ | $5.9521\pm0.1091$ | 1.005 |
| equicorr | 0.4 | 0.6 | 0.5 | $4.3692\pm0.0823$ | $4.3226\pm0.0994$ | 1.011 |
| implemented | 0.0 | 0.2 | 0.25 | $0.8375\pm0.0178$ | $0.8223\pm0.0207$ | 1.019 |
| implemented | 0.0 | 1.0 | 0.25 | $4.0489\pm0.0869$ | $3.9645\pm0.1001$ | 1.021 |

Ten of eleven within $1.5$ combined SE. The one outlier is $\varepsilon=0$, exactly the
cell the numericist themselves flag as tie-convention-dependent. **The naive-myopic arm of
`core.py` is confirmed.** Bayes arm spot-checks: 1.036, 0.946, 1.334 (the last on
$G\approx0.01$, i.e. noise on a number both agree is $\approx0$).

$G\ge0$: `frac_neg` is 0 at every $\varepsilon>0$; at $\varepsilon=0$ my simulator returns
$-3\times10^{-13}$ minima on 10–19% of draws — exact ties at machine precision, i.e. $G=0$,
not $G<0$. Consistent with both builders.

### 3.2 ⚠️ THE CONTRADICTION, AND ITS RESOLUTION

The two reports state flatly opposite $\varepsilon$ ladders for the *same* cell
(implemented $\alpha=0$, $\sigma=0.6$, naive-myopic):

| | $\varepsilon=0$ | $10^{-10}$ | $10^{-2}$ | $10^{-1}$ |
|---|---|---|---|---|
| numericist `EPS_CUSP.csv` | 2.3269 | 1.8805 | 2.0827 | 2.1674 |
| analyst `a13_results.json` | 2.3095 | 1.4647 | 1.4127 | 1.1091 |

They agree at $\varepsilon=0$ and diverge by 28% at $0^+$ and by a **factor of 2** at
$\varepsilon=0.1$, with opposite signs of $\partial G/\partial\varepsilon$. Referee ladder
(brute force, same code path that reproduced §3.1):

| $\varepsilon$ | 0 | $10^{-10}$ | $10^{-6}$ | $10^{-4}$ | $10^{-3}$ | $4{\cdot}10^{-3}$ | $10^{-2}$ | $4{\cdot}10^{-2}$ | 0.1 | 0.25 |
|---|---|---|---|---|---|---|---|---|---|---|
| referee | 2.3233 | 1.9078 | 1.9093 | 1.9280 | 1.9948 | 2.0440 | 2.0790 | 2.1078 | 2.1599 | 2.4178 |

**The referee ladder tracks the numericist to $\le1.5\%$ at every point.**

**Resolution: it is not a bug in either. `a00_lib`'s "naive" protocol is the *Bayes* belief
model, not CE.** I ran the analyst's own library directly:

```
a00_lib naive, clamp_aware=True  (default)  eps=0/0.25/1.0 : 2.2984 / 0.8956 / 0.6907
a00_lib naive, clamp_aware=False            eps=0/0.25/1.0 : 2.3477 / 3.4804 / 5.1412
core.py naive_bayes  (numericist)           eps=0/0.25/1.0 : 2.2430 / 0.9019 / 0.7386
core.py naive_ce     (numericist)           eps=0/0.25/1.0 : 2.2430 / 2.4313 / 4.1291
referee brute force, CE                     eps=0/0.25/1.0 : 2.4202 / 2.4804 / 4.1417
referee brute force, Bayes                  eps=  /0.25/1.0 :        / 0.9345 / 0.6986
```

Three distinct rules are in circulation, and the reports use different ones without
flagging it at the point of use:

1. **Bayes** — integrates the posterior through the clamp. Analyst's default; numericist's
   `naive_bayes`. $G$ **falls** in $\varepsilon$.
2. **CE-with-clip** — plug-in point estimate, clamp applied. Numericist's `naive_ce`; my
   brute force. $G$ **rises** in $\varepsilon$ at $\alpha=0$.
3. **Interior deadband, clamp-unaware** (`clamp_aware=False`). The analyst's §9.3 "CE". Not
   a belief model at all.

**Charges:**
* Against the **analyst**: `DERIVATION` §7.4's ladder *is* labelled "naive-Bayes" in the
  memo, but the executive summary, the verdict table
  ("$c_2\in[-27.3,-4.7]$", "$\varepsilon$ sign ❌") and the §10 disposition all present
  the Bayes result as unconditional; and §9.3's dismissal of the positive-slope
  configuration compares against **rule 3**, which is not the numericist's CE and
  contradicts the analyst's own assumption B1 ("the agent … knows the clamp"). Rule 3
  inflates the CE arm by $\approx24\%$ at $\varepsilon=1.0$ (5.139 vs 4.129).
* Against the **numericist**: their headline grid, their §1.5 table, their §2.2 cusp
  analysis and their §2.4 verdict table are all **CE-primary**, with Bayes relegated to
  §3.3. They do say "any closed form for $G$ that does not name a belief model is
  under-specified by a factor of two" — which is exactly right and should have been the
  headline, not a §3.3 aside.

**Which rule is environment-faithful?** Neither builder asks. IQL learns
$Q(\text{obs},a)\approx\mathbb{E}[r+\gamma\max Q'\mid \text{obs},a]$ — an expectation over
the posterior on the true state, taken through the clamp. That is **rule 1 (Bayes)**, not
rule 2 and certainly not rule 3. The analyst's default is the right one; their argument for
it is the wrong one.

**What survives both rules** (referee-verified): the downward jump at $\varepsilon=0$
exactly ($-17.9\%$ CE, $-36.6\%$ Bayes); the $\sqrt\varepsilon$ cusp with $c_1>0$; and
non-monotonicity in $\varepsilon$ (my equicorr $\rho=0.8$ CE ladder peaks at
$\varepsilon\approx4\times10^{-3}$ then falls: $6.85\to7.24\to6.62\to6.22$).

### 3.3 The floor formula

Settled in §2.4 above: analyst right, numericist wrong, no verdict consequence.

### 3.4 Stage-Nash at $\varepsilon=0$ — the one arm nobody replicated

The numericist admits this is the open item ("the iterated-best-response solver exists only
in `core.py`"). I wrote an independent exhaustive enumerator: all $3^4=81$ profiles, weak
pure-NE test (no agent strictly gains) on the actual clamped post-state, five selection
rules, deterministic chain, Cesàro average. 4000 draws × 3 resources, $\sigma=0.6$.

| family | control | welfare-best | **BR-from-zero** | maximal-move | minimal-move | welfare-worst | mean #NE | max #NE |
|---|---|---|---|---|---|---|---|---|
| implemented | 0.0 | 0.207 | **1.083** | 0.771 | 68.89 | 141.30 | 5.47 | 50 |
| implemented | 0.8 | 0.091 | **0.108** | 1.901 | 67.95 | 141.92 | 8.12 | 50 |
| equicorr | −0.33 | 0.025 | **1.930** | 0.026 | 74.83 | 148.75 | 4.89 | 50 |
| equicorr | 0.8 | 0.162 | **0.250** | 2.267 | 73.64 | 149.25 | 7.11 | 50 |

Compare: numericist's `G_grid_nash_ce_cesaro.csv` at these cells gives **1.146, 0.102,
1.952, 0.241** — my BR-from-zero column, to 3–6%. Analyst's §6.4 gives welfare-best 0.265 /
0.065 / 0.107 and minimal 72.91 / 73.16 / 71.64 and worst 140.5 / 143.6 / 144.0 — my
corresponding columns, to a few percent.

**Both builders' numbers are correct; the numericist's *interpretation* is not.**
"Equilibrium selection is a second-order perturbation, not a driver of any result here"
describes only the basin of the BR dynamic. The correspondence spans a factor of $\sim6000$.
Their NE census also under-counts: `stage_nash_pure_ne_census` tests whether each agent's
*tie-broken* best response reproduces its action, which counts strict-NE-under-a-tie-rule
(mean 1.29–3.49), not weak NE (mean 4.89–8.12, my count, textbook definition).

**And a scope condition neither builder states.** Read the sign of
$\partial G/\partial\lvert\alpha\rvert$ *down the columns*: welfare-best $0.207\to0.091$
(**negative**); BR-from-zero $1.083\to0.108$ (**negative**); minimal-move $68.9\to68.0$
(**flat**); maximal-move $0.771\to1.901$ (**positive**); welfare-worst $141.3\to141.9$
(**flat/positive**). **The canonical alignment sign is recoverable under stage-Nash only
for a subset of selection rules**, and the corpus has never specified one. The analyst
gestures at this in §10.1; it deserves to be the headline of that arm.

### 3.5 Smaller code findings

* **`run_grid.py::cell_metrics` benchmark.** The "planner" is the static lattice optimum,
  not a backward induction. Correct for the Cesàro column (the planner reaches it in ≤3 of
  100 steps and stays), and the numericist says so and quantifies the $T{=}100$ bias at
  $0.535$/step. No error, but the $T{=}100$ tables in that file are **not** a delegation
  gap and should carry a warning label wherever they are used.
* **`lattices(f)` / `TIDX`.** The $f$-independence of the transition index is asserted for
  4 values of $f$ and is in fact structural. Verified.
* **Weight-dispersion correction.** The analyst's §7.5 claims "both codebases see the same
  $+4.5\%$ excess against a derived $+4.7\%$". The numericist's grid at the cell they name
  (implemented $\alpha=0$, $\varepsilon=0.25$, CE) gives $G(0.2)/G(0.6)=0.3382$, i.e.
  $+1.46\%$ excess; median over all 56 groups is $+0.43\%$. The analyst's $0.3455$ is the
  **Bayes** grid (confirmed: `naive_bayes` gives exactly 0.3455). So the comparison is
  Bayes-to-Bayes and internally consistent, but the derived $+4.69\%$ correction
  **over-predicts by $\approx3\times$ on CE and is not a general property of the surface.**
* **Stale claim.** `NUMERIC_REPORT.md` says "`a10_crosscheck.py` … was never run". True at
  its 20:29 write time; `a10_out.txt` is dated 20:34. Cosmetic.

---

## 4. The analytic formula against the numeric surface

`referee/r4_surface.py`, on all 168 cells × 3 arms of the numericist's Cesàro grid.

### 4.1 What is an identity and what is not

* **$\mathbb{E}[G](\alpha)=\mathbb{E}[G](-\alpha)$** is an identity, provable in one line:
  $\tau_i(-\alpha;-b,\xi)=\tau_i(\alpha;b,\xi)$ pointwise and $b$ is symmetric. Both
  builders report `max_abs_tau_diff = 0.0`. ✅ No dispute.
* **$\rho(\alpha)=\alpha^2/(\alpha^2+(1-|\alpha|)^2)=\alpha^2+O(|\alpha|^3)$**, so the
  manipulated correlation is *quadratic* and the first-order $\alpha$ coefficient of any
  smooth functional of $\rho$ is exactly 0. ✅ Both right.
* **$\sigma$-homogeneity.** Median cross-$\sigma$ spread of $G/\sigma_{\rm eff}$: 0.61%
  (naive-CE), 1.16% (naive-Bayes), 3.54% (Nash-CE); with plain $\sigma$, 2.60%. $\sigma_{\rm
  eff}$ earns its keep. But it is a **units identity**: the delegated policy is provably
  $w$-free, so any reward-unit DV is degree-1 homogeneous in $\sigma$ by construction. Both
  builders say this; so does `ideation_measurement.md` §0.3, which said it first.

### 4.2 The "$\lambda$-collapse" is vacuous

The analyst presents the collapse of the three DGPs onto $(\lambda,v_\tau)$ as "not a fit;
it is an identity the DGPs satisfy". It is weaker than that: **all three DGPs are literally
the same two-parameter Gaussian family.** Verified (`r6` C5, 400 000 draws):

| $\alpha$ | DGP | measured Var | measured corr | theory |
|---|---|---|---|---|
| 0.8 | implemented | 0.6815 | 0.9402 | $v_\tau=0.68$, $\rho=0.9412$ |
| 0.8 | renormalised | 0.9989 | 0.9410 | $v_\tau=1$, same $\rho$ |
| 0.8 | equicorr($\rho{=}0.9412$) | 0.9998 | 0.9410 | identical law |

$\tau\sim N(0,\,v_\tau[(1-\rho)I+\rho\mathbf{1}\mathbf{1}'])$ in every case. Renormalised
$\alpha$ and equicorr $\rho(\alpha)$ are **the same distribution**, so their agreement
tests code, not physics. Direct brute-force check at exactly matched $(\rho,v_\tau)$
(`r6` C1; my earlier grid-interpolation version was invalid because `np.interp` clamps at
$\rho=0.8$): ratios **1.007–1.053** across $\alpha\in\{0.4,0.8\}$, $\varepsilon\in\{0,0.25,1\}$. ✅

The **non**-vacuous content, which the analyst has and the numericist does not, is that
$\Psi$ needs *both* $\lambda$ and $v_\tau$: the lattice spacing is 1 and the box is $[0,10]$,
so the problem is not scale-invariant. This matters because **the implemented $\alpha$ knob
moves both**: $\lambda=1-|\alpha|$ *and* $v_\tau=\alpha^2+(1-|\alpha|)^2\in\{1,0.52,0.68\}$.
The $\alpha$ axis is a two-dimensional path through $(\lambda,v_\tau)$, non-monotone in
$v_\tau$. (`ideation_measurement.md` §0.3 flagged the variance confound; neither builder
carries it into the verdict framing.)

### 4.3 ⚠️ The $\lambda^2$ contention law does not describe the design

The analyst's proposed replacement is
$\tilde\Psi(\lambda;\varepsilon)=\kappa_c(\varepsilon)\lambda^2+a_0(\varepsilon)$ with
$\kappa_c(\text{naive},0)=+0.0752$ "predicted, not fitted" by the $n=4$ order-statistic
bracket $[0.0482,0.1986]$.

Reading `a16b_contention_fit.py` and `a16b_out.txt`: the fit is restricted to
$\lambda>\lambda^\ast$, and $\lambda^\ast(\varepsilon{=}0)=1.60$. The window is therefore
$\lambda\in\{2.00,2.50,3.00\}$ — **three points, two free parameters, one residual degree of
freedom**, and every point lies **outside the reachable design** (equicorrelated PSD cap
$\lambda\le\sqrt{n/(n-1)}=1.1547$; implemented $\lambda=1-|\alpha|\le1$). The
$\varepsilon=0.01$ fit adds $\lambda=1.60$ — also outside. The bracket itself spans a factor
of 4.1, so landing inside it is weak evidence. And at the two $\varepsilon$ values whose
windows do reach into the design, $\kappa_c$ falls **outside** the bracket (the analyst's own
table says "no").

So: *fitted*, not predicted; on three points; outside the environment. Separately, the
$\lambda$ sweep holds the location scale at $\gamma=0.5$ for all $\lambda$, whereas the
reachable family traces the curve $\gamma(\lambda)=\sqrt{v_\tau-\tfrac34\lambda^2}$ — the
two coincide only at $\lambda=1$. So "$\lambda^\ast=1.60$ lies above the design range" is a
statement about the wrong slice of the $(\lambda,\gamma)$ plane.

**What the design actually does.** Direct fit of $\tilde\Psi=\kappa_c\lambda^2+a_0$ on the
unit-variance DGP cells of the numericist's grid, $\sigma=0.6$, 9 cells per $\varepsilon$:

| arm | $\varepsilon=0$ | 0.25 | 0.5 | 1.0 |
|---|---|---|---|---|
| naive-CE $\kappa_c$ | **−0.827** | −0.673 | −0.514 | −0.313 |
| naive-CE $R^2$ | 0.991 | 0.980 | 0.979 | 0.981 |
| naive-Bayes $\kappa_c$ | **−0.827** | −0.642 | −0.482 | −0.329 |
| Nash-CE $\kappa_c$ | **+0.177** | −0.003 | +0.002 | +0.006 |
| Nash-CE $R^2$ | 0.942 | **0.021** | 0.018 | 0.276 |

The design region is well described by $\tilde\Psi\approx a_0(\varepsilon)-|\kappa_c|\lambda^2$
with $R^2\ge0.98$ under naive — a **negative** coefficient, opposite in sign to the
analyst's headline, and $\approx11\times$ larger in magnitude. Under Nash the coefficient is
positive at $\varepsilon=0$ and **statistically zero at every $\varepsilon>0$** ($R^2=0.02$).

The analyst's §7.7 *does* say the design sits on the decreasing branch. The problem is that
their §0 executive summary and §10.2 "what a replacement must look like" present
$\kappa_c=+0.0752$ as the law of the environment. **A reader implementing that replacement
would get the alignment sign backwards.**

### 4.4 Full-surface head-to-head, referee's own fit

168 cells, Cesàro, per arm. $k$ = free coefficients + 1; AICc on cell means.

| model | $k$ | $R^2$ naive-CE | $R^2$ naive-Bayes | $R^2$ Nash-CE |
|---|---|---|---|---|
| $mn\sigma_{\rm eff}\,(a_0(\varepsilon)+\kappa_c(\varepsilon)\lambda^2)$ | 9 | **0.932** | **0.896** | **0.969** |
| $c\,\sigma(1+\varepsilon)/(1+\alpha)$ | 2 | **−0.401** | **−0.216** | **−0.594** |
| $A+B\,\sigma(1+\varepsilon)/(1+\alpha)$ | 3 | 0.264 | 0.098 | 0.051 |
| constant | 2 | 0 | 0 | 0 |
| $c\,mn\sigma_{\rm eff}$ (scale only) | 2 | 0.597 | 0.264 | 0.402 |

$\Delta$AICc for the canonical single regressor against the best model: **+494, +398,
+648**. The canonical form, fitted as intended, is **worse than a constant in all three
arms** — independently reproducing both builders' central fit result on a third codebase.

Note the row that neither builder reports: **$c\,mn\sigma_{\rm eff}$ alone — the pure units
identity, no $\alpha$, no $\varepsilon$ — reaches $R^2=0.60$.** The canonical form's only
correct factor does 60% of the work; adding its two "substantive" factors takes $R^2$ to
$-0.40$. That is the cleanest single statement of the failure.

---

## 5. ⚠️ The finding that overturns both reports: what the trained agents actually do

Both builders conclude that the naive-myopic arm is the environment-faithful one, and both
rest that on the same three-row table in `ideation_measurement.md` §0.1 — the table whose
planner column the numericist independently showed to be wrong (§0.3; I confirm: the
30-replication block-mean ceiling at $\alpha=0,\sigma=0.6$ is $-1.3431\pm0.1213$ per agent,
$P(\text{ceiling}>-1.014)=0.0000$, so the spec's $-1.014$ is unattainable).

The *realised* column of that table is fine — I pulled it from
`results/gpu_factorial_postfix/replication_results.csv` (3750 rows, 30 reps/condition) and
the values reproduce, including the striking coincidence $-2.286709$ at $(\alpha{=}0,
\sigma{=}0.6)$ vs $-2.286752$ at $(\alpha{=}0.8,\sigma{=}1.0)$. But three rows is not a
test. The full $\alpha$ grid is on disk, so I ran it (`referee/r7_iql.py`): realised
**ensemble** welfare, $\sigma=0.6$, $\varepsilon=0$, $T=100$ episode average, against my
own exact-BR delegates on the same DGP.

| $\alpha$ | IQL $\times4$ (se) | naive-CE | naive-Bayes | Nash BR-zero | Nash best |
|---|---|---|---|---|---|
| −0.8 | −5.238 (0.648) | −10.017 | −10.017 | −3.604 | −3.585 |
| −0.4 | −4.968 (0.407) | −7.005 | −7.005 | −4.116 | −3.822 |
| 0.0 | −9.147 (0.738) | −9.263 | −9.263 | −8.501 | −7.492 |
| +0.4 | −4.563 (0.377) | −6.801 | −6.801 | −4.095 | −3.791 |
| +0.8 | −5.557 (0.887) | −10.297 | −10.297 | −3.525 | −3.507 |

And on a common benchmark (static ensemble-optimal lattice point, same draws):

| $\alpha$ | $W^\ast$ | $G$ IQL | $G$ naive | $G$ Nash-BR | $G$ Nash-best |
|---|---|---|---|---|---|
| −0.8 | −2.704 | 2.534 | 7.313 | 0.900 | 0.881 |
| −0.4 | −3.089 | 1.880 | 3.917 | 1.028 | 0.733 |
| 0.0 | −6.660 | 2.487 | 2.603 | 1.841 | 0.831 |
| +0.4 | −3.075 | 1.489 | 3.727 | 1.020 | 0.716 |
| +0.8 | −2.810 | 2.747 | 7.487 | 0.715 | 0.697 |

$$G(|\alpha|{=}0.8)\,/\,G(0):\quad \textbf{IQL } 1.06 \quad\text{vs}\quad \text{naive } 2.84 \quad\text{vs}\quad \text{Nash-BR } 0.44$$

**The trained agents' delegation gap is flat in alignment.** They are bracketed by the two
exact concepts at every $\alpha$ — the analyst's weak claim survives — but they emphatically
do **not** "sit on the naive-myopic branch". They coincide with naive only at $\alpha=0$
(where naive and Nash are closest anyway, and where both builders anchored), and at the
wings naive is 4.5–4.7 welfare units worse than IQL while Nash is within 1.6–2.0.

**Mechanism.** "Naive-myopic" here means *argmax the one-step reward*. IQL maximises the
$\gamma=0.99$ discounted return over 100-step episodes. A return-maximising independent
learner learns not to overshoot — it does not need joint-action conditioning to avoid the
2-cycle, only a horizon. So the inference "IQL trains independent Q-networks ⟹ IQL is the
uncoordinated one-shot family" is a **non sequitur**, and it is the load-bearing step in
both reports' alignment sections.

**Consequence for the verdict.** On the corpus's own experiment, the alignment effect on the
delegation gap is **not distinguishable from zero** (ratio $1.06$, with the IQL SEs the
interval comfortably excludes both $2.84$ and $0.44$). Both the "naive says $+$" and the
"Nash says $-$" results are properties of idealised concepts, not of the system the papers
ran. $\partial F/\partial\alpha$ is **unidentified**, which is a stronger and more honest
retraction ground than "the sign is positive".

---

## 6. Deliverable 2, answered exactly

> Is $\sigma(1+\varepsilon)/(1+\alpha)$, up to affine reparameterisation, the first-order
> expansion of $\mathbb{E}[G]$ about $(\alpha,\varepsilon)=(0,0)$ at fixed $\sigma$?

**No.** Factor by factor, with the steelman stated before the refutation.

### 6.1 The $\varepsilon$ factor — steelmanned, then killed

**Steelman (neither builder offers this).** At $\alpha=0$ — *the expansion point* — under
CE beliefs, the canonical shape is close:

| $\varepsilon$ | exact $G$ | $G(0^+)\cdot(1+\varepsilon)$ | ratio |
|---|---|---|---|
| 0.25 | 2.4313 | 2.3536 | 1.033 |
| 0.50 | 3.0496 | 2.8243 | 1.080 |
| 1.00 | 4.1291 | 3.7658 | 1.097 |

Fitted exponent $q$ in $G\propto(1+\varepsilon)^q$ at $\alpha=0$, CE: **$q=1.12$** against a
canonical $1$. This is the best case for ADDITION and it must be conceded.

**It dies anyway, on four independent grounds:**

1. **Order.** Below $\varepsilon\approx10^{-2}$ the behaviour is $\Theta(\sqrt\varepsilon)$,
   not linear. Referee ladder: $\Delta G$ from $10^{-10}$ to $10^{-4}$ is $+0.0202$ and from
   $10^{-10}$ to $10^{-2}$ is $+0.1712$ — a factor of 8.5 for a factor of 100 in
   $\varepsilon$ ($\sqrt{}$ predicts 10, linear predicts 100). The right-derivative at
   $0^+$ is $+\infty$. **There is no first-order coefficient to compare to $\sigma$.** This
   is the *only* refutation that bites exactly at the expansion point, and it is decisive
   there.
2. **A knife-edge at $\varepsilon=0$ itself.** $\mathbb{E}[G](0)\ne\lim_{\varepsilon\to0^+}$:
   referee measures $2.3233\to1.9078$, a $-17.9\%$ drop (Bayes: $-36.6\%$). The lower clamp
   is simultaneously a lattice point and a decision threshold, so at $\varepsilon=0$ exactly
   a tie convention is load-bearing. An expansion "about $(0,0)$" is being taken about a
   point where the function is not continuous.
3. **Belief model.** Under Bayes — the model IQL actually implements — the same fit gives
   $q=\mathbf{-0.41}$. Sign reversed.
4. **Non-separability at the expansion point.** At $\alpha=0.8$ under CE, $q=\mathbf{-0.17}$.
   A product form cannot have an $\varepsilon$-exponent that changes sign with $\alpha$.

### 6.2 The $\alpha$ factor

* **Shape: fails.** $\mathbb{E}[G](\alpha)=\mathbb{E}[G](-\alpha)$ identically under both
  $\alpha$-DGPs. $1/(1+\alpha)$ is not even, so it cannot be a first-order expansion of an
  even function. And $\rho(\alpha)=O(\alpha^2)$, so the first-order coefficient is exactly 0
  under the renormalised DGP and is a *kink* with symmetric one-sided derivatives under the
  implemented one — against a canonical $-\sigma=-2.22$.
* **Sign, on the one signed axis: not identified.** Under equicorrelated $\rho$: naive gives
  $\partial G/\partial\rho>0$ (canonical says $<0$); stage-Nash-BR gives $<0$; stage-Nash
  under maximal-move selection gives $>0$; and at every $\varepsilon>0$ under Nash the
  effect is statistically zero ($R^2=0.02$, §4.3). On the corpus's own agents it is flat
  (§5).
* **Confound.** The implemented knob moves $\rho$ and $v_\tau$ together, and $v_\tau$ is
  non-monotone in $|\alpha|$ ($1,0.52,0.68$). Any $\alpha$ effect on this grid is a
  compound. (Flagged in `dgp.py` and `ideation_measurement.md` §0.3; not carried into
  either verdict.)

### 6.3 The $\sigma$ factor

**Exact, and empty.** Degree-1 homogeneity holds to 0.61% median across the grid once
$\sigma_{\rm eff}$ absorbs the clip at 1. But the delegated policy is provably $w$-free
(a positive scalar cannot move an argmin, and the per-resource argmins assemble), so any
reward-unit DV is degree-1 homogeneous by construction. It is a **normalisation, not a
comparative static**, and $\sigma_{\rm eff}$ alone reaches $R^2=0.60$ on the surface.

### 6.4 Affine reparameterisation

$A+B\,\sigma(1+\varepsilon)/(1+\alpha)$ cannot manufacture a $\sqrt\varepsilon$ cusp from a
linear term; cannot make a smooth function of $\alpha$ equal a function of $|\alpha|$;
cannot produce an $\varepsilon$-slope that flips sign with $\alpha$; and needs $B>0$ for the
$\sigma$ scale while needing $B<0$ for the Nash $\alpha$ sign. Measured: $R^2=0.264/0.098/0.051$
across the three arms, $\Delta$AICc $+388/+350/+563$.

---

## 7. Disposition of the survivor list at `01_introduction.tex:94`

The paragraph makes six adjudicable claims. Verdicts:

| # | text | verdict | ground |
|---|---|---|---|
| **1** | "supporting the directional roles of stakes, **alignment**, and entropy" | **WITHDRAW (alignment); RETAIN-AS-NORMALISATION (stakes); WITHDRAW (entropy)** | alignment: unidentified (§5, §6.2) — flat on the corpus's own agents, sign flips with concept *and* with equilibrium selection, and is zero at every $\varepsilon>0$ under Nash. stakes: exact but a units identity. entropy: no first-order term (§6.1). |
| **2** | "and the signed alignment denominator over the cooperative range" | **WITHDRAW** | $\mathbb{E}[G](\alpha)=\mathbb{E}[G](-\alpha)$ is an exact identity under both $\alpha$-DGPs, so nothing on this design is "signed" at all. On the one genuinely signed axis the sign is concept- and selection-dependent. The phrase "over the cooperative range" cannot rescue it: on $\alpha\in[0,0.8]$ under naive, $G$ *rises* 2.8×. |
| **3** | "zero friction if and only if zero stakes" | **RETAIN, with a replacement proof** | $G=\sum_m\bar W_m\Delta_m$ with $\Delta_m\ge0$ and $\bar W\propto\sigma$; $\Delta>0$ in every cell-draw computed by three independent codebases. So $\sigma=0\Rightarrow F=0$ and here $\sigma>0\Rightarrow F>0$. But it survives as a **quantization/coordination fact about $\{-1,0,1\}$ against a unit deadband**, not as a corollary of $(1+\varepsilon)/(1+\alpha)>0$. The proof at `02_axiom.tex:179–183` must be replaced, not merely re-cited. |
| **4** | "monotonicity in stakes and entropy" | **SPLIT: stakes RETAIN (as normalisation); entropy WITHDRAW** | stakes: exact, empty. entropy: non-monotone under every belief model — my CE ladder at equicorr $\rho=0.8$ rises to $\varepsilon\approx4\times10^{-3}$ then falls; under Bayes $G$ falls monotonically; under Nash $G$ falls then rises. No single-signed $\partial F/\partial\varepsilon$ exists. |
| **5** | "the irreducible $\sigma/2$ delegation baseline" | **WITHDRAW** | The object that plays this role is the quantization floor. It is large (mean 1.88, up to 6.04 welfare units), it is now in closed form to $0.25\%$ ($\operatorname{Var}\mu/2+1/48$), it is **not** proportional to $\sigma/2$ (it depends on alignment and is *exactly zero* at $\rho=-1/3$), and it **cancels identically** in any difference-against-benchmark. |
| **6** | "the selection-based bridge from friction to persistence" | **RETAIN, unconditionally** | Depends only on $F\ge0$ and comparability. $F\ge0$ is here a theorem, stronger than the paper claims. Unaffected by everything above. |
| — | "what does not survive … the divergence as $\alpha\to-1$" | **already withdrawn; strengthen** | It is not merely that the divergence fails: $\alpha<0$ was never adversarial on this design ($\rho(\alpha)=\rho(-\alpha)\ge0$), so the $\alpha\to-1$ limit was never being tested. |

**Net: two of six survive, one splits, three fall. The paragraph must be rewritten, not
patched** — "what survives the test is denominator-independent: zero friction iff zero
stakes, monotonicity in stakes and entropy, the irreducible $\sigma/2$ baseline, and the
selection bridge" retains only its first and last items, and the first needs a new proof.

Also consequential and not on the list: **Prop. Modular Governance Fragility** and the
**additive Aggregation** result (`App. principal_agent:181–196`) are algebraic consequences
of the failed product form and fall with it; and the **"12/12 Lean mapping"** is unaffected
in truth-value but three theorems now map to withdrawn propositions, so the *claim of a
12/12 match* must be withdrawn even though no Lean proof breaks.

---

## 8. Required changes to `notes/friction-sign-decision-2026-07-30.md`

**D-1 — the benchmark definition is internally inconsistent, and it is load-bearing.**
The page defines $G := W^\ast(\text{principal-executed}) - W(\text{realised})$ where
principal-executed is glossed both as "the outcome the principal's **own decision rule**
would have selected" (§ The decision) and as "**full-information** joint control maximising
ensemble welfare". These are different objects and the difference is exactly where the
Entropy Effect lives (§2.1). **MF must pick one.** If full-information: $G\ge0$ is a
theorem, $B\equiv0$ at the ensemble level, the novelty strip is dead at the ensemble level
by construction, and $\partial F/\partial\varepsilon$ is a statement about the delegated
side only. If own-decision-rule-under-the-same-$\varepsilon$: the sign of $G$ becomes an
empirical question again, the novelty strip revives, and every result in both memos would
have to be recomputed. **Recommendation: keep full-information, and state explicitly that
the Entropy Effect is thereby a claim about delegated loss net of a noise-free benchmark.**

**D-2 — the sequencing item is now discharged, with a correction.** Step 2 says "exact
backward induction on the $11^3$ lattice". The reachable set is $21^3=9261$ states almost
surely (11 only when $\mathrm{frac}(x_0)=0$, a null event under the environment's
$U[0,10]^3$ reset). Verified independently three ways. The correction makes the computation
*larger*, not approximate — **there is no discretisation error**. Amend the text and record
the result: **RETRACTION**, so no AoC v5 text, MARL discussion sentence or E&P transfer
material may assume the canonical form.

**D-3 — "Still die" list: two additions and one re-typing.**
* *Add*: **Prop. Alignment Effect** moves from "need reproof" to **"dies"** — but for the
  *identification* reason (§5), not the "wrong sign" reason. Write it as: *the sign of the
  alignment effect is a property of the delegation solution concept and, within stage-Nash,
  of an unspecified equilibrium selection rule; on the corpus's own trained agents it is
  flat. It is not identified by this design.*
* *Add*: **the $\alpha<0$ arm was never adversarial.** Any text describing $\alpha\in[-1,0)$
  as opposition must go; $\rho(\alpha)=\rho(-\alpha)$.
* *Re-type*: **Prop. Entropy Effect** — the page says "rescope to single-principal, cite
  Morris–Shin/Prat for the coordination counterexample". Too gentle. There is **no
  first-order $\varepsilon$ term at all**; the coordination counterexample is not a corner,
  it is the modal case; and the sign flips with the belief model. Rescoping cannot fix an
  expansion that does not exist.

**D-4 — the "Need reproof" list: Threshold Discontinuity is now *cheap*, not expensive.**
The page calls the Axiom-2.1 rescue "the highest-cost casualty" because "$\sigma>0$ hence
$F>0$" fails. It does **not** fail here: $F>0$ a.s. whenever $\sigma>0$ (survivor item 3),
verified across three codebases. The hypothesis survives; only its *proof* changes, from a
functional-form argument to a quantization argument. Downgrade this item.

**D-5 — the embedded units fork is settled on its factual half.** Under $\tilde F=G_+/S_{\rm
ref}$ with $S_{\rm ref}$ any reward-scale reference, $\partial\tilde F/\partial\sigma=0$
*exactly*, because the delegated policy is provably $w$-free. The Stake Effect therefore
survives for raw $F$ and dies for $\tilde F$, exactly as the page anticipates. The page's
own recommendation — both objects, explicitly typed — is correct and should be adopted.
Add: **for raw $F$, state $\partial F/\partial\sigma>0$ as a normalisation, not a finding**
(it reaches $R^2=0.60$ on the surface with no delegation content whatsoever).

**D-6 — a new paragraph is needed: "what the environment cannot answer".** The design is
$n=4$, $m=3$, $\{-1,0,+1\}$ against a unit deadband, $N(0,1)$ targets in a $[0,10]$ box so
**half of all resource optima sit exactly on the lower wall**, and an aggregate feedback
gain $\kappa\approx2$–5 that puts the delegated dynamics past the flip bifurcation. Every
sign result in both memos is downstream of those four facts. The retraction is safe because
this is *the environment the corpus's own experiments run on*; the *replacement* is not
transferable, and the page should say so before anyone builds on it.

**D-7 — flag two numbers in the upstream spec as withdrawn.** `ideation_measurement.md`
§0.1's planner-magnitude table (the $\alpha=0$ entry exceeds the analytic ceiling; referee
$P(\text{ceiling}>-1.014)=0.0000$ over 30-rep blocks) and §0.2's "49.9% variance reduction",
which is computed from it. The qualitative claims both support — exact computability of the
benchmark, and benchmark-subtraction as variance reduction — are correct.

---

## 9. What a replacement must look like (referee's version)

The analyst's structural skeleton is right and its calibration is wrong. Corrected:

$$\mathbb{E}[G]=m\,n\,\sigma_{\rm eff}(\sigma)\cdot\Psi(\lambda,v_\tau;\varepsilon,\text{protocol},\text{beliefs}),\qquad \lambda^2=v_\tau(1-\rho)$$

with, **on the reachable design** ($\lambda\le1.1547$):

* $\Psi$ **decreasing** in $\lambda^2$ under uncoordinated best response
  ($\kappa_c\in[-0.83,-0.31]$, $R^2\ge0.98$), **increasing** under stage-Nash at
  $\varepsilon=0$ ($+0.18$), and **flat** under stage-Nash at every $\varepsilon>0$ and on
  the corpus's own trained agents;
* $\varepsilon$ entering through a $\sqrt\varepsilon$ cusp at $0^+$, a discontinuity at
  $\varepsilon=0$ exactly, and a *belief-model-dependent* linear term thereafter;
* $\sigma$ entering only as $\sigma_{\rm eff}$, i.e. as a unit.

Three requirements before anyone quotes it: (i) name the belief model; (ii) name the
delegation concept **and, if stage-Nash, the equilibrium selection rule**; (iii) state that
$\lambda\le\sqrt{n/(n-1)}$ and that nothing is established outside it.

---

## 10. Files

* Referee code and outputs: `results/regret_closed_form/referee/`
  * `r1_floor.py` — the floor-constant dispute (settled: analyst)
  * `r2_bruteforce.py` → `R2_bruteforce.csv`, `R2_bruteforce_bayes.csv` — 11+3 cells by
    full-3-D forward simulation
  * `r3_epsladder.py` → `R3_epsladder.csv`, `R3_out.txt` — the $\varepsilon$-ladder
    adjudication
  * `r4_surface.py` — $\sigma$-collapse, $\lambda$-collapse, $v_\tau$, $\lambda^2$ law,
    head-to-head on 168×3 cells
  * `r5_nash_eps0.py` → `R5_nash_eps0.csv` — exhaustive pure-NE enumeration, 5 selection
    rules
  * `r6_misc.py` → `R6_C1_lambda_collapse.csv` — DGP-identity, $\sigma_{\rm eff}$, $\kappa$,
    the $\varepsilon$ steelman
  * `r7_iql.py` → `R7_iql_profile.csv` — the trained-agent profile against both exact
    concepts
* Under audit: `DERIVATION.md` (analyst), `NUMERIC_REPORT.md` (numericist),
  `analytic_checks/a00_lib.py`–`a19`, `core.py`, `run_grid.py`.
* Source of truth for the environment: `run_gpu.py`,
  `friction_marl/experiments/factorial_design.py`,
  `friction_marl/envs/resource_allocation.py`, `results/adversarial_rerun/dgp.py`,
  `results/gpu_factorial_postfix/replication_results.csv`.
