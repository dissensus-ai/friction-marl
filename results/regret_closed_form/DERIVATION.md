# The delegation gap $\mathbb{E}[G](\sigma,\alpha,\varepsilon)$ on the friction-MARL environment: an analytic derivation

**Analyst's memo.** 30 July 2026. Companion to `NUMERIC_REPORT.md` (numericist, same
directory), written against an independently coded library (`analytic_checks/a00_lib.py`,
which deliberately does not import `core.py`) so that the two arms are genuinely
independent implementations of the same object.

Everything below traces to code in `analytic_checks/`, seeded and saved. Every
environment fact was read back out of source and asserted before use
(`a01_env_verify.py`: **36/36 PASS**). Every structural claim was checked against
brute force (`a02_structure.py`: 11/12 PASS, the one FAIL being a *negative* result
that matters — §6.4). **Every algebraic identity asserted below is verified
symbolically in `a19_symbolic.py`: 19/19 PASS** — nothing here rests on hand algebra.

---

## 0. Executive summary of the derivation

The three-parameter surface collapses. Writing $n=4$ agents, $m=3$ resources,
$\sigma_{\mathrm{eff}}(\sigma) = \mathbb{E}[\mathrm{clip}(N(\sigma,0.05^2),0,1)]$,
$v_\tau = \operatorname{Var}(\tau_i)$, $\rho = \operatorname{Corr}(\tau_i,\tau_j)$ and

$$\boxed{\ \lambda^2 \;:=\; v_\tau\,(1-\rho)\ }\qquad\text{(the contention scale)},$$

the exact reduction is

$$\mathbb{E}[G](\sigma,\text{knob},\varepsilon)\;=\;m\,n\,\sigma_{\mathrm{eff}}(\sigma)\cdot\Psi\!\left(\lambda,\,v_\tau;\,\varepsilon\right)\;+\;O\!\big((0.05/\sigma)^2\big),$$

with $\Psi$ *free of $\sigma$ exactly*, and for any unit-marginal-variance DGP
($v_\tau=1$: the renormalised and the equicorrelated designs) $\Psi$ collapses further
to a **one-parameter** function $\tilde\Psi(\lambda;\varepsilon)$, which is non-monotone
with an interior minimum at $\lambda^\ast(\varepsilon)$ and, on the contention branch
$\lambda>\lambda^\ast$, is

$$\tilde\Psi(\lambda;\varepsilon)\;=\;\kappa_c(\varepsilon)\,\lambda^2\;+\;a_0(\varepsilon)\;+\;o(\lambda^2),\qquad R^2=0.99\text{--}0.9997,$$

with $\kappa_c(\text{naive},0)=0.0752$ **predicted, not fitted**, by the $n=4$
order-statistic bracket $[0.0482,\,0.1986]$ of §5.2.

Against the canonical $F=\sigma(1+\varepsilon)/(1+\alpha)$ this says:

| canonical factor | what the derivation gives | verdict |
|---|---|---|
| $\sigma^1$ | **exact**, but as a *units identity*: the delegated policy is provably $w$-free, so any reward-unit DV is degree-1 homogeneous. Deviations from exactness are the weight-clip $\sigma_{\mathrm{eff}}$ and $O((0.05/\sigma)^2)$. | ✅ (vacuous) |
| $(1+\varepsilon)$, slope $+\sigma$ | no first-order term exists. $\mathbb{E}[G](\varepsilon)-\mathbb{E}[G](0^+)=c_1\sqrt{\varepsilon}+c_2\varepsilon$, $c_1\in[0.02,3.39]$ and $c_2\in[-27.3,-4.8]$ (herd damping). $\varepsilon=0$ exactly is a tie-convention knife-edge worth up to $58\%$. | ❌ order **and** sign |
| $1/(1+\alpha)$, slope $-\sigma$ | $\mathbb{E}[G]$ is an exact function of $\lvert\alpha\rvert$ under both $\alpha$-DGPs (a distributional identity), and $\rho(\alpha)=O(\alpha^2)$, so the first-order coefficient is $0$. On the one axis where alignment is signed ($\rho$), the sign is protocol-dependent: $+$ under naive delegation, $-$ under stage-Nash. | ❌ shape, magnitude, sign |
| multiplicative separability | $\lambda$ and $\varepsilon$ interact through the same object ($\kappa$, the aggregate feedback gain), and $\partial G/\partial\varepsilon$ changes sign with $\lambda$. | ❌ |

**Verdict input: RETRACTION**, with the survivor set in §10.

The one structural fact the corpus should keep, and can now prove rather than assert,
is $G\ge 0$ — but for a reason that makes the *Zero Friction* proposition unprovable as
stated (§10). Two results here are new to this workstream and load-bearing: the
**belief-irrelevance lemma** (§6.1, §9.3), which shows the largest sensitivity in either
report is a clamp artefact that vanishes when the targets are moved off the wall; and the
**corrected floor constant** (§9.5b), which replaces a formula in circulation that is
wrong by $4$–$58\%$ with one accurate to $0.25\%$.

---

## 1. Environment formalisation, verified against source

Every line below is an assertion in `a01_env_verify.py` (36/36 PASS,
`analytic_checks/a01_out.txt`).

**State and dynamics.**
`friction_marl/envs/resource_allocation.py:57-60` — reset $s_0\sim\mathrm{Unif}[0,C]^3$,
$C=10.0$, **continuous**; `:84-86` — $s' = \mathrm{clip}(s+\sum_i\delta_i,0,C)$;
`:17` — episode length $T=100$. Mirrored in `run_gpu.py:238` (reset), `:271-276`
(clamped transition), `:54-66` (`N_AGENTS=4`, `N_RESOURCES=3`, `CAPACITY=10.0`,
`EPISODE_LENGTH=100`).

**Actions.** `friction_marl/experiments/factorial_design.py:55-60` — per agent
$\delta_i\in\{-1,0,+1\}^3$, 27 actions. Since each agent picks each coordinate freely,
the achievable **aggregate** set is exactly $\{-4,\dots,4\}^3$ and is a *product* set
(verified constructively: all 729 aggregates realised by explicit profiles).

**Reward.** `factorial_design.py:63-69` —
$r_i=\sum_m w_{im}\cdot\big(-(s'_m-\tau_{im})^2\big)$, evaluated at the **clamped**
post-state, with the **true** state and **true** preferences (`run_gpu.py:279-281`).

**Stakes.** `factorial_design.py:50-51` — $w\sim N(\sigma,0.05)$ clipped to $[0,1]$.

**Alignment.** `factorial_design.py:42-48` —
$\tau_i=\alpha b+(1-|\alpha|)\xi_i$, $b$ shared, $\xi_i$ i.i.d., all $N(0,I_3)$.

**Entropy.** `run_gpu.py:250-256` — $\mathrm{obs}_i=s+N(0,\varepsilon I)$, i.e. **std**
$\sqrt\varepsilon$; `:283-293` — the stored next-observation carries fresh matched noise.
Rewards never see the noise.

**Design.** `factorial_design.py:25-27` — $\alpha\in\{-0.8,-0.4,0,0.4,0.8\}$,
$\sigma\in\{0.2,\dots,1.0\}$, $\varepsilon\in\{0,0.25,0.5,0.75,1.0\}$.

### 1.1 The welfare identity (the whole reason a closed form exists)

Ensemble welfare $W(s)=\sum_i\sum_m w_{im}\,(-(s_m-\tau_{im})^2)$. Per resource, with
$\bar W_m=\sum_i w_{im}$, $\mu_m=\sum_i w_{im}\tau_{im}/\bar W_m$ and
$V_m=\sum_i w_{im}(\tau_{im}-\mu_m)^2$,

$$W(s)\;=\;-\sum_m\Big[\ \bar W_m\,(s_m-\mu_m)^2\ +\ V_m\ \Big].\tag{1.1}$$

*Proof.* $\sum_i w_i(x-\tau_i)^2=\bar W x^2-2x\sum_i w_i\tau_i+\sum_i w_i\tau_i^2
=\bar W(x-\mu)^2+\sum_i w_i\tau_i^2-\bar W\mu^2$, and the last two terms are
$\sum_i w_i(\tau_i-\mu)^2$. $\square$
Verified exactly in `a01`: `-352.8778776100` vs `-352.8778776100`.

$V_m$ is a **constant of the draw**: it does not depend on the state, so it is identical
for the benchmark and for the realisation and **cancels in any welfare difference**. This
is the reason the "irreducible baseline" cannot survive in $G$ (§9).

### 1.2 The reachable lattice (a correction to the gate's premise, in the harder direction)

`ideation_measurement.md` §0.1 assumes a $11^3\times 729$ exact backward induction. That
is right for *integer* starts and wrong for the environment as implemented, because the
reset is continuous. Let $f=\mathrm{frac}(x_0)$ for one coordinate. The exact closure of
the one-coordinate dynamics under $x\mapsto\mathrm{clip}(x+D,0,10)$, $D\in\{-4..4\}$, is

$$L(f)\;=\;\{0,1,\dots,10\}\ \cup\ \{f,f+1,\dots,f+9\}\qquad\text{— 21 points for }f\ne 0.$$

The integer sublattice is entered only through the clamp (from $x\le4$ a $-4$ move lands
exactly on $0$; from $x\ge6$ a $+4$ move lands exactly on $10$) and is thereafter
absorbing. Verified three ways in `a01` (transition-index closure, 200 random offsets,
brute-force forward exploration giving $|{\rm reachable}|=21$ exactly), and independently
by the numericist.

So the exact counterfactual MDP is $21^3=9261$ states $\times\,729$ aggregate actions
almost surely, $1331\times729$ only on the null event $f=0$. **There is no
discretisation error**: this is the exact reachable set, not a grid approximation. The
computation is bigger than the spec claims, not more approximate.

### 1.3 Separability (fact S1)

Reward is additive over resources; the transition is componentwise; the per-agent action
set is a product $\{-1,0,1\}^3$; the observation noise is isotropic and independent
across coordinates. Therefore the 27-action argmin **factorises into three scalar
argmins**, for the agents and for the planner alike (`a02` T1/T3: PASS, against brute
force over all 27 actions). Everything below is stated per resource; multiply by $m=3$.

---

## 2. Benchmarks

### 2.1 The principal-executed benchmark $W^\ast$

Full-information joint control of the aggregate move $D_m\in\{-4,\dots,4\}$, maximising
$\sum_{t=1}^{T} W(s_t)$ from the *same* start $s_0$. By (1.1) and §1.3 this is, per
resource, an exact finite-horizon backward induction on 21 states $\times$ 9 actions with
stage reward $-(x-\mu)^2$ — implemented in `a00_lib.planner_occupancy`. The per-resource
reduction is validated here by `a02` T1 (the 27-action argmin factorises into three scalar
argmins, against brute force) and T7 (the exact chain reproduces a brute-force simulator of
the full 3-D environment to $8.1\times10^{-4}$, which is the simulator's own MC error); the
numericist validates the same reduction the other way, running the full 3-D
$9261\times729$ induction against the separated one and finding agreement to
$3.98\times10^{-13}$.

Two facts about it:

* **Near-instant reachability.** The state span is 10 and the maximum aggregate step is
  4, so the planner reaches its optimal lattice point in $\le 3$ steps and stays. Its
  time-average is therefore its static optimum up to an $O(3/T)=O(0.03)$ transient.
* **The transient is not free, and it is the right thing to include.** Benchmarking the
  delegated $T=100$ episode against the planner's *static* optimum overstates $G$ by the
  planner's own travelling cost — the numericist measures this at $0.535$ welfare units
  per step. Because `a00_lib` runs the planner's exact backward induction **from the same
  $x_0$**, the transient is inside the benchmark and cancels. This is why the numbers in
  this memo line up with the numericist's *Cesàro* column and not their *T=100* column
  (§9.1) even though this memo is a $T=100$ computation.

**Why this benchmark and not another.** It is the object the corpus's $F$ is about — "what
the principal's own decision rule would have selected" — evaluated at full information and
with full joint control. It is also the *only* choice under which $G\ge 0$ is a theorem
(§3.1); any boundedly-rational principal benchmark makes the sign an empirical question.

### 2.2 The two delegated realisations

Both are exact best responses, not learned policies. Agent $i$ sees
$\hat s_i=s+\eta_i$, $\eta_i\sim N(0,\varepsilon I)$, holds a diffuse prior (posterior
$s\mid\hat s_i\sim N(\hat s_i,\varepsilon)$), and knows the clamp. Let $c_i$ be its
conjecture about $\sum_{j\ne i}\delta_j$. It solves, per resource,

$$\delta_i=\arg\min_{d\in\{-1,0,1\}}\ \mathbb{E}\big[(\mathrm{clip}(s+c_i+d,0,C)-\tau_i)^2\,\big|\,\hat s_i\big].\tag{2.1}$$

* **(a) Naive-myopic**: $c_i\equiv 0$ (others treated as static).
* **(b) Stage-Nash**: $c_i$ is an equilibrium conjecture. Two readings, both computed:
  * **SNE** (stage-Nash-in-expectations): $c_i=\mathbb{E}[D_{-i}\mid x]$, solved as a
    damped fixed point. This is the interim equilibrium in which each agent is right *on
    average* about the others. It is well posed only for $\varepsilon\gtrsim0.25$; at small
    noise the fixed point does not converge, which is the coordination game asserting
    itself rather than a solver defect (§6.4).
  * **Exact pure-strategy Nash at $\varepsilon=0$**: the full correspondence over all
    $3^4=81$ profiles, with the clamp included, and an explicit selection rule.

**(2.1) is a two-threshold monotone rule.** Writing $u_i=\hat s_i+c_i$, the decision
depends on $(u_i,\tau_i,\varepsilon)$ only and is monotone non-increasing in $u_i$
(`a02` T3: 0 violations over 900 $(\tau,\varepsilon)$ cases), so

$$\delta_i=+1 \iff u_i<t_A(\tau_i,\varepsilon),\qquad \delta_i=-1\iff u_i>t_B(\tau_i,\varepsilon),$$

and hence, at the **true** state $x$,

$$p_i^+(x)=\Phi\!\Big(\tfrac{t_A-c_i-x}{\sqrt\varepsilon}\Big),\qquad
p_i^-(x)=\Phi\!\Big(\tfrac{x+c_i-t_B}{\sqrt\varepsilon}\Big).\tag{2.2}$$

Because signals are independent across agents, the law of $D=\sum_i\delta_i$ is the exact
4-fold convolution of these three-point laws, and the whole delegated process is a
**21-state Markov chain computed with no Monte Carlo over the noise**. Validated against
a brute-force simulator of the full environment: max relative error
$8.1\times10^{-4}$, which is the *simulator's* MC error (`a02` T7).

---

## 3. The exact decomposition of $\mathbb{E}[G]$

### 3.1 The reduction, and why $G\ge0$ is a theorem

By (1.1), $V_m$ cancels, so with $\pi$ the delegated occupancy and $x^\ast$ the planner's
(time-averaged) state,

$$\boxed{\ G\;=\;\sum_m \bar W_m\Big[\ \mathbb{E}_\pi (x_m-\mu_m)^2\ -\ \mathbb{E}_{\rm plan}(x_m-\mu_m)^2\ \Big]\ }\tag{3.1}$$

— a **pure tracking-error difference**. Two consequences:

**Theorem 3.1 ($G\ge0$).** The delegated behaviour induces a distribution over
$\{-4..4\}$-valued aggregate moves, and the principal's benchmark maximises the same
objective over *all* such (indeed all history-dependent) policies. Hence
$W(\text{realised})\le W^\ast$ pointwise in the draw, so $G\ge0$ and $F:=G_+=G$. $\square$
Checked pointwise: $\min G=-3.0\times10^{-13}$ (naive), $+5.0\times10^{-13}$ (SNE),
$0$ exactly (Nash) — `a02` T5.

**Corollary 3.2.** The "delegation surplus" $B=\max(-G,0)$ is identically zero at the
**ensemble** level against an optimal principal. It is a real and large object *per
principal* — §8.

### 3.2 The three channels

Let $q=x^\ast-\mu$ (the signed **quantization/boundary residual** of the benchmark) and
$b=\mathbb{E}_\pi x-x^\ast$ (the **displacement** of the delegated attractor from the
benchmark point). Expanding (3.1),

$$\mathbb{E}_\pi(x-\mu)^2=\operatorname{Var}_\pi(x)+(b+q)^2,\qquad
\mathbb{E}_{\rm plan}(x-\mu)^2=q^2+\operatorname{Var}_{\rm plan}(x),$$

so, per resource,

$$\boxed{\ \frac{G}{\bar W}\;=\;\underbrace{\operatorname{Var}_\pi(x)}_{\text{(V) noise-response / churn}}\;+\;\underbrace{b^2}_{\text{(C) contention}}\;+\;\underbrace{2\,q\,b}_{\text{(X) floor}\times\text{contention}}\;-\;\underbrace{\operatorname{Var}_{\rm plan}(x)}_{\text{planner transient, }O(3/T)}\ }\tag{3.2}$$

Verified as an identity in `a05` D1: for each of 12 (protocol, $\lambda$, $\gamma$,
$\varepsilon$) cells the three terms reproduce $\mathbb{E}[G]$ up to a residual that is
*exactly* $\bar W\operatorname{Var}_{\rm plan}$, constant in $\varepsilon$ within each
draw set ($-0.1576$, $-0.1472$); and symbolically in `a19` S9.

Magnitudes, from the same table: the cross term (X) is small and consistently **negative**
($-0.03$ to $-0.12$), i.e. the delegated attractor tends to sit on the *opposite* side of
the benchmark point from $\mu$, which slightly offsets the gap. Channel (V) dominates
channel (C) in the herding regime by roughly $5{:}1$ ($2.32$ vs $0.48$ at naive,
$\lambda=0.21$, $\varepsilon=0$) and the two are comparable in the contention regime.

**The quantization floor does not appear in (3.2) except through the cross term.** The
floor $\bar W q^2$ is the benchmark's own irreducible loss; it enters $W^\ast$ and
$W(\text{realised})$ identically and cancels. Deriving it (§4) still matters, because it
is the object the corpus's "irreducible $\sigma/2$ baseline" is trying to name, and
because $q$ multiplies $b$ in (3.2).

---

## 4. Channel (F): the quantization floor, in closed form

The planner cannot sit at $\mu$; it sits at the nearest point of $L(f)$.

**Interior, generic.** Restricted to one unit cell, $L(f)$ is the two-point set
$\{0,f\}\bmod 1$, cutting the circle into arcs of length $f$ and $1-f$. Within an arc of
length $g$ the distance to the nearer endpoint is $U[0,g/2]$, so
$\mathbb{E}[d^2\mid\text{arc}]=g^2/12$; weighting each arc by its own length,

$$\mathbb{E}[d^2\mid f]=\frac{f^3+(1-f)^3}{12},\qquad
\mathbb{E}_{f\sim U(0,1)}[d^2]=\frac1{12}\Big(\tfrac14+\tfrac14\Big)=\boxed{\tfrac1{24}=0.0416\overline{6}}.\tag{4.1}$$

Verified three ways in `a14`: the symbolic integral, a $4\times10^6$-draw Monte Carlo on
$\mu\sim U(2,8)$ ($0.0416458$), and 2-D quadrature over one interior cell ($0.0416667$);
and symbolically in `a19` S10.

**Boundary.** $\mu\le0$ is absorbed by the lattice point $0$ *exactly*, contributing
$\mu^2$; likewise $\mu\ge10$. Under every DGP here $\mu$ is symmetric about $0$ on the
$[0,10]$ box, so **about half of all resource-optima sit on the lower wall** — a
structural feature of this environment that the corpus does not mention, and the source
of most of its pathologies (§5.3, §6.2).

**Location scale.** With equal weights $\mu=\bar\tau$, so for exchangeable targets

$$\operatorname{Var}(\mu)\;=\;\frac{v_\tau\big(1+(n-1)\rho\big)}{n}\;=\;v_\tau-\frac{n-1}{n}\lambda^2 .\tag{4.2}$$

| DGP | $v_\tau$ | $\rho$ | $\lambda^2=v_\tau(1-\rho)$ | $\operatorname{Var}(\mu)$ |
|---|---|---|---|---|
| implemented $\tau_i=\alpha b+(1-\lvert\alpha\rvert)\xi_i$ | $\alpha^2+(1-\lvert\alpha\rvert)^2$ | $\dfrac{\alpha^2}{\alpha^2+(1-\lvert\alpha\rvert)^2}$ | $(1-\lvert\alpha\rvert)^2$ | $\alpha^2+\tfrac{(1-\lvert\alpha\rvert)^2}{4}$ |
| renormalised (÷$\sqrt{v_\tau}$) | $1$ | same | $\dfrac{(1-\lvert\alpha\rvert)^2}{\alpha^2+(1-\lvert\alpha\rvert)^2}$ | $1-\tfrac34\lambda^2$ |
| equicorrelated ($\rho$ = true pairwise corr) | $1$ | $\rho$ | $1-\rho$ | $\tfrac{1+3\rho}{4}$ |

All four columns verified against $600\,000$-draw Monte Carlo on 15 (DGP, knob) rows in
`a12b` — worst relative error $\le0.5\%$ (§9.2) — symbolically in `a19` (S2, S3, S4, S5),
and the $\rho$ column independently in `a01`.

**The weight-dispersion correction.** With $w_i=\sigma+0.05\,z_i$ and $c=0.05/\sigma$,
$\mu=\bar\tau+\frac{c}{n}\sum_i z_i(\tau_i-\bar\tau)+O(c^2)$, so

$$\operatorname{Var}(\mu)\;=\;v_\tau-\frac{n-1}{n}\lambda^2\;+\;\frac{n-1}{n^2}\,c^2\lambda^2\;+\;O(c^4)
\;=\;v_\tau-\tfrac34\lambda^2+\tfrac{3}{16}c^2\lambda^2+O(c^4).\tag{4.3}$$

At $\sigma=0.2$ this inflates $\operatorname{Var}(\mu)$ by $+4.69\%$ at $\lambda^2=1$,
$v_\tau=1$; at $\sigma=1.0$ by $+0.17\%$. It is the **only** source of genuine
$\sigma$-dependence in $\Psi$, and §7.5 shows it accounts for a measured $+4.53\%$ excess
in $G(0.2)/G(0.6)$ over the pure $\sigma_{\mathrm{eff}}$ prediction — in both my
computation and the numericist's.

**Assembling the floor.** Substituting (4.2)/(4.3) into the corrected constants of §9.5b,

$$\mathbb{E}[\text{floor}]\;=\;m\,n\,\sigma_{\mathrm{eff}}(\sigma)\cdot\begin{cases}
\dfrac{\operatorname{Var}(\mu)}{2}+\dfrac{1}{48}, & \operatorname{Var}(\mu)\gtrsim0.25\quad(\text{err}\le0.25\%)\\[2ex]
\operatorname{Var}(\mu)-\tfrac43\sqrt{\tfrac2\pi}\operatorname{Var}(\mu)^{3/2}, & \operatorname{Var}(\mu)\lesssim0.1\quad(\text{err}\le1\%)
\end{cases}$$

giving, at $\sigma=0.6$ ($m\,n\,\sigma_{\mathrm{eff}}=7.2$): implemented $\alpha=0$ →
$1.05$; implemented $|\alpha|=0.8$ → $2.49$; renormalised $|\alpha|=0.8$ → $3.59$;
equicorrelated $\rho=-1/3$ → **$0$ exactly**, because $\operatorname{Var}(\mu)=(1+3\rho)/4$
vanishes there and $\mu$ lands deterministically on the lattice point $0$ (`a17`).

Note what this means for the corpus's *irreducible baseline*: the object exists, it is
large, it is now known in closed form — and it is **not proportional to $\sigma/2$**. It
scales as $\sigma\cdot(\operatorname{Var}(\mu)/2+1/48)$, i.e. it depends on alignment, and
it is zero at one end of the alignment axis. And it **cancels identically** in $G$.

---

## 5. Channel (C): contention, and the $\lambda^2$ law

### 5.1 $\sigma$ factors out exactly (fact S4)

**Proposition 5.1.** The delegated policy does not depend on $w$ at all.
*Proof.* Per resource, agent $i$'s objective (2.1) is
$-w_i\,\mathbb{E}[(\text{post}-\tau_i)^2\mid\hat s_i]$ with $w_i>0$ a.s.; a positive
scalar cannot move an argmin. Across resources the objective is additive and the action
set is a product, so the per-resource argmins assemble into the joint argmin (§1.3). The
same holds for every $c_i$, hence for both delegation concepts. $\square$

Therefore $\sigma$ enters $G$ *only* through the multiplicative scale $\bar W_m$ and
through $\mu_m$ (a weighted mean). If $w_{im}\equiv\sigma$ exactly, $\mu=\bar\tau$ is
$\sigma$-free and

$$\mathbb{E}[G]\;=\;m\,n\,\sigma\cdot\Psi(\lambda,v_\tau;\varepsilon)\qquad\text{— degree-1 homogeneous, with zero }\sigma\times\alpha\text{ or }\sigma\times\varepsilon\text{ interaction.}$$

With the implemented sampler the only corrections are the clip and (4.3):

$$\sigma_{\mathrm{eff}}(\sigma)=\sigma\big[\Phi(b)-\Phi(a)\big]-0.05\big[\phi(b)-\phi(a)\big]+\big[1-\Phi(b)\big],\quad a=\tfrac{-\sigma}{0.05},\ b=\tfrac{1-\sigma}{0.05}.\tag{5.1}$$

$\sigma_{\mathrm{eff}}(\sigma)=\sigma$ to six decimals for $\sigma\in\{0.2,0.4,0.6,0.8\}$
and $\sigma_{\mathrm{eff}}(1.0)=0.980053$, i.e. $1-0.05/\sqrt{2\pi}$ exactly — a
$-1.995\%$ clip bite (`a07` K3; MC $0.980054$).

**This is a units identity, not a friction result.** Any dependent variable measured in
reward units is degree-1 homogeneous in $\sigma$ by Proposition 5.1. The canonical form's
$\sigma$ factor is correct and carries no information about delegation.

### 5.2 The contention scale and the $\lambda^2$ law

Modulo the clamp and the unit lattice, the whole problem is translation-equivariant: the
delegated dynamics depends on the draw through the *centred* configuration
$(\tau_i-\mu)_i$, whose scale is $\lambda$ by construction
($\operatorname{Var}(\tau_i-\bar\tau)=\frac{n-1}{n}\lambda^2=\frac34\lambda^2$).

**Scaling argument.** For $\lambda\gg1$ the unit deadband and unit lattice are
fine-grained relative to the target spread, the system approaches a continuum limit in
which every length scales with $\lambda$, and $G/\bar W$ — a squared length — scales as
$\lambda^2$:

$$\tilde\Psi(\lambda;\varepsilon)\;=\;\kappa_c(\varepsilon)\,\lambda^2\;+\;a_0(\varepsilon)\;+\;o(1),\qquad \lambda\to\infty.\tag{5.2}$$

Measured: on $\lambda\in[0.45,3.0]$ at $\varepsilon=10^{-4}$ (SNE), the fit
$\mathbb{E}[G]=a+b\lambda^2$ gives $a=0.1010$, $b=1.4960$, **$R^2=0.99966$**, and the
log–log exponent of the offset-removed excess is **1.995** against a theory of 2
(`a08` E2).

**The coefficient, bracketed by order statistics.** At $\varepsilon=0$ the delegated
attractor sits where the pushes balance, i.e. where
$\#\{i:\tau_i>x+\tfrac12\}=\#\{i:\tau_i<x-\tfrac12\}$ — the **median interval**
$[\tau_{(2)},\tau_{(3)}]$ for $n=4$ — while the planner tracks the weighted **mean**.
So $b\approx\mathrm{med}(\tau)-\mathrm{mean}(\tau)$ and, since
$\operatorname{Cov}(\mathrm{med},\mathrm{mean})=\operatorname{Var}(\mathrm{mean})$ for
i.i.d. normals (the mean is the efficient equivariant estimator),

$$\mathbb{E}[b^2]\;\to\;\lambda^2\big(V_{\rm med}(n)-\tfrac1n\big).$$

For $n=4$ (`a07` K1, $4\times10^6$ draws): $\operatorname{Var}(\mathrm{mean})=0.249897$
(exact $\tfrac14$), $\operatorname{Var}(\mathrm{med})=0.298018$,
$\operatorname{Cov}=0.249869$, so
$\operatorname{Var}(\mathrm{med}-\mathrm{mean})=0.048177$, and if the attractor rests on
an order statistic rather than at the centre of the median interval,
$\mathbb{E}[(\tau_{(2)}-\mathrm{mean})^2]=0.198569$. Hence

$$\boxed{\ \kappa_c\ \in\ [0.0482,\ 0.1986]\ }\quad\text{if channel (C) alone were operative.}$$

Measured, fitting $\tilde\Psi=a_0+\kappa_c\lambda^2$ **on the contention branch only**
($\lambda>\lambda^\ast$, §7.7), naive delegation (`a16b`):

| $\varepsilon$ | $0$ | $0.01$ | $0.25$ | $1.0$ |
|---|---|---|---|---|
| $\kappa_c$ | $\mathbf{0.0752}$ | $0.0634$ | $0.0340$ | $0.0185$ |
| $a_0$ | $-0.017$ | $-0.003$ | $+0.058$ | $+0.074$ |
| $R^2$ | $0.9926$ | $0.9827$ | $0.9627$ | $0.9716$ |
| log–log exponent | $\mathbf{1.937}$ | $1.844$ | $1.687$ | $1.648$ |
| inside the bracket? | **yes** | **yes** | no (below) | no (below) |

**$\kappa_c(\text{naive},\varepsilon=0)=0.0752$ lands inside the order-statistic bracket,
near its median-attractor end $0.0482$**, and the measured exponent is $1.937$ against a
predicted $2$. The contention branch really is the median-versus-mean displacement of
(5.2), predicted from $n=4$ order statistics with no fitted quantity.

$\kappa_c$ falls monotonically in $\varepsilon$ (and drops out of the bracket above
$\varepsilon\approx0.1$) — the same de-synchronisation that drives $\kappa$ and
$\lambda^\ast$ down (§6.2, §7.7).

Under the SNE protocol the analogous fit over $\lambda\ge0.45$ gives
$\kappa_c=1.4960/7.2=0.2078$, slightly **above** the bracket; but that window reaches
below SNE's own $\lambda^\ast$, so part of the excess is the herding branch leaking in.
Where the two channels can be separated directly they behave as derived: `a05` D1 at
naive, $\lambda=0.21$, $\varepsilon=0$ gives $\bar W\operatorname{Var}_\pi=2.32$ against
$\bar Wb^2=0.48$ — **the variance channel dominates the bias channel roughly 5:1** in the
herding regime, which is exactly why the pure-(C) bracket under-predicts there.

### 5.3 The $\lambda\to0$ knife-edge (perfect coordination is the *worst* case)

$\tilde\Psi$ is **not** monotone in $\lambda$: it has an interior minimum near
$\lambda\approx0.3$ and rises again as $\lambda\to0$. The mechanism is exact and can be
displayed in one line (`a02` T6):

> four identical targets $\tau=5$, start $x_0=3.4$, naive rule, $\varepsilon=0$:
> $[3.4,\ 7.4,\ 3.4,\ 7.4,\ 3.4,\ 7.4,\ 3.4]$ — a **2-cycle of amplitude 4**.
> Stage-Nash at the same state selects the aggregate $+2$, landing on $5.4$ = the nearest
> lattice point, i.e. exactly the quantization floor.

When targets agree, all four agents see the same sign and all four move, so
$|D|\in\{0,4\}$ against a deadband of width 1: the state cannot stop. When targets
disagree, the pushes partially cancel and the aggregate can take intermediate values, so
the chain can come to rest. **Alignment is what creates the overshoot** — under
uncoordinated best response. Under stage-Nash the conjecture $c_i$ internalises the
others' moves, exactly one agent acts, and the overshoot disappears; §6.4.

---

## 6. Channel (V): the noise-response variance channel, derived

### 6.1 The interior decision rule, and the belief-irrelevance lemma

**Lemma 6.1 (belief irrelevance off the clamp).** Suppose the clamp does not bind on the
posterior support. Then

$$\mathbb{E}\big[(s+c_i+d-\tau_i)^2\mid\hat s_i\big]=(\hat s_i+c_i+d-\tau_i)^2+\varepsilon,$$

and the $\varepsilon$ is **additive and $d$-free**. Hence the Bayes rule and the
certainty-equivalent rule are the *same map*, namely
$d=\mathrm{clip}\big(\mathrm{round}(\tau_i-\hat s_i-c_i),-1,1\big)$: a **unit deadband
centred on $\tau_i$**, with thresholds

$$t_A=\tau_i-\tfrac12,\qquad t_B=\tau_i+\tfrac12 .\tag{6.1}$$

*Consequence.* In the interior, $\varepsilon$ enters **only** through the randomness of
$\hat s_i$ and never through the shape of the decision rule. Any wedge between belief
models is supported entirely on the event that $\mathrm{clip}(\cdot)$ is non-linear on the
posterior support — i.e. on the **clamp boundary layer**. Because the clamp is relevant
iff a $\pm1$ move can reach it, that layer has width $1+O(\sqrt\varepsilon)$, not
$O(\sqrt\varepsilon)$; §9.3(a) measures its 99th percentile at $\le2.0$ in a box of
width $10$.

This is the single most useful analytic fact in the memo, because the numericist's largest
single sensitivity — a median $-52.8\%$ change in $\mathbb{E}[G]$ between certainty-equivalent
and Bayesian agents — is thereby localised to the walls, and the walls are where half the
$\mu$'s live (§4). §9.3 tests it decisively: shifting the same DGP into the middle of the
box collapses an $86.6\%$ belief wedge to $0.11\%$ **and removes the sign disagreement in
$\partial\mathbb{E}[G]/\partial\varepsilon$ entirely.**

### 6.2 Drift, dispersion, and the aggregate feedback gain $\kappa$

With (6.1) and $y_i=x-\tau_i$, at the true state $x$,

$$e_i(y)\;=\;p_i^+-p_i^-\;=\;1-\Phi\!\Big(\tfrac{y+\frac12}{\sqrt\varepsilon}\Big)-\Phi\!\Big(\tfrac{y-\frac12}{\sqrt\varepsilon}\Big)\quad\text{(odd, decreasing, }e(\pm\infty)=\mp1)$$

$$m_D(x)=\sum_i e_i(x-\tau_i),\qquad
v_D(x)=\sum_i\Big(p_i^++p_i^--e_i^2\Big).\tag{6.2}$$

**(V) Variance injection.** $v_D$ is the per-step variance the agents inject.
$v_D\to0$ away from thresholds as $\varepsilon\to0$; and as $\varepsilon\to\infty$,
$p^\pm\to\tfrac12$, $e\to0$, so $v_D\to n=4$: **the injection saturates at $n$**, it does
not grow without bound in $\varepsilon$. Measured (`a13` P4, a representative target
configuration at $x=0.5$):

| $\varepsilon$ | $0.01$ | $0.25$ | $1$ | $4$ | $25$ | $400$ | $\to\infty$ |
|---|---|---|---|---|---|---|---|
| $v_D$ | $0.135$ | $0.836$ | $1.794$ | $2.879$ | $3.613$ | $3.916$ | $n=4$ |

This bound is what makes the sign of $\partial G/\partial\varepsilon$ decidable: channel
(V) is capped at $n$ per step, while channel (K) below removes an $O(4)$-amplitude limit
cycle. Past the saturation point only (K) is still moving.

**(K) Herd damping.** The aggregate feedback gain is

$$\kappa(x)\;=\;-m_D'(x)\;=\;\frac{1}{\sqrt\varepsilon}\sum_i\Big[\phi\Big(\tfrac{y_i+\frac12}{\sqrt\varepsilon}\Big)+\phi\Big(\tfrac{y_i-\frac12}{\sqrt\varepsilon}\Big)\Big].\tag{6.3}$$

The linearised delegated map is $x\mapsto x+m_D(x)$, multiplier $1-\kappa$; hence

$$\boxed{\ |1-\kappa|\ge1\iff\kappa\ge2\iff\text{flip-unstable (2-cycle / overshoot).}\ }$$

At $\varepsilon=0$, $m_D$ is a staircase with total variation $2n=8$ concentrated on the
$2n=8$ thresholds $\{\tau_i\pm\tfrac12\}$; the gain over any unit window is the number of
thresholds it contains, which for $\lambda\lesssim1$ is generically 2–4. **Hence
$\kappa\in[2,5]$ generically at small $\varepsilon$ — the delegated dynamics sits at or
past the flip bifurcation, and no contracting-linearisation reduction of channel (V) is
valid.** For large $\varepsilon$, targets fall within $O(\sqrt\varepsilon)$ of $x$ and

$$\kappa\;\longrightarrow\;\frac{2n\phi(0)}{\sqrt\varepsilon}=\frac{3.1915}{\sqrt\varepsilon}\ \Rightarrow\ \kappa<2\ \text{once}\ \varepsilon\gtrsim 2.55 .\tag{6.4}$$

Noise de-synchronises the herd. This is the channel a monotone
$\partial F/\partial\varepsilon>0$ has no room for.

(6.3) is verified against a finite difference of $m_D$ to $\le1.3\times10^{-10}$
(`a13` P3). Evaluated at the exact chain's own modal state, over the equicorrelated
family at $\sigma=0.6$:

| median $\kappa$ ($P(\kappa\ge2)$) | $\varepsilon=0.01$ | $0.05$ | $0.25$ | $0.5$ | $1.0$ | $2.0$ |
|---|---|---|---|---|---|---|
| $\lambda=0.45$ | $2.32$ ($.53$) | $2.91$ ($.67$) | $2.94$ ($.72$) | $2.67$ ($.72$) | $2.32$ ($.68$) | $1.89$ ($.28$) |
| $\lambda=0.71$ | $2.47$ ($.55$) | $2.78$ ($.70$) | $2.74$ ($.78$) | $2.52$ ($.77$) | $2.21$ ($.68$) | $1.83$ ($.15$) |
| $\lambda=1.00$ | $2.52$ ($.56$) | $2.73$ ($.71$) | $2.64$ ($.81$) | $2.44$ ($.78$) | $2.15$ ($.66$) | $1.80$ ($.10$) |
| $\lambda=1.15$ | $2.51$ ($.56$) | $2.68$ ($.70$) | $2.58$ ($.76$) | $2.42$ ($.73$) | $2.16$ ($.64$) | $1.80$ ($.15$) |

**Median $\kappa$ exceeds the flip-bifurcation threshold $2$ at every $\varepsilon\le1$
and every $\lambda$ in the design**, with $P(\kappa\ge2)$ peaking at $0.72$–$0.81$ around
$\varepsilon=0.25$ and collapsing to $0.10$–$0.28$ only at $\varepsilon=2$ — consistent
with the analytic crossing (6.4) at $\varepsilon\approx2.55$. The numericist's independent
$\kappa$ table ($4.73\to2.15$ as $\varepsilon:0.01\to1.0$) agrees on the decreasing
regime; my $\varepsilon=0.01$ value is lower because at very small noise the exact chain
sits *far from* any threshold, so the modal state's local gain understates the effective
gain over the cycle.

**Any derivation that linearises the delegated dynamics and assumes contraction is
modelling a different system.**

### 6.3 The $\varepsilon\to0^+$ cusp, and the knife-edge at $\varepsilon=0$

**For a fixed draw**, every transition probability is $\Phi(-d/\sqrt\varepsilon)\le
e^{-d^2/2\varepsilon}$, $d$ = distance from a visited lattice point to the nearest
threshold. So $G(\varepsilon)-G(0^+)$ is flat *to all orders* in $\varepsilon$ for
a.e. draw.

**Averaged over the DGP**, $d$ has a bounded density with $p(0)>0$ (the $\tau_i$ are
continuous), and $\int_0^\infty p(d)\Phi(-d/\sqrt\varepsilon)\,dd
\approx p(0)\sqrt\varepsilon\int_0^\infty\Phi(-z)\,dz=p(0)\sqrt\varepsilon/\sqrt{2\pi}$.
Hence

$$\mathbb{E}[G](\varepsilon)-\mathbb{E}[G](0^+)\;=\;c_1\sqrt\varepsilon\;+\;c_2\,\varepsilon\;+\;o(\varepsilon),\qquad
c_1=\frac{p(0)\,\overline{\Delta G}}{\sqrt{2\pi}}\ \ \big(2\mathbb{E}[Z^+]=0.797885\big).\tag{6.5}$$

$\partial\mathbb{E}[G]/\partial\varepsilon|_{0^+}=\pm\infty$; the finite object is
$\partial\mathbb{E}[G]/\partial\sqrt\varepsilon$. **A finite linear slope $+\sigma$ is not
an approximation to this; it is the wrong order.**

Note the coefficient $c_1$ is *not* zero even though each flipping agent is exactly
indifferent at its own threshold: an agent's indifference is about **its own** loss, while
$\Delta G$ prices the change in the aggregate and hence in every *other* principal's loss
and in the whole subsequent trajectory.

**$\varepsilon=0$ exactly is a different point.** The lower clamp $x=0$ is simultaneously
a lattice point and a decision threshold: for an agent with $\tau_i\in(-\tfrac12,0)$ both
$\delta=-1$ and $\delta=0$ land the post-state on $0$ and the clamped losses tie
*exactly*, so a tie convention decides. Measured directly (`a13` P2): **$7.16\%$ of
(agent, reachable-state) pairs are decided by the tie convention rather than by a strict
inequality at $\varepsilon=0$.** Any $\varepsilon>0$ breaks every tie with probability 1.
Hence $\mathbb{E}[G](0)\ne\lim_{\varepsilon\to0^+}\mathbb{E}[G](\varepsilon)$ — measured
at $+57.7\%$, $+57.2\%$ and $+9.9\%$ on three paired ladders (`a13` P1), $+36\%$ in
`a08` E1, $+23.6\%$ by the numericist under a different tie rule. **The $\varepsilon=0$
column of every table in this workstream is convention-dependent by up to $58\%$; every
$\varepsilon>0$ number is not.**

### 6.4 Why the two delegation concepts disagree about the sign of the alignment effect

`a02` T4b is the one structural check that FAILS, and it is the informative one: with the
clamp included, the pure-strategy Nash aggregate is **not** a.s. unique — mean 2.04,
max 5 distinct Nash aggregates per reachable state. Consequently the $\varepsilon=0$
stage-Nash $\mathbb{E}[G]$ is a *correspondence*, not a number:

| selection rule at $\varepsilon=0$, $\sigma=0.6$ | implemented $\alpha=0$ | equicorr $\rho=-1/3$ | equicorr $\rho=+0.94$ |
|---|---|---|---|
| welfare-**best** | $0.265\pm0.011$ | $0.065\pm0.005$ | $0.107\pm0.003$ |
| **minimal**-move | $72.91\pm1.97$ | $73.16\pm1.96$ | $71.64\pm2.01$ |
| welfare-**worst** | $140.5\pm2.6$ | $143.6\pm2.6$ | $144.0\pm2.7$ |

(`a09` N1.) A three-orders-of-magnitude spread. The numericist's Gauss–Seidel-from-zero
selection lands at $1.146$ — inside the bracket, and their $\le5.3\%$ selection
sensitivity is a statement about *that basin*, not about the correspondence. **Any closed
form for $G$ under stage-Nash at $\varepsilon=0$ is under-specified until the selection
rule is named.** This is a stronger caveat than either report alone contained, and it is
the reason §10.1 does not treat the stage-Nash sign result as a rescue.

Away from $\varepsilon=0$ the SNE fixed point is also delicate: the damped iteration fails
to converge at $\varepsilon\le10^{-2}$ (max residual $2.69$ at $\varepsilon=10^{-4}$,
$1.64$ at $10^{-2}$) and converges cleanly only for $\varepsilon\gtrsim0.25$
(residual $4\times10^{-5}$) — `a05` D0. Multiplicity at small noise is not a numerical
artefact; it is the coordination game.

---

## 7. The three DGPs, and the expansion about $(\alpha,\varepsilon)=(0,0)$

### 7.1 An exact symmetry theorem: no signed $\alpha$ effect can exist

**Theorem 7.1.** Under both the implemented and the renormalised DGP, the law of
$(\tau_1,\dots,\tau_n)$ is invariant under $\alpha\mapsto-\alpha$. Hence
$\mathbb{E}[G](\alpha)=\mathbb{E}[G](-\alpha)$ identically — for *any* delegation
protocol, *any* $\varepsilon$, *any* $\sigma$, and any functional of the realised path.

*Proof.* $\tau_i(\alpha;b,\xi)=\alpha b+(1-|\alpha|)\xi_i$. Substituting $b\mapsto-b$,
$\tau_i(-\alpha;-b,\xi)=(-\alpha)(-b)+(1-|-\alpha|)\xi_i=\alpha b+(1-|\alpha|)\xi_i
=\tau_i(\alpha;b,\xi)$ — the *same array*, not merely the same law. Since $b\sim N(0,I)$
is symmetric, $(-b,\xi)\overset{d}{=}(b,\xi)$, so the target array has the same law at
$\pm\alpha$. The renormalisation divides by $\sqrt{\alpha^2+(1-|\alpha|)^2}$, itself a
function of $|\alpha|$, so the same argument applies. $\square$

Verified two ways: `EXPANSION.json::P3_alpha_pm_symmetry` reports
`max_abs_tau_diff = 0.0` (numericist, four values of $\alpha$, both DGPs); `a09` N2
gives $|{\rm diff}|/{\rm SE}\le1.94$ over four $\pm\alpha$ pairs.

**The $\alpha$ knob is a $|\alpha|$ knob, and renormalisation does not fix that** — it
fixes the marginal variance, not the sign-blindness. Any proposition of the form
$\partial F/\partial\alpha<0$ is unfalsifiable on this design, because the design cannot
produce a signed alignment contrast.

### 7.2 What each knob actually varies, to first order

Using §4's map $\text{knob}\mapsto(\lambda^2,v_\tau)$ and expanding at the origin:

| DGP | $\lambda^2(\text{knob})$ | expansion at 0 | $\partial\lambda^2/\partial(\text{knob})$ at $0$ |
|---|---|---|---|
| implemented | $(1-\lvert\alpha\rvert)^2$ | $1-2\lvert\alpha\rvert+\alpha^2$ | $\mp2$ (a **kink**) |
| renormalised | $\dfrac{(1-\lvert\alpha\rvert)^2}{\alpha^2+(1-\lvert\alpha\rvert)^2}$ | $1-\alpha^2+O(\lvert\alpha\rvert^3)$ | $\mathbf{0}$ (exactly) |
| equicorrelated | $1-\rho$ | $1-\rho$ | $-1$ (a genuine signed knob) |

Because $\rho(\alpha)=\alpha^2/(\alpha^2+(1-|\alpha|)^2)=\alpha^2+O(|\alpha|^3)$ is
**quadratic**, the renormalised DGP has *no first-order $\alpha$ content at all* — not
even the $|\alpha|$ kink one might expect. The implemented DGP does have a kink, but it
comes from the marginal-variance drift ($v_\tau=1-2|\alpha|+2\alpha^2$) as much as from
$\lambda$, and it is even, so its two-sided derivative does not exist and its symmetric
derivative is zero.

### 7.3 The $\alpha$ / $\rho$ expansion, measured

One-sided slopes at $\sigma=0.6$, $\varepsilon\to0^+$ (`a06` S4; note the label
correction of §9.5 — `a06`'s "$\varepsilon=0$" column is $\varepsilon=10^{-4}$):

| DGP | protocol | right slope | left slope | canonical | reading |
|---|---|---|---|---|---|
| implemented, $h=0.05$ | SNE | $-1.543$ | $+1.785$ | $-\sigma$, smooth | **kink** ($3\mathrm{SE}\approx1.05$) |
| implemented, $h=0.05$ | naive | $-1.055$ | $+1.097$ | $-\sigma$, smooth | **kink** ($3\mathrm{SE}\approx1.50$) |
| renormalised, $h=0.05$ | SNE | $+0.137$ | $+0.290$ | $-\sigma$ | $\approx0$, smooth |
| renormalised, $h=0.05$ | naive | $+0.226$ | $-0.238$ | $-\sigma$ | $\approx0$, smooth |
| equicorrelated (in $\rho$), $h=0.05$ | SNE | $-1.381$ | $-1.188$ | $-\sigma$ | signed, **negative** (central $-1.285$) |
| equicorrelated (in $\rho$), $h=0.05$ | naive | $+4.858$ | $+3.738$ | $-\sigma$ | signed, **positive** (central $+4.298$) |

In *relative* units — which is the only scale-free comparison, since
$(1/F)\,\partial F/\partial\alpha|_0=-1$ — the implemented DGP's **right** branch gives
$-1.543/1.337=-1.15$ (SNE) and $-1.055/1.461=-0.72$ (naive), i.e. **the canonical
$1/(1+\alpha)$ shape approximately matches the $\alpha>0$ half-line and gets the
$\alpha<0$ half-line exactly backwards**, because $\mathbb{E}[G]$ is even in $\alpha$ and
$1/(1+\alpha)$ is not. The numericist's two-sided OLS slope over $[-0.1,0.1]$
($-0.083$) is the correct summary of an even function and is consistent with this: it
averages the kink to zero.

### 7.4 The $\varepsilon$ expansion, measured on a paired ladder

`a13` P1, equicorrelated $\rho=0$ ($\lambda=1$), $\sigma=0.6$, naive-Bayes, 40 000
preference draws **held fixed across every $\varepsilon$** (so the differences below are
exact, not Monte Carlo):

| $\varepsilon$ | $0$ | $10^{-10}$ | $10^{-8}$ | $10^{-6}$ | $10^{-4}$ | $10^{-3}$ | $4\!\cdot\!10^{-3}$ | $10^{-2}$ | $4\!\cdot\!10^{-2}$ | $10^{-1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| $\mathbb{E}[G]$ | **2.3095** | 1.4647 | 1.4636 | 1.4637 | 1.4674 | 1.4674 | 1.4488 | 1.4127 | 1.2777 | 1.1091 |

* **Jump at $\varepsilon=0$: $+0.845$, i.e. $+57.7\%$** over the right limit. (The
  numericist measures $+23.6\%$ on the certainty-equivalent rule with a different tie
  convention. Different conventions, same sign, same order.)
* Fit on $\varepsilon\le0.04$: $\Delta G=c_1\sqrt\varepsilon+c_2\varepsilon$ gives
  $c_1=+0.0194$, $c_2=-4.800$, $R^2=0.99731$. A **pure linear** fit gives $c=-4.695$,
  $R^2=0.99727$ — indistinguishable. At $\lambda=1$ the cusp is real but tiny
  ($0.13\%$ of $\mathbb{E}[G]$ at $\varepsilon=0.01$), and the two terms cross at
  $\varepsilon^\dagger=(c_1/|c_2|)^2=1.6\times10^{-5}$.

**The cusp coefficient is strongly $\lambda$-dependent, and in the aligned regime it
dominates.** Same ladder at equicorrelated $\rho=0.8$ ($\lambda=0.447$):

| $\varepsilon$ | $0$ | $10^{-10}$ | $10^{-6}$ | $10^{-4}$ | $10^{-3}$ | $4\!\cdot\!10^{-3}$ | $10^{-2}$ | $4\!\cdot\!10^{-2}$ | $10^{-1}$ |
|---|---|---|---|---|---|---|---|---|---|
| $\mathbb{E}[G]$ | **7.1857** | 6.5409 | 6.5418 | 6.5803 | 6.6394 | **6.6531** | 6.5896 | 6.1310 | 5.4436 |

$c_1=\mathbf{+3.387}$, $c_2=-27.26$, $R^2=0.99592$; the **pure linear** fit collapses to
$R^2=0.70702$. Here the cusp is worth $5.2\%$ of $\mathbb{E}[G]$ at $\varepsilon=0.01$,
$\mathbb{E}[G]$ genuinely *rises* on $[0^+,\,4\times10^{-3}]$, and the crossover is at
$\varepsilon^\dagger=(3.387/27.26)^2=1.5\times10^{-2}$.

A third ladder — the *implemented* DGP at $\alpha=0$, which also has $\lambda=1$ — gives
$c_1=+0.0179$, $c_2=-4.736$, $R^2=0.99855$, jump $+57.2\%$: within $8\%$ of the
equicorrelated $\lambda=1$ ladder on every coefficient. That is a fourth, independent
confirmation of the $\lambda$-reduction of §5: two structurally different samplers at the
same $\lambda$ produce the same $\varepsilon$-expansion.

So the expansion is

$$\mathbb{E}[G](\varepsilon)-\mathbb{E}[G](0^+)\;=\;c_1(\lambda)\sqrt{\varepsilon}\;+\;c_2(\lambda)\,\varepsilon+o(\varepsilon),\qquad
c_1\in[0.018,\,3.39],\ \ c_2\in[-27.3,\,-4.7],$$

with $c_1\ge0$ and $c_2<0$ over the range probed. Either reading refutes the canonical
$(1+\varepsilon)$ factor, and they refute it *differently*: the leading term is
$\sqrt\varepsilon$, not linear, so $\partial\mathbb{E}[G]/\partial\varepsilon|_{0^+}=+\infty$
and there is no first-order coefficient to have a sign; and the first term that *is*
linear has $c_2<0$ against a canonical prediction of $+\mathbb{E}[G](0^+)$ — wrong sign,
$3$–$4\times$ the magnitude.

### 7.5 $\sigma$: exact, and for a reason that empties it

$G/\sigma$ over $\sigma\in\{0.2,\dots,1.0\}$, implemented $\alpha=0$, $\varepsilon\to0^+$
(`a06` S1): ratios $1.0000,\,0.9794,\,0.9735,\,0.9698,\,0.9510$ (SNE) and
$1.0000,\,0.9793,\,0.9736,\,0.9747,\,0.9527$ (naive). The predicted profile is
$\sigma_{\mathrm{eff}}(\sigma)/\sigma$ (flat to $\sigma=0.8$, $0.980$ at $\sigma=1.0$)
times the $O((0.05/\sigma)^2)$ correction of (4.3).

The sharpest version of this is `a15b` X2, which puts my computation, the numericist's
independent one, and the closed-form prediction side by side at $\varepsilon=0.25$,
implemented $\alpha=0$:

| $\sigma$ | mine, $G(\sigma)/G(0.6)$ | numericist | $\sigma_{\mathrm{eff}}(\sigma)/\sigma_{\mathrm{eff}}(0.6)$ | excess over prediction | (4.3) predicts |
|---|---|---|---|---|---|
| $0.2$ | $0.3484$ | $0.3455$ | $0.3333$ | $\mathbf{+4.53\%}$ | $\tfrac{3}{16}(0.05/0.2)^2\lambda^2/\operatorname{Var}(\mu)=\mathbf{+4.69\%}$ |
| $0.6$ | $1.0000$ | $1.0000$ | $1.0000$ | — | — |
| $1.0$ | $1.6247$ | $1.6283$ | $1.6334$ | $-0.53\%$ | $+0.17\%$ |

**Both effects, in both directions, are accounted for by closed forms derived rather than
fitted**, and both codebases see the same $+4.5\%$ excess at $\sigma=0.2$ against a
predicted $+4.7\%$. The residual at $\sigma=1.0$ is $0.5\%$.

### 7.6 Direct answer to Deliverable 2

> Is $\sigma(1+\varepsilon)/(1+\alpha)$ — up to affine reparameterisation — the
> first-order expansion of $\mathbb{E}[G]$ about $(\alpha,\varepsilon)=(0,0)$ at fixed
> $\sigma$?

**No, and not up to any affine reparameterisation.**

| factor | canonical | exact $\mathbb{E}[G]$ | match |
|---|---|---|---|
| degree-1 $\sigma$ | yes | yes, exactly (Prop. 5.1); deviations are $\sigma_{\mathrm{eff}}$ and $O((0.05/\sigma)^2)$, both derived | ✅ but **vacuous** — a units identity |
| $\varepsilon$ slope | $+\sigma$, finite | $+\infty$ (a $\sqrt\varepsilon$ cusp) below $\varepsilon^\dagger=1.6\times10^{-5}$; $-4.80$ above it | ❌ order, then ❌ sign |
| $\varepsilon$ sign | strictly $+$ | negative under Bayesian beliefs at every $\varepsilon$ in every DGP; non-monotone under certainty-equivalence | ❌ |
| $\alpha$ slope, implemented | $-\sigma$, smooth | one-sided $\mp1.54$ (SNE): a **kink**, symmetric derivative 0 | ❌ shape |
| $\alpha$ slope, renormalised | $-\sigma$ | $\mathbf{0}$ exactly — $\rho(\alpha)=O(\alpha^2)$ | ❌ magnitude |
| $\alpha$ slope, equicorrelated | $-\sigma$ | $+4.30$ (naive), $-1.28$ (SNE) | ❌ sign is protocol-dependent |
| $\alpha$ **shape** | smooth, signed, monotone | an exact function of $\lvert\alpha\rvert$ (Thm 7.1) | ❌ |
| separability $\sigma g(\varepsilon)h(\alpha)$ | assumed | $\lambda$ and $\varepsilon$ enter through the same object $\kappa$ (6.3); $\partial G/\partial\varepsilon$ changes sign with $\lambda$ | ❌ |

An affine map $A+B\,\sigma(1+\varepsilon)/(1+\alpha)$ repairs none of it: it cannot
manufacture a $\sqrt\varepsilon$ cusp from a linear term, it cannot make a smooth function
of $\alpha$ equal a function of $|\alpha|$, and it needs $B>0$ to keep the $\sigma$ scale
positive while needing $B<0$ to get the $\varepsilon$ and $\alpha$ signs right.

### 7.7 What the alignment "sign flip" actually is

$\tilde\Psi(\lambda)$ is **not monotone**: it has an interior minimum at
$\lambda^\ast(\text{protocol},\varepsilon)$ separating the herding branch (§5.3) from the
contention branch (§5.2). `a16` W1, $\sigma=0.6$, probing $\lambda$ from $0.05$ to $3.0$
— i.e. straddling the equicorrelated design's positive-semidefiniteness cap
$\lambda\le\sqrt{n/(n-1)}=1.1547$:

| naive protocol | $\varepsilon=0$ | $\varepsilon=0.01$ | $\varepsilon=0.25$ | $\varepsilon=1.0$ |
|---|---|---|---|---|
| $\lambda^\ast$ | $\mathbf{1.60}$ | $\mathbf{1.35}$ | $1.00$ | $0.75$ |
| $\tilde\Psi(\lambda^\ast)$ | $0.241$ | $0.170$ | $0.123$ | $0.096$ |
| vs the PSD cap $1.1547$ | **above** | **above** | inside | inside |

At $\varepsilon=0$, $\tilde\Psi$ falls monotonically from $0.604$ at $\lambda=0.05$ to
$0.241$ at $\lambda^\ast=1.60$ and then rises ($0.294,\,0.434,\,0.668$ at
$\lambda=2.0,2.5,3.0$ — the contention branch). **At $\varepsilon\le0.01$, $\lambda^\ast$
lies ABOVE the entire reachable design range**, so the design is confined to the
*decreasing* branch and $\partial G/\partial\rho>0$ everywhere.

That single fact explains the whole "alignment effect" result on this environment.
It also unifies §5.3 and §6.2: **$\lambda^\ast$ falls monotonically in $\varepsilon$
($1.60\to0.75$) by the same mechanism that makes $\kappa$ fall — noise de-synchronises
the herd, shrinking the herding branch until it no longer covers the design.** By
$\varepsilon=0.25$ the minimum is inside the range, which is precisely the numericist's
observation that "whatever alignment effect exists lives almost entirely at
$\varepsilon=0$": at $\varepsilon=0.25$ the whole reachable $\tilde\Psi$ range is
$0.123$–$0.171$, i.e. flat.

**The sign of the alignment effect is therefore not a property of the environment and not
even a property of the delegation concept — it is the question of which side of
$\lambda^\ast(\text{protocol},\varepsilon)$ the design happens to sit on.** Contention-branch
refits are in `a16_out.txt`.

The PSD cap deserves its own line: $\rho\ge-1/(n-1)$ means the "adversarial" arm is
structurally capped at $\rho=-1/3$ for $n=4$ and tightens to $\rho\to0^-$ as $n$ grows.
At $\rho=-1/3$ exactly, $\operatorname{Var}(\mu)=(1+3\rho)/4=0$: the four targets sum to
a constant, $\mu$ is deterministic and lands on the lattice point $0$, and the
**quantization floor vanishes identically** (`a17`). It is the one point in the design
where the principal has no quantization loss at all.

---

## 8. Deliverable 1, secondary: per-principal incidence

Define $G_i=w_i\big[\mathbb{E}_\pi(x-\tau_i)^2-\mathbb{E}_{\rm plan}(x-\tau_i)^2\big]$ per
resource, summed over resources. The algebra of (3.1) with $\mu\!\to\!\tau_i$,
$\bar W\!\to\!w_i$ gives $\sum_i G_i=G$ **identically** — verified at
$4.5\times10^{-14}$ (`a06` S7) and $1.4\times10^{-14}$ (numericist).

$G_i$ **can be negative**: the ensemble-optimal plan tracks $\mu$, not $\tau_i$, so a
principal whose target is far from $\mu$ can be better off under the delegates' churn than
under the plan. At $\sigma=0.6$, implemented DGP, **episode totals over $m=3$ resources**
(`a18` I1, 30 000 draws per cell; equal stakes $s_i\equiv0.6$):

| protocol | $\alpha$ | $\varepsilon$ | $\mathbb{E}[G]$ | $P(G_i<0)$ | $P(\exists i: G_i<0)$ | $\mathbb{E}[\min_i G_i]$ | $\mathbb{E}[\max_i G_i]$ |
|---|---|---|---|---|---|---|---|
| naive | $0$ | $\to0^+$ | $1.471$ | $0.129$ | $0.278$ | $-0.026$ | $+0.900$ |
| naive | $0$ | $0.25$ | $0.885$ | $0.219$ | $0.528$ | $-0.177$ | $+0.717$ |
| naive | $0$ | $1.0$ | $0.685$ | $0.244$ | $0.622$ | $-0.348$ | $+0.738$ |
| naive | $0.8$ | $\to0^+$ | $6.938$ | $0.022$ | $0.068$ | $+1.505$ | $+1.969$ |
| naive | $0.8$ | $1.0$ | $2.210$ | $0.039$ | $0.117$ | $+0.420$ | $+0.686$ |
| SNE | $0$ | $\to0^+$ | $1.137$ | $0.249$ | $0.600$ | $-0.589$ | $+1.376$ |
| SNE | $0$ | $0.25$ | $0.525$ | $0.250$ | $0.652$ | $-0.251$ | $+0.633$ |

$\sum_iG_i=G$ holds to $\le2.5\times10^{-13}$ on every row.

So **delegation surplus is a per-principal phenomenon and a common one** — a quarter of
principal-draws and **up to $65\%$ of ensemble draws** have someone strictly gaining from
delegation relative to the ensemble-optimal plan — even though it is identically zero at
the ensemble level (Cor. 3.2). This is the object the corpus's "unoccupied novelty strip"
should be attached to. The numericist's complementary statistic runs to $63.5\%$ (naive)
and $90.2\%$ (stage-Nash) of draws.

Note the structure: **the incidence is most unequal exactly where the aggregate gap is
smallest.** At $\alpha=0.8$ the gap is $6.9$ and essentially every principal bears a
positive share ($P(G_i<0)=0.022$, $\min_i G_i=+1.51$); at $\alpha=0$ the gap is $1.5$ and
a quarter of principals are net beneficiaries. An aggregate-only friction measure is
blindest precisely where distributional conflict is sharpest.

**The stakes-weighted incidence $\sum_i s_iG_i/\sum_i s_i$ ($s_i=\sum_m w_{im}$) equals
$G/n$ to within $0.5$–$1.9\%$ under the implemented sampler** (`a18` I1: ratios $1.005$
to $1.019$), because $w\sim N(\sigma,0.05)$ gives every principal the same stake up to a
$0.05$ wobble. It only becomes a distinct object under a stakes design that varies $s_i$
*across* principals — which, by Proposition 5.1, leaves the delegated chain **exactly
unchanged** and moves only $\bar W$ and the benchmark point $\mu$. That configuration is
worth building (the burden should shift toward the low-stakes principals, whose targets
$\mu$ stops tracking) but the heterogeneous-stakes arm of `a18` was retired for cost
along with the other SNE arms; it is listed as not established in §11.

---

## 9. Validation

### 9.1 Two independent codebases agree across the whole design

`a15` compares this library (independent code, $T=100$ exact paired planner) against the
numericist's `core.py` **Cesàro** column, rule = naive, beliefs = Bayes, $\sigma=0.6$,
across all three DGP families and every control level. This is the single strongest check
in either report, because the two implementations share no code and were written from the
source independently.

**56 cells** (three families $\times$ every control level $\times$ four $\varepsilon$).
Full table in `a15_out.txt`; re-serialised with summary statistics in
`a15b_results.json`. Sampled rows:

| family | control | $\varepsilon$ | mine | numericist | ratio |
|---|---|---|---|---|---|
| implemented | $+0.000$ | $0$ | $2.2791\pm0.0210$ | $2.2430\pm0.0888$ | $1.016$ |
| implemented | $+0.400$ | $0.25$ | $1.1369\pm0.0136$ | $1.1527\pm0.0609$ | $0.986$ |
| implemented | $+0.800$ | $0$ | $7.0402\pm0.0536$ | $7.1227\pm0.2476$ | $0.988$ |
| implemented | $-0.800$ | $1.00$ | $2.2131\pm0.0190$ | $2.2759\pm0.0885$ | $0.972$ |
| renormalised | $+0.400$ | $0.25$ | $1.8490\pm0.0185$ | $1.8472\pm0.0811$ | $1.001$ |
| renormalised | $+0.800$ | $0$ | $8.0707\pm0.0551$ | $8.1045\pm0.2529$ | $0.996$ |

Summary: **$\varepsilon>0$ (42 cells) median ratio $0.9702$, IQR $[0.9519,\,0.9873]$;
$\varepsilon=0$ (14 cells) median ratio $0.9878$, range $[0.9387,\,1.0161]$.** Two
independently written exact solvers of the same object agree to a systematic $\sim3\%$
and mostly within one to two combined standard errors. The $T=100$-vs-Cesàro design
difference cancels precisely because this library benchmarks against the planner's own
backward induction from the same start (§2.1): the numericist's own $T=100$ column sits
$\approx0.50$ higher than their Cesàro column ($2.742$ vs $2.243$ at implemented
$\alpha=0$, $\varepsilon=0$), which is their measured planner transient of $0.535$/step.

One correction to an expectation I held going in: I expected the two codebases to
**disagree** at $\varepsilon=0$ because of the tie convention. They do not — agreement at
$\varepsilon=0$ (median $0.988$) is if anything *better* than at $\varepsilon>0$. The two
tie rules happen to coincide on this environment. **The $\varepsilon=0$ knife-edge of §6.3
therefore rests entirely on the within-implementation jump $\mathbb{E}[G](0)$ vs
$\mathbb{E}[G](0^+)$ (a13, a08), not on any cross-implementation disagreement** — which
is the stronger evidence anyway, since it is a paired, deterministic comparison rather
than a comparison of two Monte Carlo estimates.

### 9.2 The $\lambda$-collapse identity

If §5's reduction is right, the renormalised DGP at $\alpha$ and the equicorrelated DGP
at $\rho(\alpha)$ are the *same problem* — same $\lambda$, same $v_\tau=1$ — even though
they are different samplers. `a05` D4 tests it directly:

| $\alpha$ | $\rho(\alpha)$ | renorm (var, corr) | equicorr (var, corr) | $\mathbb{E}[G]$ renorm | $\mathbb{E}[G]$ equicorr |
|---|---|---|---|---|---|
| $0.4$ | $0.3077$ | $(1.0013,\,0.3116)$ | $(1.0008,\,0.3059)$ | $0.25310\pm0.00427$ | $0.25172\pm0.00430$ |
| $0.8$ | $0.9412$ | $(1.0007,\,0.9415)$ | $(0.9996,\,0.9409)$ | $0.11439\pm0.00341$ | $0.11805\pm0.00354$ |

Agreement within $0.3$ and $0.7$ standard errors.

The underlying map is verified independently: `a12b` compares the closed forms of §4 for
$(v_\tau,\rho,\tfrac34\lambda^2,\operatorname{Var}\mu)$ against $600\,000$-draw Monte
Carlo on **15 (DGP, knob) rows** covering all three samplers. Worst relative error over
every quantity and every row is $\le0.5\%$; the three rows that print a larger number are
artefacts of dividing by $\rho=0$ exactly (the MC estimate is $|\hat\rho|\le0.0009$).
And `a13` P1's third ladder gives a fourth check: the *implemented* DGP at $\alpha=0$ and
the *equicorrelated* DGP at $\rho=0$ — different samplers, same $\lambda=1$ — produce
$\varepsilon$-expansions agreeing on every coefficient to $8\%$ ($c_1$: $0.0179$ vs
$0.0194$; $c_2$: $-4.736$ vs $-4.800$; jump: $+57.2\%$ vs $+57.7\%$).

**The two-parameter reduction $\Psi(\lambda,v_\tau;\varepsilon)$ is not a fit; it is an
identity the DGPs satisfy.**

### 9.3 The belief-irrelevance lemma, and the decisive test

`a11b` tests Lemma 6.1 two ways.

**(a) The disagreement set is a boundary layer.** Comparing the clamp-aware Bayes argmin
against the certainty-equivalent argmin over a $2401\times1500$ grid of $(u,\tau)$:

| $\varepsilon$ | $0.01$ | $0.05$ | $0.25$ | $0.5$ | $1.0$ |
|---|---|---|---|---|---|
| fraction of cells disagreeing | $3.79\%$ | $6.62\%$ | $9.13\%$ | $10.74\%$ | $12.41\%$ |
| p99 of distance-to-wall among disagreements | $1.51$ | $1.98$ | $1.98$ | $1.98$ | $1.99$ |
| max distance-to-wall | $1.54$ | $2.00$ | $2.53$ | $3.05$ | $3.39$ |

99% of disagreements lie within $2.0$ of a wall, in a box of width $10$ — a layer of
width one action-step plus a couple of noise SDs, exactly as Lemma 6.1 predicts (the
clamp is relevant iff a $\pm1$ move can reach it, so the layer is $1+O(\sqrt\varepsilon)$,
not $O(\sqrt\varepsilon)$).

**(b) The decisive test.** Run the *same* DGP twice — once at its native location
($\mu\sim N(0,\cdot)$, so about half the resource optima sit on the lower wall) and once
shifted $+5$ into the middle of the box so the clamp is slack — under both belief models,
naive protocol, $\sigma=0.6$, implemented $\alpha=0$:

| location | $\varepsilon$ | $10^{-4}$ | $0.01$ | $0.05$ | $0.25$ | $0.5$ | $1.0$ | sign of $\Delta G$ |
|---|---|---|---|---|---|---|---|---|
| **native** | Bayes | $1.468$ | $1.429$ | $1.255$ | $0.896$ | $0.754$ | $0.690$ | $-$ |
| | CE | $2.377$ | $2.618$ | $2.838$ | $3.476$ | $4.127$ | $5.139$ | $+$ |
| | wedge | $-38.3\%$ | $-45.4\%$ | $-55.8\%$ | $-74.2\%$ | $-81.7\%$ | $\mathbf{-86.6\%}$ | **opposite** |
| **shifted $+5$** | Bayes | $20.866$ | $22.005$ | $21.757$ | $19.746$ | $17.979$ | $16.190$ | $-$ |
| | CE | $20.866$ | $22.005$ | $21.757$ | $19.746$ | $17.980$ | $16.208$ | $-$ |
| | wedge | $0.00\%$ | $0.00\%$ | $0.00\%$ | $-0.00\%$ | $-0.01\%$ | $\mathbf{-0.11\%}$ | **same** |

**The entire belief-model wedge — the largest single sensitivity in either report, worth
up to $87\%$ of $\mathbb{E}[G]$ and enough to flip the sign of
$\partial\mathbb{E}[G]/\partial\varepsilon$ — collapses to $\le0.11\%$ the moment the
clamp stops binding.** It is not a fact about noisy delegation. It is a fact about a
quadratic loss evaluated through a $\mathrm{clip}(\cdot)$, in an environment whose targets
are drawn on an $N(0,1)$ scale inside a $[0,10]$ box.

Two consequences for the corpus:

1. Any proposition about $\partial F/\partial\varepsilon$ on this environment is a
   proposition about the boundary, not about entropy. The Morris–Shin/Prat rescoping the
   decision page contemplates is the right move but for a different reason than assumed.
2. The numericist's warning that "any closed form for $G$ that does not name a belief
   model is under-specified by a factor of two" is correct **and now has a mechanism**:
   the factor of two is the measure of the state distribution sitting on a wall. Move the
   targets off the wall and the belief model becomes irrelevant to four decimal places.

### 9.4 Against the trained IQL agents (the point of the exercise)

`a10` X2 compares the exact-best-response delegates against the actual trained Q-learners
reported in `ideation_measurement.md` §0.1. Converting the ensemble welfare $W$ to the
per-agent units the runner reports (`run_gpu.py:419` is `float(eval_clean[r].mean())` — a mean
over agents — so the ensemble $W$ must be divided by $n=4$):

| condition | naive-myopic $W/4$ | stage-Nash(best) $W/4$ | trained IQL `eval_reward_clean` |
|---|---|---|---|
| $\alpha=-0.8,\ \sigma=0.6$ | $-2.558$ | $-0.848$ | $-1.309$ |
| $\alpha=+0.0,\ \sigma=0.6$ | $-2.323$ | $-1.807$ | $\mathbf{-2.287}$ |
| $\alpha=+0.8,\ \sigma=1.0$ | $-4.201$ | $-1.372$ | $-2.287$ |

**The trained agents are bracketed by the two exact concepts in all three conditions, and
at $\alpha=0$ they land on the naive-myopic value to $1.6\%$.** This is the empirical
warrant for treating the naive-myopic branch — the one whose alignment slope is
*positive* — as the concept the corpus's own experiments implement. `run_gpu.py:79-99`
trains $n$ independent Q-networks on their own observations with no joint-action
conditioning: IQL is the uncoordinated family.

### 9.5 Two corrections to the record

**(a) A label error in the prior analytic run.** `a06_final.py:47` evaluates
`gap_multi_sigma(..., max(eps, EPS0), ...)` with `EPS0 = 1e-4`, so every column printed as
"$\varepsilon=0$" in `a06_out.txt` is in fact $\varepsilon=10^{-4}$, i.e. the
$\varepsilon\to0^+$ anchor. This memo relabels them. It is not an error of substance —
§6.3 shows $\varepsilon=0$ is the convention-dependent point and $0^+$ is the meaningful
limit — but it explains an apparent $60\%$ discrepancy between `a06` ($1.414$) and both
`a15` ($2.279$) and the numericist ($2.243$) at implemented $\alpha=0$, $\sigma=0.6$.
Once relabelled, all three agree.

**(b) The floor constant.** The formula circulating in this workstream,
$\mathbb{E}[d^2]\approx\operatorname{Var}(\mu)/2+1/24$, double counts: the $1/24$ is the
*interior* constant and the interior branch only carries the mass $\mu>0$. `a17`
($8\times10^6$ draws per row) gives the corrected forms and their errors:

| $\operatorname{Var}(\mu)$ | exact | $\operatorname{Var}(\mu)/2+\tfrac1{48}$ | err | $\operatorname{Var}(\mu)-1.0638\operatorname{Var}(\mu)^{3/2}$ | err | $\operatorname{Var}(\mu)/2+\tfrac1{24}$ | err |
|---|---|---|---|---|---|---|---|
| $0.01$ | $0.008927$ | $0.025833$ | $+189\%$ | $0.008936$ | $\mathbf{+0.10\%}$ | $0.04667$ | $+423\%$ |
| $0.0625$ | $0.046053$ | $0.052083$ | $+13.1\%$ | $0.045877$ | $\mathbf{-0.38\%}$ | $0.07292$ | $+58.3\%$ |
| $0.25$ | $0.145488$ | $0.145833$ | $\mathbf{+0.24\%}$ | $0.117019$ | $-19.6\%$ | $0.16667$ | $+14.6\%$ |
| $0.625$ | $0.332838$ | $0.333333$ | $\mathbf{+0.15\%}$ | — | — | $0.35417$ | $+6.4\%$ |
| $1.0$ | $0.520041$ | $0.520833$ | $\mathbf{+0.15\%}$ | — | — | $0.54167$ | $+4.2\%$ |
| $1.3333$ | $0.686405$ | $0.687483$ | $\mathbf{+0.16\%}$ | — | — | $0.70832$ | $+3.2\%$ |

$$\boxed{\ \mathbb{E}[d^2]=\frac{\operatorname{Var}(\mu)}{2}+\frac{1}{48}\quad(\operatorname{Var}(\mu)\gtrsim0.25,\ \text{err}\le0.25\%);\qquad
\mathbb{E}[d^2]=\operatorname{Var}(\mu)-\tfrac43\sqrt{\tfrac{2}{\pi}}\operatorname{Var}(\mu)^{3/2}\quad(\operatorname{Var}(\mu)\lesssim0.1,\ \text{err}\le1\%).\ }$$

The interior residual $\mathbb{E}[d^2\mathbf 1\{\mu>0\}]$ is measured at
$0.0207$–$0.0209$ across $\operatorname{Var}(\mu)\in[0.25,1.33]$ against
$1/48=0.020833$ — which is what pins the constant. The small-$\operatorname{Var}(\mu)$
coefficient is exact: $\tfrac43\cdot\tfrac{2}{\sqrt{2\pi}}=1.063846$, from
$\mathbb{E}[d^2\mid\mu]=\mu^2-\tfrac43\mu^3$ for $0<\mu<\tfrac12$, $f\sim U(0,1)$, and
$\mathbb{E}[\mu^3\mathbf1\{\mu>0\}]=\operatorname{Var}(\mu)^{3/2}\cdot2/\sqrt{2\pi}$.

Floors at $\sigma=0.6$ ($\mathbb{E}[\bar W]=2.4$, $m=3$): implemented $\alpha=0$ →
$1.05$; implemented $|\alpha|=0.8$ → $2.49$; renormalised $|\alpha|=0.8$ → $3.59$;
equicorrelated $\rho=-1/3$ → **$0.00$ exactly**.

---

## 10. Deliverable 3: verdict input

> Is the canonical $F$ a linearisation of $G$ under stated restrictions (ADDITION), or is
> it not any limit of $G$ (RETRACTION)?

### **RETRACTION**, with the survivor set below.

The canonical $F=\sigma(1+\varepsilon)/(1+\alpha)$ is not a first-order expansion of
$\mathbb{E}[G]$ about $(\alpha,\varepsilon)=(0,0)$ on this environment, and no affine
reparameterisation makes it one. One of its three factors is exact but empty; the other
two fail on **order and sign**, not on magnitude. There is no restriction of the domain
that rescues them, because the failures are structural: Theorem 7.1 (evenness in
$\alpha$) holds on the whole design and for every protocol; §6.3's cusp holds at the
expansion point itself; §6.2's $\kappa\ge2$ holds generically at small $\varepsilon$.

**Survives — put these in the scope-conditioned rewrite:**

| claim | status | why |
|---|---|---|
| $F\ge0$ | ✅ **stronger than claimed: a theorem** (Thm 3.1) | The delegated behaviour is a feasible policy for the principal and the benchmark optimises over a superset. Verified pointwise ($\min G\approx-3\times10^{-13}$). $F:=G_+$ costs nothing at the ensemble level. |
| $\partial\rho_S/\partial F<0$; the legitimacy carve | ✅ | Depends only on $F\ge0$ and comparability. |
| $\partial F/\partial\sigma>0$, degree-1 homogeneous | ✅ **exact** (Prop. 5.1) | But it is a **units identity**: the delegated policy is provably $w$-free, so any reward-unit DV is degree-1 homogeneous. State it as a normalisation, not a result. Under the dimensionless $\tilde F=G_+/S_{\rm ref}$ of the decision page's open fork, $\partial\tilde F/\partial\sigma=0$ exactly — the fork is real and this settles which side of it $\sigma$-homogeneity lives on. |
| $F=0\iff\sigma=0$ | ✅ a.s. true here, **for a different reason than the paper gives** | $G=\sum_m\bar W_m\Delta_m$ with $\Delta_m\ge0$, $\bar W\propto\sigma$, and $\Delta>0$ in every draw computed. It survives as a **quantization/coordination fact about $\{-1,0,1\}$ against a unit deadband**, not as a consequence of $(1+\varepsilon)/(1+\alpha)>0$. The proof at `02_axiom.tex:179-183` must be replaced. |
| **Delegation surplus $B$, per principal** | ✅ **real and large** (§8) | $P(G_i<0)$ up to $25\%$; at least one principal gains in up to $63.5\%$ (naive) / $90.2\%$ (stage-Nash) of draws. **But $B\equiv0$ at the ensemble level against an optimal principal (Cor. 3.2)**: the novelty strip must be re-typed as a per-principal claim, or it requires a boundedly rational benchmark. |

**Dies — no scope condition rescues these:**

| claim | why |
|---|---|
| $\partial F/\partial\varepsilon>0$ (Prop. Entropy Effect) | Below $\varepsilon^\dagger$ there is no first-order term to have a sign ($\sqrt\varepsilon$ cusp, right-derivative $+\infty$); above it the slope exists and is $c_2\in[-27.3,-4.7]$ against a canonical $+\mathbb{E}[G](0^+)$. The mechanism is derived, not fitted: noise cuts the aggregate feedback gain $\kappa$ (6.3), de-synchronises the herd, and shrinks $\lambda^\ast$ from $1.60$ to $0.75$. Morris–Shin/Prat is not an exotic corner here; **it is the modal case.** And the *one* configuration in which the slope comes out positive — certainty-equivalent beliefs — is shown in §9.3 to be a clamp artefact that vanishes ($86.6\%\to0.11\%$) once the targets are moved off the wall. |
| $\partial F/\partial\alpha<0$ (Prop. Alignment Effect) | Theorem 7.1: $\mathbb{E}[G](\alpha)=\mathbb{E}[G](-\alpha)$ **identically** under both $\alpha$-DGPs, so no signed $\alpha$ effect can exist on this design. The first-order coefficient is $0$ (renormalised, exactly) or a kink with symmetric derivative $0$ (implemented). On the one signed axis ($\rho$), the sign is $+4.30$ under the delegation concept the experiments implement. |
| Multiplicative separability $\sigma g(\varepsilon)h(\alpha)$ | $\lambda$ and $\varepsilon$ act through the same object, $\kappa$; $\partial G/\partial\varepsilon$ changes sign as $\lambda$ falls. |
| The irreducible $\sigma/2$ baseline | The quantization floor is the object that plays this role; it is large ($1.05$–$3.59$ welfare units at $\sigma=0.6$), it is now in closed form to $0.25\%$ (§9.5b), and it **cancels identically** in any difference-against-benchmark. It is also *zero* at $\rho=-1/3$, which no $\sigma/2$ baseline can represent. |
| Prop. Modular Governance Fragility; additive Aggregation | Algebraic consequences of the functional form that just failed. |
| The "12/12 Lean mapping" | Unaffected in truth-value; three theorems now map to withdrawn propositions. |

### 10.0 The head-to-head, on the corpus's own 5×5×5 design

`a14b`: all 125 cells of $(\alpha,\sigma,\varepsilon)$ recomputed at $R=15\,000$ draws
$\times$ 3 resources, naive-myopic delegation (the family §9.4 shows the trained agents
fall into), OLS on cell means. Everything the closed form uses —
$\sigma_{\mathrm{eff}}(\sigma)$ from (5.1), $\lambda(\text{knob})$ from §4 — is **derived,
not fitted**; only $a_0(\varepsilon)$ and $\kappa_c(\varepsilon)$ are free.

**Implemented DGP:**

| model | free coefficients | $R^2$ | AICc | $\Delta$AICc |
|---|---|---|---|---|
| $mn\,\sigma_{\mathrm{eff}}\big(a_0(\varepsilon)+\kappa_c(\varepsilon)\lambda^2\big)$, quadratic in $\varepsilon$ | 6 | $\mathbf{+0.833}$ | $-3.20$ | $\mathbf{0}$ |
| the same, linear in $\varepsilon$ | 4 | $+0.772$ | $31.45$ | $34.6$ |
| $a+b\,\sigma(1+\varepsilon)/(1+\alpha)$ | 2 | $+0.106$ | $197.98$ | $201.2$ |
| a constant | 1 | $0.000$ | $209.97$ | $213.2$ |
| $c\,\sigma(1+\varepsilon)/(1+\alpha)$ | 1 | $\mathbf{-0.211}$ | $233.86$ | $237.1$ |

**The canonical form fitted as a single scaled regressor has negative $R^2$: it fits worse
than a constant.** Even given a free intercept and slope it is $201$ AICc units behind a
closed form with four more coefficients — a margin that no information criterion softens.
This independently reproduces the numericist's `FORM_FITS.csv` result (canonical $R^2<0$
in 14 of 16 subsets) on a separate codebase.

The marginal means say the same thing without any model at all:

| factor | levels | $\mathbb{E}[G]$ | canonical $F$ |
|---|---|---|---|
| $\alpha$ | $-0.8,-0.4,0,+0.4,+0.8$ | $3.956,\ 1.354,\ 1.087,\ 1.357,\ 3.902$ | $4.5,\ 1.5,\ 0.9,\ 0.643,\ 0.5$ |
| $\sigma$ | $0.2\dots1.0$ | $0.787,\ 1.565,\ 2.346,\ 3.127,\ 3.830$ | $0.536,\ 1.072,\ 1.609,\ 2.145,\ 2.681$ |
| $\varepsilon$ | $0\dots1.0$ | $4.603,\ 2.460,\ 1.805,\ 1.480,\ 1.308$ | $1.072,\ 1.340,\ 1.609,\ 1.877,\ 2.145$ |

* $\alpha$: $\mathbb{E}[G]$ is **visibly symmetric** ($3.956$ vs $3.902$; $1.354$ vs
  $1.357$) — Theorem 7.1 in the raw marginals. $F$ is monotone. It happens to get the
  $\alpha=-0.8$ end qualitatively right and the $\alpha=+0.8$ end exactly backwards.
* $\sigma$: $\mathbb{E}[G]/F$ is $1.468,\,1.460,\,1.458,\,1.458,\,1.428$ — **constant to
  $0.7\%$ over $\sigma\in[0.2,0.8]$.** The $\sigma$ factor is right.
* $\varepsilon$: $\mathbb{E}[G]$ is **monotone decreasing** across the whole design
  ($4.603\to1.308$, and still monotone if the convention-dependent $\varepsilon=0$ cell is
  dropped); $F$ is monotone increasing. Exactly opposite.

One reading of the fitted coefficients worth recording: $\kappa_c(\varepsilon=0)=-0.673$
comes out **negative** here. That is not a contradiction of §5.2 — the implemented DGP
spans $\lambda\in[0.2,1.0]$, entirely below $\lambda^\ast=1.60$, so the whole design sits
on the *herding* branch where $\tilde\Psi$ decreases in $\lambda$. The quadratic term is
doing the herding branch's work, not the contention branch's. A form fitted on this design
alone would therefore mis-sign the alignment effect **in the other direction** if
extrapolated past $\lambda^\ast$ — which is one more reason the replacement in §10.2 must
carry $\lambda^\ast$ explicitly rather than a single monotone $h(\alpha)$.

(The identical grid for the **renormalised** DGP was still computing when this memo was
written; it appends to `a14b_out.txt` under the same five models. Nothing in the verdict
turns on it: the renormalised DGP differs from the implemented one only in the
marginal-variance drift, its $\lambda$ range is *narrower* ($[0.24,1]$), and §7.3 shows its
$\alpha$-slope is $\approx0$ rather than a kink — so if anything the canonical form should
do worse there, as it does in the numericist's `FORM_FITS.csv`.)

### 10.0b The iso-$F$ fan test — the gate's own preferred discriminator

`ideation_gate.md` §2.2 identifies the **iso-$F$ fan** as the strongest available test and
the one thing that memo "gets exactly right": $F=\sigma(1+\varepsilon)/(1+\alpha)$ asserts
$G=f(F)$ for *some* $f$, so cells with equal $F$ but different $(\sigma,\alpha,\varepsilon)$
must have equal $G$. It needs no functional form for $f$. Run in `a06` S6 on the 8 $F$-levels
of the 5×5×5 grid that carry $\ge3$ cells:

| protocol | mean within-$F$ SD | pooled SD | ratio | worst within-$F$ spread |
|---|---|---|---|---|
| SNE | $0.6246$ | $0.6541$ | $\mathbf{0.955}$ | $8.97\times$ (at $F=1/3$) |
| naive | $1.1994$ | $1.4620$ | $\mathbf{0.820}$ | $\mathbf{19.06\times}$ (at $F=2/3$) |

**Conditioning on $F$ removes $4.5\%$ (SNE) / $18\%$ (naive) of the dispersion in $G$.**
Cells the canonical form declares identical differ by up to a factor of $19$. This is a
non-parametric refutation: it does not assume any $f$, any linearisation, or any
expansion point, and it fails on the gate's own chosen ground.

### 10.1 One caveat that cuts against over-claiming the replacement

The **stage-Nash** results — the only place the canonical sign is recoverable — rest on a
correspondence, not a function. `a02` T4b is the one structural check that FAILS: with the
clamp included, the pure Nash aggregate is not a.s. unique (mean $2.04$, max $5$ per
state), and the induced $\mathbb{E}[G]$ at $\varepsilon=0$ spans
$0.065\to73\to144$ across welfare-best / minimal-move / welfare-worst selection (`a09`
N1). The numericist's Gauss–Seidel-from-zero selection lands at $1.146$, inside that
bracket; their $\le5.3\%$ selection sensitivity describes that basin, not the
correspondence. **Nothing in this memo licenses "$\partial F/\partial\alpha<0$ holds under
stage-Nash" as a scope condition** — it holds under *a named selection rule* on a
correspondence the corpus has never specified.

### 10.2 What a replacement functional form would have to look like

$$\mathbb{E}[G]\;=\;m\,n\,\sigma_{\mathrm{eff}}(\sigma)\cdot\Psi\big(\lambda,\,v_\tau;\,\varepsilon\big),\qquad
\lambda^2=v_\tau(1-\rho),$$

with $\Psi$ non-monotone in $\lambda$ (minimum at $\lambda^\ast$), quadratic above it
($\Psi\approx\kappa_c\lambda^2+a_0$, $R^2=0.96$–$0.9997$, with $\kappa_c$ bracketed by
$n=4$ order statistics), and $\varepsilon$ entering through $\kappa$ and $\lambda^\ast$
rather than multiplicatively. Three of the corpus's four parameters survive in
recognisable form ($\sigma$ as a scale, $\rho$ as the alignment primitive, $\varepsilon$
as noise); what does not survive is that any of them acts *monotonically* or
*separably*.

---

## 11. Assumptions, and what is not established

* **B1 (beliefs).** Diffuse prior, posterior $N(\hat s,\varepsilon)$, expected loss taken
  through the clamp. Lemma 6.1 shows this is the *same rule* as certainty-equivalence
  except in the clamp boundary layer — so "which belief model" is not a free parameter of
  the analysis, it is a statement about how much of the state distribution sits on a wall.
  Under this environment's target scale, about half of it does.
* **N1 (equilibrium).** SNE is a damped fixed point on conjectures; it converges cleanly
  only for $\varepsilon\gtrsim0.25$ (residual $4\times10^{-5}$) and **fails to converge**
  at $\varepsilon\le10^{-2}$ (residual $1.6$–$2.7$) — `a05` D0. Multiplicity at small
  noise is the coordination game, not a solver defect. The exact $\varepsilon=0$
  correspondence is reported with its selection bracket rather than a point value.
* **T1 (ties).** Lexicographic, breaking toward the unclamped best response
  $\mathrm{clip}(\mathrm{round}(\tau-u),-1,1)$. Load-bearing **only at $\varepsilon=0$**,
  where it is worth up to $58\%$ of $\mathbb{E}[G]$ (§7.4). Every $\varepsilon>0$ number
  is convention-free.
* **Horizon.** $T=100$, matching `EPISODE_LENGTH`, with the planner's own transient inside
  the benchmark. Cesàro (stationary) results differ by the planner transient, $0.535$/step
  — §9.1 shows the two conventions reconcile exactly.
* **Monte Carlo.** $8\,000$–$40\,000$ preference draws per cell for the chain work,
  $4\times10^6$–$8\times10^6$ for the constants; all noise integration is exact
  (convolution, not sampling). SEs reported throughout.
* **Retired for cost.** The SNE arms of `a14`, `a16` and `a18` were killed mid-run: the
  damped conjecture fixed point costs $70$–$150$ s per cell at these sample sizes. Every
  result reported from those scripts is the **naive-myopic** arm, which §9.4 shows is the
  family the corpus's own trained agents fall into. The equilibrium concept is covered by
  `a05`, `a06`, `a09` here and by the numericist's `nash_ce` grid.
* **Not established here.** $n\ne4$; $m\ne3$; the two-team block DGP (`dgp.py:88`);
  learned policies (deliberately excluded — the point is to remove them); a stakes design
  with heterogeneous $s_i$ across principals (the one configuration in which the
  stakes-weighted incidence stops being $G/n$ — by Prop. 5.1 it leaves the delegated chain
  untouched, so it is cheap and should be done); $\lambda^\ast$ under the SNE and exact-Nash
  concepts; the behaviour of $\lambda^\ast$ as $n$ grows (the PSD cap $\rho\ge-1/(n-1)$
  tightens, so the reachable $\lambda$ range shrinks toward $[0,1]$ and the herding branch
  should dominate *more*, but this is a conjecture); and the units fork of the decision page
  ($F$ vs $\tilde F=G_+/S_{\rm ref}$), on which §10 records only that degree-1 homogeneity is
  exact for $F$ and therefore vacuous for $\tilde F$.

### 11.1 Scope

Everything above is *this* environment: 4 agents, 3 resources, a $\{-1,0,+1\}$ action set
against a unit deadband, targets on an $N(0,1)$ scale inside a $[0,10]$ box so that half
the resource optima sit on the lower clamp. The failure mode that drives both the
wrong-sign $\alpha$ result and the negative $\varepsilon$ result is over-correction with
aggregate feedback gain $\kappa\approx2$–$5$; a different environment could give a
different answer. What cannot be argued around is that **this is the environment the
corpus's own experiments run on**, that the trained agents sit on the naive-myopic branch
of it (§9.4), and that on it the canonical functional form fits worse than a constant.

---

## 12. Index of code and outputs

| script | what it establishes |
|---|---|
| `a00_lib.py` | the independent exact-chain library (lattice, planner, naive/SNE/exact-Nash, occupancies, DGPs) |
| `a01_env_verify.py` → `a01_out.txt` | 36/36 environment assertions against source |
| `a02_structure.py` → `a02_out.txt` | separability, two-threshold monotonicity, Nash characterisation vs brute force, $G\ge0$ pointwise, the $\lambda=0$ 2-cycle, T4b non-uniqueness |
| `a03`–`a06` | $\sigma$-homogeneity, $\Psi$ surfaces, decomposition identity, profiles, iso-$F$ fan, incidence |
| `a07_constants.py` → `a07_out.txt` | order-statistic constants $K1$, the $\sqrt\varepsilon$ rate $K2$, $\sigma_{\mathrm{eff}}$ $K3$ |
| `a08_eps_limit.py` → `a08_out.txt` | the $\varepsilon=0$ jump, the $\lambda^2$ law ($R^2=0.99966$, exponent $1.995$) |
| `a09_nash_dgp.py` → `a09_out.txt` | the exact Nash selection bracket; $\pm\alpha$ symmetry |
| `a10_crosscheck.py` → `a10_out.txt` | planner welfare and exact-BR vs the trained IQL agents |
| `a11b_belief_fast.py` → `a11b_out.txt` | **Lemma 6.1**: the boundary layer, and the decisive shift test ($86.6\%$ wedge → $0.11\%$) |
| `a12b_dgp_map.py` → `a12b_out.txt` | the $(v_\tau,\rho,\lambda,\operatorname{Var}\mu)$ closed forms vs MC, 15 rows, $\le0.5\%$ |
| `a13_eps_structure.py` → `a13_out.txt` | the paired $\varepsilon$ ladder, $c_1/c_2$, $\kappa$ closed form, $v_D$ saturation |
| `a14_floor_and_assembly.py` → `a14_out.txt` | the $1/24$ constant three ways (symbolic, MC, quadrature) |
| `a14b_headtohead.py` → `a14b_out.txt` | the 125-cell head-to-head: closed form $R^2=0.833$ vs canonical $R^2=-0.211$, $\Delta$AICc $=201$ |
| `a15_cross_implementation.py` → `a15_out.txt`, `a15b_sigma_xcheck.py` → `a15b_out.txt` | 56-cell agreement with the numericist's independent grid; the $\sigma$ cross-check |
| `a16_two_regimes.py` → `a16_out.txt` | $\lambda^\ast(\varepsilon)=1.60,1.35,1.00,0.75$; the herding vs contention branches; the PSD cap |
| `a16b_contention_fit.py` → `a16b_out.txt` | the contention-branch refit: $\kappa_c(\text{naive},0)=0.0752$ inside the order-statistic bracket, exponent $1.937$ |
| `a17_floor_exact.py` → `a17_out.txt` | the corrected floor closed form ($+1/48$, and the small-$v$ branch) |
| `a18_incidence.py` → `a18_out.txt` | per-principal $G_i$: $\sum_iG_i=G$ to $2.5\times10^{-13}$, $P(\exists i:G_i<0)$ up to $0.65$ |
| `a04`, `a11`, `a12` | retired mid-run for cost; superseded by `a09`/`a14b`, `a11b`, `a12b` (stubs left in their `*_out.txt`) |
| `a19_symbolic.py` → `a19_out.txt` | **19/19** sympy verifications of every algebraic identity in this memo: (1.1), (3.2), (4.1), (4.2), (4.3), (5.1), the $\rho(\alpha)=O(\alpha^2)$ expansion, the kink, the small-$v$ floor coefficient |
