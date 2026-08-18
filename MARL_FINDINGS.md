# MARL Findings — Corrected Summary

**Last substantive revision:** 2026-08-19 (supersedes the 2026-02-13 version of this file)

This file used to carry the February 2026 interpretation of the 5x5x5 factorial. That
interpretation was overturned in June 2026 when the preference data-generating process
was audited and re-run with a genuinely signed design. The corrected findings are
summarized in the [README](README.md#findings-corrected-supersedes-the-february-2026-interpretation)
and documented in full, with designs, verification gates, and caveats, in:

| Write-up | Contents |
|---|---|
| [`results/adversarial_rerun/FINDINGS.md`](results/adversarial_rerun/FINDINGS.md) | The DGP bug, the signed target-correlation redesign, and the functional-form refit |
| [`results/reviewer3_controls/FINDINGS.md`](results/reviewer3_controls/FINDINGS.md) | Separable-pool control and the qualified n=2 latent-target rho=-1 opposition test |
| [`results/partial_sharing/FINDINGS.md`](results/partial_sharing/FINDINGS.md) | Contention-degree dose-response (slope scales monotonically with the fraction of shared resources) |
| [`results/reviewer3_followup/README.md`](results/reviewer3_followup/README.md) | Frozen held-out evaluation, feasible targets, exact dynamic oracle, and follow-up gates |

**One-paragraph version:** the original sampler was sign-blind — its realized cross-agent
target-coordinate correlation is alpha^2/(alpha^2+(1-|alpha|)^2), a function of |alpha| only and
never negative, so the "adversarial" arm never existed (alpha = +-0.8 both realize
correlation +0.94; `results/adversarial_rerun/dgp_verification.csv`). On genuinely signed
DGPs the gap-vs-correlation relationship is monotone: cooperation lowers the coordination
gap. Under legacy Gaussian-at-zero targets, opposition is mixed across outcome protocols, stakes
and learners; the n=2 target sampler
reaches rho=-1 without the PSD or within-team confounds, but carries an agent-count confound and
realizes only -0.468 correlation between feasible optima. Under feasible-centred targets,
opposition is worse than indifference at both stakes and for both learners after Holm correction.
The consent-friction functional's single-index form
sigma/(1+rho) is rejected against an independent-effects model (Delta-AICc 22.5 IQL / 17.7 VDN;
[`functional_form_refit.csv`](results/adversarial_rerun/functional_form_refit.csv)). Dominance of the implemented sigma reward scale is mechanical
(degree-1 homogeneity of the reward in the weights), not independent evidence that theoretical stakes dominate. The post-control target-correlation estimate is
bounded to shared-state contention: the fully separable IQL endpoint is structurally rho-invariant
and the VDN endpoint has no detected rho slope despite team-reward coupling, without establishing
equivalence, while the interaction-preserving
partial-sharing series detects graded slope modulation. Frozen
held-out evaluation and an exact finite-horizon oracle also withdraw the old “entirely
feasibility” attribution: feasibility supplies about 54–63% of the feasible-centred gradient and
the residual frozen-policy component about 37–46%. In a reduced feasible-centred signed crossing,
observation noise raises the mean clean frozen gap, but its penalty grows toward cooperation
(`rho x epsilon = +1.559` IQL / `+.962` VDN), opposite the proposed ratio's cross-effect; the
penalty is not detected in most opposed cells. All reviewer follow-ups are exploratory and
remain unpublished until a checked v3 record is actually published.

---

## Superseded interpretations (honest changelog)

The 2026-02-13 version of this file (preserved in git history) made five headline claims
from the original factorial. Their fates:

1. **"Preference structure > preference direction" (the U-shape)** — **retracted.**
   The claim that both cooperative and adversarial nominal alignment beat unstructured indifference
   was an artifact of the sign-blind DGP: alpha<0 conditions were in fact *positively*
   correlated, identically to alpha>0, so the design compared |correlation| levels, not
   signed target-coordinate correlation. With a real signed axis the symmetry collapses (the signed-linear term
   explains 99% IQL / 95% VDN of the rho-variance) and the positively correlated target endpoint
   has the smaller gap. The proposed
   companion-paper framing "structured disagreement improves coordination" was withdrawn
   with it.

2. **"Stakes dominate structure" (claimed eta^2 = 0.397 vs 0.096)** — **reinterpreted.**
   The direction survives (higher stakes raise the raw gap) but the dominance is mechanical:
   reward is degree-1 homogeneous in the preference weights, and the sigma effect essentially
   collapses under sigma-normalization (signed-run eta^2 = 0.0002-0.0004). It is an arithmetic property of the reward scale, not
   a structural fact about coordination. The February eta^2 figures are from the artifact
   run and should not be quoted.

3. **"Friction equalizes" (claimed 28x cross-agent variance reduction at alpha=+-0.8 vs
   alpha=0)** — **retracted as a friction claim.** Under the broken DGP, alpha=+-0.8 both
   meant high *positive* correlation, so this observation was "strongly correlated
   preferences produce more symmetric outcomes" — unsurprising and not about structured
   opposition. The consent-efficiency-tradeoff gloss built on it is withdrawn.

4. **"Reward convergence without policy convergence"** — **descriptive statistic retained,
   theoretical gloss withdrawn.** The observation (reward stabilizes while policies keep
   moving) is a property of the original runs, but its framing as "the strongest empirical
   confirmation of the AoC's core theoretical claim" over-read it; it is consistent with
   ordinary non-stationarity in independent learners.

5. **"Observation noise barely matters"** — **retracted as a portable claim.** On the
   post-fix factorial's frozen clean evaluation, the epsilon=0 to epsilon=1 contrast is real
   and signed, while a pooled independent-sample equivalence test bounds it to 21.5% of the design's largest stakes
   contrast on that scale. The repaired feasible-centred signed crossing instead finds a much
   larger positive mean penalty and a positive `rho x epsilon` interaction: the penalty grows
   toward cooperation, contrary to the canonical ratio, and is not detected across most opposed
   cells. The first-stage bound was scale- and mixture-specific; the February eta^2 figure should
   not be quoted.

Nothing was silently deleted: `git log -- MARL_FINDINGS.md README.md` recovers the full
superseded text. The reason for the rewrite is currency, not concealment — these two files
are the repository's public statement of results, and until this revision they still
presented the February artifact as the finding.
