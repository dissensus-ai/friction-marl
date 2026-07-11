# MARL Findings — Corrected Summary

**Last substantive revision:** 2026-07-11 (supersedes the 2026-02-13 version of this file)

This file used to carry the February 2026 interpretation of the 5x5x5 factorial. That
interpretation was overturned in June 2026 when the preference data-generating process
was audited and re-run with a genuinely signed design. The corrected findings are
summarized in the [README](README.md#findings-corrected-supersedes-the-february-2026-interpretation)
and documented in full, with designs, verification gates, and caveats, in:

| Write-up | Contents |
|---|---|
| `results/adversarial_rerun/FINDINGS.md` | The DGP bug, the signed redesign, the monotone alignment law, the functional-form refit |
| `results/reviewer3_controls/FINDINGS.md` | Separable-pool control (bounds the law to shared-state contention) and n=2 clean rho=-1 opposition |
| `results/partial_sharing/FINDINGS.md` | Contention-degree dose-response (slope scales monotonically with the fraction of shared resources) |

**One-paragraph version:** the original sampler was sign-blind — its realized cross-agent
preference correlation is alpha^2/(alpha^2+(1-|alpha|)^2), a function of |alpha| only and
never negative, so the "adversarial" arm never existed (alpha = +-0.8 both realize
correlation +0.94; `results/adversarial_rerun/dgp_verification.csv`). On genuinely signed
DGPs the gap-vs-correlation relationship is monotone: cooperation lowers the coordination
gap, opposition never beats indifference (including at a clean, unconfounded rho=-1 with
n=2). The consent-friction functional's single-index form sigma/(1+rho) is rejected against
an independent-effects model (Delta-AIC ~24 IQL / ~19 VDN;
`results/adversarial_rerun/functional_form_refit.csv`). Sigma-dominance is mechanical
(degree-1 homogeneity of the reward in the weights). The surviving alignment effect is
bounded to shared-state contention: it flattens with separable resource pools and scales
monotonically with the fraction of resources shared.

---

## Superseded interpretations (honest changelog)

The 2026-02-13 version of this file (preserved in git history) made five headline claims
from the original factorial. Their fates:

1. **"Preference structure > preference direction" (the U-shape)** — **retracted.**
   The claim that both cooperative and adversarial alignment beat unstructured indifference
   was an artifact of the sign-blind DGP: alpha<0 conditions were in fact *positively*
   correlated, identically to alpha>0, so the design compared |correlation| levels, not
   signed alignment. With a real signed axis the symmetry collapses (the signed-linear term
   explains 99% IQL / 95% VDN of the rho-variance) and only alignment helps. The proposed
   companion-paper framing "structured disagreement improves coordination" was withdrawn
   with it.

2. **"Stakes dominate structure" (claimed eta^2 = 0.397 vs 0.096)** — **reinterpreted.**
   The direction survives (higher stakes raise the raw gap) but the dominance is mechanical:
   reward is degree-1 homogeneous in the preference weights, and the sigma effect vanishes
   entirely under sigma-normalization. It is an arithmetic property of the reward scale, not
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

5. **"Observation noise barely matters"** — **direction retained.** epsilon had a near-zero
   effect in the original runs, which is why the signed-DGP re-runs fixed epsilon=0. No
   corrected effect size is claimed; the February eta^2 figure is from the artifact run.

Nothing was silently deleted: `git log -- MARL_FINDINGS.md README.md` recovers the full
superseded text. The reason for the rewrite is currency, not concealment — these two files
are the repository's public statement of results, and until this revision they still
presented the February artifact as the finding.
