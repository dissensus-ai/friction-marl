# Friction-MARL

**An artifact-control battery for mixed-motive multi-agent reinforcement learning**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Code and evidence repository for *Four Ways to Manufacture a Coordination Finding: A Control Battery for Mixed-Motive MARL* (DAI-2606). The study tests a candidate ratio introduced in [The Axiom of Consent](https://arxiv.org/abs/2601.06692), but the contribution of record is the methodological control battery and its empirical findings.

## Overview

A four-agent resource-allocation MARL testbed originally designed to evaluate the following candidate coordination-friction ratio:

```
F = sigma * (1 + epsilon) / (1 + alpha)
```

Agents manage shared continuous resources under Independent Q-Learning (IQL), with a value-decomposition learner (VDN) run alongside on control grids. The original design is a 5x5x5 factorial over a nominal preference-alignment label (alpha), an implemented reward-scale factor (sigma), and observation noise (epsilon) — 125 conditions x 30 replications. Sigma is only a proxy for theoretical stakes unless a separate calibration is supplied. Later runs add a redesigned signed preference DGP and a control battery: separable and partial-sharing environments, n=2 strong opposition, frozen held-out evaluation, feasible target support, and an exact finite-horizon oracle. In the corrected designs, rho is signed cross-agent correlation among sampled target coordinates (ideal points) inside one fixed quadratic reward family; it is not a direct measure of objective- or reward-function alignment.

The suite ships the theory's own functional alongside the evidence rejecting its specific form. That is the point: the repository documents both a set of corrected findings and, as a methodological result in its own right, how the original experiment design manufactured a spurious one.

## Findings (corrected; supersedes the February 2026 interpretation)

### 1. The original U-shape was a data-generating-process artifact

The original sampler, `target_i = alpha*base + (1-|alpha|)*noise_i`, realizes a cross-agent target-coordinate correlation of

```
rho(alpha) = alpha^2 / (alpha^2 + (1-|alpha|)^2)
```

— a function of |alpha| only, **never negative**. alpha = -0.8 and alpha = +0.8 both realize correlation +0.94 (verified in [`dgp_verification.csv`](results/adversarial_rerun/dgp_verification.csv)); the "adversarial" arm never existed. The U-shape reported from the original factorial ("both cooperative and adversarial nominal alignment beat indifference") was a symmetry artifact of a sign-blind DGP, not a finding. This is the teachable failure the repo preserves: always verify the realized statistic your design claims to manipulate.

### 2. Signed target-coordinate correlation lowers the shared-state gap, conditionally on target support

With cross-agent correlation rho controlled directly and genuinely signed
([`results/adversarial_rerun/`](results/adversarial_rerun/)), the gap-vs-rho relationship is
monotone, not U-shaped. The archived estimates below use Gaussian-at-zero targets and the
late-training outcome; they are not transported to other target supports:

- **Cooperation beats indifference**: cooperative (rho=+0.8) minus neutral gap = -0.71 (p=0.048, IQL) and -0.92 (p=0.0018, VDN).
- **Legacy-support opposition is mixed**: on the archived late-training outcome at the equicorrelated bound rho=-1/3, the point estimates are slightly lower than indifference (-0.154 IQL, p=0.60; -0.159 VDN, p=0.54), with wide intervals. On frozen legacy evaluation, all four stake-by-learner point estimates instead favour an opposition penalty, but only IQL at sigma=1 survives Holm (+1.016, adjusted p=.034). At latent-target rho=-1, the n=2 design removes the PSD and within-team confounds but introduces an agent-count confound and realizes only -0.468 correlation between feasible optima. There, opposition shows no advantage for VDN (+0.35, p=0.37) and is fragilely worse for IQL (+1.27, Welch p=0.041 pooled only; MWU p=0.24)—a mixed result, not equivalence ([`exp2_n2_headline.csv`](results/reviewer3_controls/exp2_n2_headline.csv)). The feasible-centred follow-up below consistently detects an opposition penalty.
- The VDN/team-reward bundle reproduces the IQL/individual-reward shape (gap-vs-rho curve correlation 0.976) while shifting the level. Because architecture and reward objective change together, this is a cross-bundle replication, not an isolated effect of centralization.

### 3. The epsilon-zero implementation residual remains partly explained, not closed

A frozen matched-seed diagnostic changed only CPU-network bias initialization to
match the GPU arm's zero biases ([protocol](results/zero_bias_eps0/PROTOCOL.md),
[final summary](results/zero_bias_eps0/analysis/summary.json)). At the planned
maximum of 30 replications per cell (750 runs), zero initialization improved CPU
mean reward by +0.0412 (95% condition-bootstrap CI [0.0315, 0.0514]) and moved
the CPU-minus-GPU offset from -0.1658 to -0.1246. The primary CI [-0.2322,
-0.0323] was neither contained within nor wholly outside the +/-0.10 equivalence
margin, so the frozen verdict is **inconclusive at the maximum**. Bias
initialization is a real contributor, but not an established full explanation.
The interrupted predecessor in `results/crossval_zerobias_eps0/` is provenance
only and is not merged into this result.

At the aggregate 125-condition level, the repaired CPU-minus-GPU offset is
-0.0261 (90% CI [-0.0652, 0.0131]). The paired condition-level TOST establishes
equivalence at +/-0.10 (p=0.0011) but not at +/-0.05 (p=0.156); the calculation
and both one-sided components are in
[`crossval_equivalence.csv`](results/full_factorial_postfix/crossval_equivalence.csv).

### 4. The functional's single-index form is rejected

Fitting competing functional forms to the signed-DGP condition means ([`functional_form_refit.csv`](results/adversarial_rerun/functional_form_refit.csv)): the friction form M1, gap ~ sigma/(1+rho), gives R^2 = 0.666 with AICc = 29.8 (IQL), while the independent-effects model M4 gives R^2 = 0.935 with AICc = 7.4. Delta-AICc(M1 - M4) is 22.5 for IQL and 17.7 for VDN. The kernel variables matter directionally, but the single-index composition F is rejected as a predictor of coordination failure.

### 5. Reward-scale dominance is mechanical

Reward is degree-1 homogeneous in the preference weights, so sigma dominates the raw gap (eta^2 = 0.22-0.24) and its effect essentially collapses after sigma-normalization (eta^2 = 0.0002-0.0004) — see [`results/adversarial_rerun/FINDINGS.md`](results/adversarial_rerun/FINDINGS.md), section 4C. Higher implemented reward scale therefore raises the raw gap mechanically; this is not independent evidence that theoretical stakes dominate coordination. The remaining effect of signed target-coordinate correlation is real but modest in this design (normalized eta^2 = 0.036-0.058).

### 6. The target-correlation effect is bounded to shared-state contention

The control battery locates the mechanism:

- **Separable pools** ([`exp1_separable_headline.csv`](results/reviewer3_controls/exp1_separable_headline.csv)): give each agent its own independent pool and the gap-vs-rho law is removed by construction for IQL because agents share neither state nor reward and have rho-invariant marginal targets. For VDN, which retains team-reward coupling during training, no rho slope is detected; equivalence is not established. This is a negative endpoint and scope check, not unique mechanism identification.
- **Partial sharing** ([paired data](results/reviewer3_followup/partial_legacy_paired.csv), [interaction](results/reviewer3_followup/partial_frozen_interactions.csv), [shape tests](results/reviewer3_followup/partial_frozen_shape_tests.csv)): on clean frozen held-out evaluation, slopes steepen monotonically from −0.219/−0.125 at the separable endpoints to −1.156/−1.162 under full sharing (IQL/VDN). The problem-block-clustered rho-by-w interaction is −0.916 (95% CI [−1.304, −0.528], p=3.7e-6) / −1.023 ([−1.260, −0.787], p=2.0e-17). Categorical modulation is detected, while departures from a linear ramp are not (p=.890/.191). The old unpaired late-training analysis remains under `results/partial_sharing/` as provenance, not as the result of record.
- **n=2 latent-target opposition** ([`results/reviewer3_controls/`](results/reviewer3_controls/)): the equicorrelated sampler reaches target rho=-1.0000, but the box constraint attenuates feasible-optimum correlation to -0.468 and reducing to two agents introduces an agent-count confound (finding 2 above).

### 7. Frozen evaluation and a dynamic oracle overturn the old attribution

The legacy outcome averages the final 100 training episodes, mixing learned-policy quality with
exploration, parameter updates and reset dynamics. The follow-up freezes each policy and scores
100 held-out resets under clean greedy, matched-noise and 1%-exploration protocols
([`results/reviewer3_followup/`](results/reviewer3_followup/)). An exact 100-step oracle separates
continuous feasibility, reachable-lattice cost, travel transient and the frozen-policy residual;
an independent implementation passes 54/54 brute-force cases.

Under legacy Gaussian-at-zero targets, the clean frozen slope is -1.156 IQL / -1.162 VDN. The
dynamic-controller residual slopes are +0.136 (p=.463) IQL and +0.130 (p=.051) VDN: neither is
detected, but equivalence is not established, and continuous feasibility supplies more than the
point-estimated full gradient. With the
same signed Gaussian translated to the centre of the feasible box, every target is reachable and
the slope steepens to -3.602 / -4.210. Continuous feasibility supplies about 63% IQL / 54% VDN,
while the residual to the exact dynamic controller supplies about 37% / 46%. Within the feasible
design, opposition is worse than indifference at both stakes and for both learners after Holm
correction. The old “entirely feasibility” attribution and the pooled opposition null are therefore
legacy-support results, not general conclusions.

### 8. Observation noise has a positive mean cost, but the proposed interaction fails

A reduced feasible-centred signed grid crosses rho in `{-0.33, 0, .8}` with epsilon in `{0,1}`
on matched latent problems and initializations. On the primary clean frozen outcome, training with
noise raises the normalized gap on average by +2.232 (95% CI [1.635, 2.828], Holm p=1.1e-11) for
IQL and +0.714 ([0.505, 0.923], Holm p=2.1e-10) for VDN
([analysis](results/reviewer3_followup/signed_entropy_interactions.csv)). The linear rho-by-epsilon
interaction is positive: +1.559 ([0.329, 2.789], Holm p=.0130) IQL and +0.962
([0.574, 1.350], Holm p=2.4e-6) VDN. The noise penalty therefore grows toward cooperation, the
opposite of the negative cross-effect implied by `sigma*(1+epsilon)/(1+rho)`. The mean direction
is not universal cell by cell: the pooled opposed-end penalty is not detected for either learner
after correction, and most opposed stake-specific cells are non-detections. The first-stage
small-effect bound is not portable to this target support or mixture.

Full write-ups with designs, verification gates, and caveats: [`adversarial_rerun/FINDINGS.md`](results/adversarial_rerun/FINDINGS.md), [`reviewer3_controls/FINDINGS.md`](results/reviewer3_controls/FINDINGS.md), and [`partial_sharing/FINDINGS.md`](results/partial_sharing/FINDINGS.md). A changelog of what the superseded February interpretation claimed and why it was withdrawn is in [`MARL_FINDINGS.md`](MARL_FINDINGS.md).

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python run_experiments.py \
  --n-replications 1 --n-episodes 2 \
  --alphas 0 --sigmas 0.6 --epsilons 0 \
  --output-dir ./results/quickstart_smoke
```

This command is a portable smoke test of the reference CPU implementation; it is
not a regeneration of the results of record. `requirements.txt` pins the software versions used for the reported release and documents the
ROCm, CUDA, and CPU PyTorch installation variants. Installing only the package metadata leaves
dependencies broad and is not the environment-of-record route. The tracked `uv.lock` captures a
newer local verification/development environment; it is not provenance for the archived training
runs and does not supersede `requirements.txt`.

## CLI

```bash
python run_experiments.py \
  --n-agents 4 \
  --n-resources 3 \
  --n-replications 30 \
  --n-episodes 1000 \
  --output-dir ./results \
  --seed 123
```

Outputs: raw result CSVs in the chosen directory, with analysis tables and plots in its
`analysis/` subdirectory.

The defaults execute the original 125-condition x 30-replication x 1,000-episode
sign-blind IQL protocol. They do **not** reproduce the paper's corrected signed DGP,
VDN/MAPPO comparisons, frozen evaluations, feasible-target control, or dynamic-oracle
decomposition.

Reproduction commands for the signed-DGP re-run and each control experiment are at the bottom of the corresponding `FINDINGS.md`.

## Project Layout

```
friction-marl/
├── friction_marl/
│   ├── envs/          # Resource allocation environments
│   ├── agents/        # Q-learning agent implementations
│   ├── experiments/   # Experiment configurations
│   └── utils/         # Analysis and plotting utilities
├── results/
│   ├── adversarial_rerun/    # Signed-DGP redesign: monotone law, functional-form refit
│   ├── reviewer3_controls/   # Separable-pool + n=2 strong-opposition controls
│   ├── partial_sharing/      # Contention-degree dose-response
│   ├── reviewer3_followup/   # Frozen evaluation, feasible targets, dynamic oracle
│   └── ...                   # Original factorial runs and analyses
├── run_experiments.py # Portable reference CPU factorial / smoke-test entry point
├── requirements.txt   # Dependencies
└── pyproject.toml     # Package configuration
```

## Data

The public Zenodo concept DOI is [10.5281/zenodo.20636467](https://doi.org/10.5281/zenodo.20636467).
Published v2 includes the February factorial and June control battery, but not the post-fix
factorials or the August reviewer follow-up. Those additions remain unpublished until a checked
v3 record is published. Files under `release/zenodo-v3/` are local release-candidate artifacts,
not a public record; their embedded source commit, inventories and `SHA256SUMS` must all be checked
before use. The concept DOI resolves only to the latest version actually published by Zenodo.

## Related Papers

- Axiom of Consent (DAI-2601): [arXiv:2601.06692](https://arxiv.org/abs/2601.06692)
- ROM (DAI-2503): [arXiv:2601.06363](https://arxiv.org/abs/2601.06363)

## Authors

- **Murad Farzulla** -- [Dissensus](https://dissensus.ai)
  - ORCID: [0009-0002-7164-8704](https://orcid.org/0009-0002-7164-8704)

## License

MIT License
