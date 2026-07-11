# Friction-MARL

**A multi-agent reinforcement learning test of the consent-friction functional**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Code companion to [Axiom of Consent: Friction Dynamics in Multi-Agent Coordination](https://arxiv.org/abs/2601.06692) (DAI-2601). The empirical results here are written up in the companion manuscript *Testing the Consent-Friction Functional: Preference-Design Artifacts and the Contention-Bounded Limits of Coordination Friction in Multi-Agent Reinforcement Learning* (DAI-2606).

## Overview

A four-agent resource-allocation MARL testbed for the coordination-friction functional proposed in the Axiom of Consent:

```
F = sigma * (1 + epsilon) / (1 + alpha)
```

Agents manage shared continuous resources under Independent Q-Learning (IQL), with a value-decomposition learner (VDN) run alongside on control grids. The original design is a 5x5x5 factorial over preference alignment (alpha), preference intensity / stakes (sigma), and observation noise (epsilon) — 125 conditions x 30 replications. Later runs add a redesigned signed preference DGP and a control battery (separable resource pools, partial sharing, n=2 strong opposition).

The suite ships the theory's own functional alongside the evidence rejecting its specific form. That is the point: the repository documents both a set of corrected findings and, as a methodological result in its own right, how the original experiment design manufactured a spurious one.

## Findings (corrected; supersedes the February 2026 interpretation)

### 1. The original U-shape was a data-generating-process artifact

The original sampler, `target_i = alpha*base + (1-|alpha|)*noise_i`, realizes a cross-agent preference correlation of

```
rho(alpha) = alpha^2 / (alpha^2 + (1-|alpha|)^2)
```

— a function of |alpha| only, **never negative**. alpha = -0.8 and alpha = +0.8 both realize correlation +0.94 (verified empirically in `results/adversarial_rerun/dgp_verification.csv`); the "adversarial" arm never existed. The U-shape reported from the original factorial ("both cooperative and adversarial alignment beat indifference") was a symmetry artifact of a sign-blind DGP, not a finding. This is the teachable failure the repo preserves: always verify the realized statistic your design claims to manipulate.

### 2. On genuinely signed DGPs, only alignment helps

With the cross-agent correlation rho controlled directly and genuinely signed (`results/adversarial_rerun/`), the gap-vs-rho relationship is monotone, not U-shaped:

- **Cooperation beats indifference**: cooperative (rho=+0.8) minus neutral gap = -0.71 (p=0.048, IQL) and -0.92 (p=0.0018, VDN).
- **Opposition never beats indifference**: statistically tied at the equicorrelated bound rho=-1/3 (+0.15, p=0.60 IQL; +0.16, p=0.54 VDN), and at a genuine, unconfounded rho=-1 (n=2 agents, so no PSD cap and no within-team-cooperation confound) opposition ties indifference for VDN (+0.35, p=0.37) and is at most fragilely worse for IQL (+1.27, Welch p=0.041 pooled only; MWU p=0.24) — it never wins (`results/reviewer3_controls/exp2_n2_headline.csv`).
- VDN preserves the IQL shape (gap-vs-rho curve correlation 0.976) while uniformly lowering the gap; centralized training does not manufacture a U either.

### 3. The functional's single-index form is rejected

Fitting competing functional forms to the signed-DGP condition means (`results/adversarial_rerun/functional_form_refit.csv`): the friction form M1, gap ~ sigma/(1+rho), gives R^2 = 0.67 with AIC = 25.8 (IQL), while the independent-effects model M4 (rho and sigma entering separately) gives R^2 = 0.93 with AIC = 1.7. Delta-AIC(M1 - M4) is approximately 24 for IQL and 19 for VDN. The kernel variables matter directionally, but the single-index composition F is rejected as a predictor of coordination failure.

### 4. Stakes dominance is mechanical

Reward is degree-1 homogeneous in the preference weights, so sigma dominates the raw gap (eta^2 = 0.22-0.24) and its effect vanishes entirely after sigma-normalization (eta^2 = 0.000) — see `results/adversarial_rerun/FINDINGS.md`, section 4C. The direction (higher stakes raise friction) holds, but the dominance is arithmetic, not structural. The remaining structural effect of preference correlation is real but modest (normalized eta^2 = 0.036-0.058).

### 5. The alignment effect is bounded to shared-state contention

The control battery locates the mechanism:

- **Separable pools** (`results/reviewer3_controls/exp1_separable_headline.csv`): give each agent its own independent resource pool and the gap-vs-rho law flattens — shared-pool slope -0.755 (p=1.7e-4, IQL) / -0.838 (p=7.3e-7, VDN) vs separable slope -0.029 (p=0.87) / -0.053 (p=0.76). The friction is contention over shared state, not preference correlation per se.
- **Partial sharing** (`results/partial_sharing/slope_vs_overlap.csv`, `interaction_test.csv`): varying the fraction of resources that are common-pool, w in {0, 1/3, 2/3, 1}, the slope scales monotonically with contention degree from ~0 to ~-0.75; the rho-by-w interaction is -0.765 (p=3.4e-3, IQL) / -0.731 (p=2.4e-3, VDN). Individually significant slopes appear only once a majority of resources are shared.
- **n=2 clean opposition** (`results/reviewer3_controls/`): the equicorrelated design at n=2 reaches an exact realized rho of -1.0000, closing the strong-opposition gap the n=4 design could not test cleanly (finding 2 above).

Full write-ups with designs, verification gates, and honest caveats: `results/adversarial_rerun/FINDINGS.md`, `results/reviewer3_controls/FINDINGS.md`, `results/partial_sharing/FINDINGS.md`. A changelog of what the superseded February interpretation claimed and why it was withdrawn is in `MARL_FINDINGS.md`.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python run_experiments.py --output-dir ./results
```

## CLI

```bash
python run_experiments.py \
  --n-agents 4 \
  --n-resources 3 \
  --n-replications 30 \
  --n-episodes 10000 \
  --output-dir ./results \
  --seed 123
```

Outputs: Raw results CSV in `results/`, analysis tables and plots in `results/analysis/`.

Default settings are computationally heavy (125 conditions x 30 replications x 10,000 episodes). Reduce episodes or replications for quick tests.

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
│   └── ...                   # Original factorial runs and analyses
├── run_experiments.py # Main entry point
├── requirements.txt   # Dependencies
└── pyproject.toml     # Package configuration
```

## Data

Full replication dataset on Zenodo: concept DOI [10.5281/zenodo.20636467](https://doi.org/10.5281/zenodo.20636467) (v2 adds the signed-DGP and control-battery runs).

## Related Papers

- Axiom of Consent (DAI-2601): [arXiv:2601.06692](https://arxiv.org/abs/2601.06692)
- ROM (DAI-2503): [arXiv:2601.06363](https://arxiv.org/abs/2601.06363)

## Authors

- **Murad Farzulla** -- [Dissensus AI](https://dissensus.ai)
  - ORCID: [0009-0002-7164-8704](https://orcid.org/0009-0002-7164-8704)

## License

MIT License
