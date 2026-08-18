# Reviewer3 follow-up — frozen policy, feasible support, and dynamic oracle

Artifact tier: **local release-candidate evidence, unpublished**. These files answer the
13 August 2026 Reviewer3 report. They do not become part of the public replication
record until the final source/data package is committed, independently extracted,
checksummed, and published as a new Zenodo version.

## Estimands

- `mean_reward`: final 100 training episodes; learning continues, observation noise
  and the 1% action-exploration floor remain active.
- `eval_reward_clean`: frozen greedy policy on 100 held-out reset seeds, without
  observation noise. This is the primary learned-policy outcome.
- `eval_reward_matched`: the same frozen policy and reset bank, with the training
  condition's observation noise restored.
- `eval_reward_explore_001`: the same reset bank and observation condition, with the
  1% action-exploration floor restored but no parameter updates.
- Dynamic-oracle fields split the clean frozen gap into continuous-box feasibility,
  exact reachable-lattice quantization, T=100 travel/transient cost, and the residual
  policy-to-oracle gap. The last term is not called a pure learning shortfall: for IQL
  it also includes decentralized control and individual-versus-team objective mismatch.

All signed outputs use seed `20260813`, 30 replication blocks per cell, 1,000
training episodes, 100 held-out evaluation episodes, and separate GPUs for IQL and
VDN. `condition_seed`, `problem_instance_sha256`, and `latent_problem_sha256` make
the matched problem instances auditable at row level.

## Gates and regenerators

- `verify_dynamic_oracle.py`: independent two-step brute force, including integer and
  boundary resets.
- `verify_partial_endpoints.py`: full end-to-end equality for both IQL and VDN at
  `k=0` (separable) and `k=m` (shared).
- `augment_instance_provenance.py`: deterministic reconstruction of row-level problem
  hashes for files generated before the hash columns were added to the runners.
- `assemble_partial_followup.py`: reuses the exact frozen endpoints and inserts only
  the newly trained `k=1,2` arms.
- `analyze_followup.py`: HC3 outcome/component slopes, paired support changes,
  within-stake and pooled contrasts, and problem-block-clustered partial-sharing
  interaction.
- `analyze_signed_entropy.py`: paired signed `rho x epsilon` analysis on clean,
  matched-noise and 1%-exploration frozen evaluations.

Smoke-test directories are implementation diagnostics, not evidence and must be
excluded from any release inventory.

## Results of record

- The clean frozen shared-state slope is detected under legacy support (`-1.156` IQL; `-1.162` VDN)
  and steepens under feasible-centred support (`-3.602`; `-4.210`). The paired support changes are
  `-2.446` and `-3.049` (both Holm-adjusted `p<1e-8`).
- Under legacy support the dynamic-controller residual slopes are `+0.136` (`p=.463`) IQL and
  `+0.130` (`p=.051`) VDN. Neither is detected, but equivalence is not established. Under
  feasible-centred support continuous feasibility supplies about 63%/54% of the cooperative
  gradient and the residual supplies about 37%/46% (IQL/VDN). The old binary
  feasibility-versus-learning attribution is withdrawn.
- Paired partial sharing gives clean frozen slopes from `-0.219/-0.125` at `w=0` to
  `-1.156/-1.162` at `w=1`. The problem-block-clustered `rho x w` interaction is `-0.916`
  (`p=3.7e-6`) IQL and `-1.023` (`p=2.0e-17`) VDN. All partially shared slopes are detected;
  departures from a linear ramp are not (`p=.890/.191`).
- In the feasible-centred signed noise crossing, the mean clean frozen noise penalty is `+2.232`
  IQL and `+.714` VDN. The `rho x epsilon` interaction is positive (`+1.559`, Holm `p=.0130`;
  `+.962`, Holm `p=2.4e-6`), opposite the canonical ratio's implied cross-effect. The penalty is
  not detected in most opposed cells, so the direction is scope- and mixture-specific rather than
  universal.

All estimates are exploratory controlled demonstrations. Holm correction controls the stated
families; it does not make the protocols preregistered or independently replicated.
