# The team/block DGP is sign-blind — control check #1 applied to our own robustness design

**Date:** 30 Jul 2026
**Status:** verified analytically and empirically. No re-run required; the fix is analytic.
**Reproduce:** `PYTHONPATH=<repo> python3 results/mode_comparison_postfix/verify_team_dgp.py`

## What prompted it

The two adversarial arms disagreed. Cluster-robust slopes of held-out eval reward on the column
labelled `rho` (CR1, clustered on replication, sigma as fixed effect):

| design | iql | vdn | mappo |
|---|---|---|---|
| equicorr | **+0.801** (p 1.9e-3) | **+0.717** (p 1.9e-5) | +0.297 (p 0.105) |
| team | **-0.425** (p 2.5e-2) | -0.070 (p 0.595) | -0.044 (p 0.724) |

Same nominal rho range (-0.33 to +0.8), same learner, opposite significant sign. Both designs claim
to manipulate signed cross-agent preference correlation, so at most one of them does.

## What the code does

`results/adversarial_rerun/dgp.py`, `sample_agent_params_team`:

```python
opp   = float(opp)
shared = opp
idio  = np.sqrt(max(1.0 - opp * opp, 0.0))
sign  = 1.0 if i < half else -1.0
targets[i] = shared * sign * base + idio * noise
```

Covariance between agents i and j is `+opp**2` within team and `-opp**2` between. **The sign of the
control parameter cancels.** For n=4 (2 teams of 2) there are 2 within-team pairs and 4 between-team
pairs, so the mean pairwise correlation is

    mean_corr(opp) = (2*(+opp^2) + 4*(-opp^2)) / 6 = -opp^2 / 3

which is always <= 0, peaks at opp = 0, and is symmetric in the sign of opp.

## Empirical confirmation (control check #1)

Realised mean pairwise correlation, 4000 draws, seed 7, n_agents=4, n_resources=3:

| nominal | equicorr realised | team realised | -opp^2/3 predicted |
|---|---|---|---|
| -0.33 | -0.330 | -0.038 | -0.036 |
| -0.20 | -0.199 | -0.016 | -0.013 |
| -0.10 | -0.099 | -0.007 | -0.003 |
| 0.00 | 0.001 | -0.004 | 0.000 |
| 0.20 | 0.202 | -0.018 | -0.013 |
| 0.40 | 0.402 | -0.058 | -0.053 |
| 0.60 | 0.602 | -0.124 | -0.120 |
| 0.80 | 0.801 | -0.215 | -0.213 |

Sign test, 8000 draws, seed 11:

| control | realised | mirror | difference |
|---|---|---|---|
| opp = -0.10 | -0.0056 | opp = +0.10 -> -0.0055 | -0.00011 |
| opp = -0.20 | -0.0156 | opp = +0.20 -> -0.0154 | -0.00021 |
| opp = -0.33 | -0.0385 | opp = +0.33 -> -0.0382 | -0.00030 |

Equicorr realises its nominal rho to three decimals at every level. Team never does, is always
negative, and does not distinguish the sign of its control.

## Why this produced a spurious slope

The swept axis (-0.33, -0.2, -0.1, 0, 0.2, 0.4, 0.6, 0.8) samples a symmetric-in-|opp| response
**asymmetrically** — the positive wing extends to 0.8 while the negative wing stops at -0.33. Realised
correlation therefore falls much further on the right. An OLS line through an asymmetrically sampled
inverted-U returns a non-zero slope with no signed effect present. That is the iql -0.425.

## What this does NOT say

The team design is not broken as a **block-opposition** design, and its docstring claim is true: at
opp = 1 two agents on opposing teams are perfectly anti-correlated, a regime the equicorrelated
construction cannot reach (PSD bound rho >= -1/(n-1) = -1/3 at n=4). What is invalid is treating its
control parameter as a signed rho and regressing on it — which the `rho` column name in
`adv_runner.py:95` (`r["rho"] = r.pop("alpha")`) actively invites.

## Actions

1. **Analytic, not experimental.** Analyse the team arm as a within- vs between-team contrast, which is
   what it measures. Drop it from every signed-rho claim. No re-run.
2. **Rename the column** in the team branch — `opp`, not `rho` — so the next analysis cannot repeat this.
3. **Signed-rho claims rest on equicorr alone**, which passes check #1 cleanly.
4. **This is a fourth worked case for the paper**, and a stronger one than the three it already reports,
   because it is the battery firing on the authors' own robustness design rather than on a stylised
   example. Note the recursion honestly: the team design was built to fix the sign-blindness of the
   original alpha DGP (artifact #2) and reproduced it.
