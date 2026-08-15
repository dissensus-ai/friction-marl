"""Redesigned preference-generation DGP for the friction-MARL adversarial re-run.

WHY THIS EXISTS
---------------
The original `factorial_design.sample_agent_params` builds targets as

    tau_i = alpha * b + (1 - |alpha|) * xi_i           (b shared across agents)

For n agents the population cross-agent correlation of the target vectors is

    rho(alpha) = alpha^2 / (alpha^2 + (1-|alpha|)^2)

which is a function of |alpha| ONLY and is always >= 0. At alpha = +-0.8 it is
~0.94 for BOTH signs -- so the "adversarial" arm (alpha<0) is in fact just as
POSITIVELY correlated as the cooperative arm. alpha<0 was never adversarial. The
apparent "U-shape" (alpha=0 worst) is therefore a |alpha|-symmetry artefact: the
only genuinely unstructured point is alpha=0 (rho=0); both wings are structured
(rho~0.94) regardless of sign.

THE FIX
-------
We parameterise the design directly by the cross-agent correlation rho and make it
genuinely SIGNED: rho < 0 = real opposition, rho = 0 = indifference, rho > 0 =
alignment.

(A) Equicorrelated continuum  [primary]
    For each resource we draw the n-vector of agent-targets from N(0, Sigma(rho))
    with Sigma(rho) = (1-rho) I + rho 11'. This is PSD iff
        rho >= -1/(n-1)   (= -1/3 for n=4).
    Clean signed continuum; the adversarial arm is CAPPED at -1/3 (every agent
    cannot be strongly anti-correlated with three others at once). Marginal target
    variance is held at 1 for every rho (unlike the original, whose target variance
    drifted with alpha), so reward magnitudes are comparable across the sweep and
    to the original alpha=0 reference (which also had unit-variance targets).

(B) Two-team block structure  [robustness, strong opposition]
    4 agents -> 2 teams of 2. Team A shares +base, team B shares -base, strength
    `opp` in [0,1]. Within-team corr = +opp^2, between-team corr = -opp^2. At opp=1
    two agents on opposing teams are perfectly ANTI-correlated (rho_between = -1):
    a genuinely strong adversarial regime unreachable by the equicorrelated bound.
    Mean pairwise corr (6 pairs: 2 within, 4 between) = -opp^2/3.

Both return (weights, targets) with the SAME signature/shape as the original
`sample_agent_params`, so they are drop-in for the existing IQL / VDN runners
(weights ~ N(sigma, 0.05) clipped to [0,1] -- identical stakes semantics).
"""
from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------- #
#  Weights (stakes) -- identical to the original; sigma is preference INTENSITY  #
# ----------------------------------------------------------------------------- #
def _sample_weights(rng, n_agents, n_resources, sigma):
    w = rng.normal(loc=sigma, scale=0.05, size=(n_agents, n_resources))
    return np.clip(w, 0.0, 1.0)


# ----------------------------------------------------------------------------- #
#  (A) Equicorrelated targets: pairwise corr -> rho in [-1/(n-1), 1]             #
# ----------------------------------------------------------------------------- #
def _equicorr_sqrt(n: int, rho: float) -> np.ndarray:
    """Symmetric PSD square root of Sigma(rho) = (1-rho) I + rho 11'.

    Valid for rho in [-1/(n-1), 1]; eigh-based so it is stable at the boundary
    (smallest eigenvalue -> 0) where Cholesky would fail.
    """
    Sigma = (1.0 - rho) * np.eye(n) + rho * np.ones((n, n))
    evals, evecs = np.linalg.eigh(Sigma)
    evals = np.clip(evals, 0.0, None)
    return (evecs * np.sqrt(evals)) @ evecs.T


def sample_agent_params_equicorr(rng, n_agents, n_resources, rho, sigma):
    """Drop-in sampler. `rho` is the target cross-agent correlation (signed)."""
    L = _equicorr_sqrt(n_agents, float(rho))
    # one independent n-vector per resource: z[:, j] ~ N(0, I_n)
    z = rng.standard_normal((n_agents, n_resources))
    targets = L @ z                      # each column ~ N(0, Sigma(rho)); unit marginal var
    weights = _sample_weights(rng, n_agents, n_resources, sigma)
    return weights, targets.astype(np.float64)


def sample_agent_params_equicorr_feasible(
    rng, n_agents, n_resources, rho, sigma, *, capacity=10.0
):
    """Equicorrelated targets generated wholly inside the feasible state box.

    This is a construct-validity control for the legacy ``N(0,1)`` targets, roughly
    half of whose coordinates lie below the environment's lower bound.  We translate
    the same unit-variance Gaussian design to the box centre and reject the vanishingly
    rare draw outside ``[0, capacity]``.  With centre 5 and sd 1, rejection is below
    one draw in 100,000 for the 4-agent x 3-resource condition, so the covariance is
    effectively unchanged while every delivered ideal point is feasible.  The design
    is translation-equivalent to centring the state box at zero.
    """
    L = _equicorr_sqrt(n_agents, float(rho))
    centre = float(capacity) / 2.0
    for _ in range(10_000):
        z = rng.standard_normal((n_agents, n_resources))
        targets = centre + L @ z
        if np.all((targets >= 0.0) & (targets <= capacity)):
            weights = _sample_weights(rng, n_agents, n_resources, sigma)
            return weights, targets.astype(np.float64)
    raise RuntimeError("failed to draw feasible equicorrelated targets after 10,000 attempts")


# ----------------------------------------------------------------------------- #
#  (B) Two-team block targets: strong opposition between teams                   #
# ----------------------------------------------------------------------------- #
def sample_agent_params_team(rng, n_agents, n_resources, opp, sigma):
    """Drop-in sampler. `opp` in [0,1] = between-team opposition strength.

    Teams = first half vs second half of agents. Unit marginal target variance.
    within-team corr = +opp^2 ; between-team corr = -opp^2.
    """
    opp = float(opp)
    base = rng.standard_normal(n_resources)
    half = n_agents // 2
    shared = opp
    idio = np.sqrt(max(1.0 - opp * opp, 0.0))
    targets = np.empty((n_agents, n_resources))
    for i in range(n_agents):
        sign = 1.0 if i < half else -1.0
        noise = rng.standard_normal(n_resources)
        targets[i] = shared * sign * base + idio * noise
    weights = _sample_weights(rng, n_agents, n_resources, sigma)
    return weights, targets.astype(np.float64)


# ----------------------------------------------------------------------------- #
#  Empirical verification of the realised correlation                           #
# ----------------------------------------------------------------------------- #
def realised_mean_pairwise_corr(sampler, control, sigma, n_agents, n_resources,
                                n_draws, seed, resources_as_samples=False):
    """Empirically estimate the mean pairwise cross-agent target correlation.

    Two estimators (both reported because m=n_resources is small):
      * pooled : stack all (n_draws x n_resources) target components and compute
                 the Pearson corr between agent-i and agent-j component series
                 (this is the *population* corr the DGP targets; the cleanest read).
      * per_draw: corr estimated within each draw using the m resources as samples
                 then averaged (this is what an observer with one 3-resource draw
                 would measure -- noisy for m=3, reported for honesty).
    Returns (pooled_mean, per_draw_mean).
    """
    rng = np.random.default_rng(seed)
    A = n_agents
    # collect per-agent component series across draws
    comp = [[] for _ in range(A)]
    per_draw_corrs = []
    for _ in range(n_draws):
        _, t = sampler(rng, A, n_resources, control, sigma)  # (A, M)
        for i in range(A):
            comp[i].append(t[i])
        # per-draw corr using resources as samples (m=3)
        if n_resources >= 2:
            c = np.corrcoef(t)            # (A,A)
            iu = np.triu_indices(A, k=1)
            vals = c[iu]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                per_draw_corrs.append(vals.mean())
    comp = [np.concatenate(c) for c in comp]      # each length n_draws*M
    M = np.corrcoef(np.stack(comp))               # pooled (A,A)
    iu = np.triu_indices(A, k=1)
    pooled = M[iu].mean()
    per_draw = float(np.mean(per_draw_corrs)) if per_draw_corrs else float("nan")
    return float(pooled), per_draw
