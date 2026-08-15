"""Exact finite-horizon oracle for the shared-pool resource environment.

The manuscript's historical ``R_joint`` benchmark is a continuous, static optimum.
That is not the reward an optimal controller can collect over a 100-step episode:
aggregate actions are integer-valued, the reachable lattice depends on the reset's
fractional part, and the controller pays a short travelling transient.  This module
computes all three reference levels on the *same held-out reset states* used to score
a frozen policy:

``continuous_static``
    Best mean-agent reward at any state in the continuous box.
``reachable_static``
    Best mean-agent reward on the exact lattice reachable from that reset.
``dynamic_oracle``
    Exact T-step reward under joint control, by backward induction.

The one-resource reduction is exact because both reward and transition factor over
resources and the aggregate action set is the product ``{-n_agents,...,+n_agents}^m``.
The 21-state lattice construction and transition index are independently audited in
``results/regret_closed_form``; the implementation here is deliberately self-contained
so the experimental runner does not depend on analysis-report modules.
"""
from __future__ import annotations

import numpy as np


def _lattices(fractional_part: np.ndarray, capacity: int) -> np.ndarray:
    """Return exact reachable lattices, ordered integer/shifted by floor cell."""
    f = np.asarray(fractional_part, dtype=float)
    out = np.empty((f.size, 2 * capacity + 1), dtype=float)
    for k in range(capacity + 1):
        out[:, 2 * k] = k
    for k in range(capacity):
        out[:, 2 * k + 1] = f + k
    return out


def _transition_index(capacity: int, n_agents: int) -> np.ndarray:
    """Transition lookup; independent of the nonzero fractional lattice offset."""
    probe = _lattices(np.array([0.3719]), capacity)[0]
    deltas = np.arange(-n_agents, n_agents + 1, dtype=float)
    nxt = np.clip(probe[:, None] + deltas[None, :], 0.0, float(capacity))
    idx = np.abs(nxt[..., None] - probe[None, None, :]).argmin(axis=-1)
    residual = np.abs(nxt - probe[idx]).max()
    if residual > 1e-10:
        raise AssertionError(f"reachable-lattice transition residual {residual:.3e}")
    return idx


def shared_pool_oracle_components(
    weights: np.ndarray,
    targets: np.ndarray,
    initial_states: np.ndarray,
    *,
    horizon: int = 100,
    capacity: int = 10,
) -> dict[str, np.ndarray]:
    """Compute exact reference rewards for each replication.

    Parameters
    ----------
    weights, targets
        Arrays shaped ``(replication, agent, resource)``.
    initial_states
        Held-out reset states shaped ``(episode, replication, resource)``.

    Returns
    -------
    dict of arrays shaped ``(replication,)``. Rewards use the manuscript's
    mean-across-agents, sum-across-resources convention and are per timestep.
    """
    w = np.asarray(weights, dtype=float)
    tau = np.asarray(targets, dtype=float)
    x0 = np.asarray(initial_states, dtype=float)
    if w.shape != tau.shape or w.ndim != 3:
        raise ValueError("weights and targets must share shape (replication, agent, resource)")
    if x0.ndim != 3 or x0.shape[1] != w.shape[0] or x0.shape[2] != w.shape[2]:
        raise ValueError("initial_states must have shape (episode, replication, resource)")
    if int(capacity) != capacity:
        raise ValueError("the exact interleaved lattice requires an integer capacity")

    n_rep, n_agents, n_resources = w.shape
    n_eval = x0.shape[0]
    weight_sum = w.sum(axis=1)  # (R,M), kept as a sum for the weighted target
    safe_sum = np.where(weight_sum > 0, weight_sum, 1.0)
    target_bar = (w * tau).sum(axis=1) / safe_sum
    target_bar = np.where(weight_sum > 0, target_bar, 0.0)

    # Mean-agent quadratic decomposition for each resource:
    # mean_i[-w_i(x-tau_i)^2] = -coef*(x-target_bar)^2 - constant.
    coef = weight_sum / n_agents
    constant = (w * (tau - target_bar[:, None, :]) ** 2).sum(axis=1) / n_agents
    continuous_state = np.clip(target_bar, 0.0, float(capacity))
    continuous_static = (
        -coef * (continuous_state - target_bar) ** 2 - constant
    ).sum(axis=1)

    # Flatten eval episodes and replications so one vectorized backward induction
    # handles all held-out starts for a resource.
    flat_x0 = x0.reshape(n_eval * n_rep, n_resources)
    rep_index = np.tile(np.arange(n_rep), n_eval)
    reachable_total = np.zeros(n_eval * n_rep, dtype=float)
    dynamic_total = np.zeros(n_eval * n_rep, dtype=float)
    tidx = _transition_index(int(capacity), n_agents)

    for resource in range(n_resources):
        starts = flat_x0[:, resource]
        frac = starts - np.floor(starts)
        # At an exact integer the interleaved representation contains duplicate
        # states (0,0,1,1,...,10). That is harmless: their stage values and
        # continuations coincide, while retaining the common transition table.
        frac = np.where(frac <= 1e-12, 0.0, frac)
        lattice = _lattices(frac, int(capacity))
        k0 = np.floor(starts).astype(int).clip(0, int(capacity))

        c = coef[rep_index, resource]
        mu = target_bar[rep_index, resource]
        v = constant[rep_index, resource]
        stage = -c[:, None] * (lattice - mu[:, None]) ** 2 - v[:, None]
        reachable_total += stage.max(axis=1)

        value = np.zeros_like(stage)
        for _ in range(horizon):
            value = (stage[:, tidx] + value[:, tidx]).max(axis=2)
        is_integer = np.abs(frac) <= 1e-12
        start_index = np.where(is_integer, 2 * k0, 2 * k0 + 1)
        dynamic_total += value[np.arange(value.shape[0]), start_index] / horizon

    reachable_static = reachable_total.reshape(n_eval, n_rep).mean(axis=0)
    dynamic_oracle = dynamic_total.reshape(n_eval, n_rep).mean(axis=0)
    tol = 2e-8
    if np.any(continuous_static + tol < reachable_static):
        raise AssertionError("reachable static reward exceeds continuous-box optimum")
    if np.any(reachable_static + tol < dynamic_oracle):
        raise AssertionError("dynamic oracle exceeds reachable static optimum")

    return {
        "continuous_static": continuous_static,
        "reachable_static": reachable_static,
        "dynamic_oracle": dynamic_oracle,
        "quantization_cost": continuous_static - reachable_static,
        "transient_cost": reachable_static - dynamic_oracle,
    }
