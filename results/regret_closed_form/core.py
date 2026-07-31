"""Exact computation of the signed delegation gap G(sigma, alpha, epsilon)
for the friction-MARL resource-allocation environment.

EVERYTHING HERE IS EXACT FOR THE ACTUAL ENVIRONMENT (no learned policies, no
reward shaping, no discretisation of a continuous control problem) except where
explicitly flagged: the stage-Nash noise integration is Monte Carlo, and the
long-run average uses a declared finite Cesaro truncation.

Environment facts verified from source, 2026-07-30
----------------------------------------------------------------------------
* friction_marl/envs/resource_allocation.py
    - state in R^3, reset ~ Uniform[0, C]^3 with C = 10.0        (:57-60)
    - joint step: new = clip(state + sum_i delta_i, 0.0, C)      (:84-86)
    - episode_length = 100                                        (:17)
* friction_marl/experiments/factorial_design.py
    - per-agent action delta in {-1,0,+1}^3 (build_action_map,    :55-60)
    - r_i = sum_m w_im * (-(s'_m - tau_im)^2) (compute_rewards,   :63-69)
    - w_im ~ N(sigma, 0.05) clipped to [0,1]                      (:50-51)
    - tau_i = alpha*b + (1-|alpha|)*xi_i, b shared                (:42-48)
* run_gpu.py
    - obs_i = s + N(0, eps*I), std sqrt(eps)                      (:250-256)
    - rewards use the TRUE state and TRUE (w, tau)                (:279-281)
    - n_agents = 4, n_resources = 3, CAPACITY = 10.0              (:54-65)

Structural facts derived and used throughout (validated in validate_lattice.py)
----------------------------------------------------------------------------
S1  SEPARABILITY. Reward is additive over resources; the transition is
    componentwise; the per-agent action set is the product {-1,0,1}^3; the
    observation noise is isotropic. Hence the planner problem, the naive-myopic
    problem and the stage-Nash problem all factorise EXACTLY into 3 independent
    scalar problems.

S2  LATTICE CLOSURE. With f = frac(x0) for one coordinate, the reachable set is
        L(f) = {f, f+1, ..., f+9}  U  {0, 1, ..., 10}
    -- 21 points for generic f, 11 if f == 0. The integer sublattice is entered
    only through the clamp. The "1331 states x 729 actions" claim of
    ideation_measurement.md sec 0.1 is correct ONLY for integer start states;
    under the environment's Uniform[0,C] reset the exact closure is almost
    surely 21^3 = 9261 states x 729 aggregate actions.

S3  WELFARE ALGEBRA. Wbar_m = sum_i w_im, sstar_m = (sum_i w_im tau_im)/Wbar_m,
        W(s) = -sum_m [ Wbar_m (s_m - sstar_m)^2 + V_m ],
        V_m  = sum_i w_im (tau_im - sstar_m)^2.
    V_m is common to benchmark and realisation so it CANCELS in G. Hence
        G = sum_m Wbar_m [ E_pi (x_m - sstar_m)^2 - (xlat_m - sstar_m)^2 ],
    xlat_m = argmin_{x in L(f_m)} (x - sstar_m)^2 -- the QUANTIZATION FLOOR.
    G is a pure tracking-error difference.

S4  STAKES DO NOT ENTER THE DELEGATED POLICY. Agent i maximises
    -w_im (post - tau_im)^2 with w_im > 0; a positive scalar cannot change an
    argmax. The delegated behaviour is EXACTLY independent of sigma and of w.
    sigma enters G only through Wbar_m (a multiplicative scale) and sstar_m
    (the w-weighted mean of targets). G is therefore degree-1 homogeneous in
    sigma up to (a) the fixed weight dispersion 0.05 and (b) the clip at 1.0.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

# --------------------------------------------------------------------------- #
#  Environment constants (asserted against source in validate_lattice.py)      #
# --------------------------------------------------------------------------- #
C_CAP = 10.0
N_AGENTS = 4
N_RESOURCES = 3
EPISODE_LENGTH = 100
DELTA_SET = np.array([-1.0, 0.0, 1.0])
W_SCALE = 0.05
TIE_ORDER = np.array([1, 0, 2])   # tie rule T1: prefer |delta| small, then delta small


# --------------------------------------------------------------------------- #
#  1. Preference DGPs                                                          #
# --------------------------------------------------------------------------- #
def dgp_implemented(b, xi, alpha):
    """tau_i = alpha*b + (1-|alpha|)*xi_i (factorial_design.py:42-48).
    b: (D, M);  xi: (D, A, M)."""
    return alpha * b[:, None, :] + (1.0 - abs(alpha)) * xi


def dgp_renormalised(b, xi, alpha):
    """Construct-clean variant (ideation_measurement.md sec 0.3): divide by
    sqrt(alpha^2 + (1-|alpha|)^2) so marginal target variance is 1 for all
    alpha and alpha moves cross-agent CORRELATION only."""
    return dgp_implemented(b, xi, alpha) / np.sqrt(alpha ** 2 + (1.0 - abs(alpha)) ** 2)


def _equicorr_sqrt(n: int, rho: float) -> np.ndarray:
    """Verbatim algebra of results/adversarial_rerun/dgp.py:63-72."""
    Sigma = (1.0 - rho) * np.eye(n) + rho * np.ones((n, n))
    evals, evecs = np.linalg.eigh(Sigma)
    evals = np.clip(evals, 0.0, None)
    return (evecs * np.sqrt(evals)) @ evecs.T


def dgp_equicorr(z, rho):
    """targets = L(rho) @ z (dgp.py:75-82). rho IS the pairwise correlation.
    z: (D, A, M)."""
    L = _equicorr_sqrt(z.shape[-2], float(rho))
    return np.einsum("ij,djm->dim", L, z)


def weights_from_z(z, sigma):
    """w = clip(sigma + 0.05*z, 0, 1) -- factorial_design.py:50-51, in
    common-random-number form so cross-sigma contrasts are not MC-confounded."""
    return np.clip(sigma + W_SCALE * z, 0.0, 1.0)


# --------------------------------------------------------------------------- #
#  2. Exact reachable lattice (fact S2)                                        #
# --------------------------------------------------------------------------- #
def lattice(f: float, cap: float = C_CAP) -> np.ndarray:
    ints = np.arange(0.0, cap + 1e-12, 1.0)
    if f <= 1e-12 or f >= 1.0 - 1e-12:
        return ints
    shifted = f + np.arange(0.0, cap - f - 1e-12, 1.0)
    return np.sort(np.concatenate([ints, shifted]))


def lattice_transition_index(L, deltas, cap: float = C_CAP):
    """idx[s, d] = index in L of clip(L[s] + deltas[d], 0, cap).
    Raises if the lattice is not closed -- this IS the S2 assertion."""
    nxt = np.clip(L[:, None] + deltas[None, :], 0.0, cap)
    idx = np.abs(nxt[..., None] - L[None, None, :]).argmin(axis=-1)
    resid = np.abs(nxt - L[idx]).max()
    if resid > 1e-9:
        raise AssertionError(f"lattice not closed: max residual {resid:.3e}")
    return idx


# --------------------------------------------------------------------------- #
#  3. Agent decision objectives                                                #
# --------------------------------------------------------------------------- #
def obj_ce(mu, t, cap: float = C_CAP):
    """Certainty-equivalence loss (clip(mu) - t)^2, mu = believed post-state."""
    return (np.clip(mu, 0.0, cap) - t) ** 2


def obj_bayes(mu, t, eps, cap: float = C_CAP):
    """E[(clip(z)-t)^2] with z ~ N(mu, eps); exact truncated-normal moments.

    Belief model (DECLARED ASSUMPTION B1): diffuse Lebesgue prior on the true
    state + Gaussian likelihood => posterior N(shat, eps). The agent knows the
    clamp; it does not know the lattice.
    """
    if eps <= 0.0:
        return obj_ce(mu, t, cap)
    sd = np.sqrt(eps)
    a = (0.0 - mu) / sd
    bb = (cap - mu) / sd
    Fa, Fb = norm.cdf(a), norm.cdf(bb)
    fa, fb = norm.pdf(a), norm.pdf(bb)
    mid = Fb - Fa
    m1 = sd * (fa - fb)
    m2 = sd * sd * (mid + a * fa - bb * fb)
    d = mu - t
    return m2 + 2.0 * d * m1 + d * d * mid + (t ** 2) * Fa + ((cap - t) ** 2) * (1.0 - Fb)


TIE_TOL = 1e-9


def _argmin_tie(loss, secondary=None):
    """argmin over the last axis (candidates ordered [-1, 0, +1]).

    TIE RULE T1 (declared).  The clamp creates EXACT ties whenever every
    candidate post-state clamps to the same boundary (e.g. a signal shat < -1
    makes all three deltas map to 0).  Breaking those arbitrarily destroys
    monotonicity of the decision rule in shat and is not behaviourally neutral,
    so the rule is lexicographic:
      1. smallest clamped expected loss;
      2. among those, smallest UNCLAMPED squared error (shat + D + delta - t)^2
         -- i.e. the agent that cannot tell the clamp apart still moves toward
         its target;
      3. among those, smallest |delta|, then smallest delta.
    """
    if secondary is None:
        return TIE_ORDER[loss[..., TIE_ORDER].argmin(axis=-1)]
    m = loss.min(axis=-1, keepdims=True)
    sec = np.where(loss <= m + TIE_TOL, secondary, np.inf)
    m2 = sec.min(axis=-1, keepdims=True)
    key = np.where(sec <= m2 + TIE_TOL, 0.0, 1.0)
    return TIE_ORDER[key[..., TIE_ORDER].argmin(axis=-1)]


# --------------------------------------------------------------------------- #
#  4. Naive-myopic delegation -- EXACT (thresholds + convolution)              #
# --------------------------------------------------------------------------- #
def _naive_decision(shat, t, eps, bayes, cap=C_CAP):
    """0/1/2 for delta = -1/0/+1. Others assumed static (shift 0)."""
    mu = shat[..., None] + DELTA_SET
    tt = t[..., None]
    loss = obj_bayes(mu, tt, eps, cap) if bayes else obj_ce(mu, tt, cap)
    return _argmin_tie(loss, (mu - tt) ** 2)


def naive_thresholds(t, eps, bayes, cap=C_CAP, n_scan=2401, n_bisect=45,
                     chunk=512):
    """Chunked wrapper: the scan array is (n_targets, n_scan, 3), so it must be
    blocked or it dominates the runtime through allocation alone."""
    t = np.asarray(t, float)
    flat = t.ravel()
    if flat.size <= chunk:
        a, b = _naive_thresholds_block(flat, eps, bayes, cap, n_scan, n_bisect)
        return a.reshape(t.shape), b.reshape(t.shape)
    o1 = np.empty(flat.size)
    o2 = np.empty(flat.size)
    for lo in range(0, flat.size, chunk):
        hi = min(lo + chunk, flat.size)
        o1[lo:hi], o2[lo:hi] = _naive_thresholds_block(
            flat[lo:hi], eps, bayes, cap, n_scan, n_bisect)
    return o1.reshape(t.shape), o2.reshape(t.shape)


def _naive_thresholds_block(t, eps, bayes, cap=C_CAP, n_scan=2401, n_bisect=45):
    """Thresholds th1 <= th2 of the (verified monotone) naive decision rule:
    delta = +1 for shat < th1, 0 for th1 < shat < th2, -1 for shat > th2.
    +/-inf encode 'never'. Returns arrays shaped like t."""
    t = np.asarray(t, float)
    shape = t.shape
    tf = t.ravel()
    lo_x, hi_x = -8.0, cap + 8.0
    grid = np.linspace(lo_x, hi_x, n_scan)
    dec = 2 - _naive_decision(grid[None, :], tf[:, None], eps, bayes, cap)
    if np.any(np.diff(dec, axis=1) < 0):
        raise AssertionError("naive decision rule not monotone in shat")

    def first_cross(level):
        ge = dec >= level
        any_ge = ge.any(axis=1)
        j = np.where(any_ge, ge.argmax(axis=1), 0)
        below = any_ge & (j == 0)
        act = any_ge & (j > 0)
        lo = np.where(act, grid[np.maximum(j - 1, 0)], lo_x)
        hi = np.where(act, grid[np.minimum(j, n_scan - 1)], hi_x)
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            gm = (2 - _naive_decision(mid, tf, eps, bayes, cap)) >= level
            hi = np.where(gm, mid, hi)
            lo = np.where(gm, lo, mid)
        th = np.where(act, 0.5 * (lo + hi), np.inf)
        return np.where(below, -np.inf, th)

    return first_cross(1).reshape(shape), first_cross(2).reshape(shape)


def naive_action_probs(L, th1, th2, eps):
    """EXACT P(delta_i = d | x) on the lattice. Returns (..., nS, 3) with
    columns [-1, 0, +1]."""
    t1, t2 = th1[..., None], th2[..., None]
    if eps <= 0.0:
        p_plus = (L < t1).astype(float)
        p_zero = ((L >= t1) & (L < t2)).astype(float)
    else:
        sd = np.sqrt(eps)
        p_plus = norm.cdf((t1 - L) / sd)
        p_zero = norm.cdf((t2 - L) / sd) - p_plus
    return np.stack([1.0 - p_plus - p_zero, p_zero, p_plus], axis=-1)


def aggregate_delta_pmf(p_agents):
    """Convolve A independent {-1,0,+1} pmfs. (D, A, nS, 3) -> (D, nS, 2A+1)."""
    A = p_agents.shape[1]
    out = p_agents[:, 0]
    for i in range(1, A):
        nxt = p_agents[:, i]
        new = np.zeros(out.shape[:-1] + (out.shape[-1] + 2,))
        for k in range(3):
            new[..., k:k + out.shape[-1]] += out * nxt[..., k][..., None]
        out = new
    assert out.shape[-1] == 2 * A + 1
    return out


# --------------------------------------------------------------------------- #
#  5. Stage-Nash delegation -- Monte Carlo over the private signals            #
# --------------------------------------------------------------------------- #
def stage_nash_pmf(L2d, t, eps, rng, n_noise=512, bayes=False, cap=C_CAP,
                   max_sweeps=100, chunk=8, start="zeros"):
    """P(sum_i delta_i = D | x) under a stage-Nash profile.

    Equilibrium concept (DECLARED ASSUMPTION N1): interim equilibrium of the
    one-shot game. Agent i observes shat_i = x + N(0,eps) privately, treats the
    other agents' realised equilibrium deltas as known, and best-responds under
    its own belief. Solved by Gauss-Seidel iterated best response from the
    all-zero profile, agent order 0..3, tie rule T1, cycle cut at max_sweeps.
    Non-convergent signal draws keep the last iterate and are counted.

    t: (D, A). Returns (pmf (D, nS, 2A+1), diagnostics dict).
    """
    L2d = np.atleast_2d(L2d)
    nD_draw, A = t.shape
    if L2d.shape[0] == 1:
        L2d = np.broadcast_to(L2d, (nD_draw, L2d.shape[1]))
    nS = L2d.shape[1]
    nD = 2 * A + 1
    pmf = np.zeros((nD_draw, nS, nD))
    sd = np.sqrt(max(eps, 0.0))
    shifts = np.arange(-A, A + 1, dtype=float)
    off = np.array([-1, 0, 1])
    n_bad = 0
    sweeps_used = 0
    for lo in range(0, nD_draw, chunk):
        hi = min(lo + chunk, nD_draw)
        c = hi - lo
        tc = t[lo:hi]
        z = rng.standard_normal((c, nS, n_noise, A))
        shat = L2d[lo:hi][:, :, None, None] + sd * z
        mu = shat[..., None] + shifts
        tt = tc[:, None, None, :, None]
        loss = obj_bayes(mu, tt, eps, cap) if bayes else obj_ce(mu, tt, cap)
        unc = (mu - tt) ** 2
        del mu, shat, z
        if start == "zeros":
            delta = np.zeros((c, nS, n_noise, A))
        elif start == "naive":
            # start the BR dynamic at the naive-myopic profile instead of the
            # all-zero profile: an equilibrium-SELECTION sensitivity check,
            # since 13-31% of (state, signal) cells admit multiple pure NE.
            base0 = A + off
            delta = DELTA_SET[_argmin_tie(loss[..., base0], unc[..., base0])]
        elif start == "random":
            delta = DELTA_SET[rng.integers(0, 3, size=(c, nS, n_noise, A))]
        else:
            raise ValueError(start)
        changed = np.ones((c, nS, n_noise), dtype=bool)
        for sweep in range(max_sweeps):
            changed = np.zeros((c, nS, n_noise), dtype=bool)
            for i in range(A):
                Dm = delta.sum(axis=-1) - delta[..., i]
                base = (Dm + A).astype(np.int64)[..., None] + off
                li = np.take_along_axis(loss[..., i, :], base, axis=-1)
                si = np.take_along_axis(unc[..., i, :], base, axis=-1)
                newd = DELTA_SET[_argmin_tie(li, si)]
                changed |= (newd != delta[..., i])
                delta[..., i] = newd
            sweeps_used = max(sweeps_used, sweep + 1)
            if not changed.any():
                break
        n_bad += int(changed.sum())
        D = delta.sum(axis=-1)
        for k in range(nD):
            pmf[lo:hi, :, k] = (D == (k - A)).mean(axis=-1)
        del loss, unc, delta
    return pmf, {"n_noise": n_noise,
                 "nonconverged_frac": n_bad / (nD_draw * nS * n_noise),
                 "max_sweeps_used": sweeps_used}


def stage_nash_pure_ne_census(L, t, eps, rng, n_noise=256, bayes=False,
                              cap=C_CAP, chunk=8):
    """Exhaustive census of PURE stage-Nash equilibria over all 3^A profiles.
    Returns mean number of pure NE per (state, signal draw) and the fraction
    with zero / exactly one / more than one."""
    L2d = np.atleast_2d(L)
    nD_draw, A = t.shape
    if L2d.shape[0] == 1:
        L2d = np.broadcast_to(L2d, (nD_draw, L2d.shape[1]))
    nS = L2d.shape[1]
    shifts = np.arange(-A, A + 1, dtype=float)
    profiles = np.array(np.meshgrid(*([[-1.0, 0.0, 1.0]] * A), indexing="ij"))
    profiles = profiles.reshape(A, -1).T                     # (3^A, A)
    counts = []
    sd = np.sqrt(max(eps, 0.0))
    off = np.array([-1, 0, 1])
    dm_vals = np.arange(-(A - 1), A)                          # -3..3
    for lo in range(0, nD_draw, chunk):
        hi = min(lo + chunk, nD_draw)
        c = hi - lo
        tc = t[lo:hi]
        z = rng.standard_normal((c, nS, n_noise, A))
        shat = L2d[lo:hi][:, :, None, None] + sd * z
        mu = shat[..., None] + shifts
        tt = tc[:, None, None, :, None]
        loss = obj_bayes(mu, tt, eps, cap) if bayes else obj_ce(mu, tt, cap)
        unc = (mu - tt) ** 2
        del mu, shat, z
        # br[..., i, j] = best response of agent i to others' total dm_vals[j]
        br = np.zeros((c, nS, n_noise, A, dm_vals.size))
        for j, dm in enumerate(dm_vals):
            base = int(dm + A) + off
            br[..., j] = DELTA_SET[_argmin_tie(loss[..., base], unc[..., base])]
        ok = np.ones((c, nS, n_noise, profiles.shape[0]), dtype=bool)
        for i in range(A):
            dmi = profiles.sum(axis=1) - profiles[:, i]
            jj = (dmi - dm_vals[0]).astype(int)
            ok &= (br[..., i, :][..., jj] == profiles[:, i])
        counts.append(ok.sum(axis=-1).ravel())
        del loss, unc, br, ok
    cnt = np.concatenate(counts)
    return {"mean_n_pure_ne": float(cnt.mean()),
            "frac_zero": float((cnt == 0).mean()),
            "frac_one": float((cnt == 1).mean()),
            "frac_multi": float((cnt > 1).mean())}


# --------------------------------------------------------------------------- #
#  6. Chains and long-run averages                                             #
# --------------------------------------------------------------------------- #
def transition_from_pmf(pmf, tidx_agg):
    """pmf (D, nS, nDelta) over aggregate delta -> row-stochastic P (D, nS, nS)."""
    nS, nDel = tidx_agg.shape
    P = np.zeros(pmf.shape[:-1] + (nS,))
    for k in range(nDel):
        col = tidx_agg[:, k]
        for s in range(nS):
            P[..., s, col[s]] += pmf[..., s, k]
    return P


def cesaro_occupation(P, start, n_square=9, t_avg=512):
    """Long-run Cesaro OCCUPATION measure. Burn-in by repeated squaring
    (2^n_square = 512 steps), then average the state distribution over t_avg
    further steps -- this is the Cesaro limit and it is correct for periodic
    chains (the epsilon = 0 dynamics are frequently 2-cyclic)."""
    Pk = P
    for _ in range(n_square):
        Pk = Pk @ Pk
    d = np.einsum("ds,dst->dt", start, Pk)
    acc = np.zeros_like(d)
    for _ in range(t_avg):
        acc += d
        d = np.einsum("ds,dst->dt", d, P)
    return acc / t_avg


def horizon_occupation(P, start, T=EPISODE_LENGTH):
    """Occupation measure of the POST-states over the environment's actual
    T-step episode (reward is on the post-state, so step 1 follows one
    transition)."""
    d = start.copy()
    acc = np.zeros_like(d)
    for _ in range(T):
        d = np.einsum("ds,dst->dt", d, P)
        acc += d
    return acc / T


def cesaro_average(P, start, r, n_square=9, t_avg=512):
    return (cesaro_occupation(P, start, n_square, t_avg) * r).sum(axis=-1)


def horizon_average(P, start, r, T=EPISODE_LENGTH):
    return (horizon_occupation(P, start, T) * r).sum(axis=-1)
