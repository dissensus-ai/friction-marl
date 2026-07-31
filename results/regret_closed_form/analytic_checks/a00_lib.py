"""Independent exact-chain library for the analytic derivation of E[G](sigma,alpha,eps).

Written from scratch (deliberately NOT importing ../core.py) so that the analyst's
numbers and the numerics agent's numbers are independent implementations of the
same object.

ENVIRONMENT FACTS, each verified against source in a01_env_verify.py
--------------------------------------------------------------------
run_gpu.py:54-66      N_AGENTS=4, N_RESOURCES=3, CAPACITY=10.0, EPISODE_LENGTH=100
run_gpu.py:238        reset s0 ~ Uniform[0,CAPACITY]^3  (CONTINUOUS)
run_gpu.py:250-256    obs_i = s + N(0, eps*I), i.e. std = sqrt(eps) per component
run_gpu.py:271-276    s' = clamp(s + sum_i delta_i, 0, CAPACITY)
run_gpu.py:279-281    r_i = sum_m w_im * ( -(s'_m - tau_im)^2 )  -- TRUE state, TRUE w, TRUE tau
factorial_design.py:42-48   tau_i = alpha*b + (1-|alpha|)*xi_i, b shared across agents
factorial_design.py:50-51   w ~ N(sigma, 0.05) clipped to [0,1]
factorial_design.py:55-60   per-agent action = {-1,0,1}^3 (27 actions)
adversarial_rerun/dgp.py:75-82   equicorr: targets = L(rho) @ z, unit marginal variance

DERIVED STRUCTURE USED THROUGHOUT
---------------------------------
(D1) The set of achievable AGGREGATE deltas is exactly {-4,...,4}^3 and is a
     PRODUCT set, because each agent freely chooses each coordinate of its own
     delta vector.  => the planner's control problem factorises over resources.
(D2) Reward is additive over resources; noise is isotropic and independent
     across resources; the transition is componentwise.  => the delegated
     best-response problem ALSO factorises over resources, exactly.
(D3) Per resource: with Wbar = sum_i w_i, mu = sum_i w_i tau_i / Wbar,
     V = sum_i w_i (tau_i - mu)^2,
         W(x) = -[ Wbar (x - mu)^2 + V ].
     V is identical for benchmark and realisation, so it CANCELS in G.
(D4) With frac f = x0 - floor(x0), the reachable state set of ONE coordinate is
         L(f) = {0,1,...,10} u {f, f+1, ..., f+9}      (21 points, generic f)
     closed under x -> clamp(x + D, 0, 10), D in {-4..4}.  The integer sublattice
     is absorbing; the f-coset is transient (entered only at t=0, left only via
     the clamp).
(D5) The delegated policy does not depend on w at all (a positive scalar cannot
     move an argmax of a per-resource quadratic), so sigma enters G only through
     the multiplicative scale Wbar and through mu.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm
from scipy.special import ndtr as _ndtr

CAP = 10.0
N_AGENTS = 4
N_RES = 3
T_EPISODE = 100
W_DISP = 0.05
NSTATE = 21          # 11 integers + 10 coset points, generic f
DELTAS = np.arange(-4, 5)   # aggregate delta alphabet, 9 values


# --------------------------------------------------------------------------- #
# 1.  Lattice + transition index map (independent of the draw; see D4)         #
# --------------------------------------------------------------------------- #
def build_transition_index() -> np.ndarray:
    """(21, 9) int array: TIDX[j, a] = index of clamp(value(j) + DELTAS[a]).

    index 0..10   -> integer value j
    index 11..20  -> coset value f + (j-11), k = j-11 in 0..9
    """
    tidx = np.empty((NSTATE, len(DELTAS)), dtype=np.int64)
    for a, D in enumerate(DELTAS):
        for j in range(11):                       # integer states
            tidx[j, a] = int(np.clip(j + D, 0, 10))
        for k in range(10):                       # coset states, value f+k, 0<f<1
            kd = k + D
            if kd < 0:                            # f+kd <= f-1 < 0   -> clamp low
                tidx[11 + k, a] = 0
            elif kd > 9:                          # f+kd >= f+10 > 10 -> clamp high
                tidx[11 + k, a] = 10
            else:
                tidx[11 + k, a] = 11 + kd
    return tidx


TIDX = build_transition_index()


def state_values(frac: np.ndarray) -> np.ndarray:
    """(R, 21) state values for a batch of fractional parts frac (R,)."""
    R = frac.shape[0]
    v = np.empty((R, NSTATE))
    v[:, :11] = np.arange(11.0)[None, :]
    v[:, 11:] = frac[:, None] + np.arange(10.0)[None, :]
    return v


def start_index(x0: np.ndarray) -> np.ndarray:
    """(R,) start index for x0 in [0, 10). Generic frac>0 -> coset index."""
    k0 = np.floor(x0).astype(np.int64)
    frac = x0 - k0
    idx = np.where(frac > 0, 11 + np.minimum(k0, 9), np.minimum(k0, 10))
    return idx


# --------------------------------------------------------------------------- #
# 2.  Preference DGPs                                                          #
# --------------------------------------------------------------------------- #
def targets_implemented(b, xi, alpha):
    """factorial_design.sample_agent_params: tau_i = alpha*b + (1-|alpha|)*xi_i."""
    return alpha * b[:, None] + (1.0 - abs(alpha)) * xi


def targets_renormalised(b, xi, alpha):
    """Gate item 5 / measurement spec sec 0.3: unit marginal variance."""
    s = np.sqrt(alpha ** 2 + (1.0 - abs(alpha)) ** 2)
    return (alpha * b[:, None] + (1.0 - abs(alpha)) * xi) / s


def equicorr_sqrt(n, rho):
    Sig = (1.0 - rho) * np.eye(n) + rho * np.ones((n, n))
    ev, U = np.linalg.eigh(Sig)
    ev = np.clip(ev, 0.0, None)
    return (U * np.sqrt(ev)) @ U.T


def targets_equicorr(z, rho):
    """dgp.sample_agent_params_equicorr: rho IS the pairwise correlation."""
    L = equicorr_sqrt(z.shape[1], float(rho))
    return z @ L.T


def rho_of_alpha(alpha):
    """Population cross-agent target correlation induced by the implemented DGP."""
    a2 = alpha ** 2
    return a2 / (a2 + (1.0 - abs(alpha)) ** 2)


def lam_of_alpha_implemented(alpha):
    """Dispersion scale of the CENTRED targets under the implemented DGP."""
    return 1.0 - abs(alpha)


def lam_of_rho(rho):
    """Dispersion scale of the CENTRED targets at unit marginal variance."""
    return np.sqrt(1.0 - rho)


# --------------------------------------------------------------------------- #
# 3.  Delegated behaviour: per-agent action probabilities given the TRUE state #
# --------------------------------------------------------------------------- #
# Agent i's myopic best response on ONE resource. Its information is
# shat_i = s + eta_i, eta_i ~ N(0, eps); it holds a flat prior on s, so
# s | shat_i ~ N(shat_i, eps).  Let c_i be its conjecture about sum_{j!=i} delta_j.
# It solves
#     delta_i = argmin_{d in {-1,0,1}}  E_{s|shat_i} [ (clamp(s + c_i + d) - tau_i)^2 ]
# CLAMP-AWARE (this is literally "argmax own expected r_i", since r_i is
# evaluated at the CLAMPED post-state).  Writing u = shat_i + c_i, the decision
# depends on (u, tau_i, eps) only, and is monotone non-increasing in u, so it is
# described by two thresholds tA(tau,eps) >= ... :
#     delta_i = +1  iff u < tA ;   -1  iff u > tB ;   0 otherwise    (tA <= tB)
# Hence, given the TRUE state x:
#     p_i^+ = Phi( (tA - c_i - x)/sqrt(eps) ),  p_i^- = Phi( (x + c_i - tB)/sqrt(eps) )
#
# c_i = 0            -> NAIVE-MYOPIC (others treated as static)
# c_i = E[D_{-i}|x]  -> STAGE-NASH-IN-EXPECTATIONS (SNE)
# At eps = 0 the exact pure-strategy Nash correspondence is used instead (a02).
# --------------------------------------------------------------------------- #
def expected_clamped_loss(m, tau, eps, cap=CAP):
    """E[(clamp(Z,0,cap) - tau)^2] for Z ~ N(m, eps). Exact, vectorised."""
    if eps <= 0:
        return (np.clip(m, 0.0, cap) - tau) ** 2
    sd = np.sqrt(eps)
    a = (0.0 - m) / sd
    b = (cap - m) / sd
    Fa, Fb = _ndtr(a), _ndtr(b)
    fa = np.exp(-0.5 * a * a) * 0.3989422804014327
    fb = np.exp(-0.5 * b * b) * 0.3989422804014327
    mid = ((m - tau) ** 2 * (Fb - Fa)
           + 2.0 * (m - tau) * sd * (fa - fb)
           + eps * ((Fb - Fa) + (a * fa - b * fb)))
    return tau ** 2 * Fa + (cap - tau) ** 2 * (1.0 - Fb) + mid


def _delta_star(u, tau, eps, tie=1e-9):
    """argmin over d in {-1,0,1}; returns integer array in {-1,0,1}.

    TIE RULE (declared, and load-bearing at the clamp): when two deltas give the
    SAME expected clamped loss -- which happens exactly when the agent believes
    every move is absorbed by the wall -- break toward the UNCLAMPED best
    response clip(round(tau-u),-1,1).  This is the trembling-hand refinement
    toward the interior problem; the alternative ("do nothing") makes an agent at
    the wall stop counteracting the others' pushes, which is a strictly weaker
    best response in the multi-agent game (and the stage-Nash protocol resolves
    it the same way).  Sensitivity to this rule is reported in a02.
    """
    d_int = np.clip(np.round(tau - u), -1.0, 1.0)
    losses = np.stack([expected_clamped_loss(u + d, tau, eps)
                       + tie * np.abs(d - d_int) for d in (-1.0, 0.0, 1.0)],
                      axis=-1)
    return np.argmin(losses, axis=-1) - 1


def thresholds(tau, eps, lo=-60.0, hi=80.0, n_bis=64):
    """(R,n) targets -> tA, tB with delta=+1 iff u<tA, -1 iff u>tB."""
    shape = tau.shape
    # tA = sup{u : delta*(u) == +1};  predicate P1(u) = (delta*(u) == +1) is
    # monotone non-increasing in u.
    loA = np.full(shape, lo)
    hiA = np.full(shape, hi)
    p_lo = _delta_star(loA, tau, eps) == 1
    for _ in range(n_bis):
        mid = 0.5 * (loA + hiA)
        p = _delta_star(mid, tau, eps) == 1
        loA = np.where(p, mid, loA)
        hiA = np.where(p, hiA, mid)
    tA = np.where(p_lo, 0.5 * (loA + hiA), -np.inf)
    # tB = inf{u : delta*(u) == -1}; P2(u) = (delta*(u) == -1) monotone non-decreasing
    loB = np.full(shape, lo)
    hiB = np.full(shape, hi)
    p_hi = _delta_star(hiB, tau, eps) == -1
    for _ in range(n_bis):
        mid = 0.5 * (loB + hiB)
        p = _delta_star(mid, tau, eps) == -1
        hiB = np.where(p, mid, hiB)
        loB = np.where(p, loB, mid)
    tB = np.where(p_hi, 0.5 * (loB + hiB), np.inf)
    tB = np.maximum(tB, tA)
    return tA, tB


def probs_from_thresholds(tA, tB, c, xval, eps):
    """tA,tB: (R,n); c: (R,S,n) or scalar; xval: (R,S) -> p_plus, p_minus (R,S,n)."""
    A = tA[:, None, :]
    B = tB[:, None, :]
    X = xval[:, :, None]
    u = X + c                      # agent's effective pre-move position estimate
    if eps <= 0:
        pp = (u < A).astype(float)
        pm = (u > B).astype(float)
    else:
        inv = 1.0 / np.sqrt(eps)
        pp = _ndtr((A - u) * inv)
        pm = _ndtr((u - B) * inv)
    return pp, pm


def agent_probs(t, xval, eps, protocol="naive", n_iter=600, damp=0.25,
                clamp_aware=True, report_conv=False):
    """(R,S,n) p_plus, p_minus for the requested behavioural protocol."""
    if clamp_aware:
        tA, tB = thresholds(t, eps)
    else:                                       # interior deadzone rule
        tA, tB = t - 0.5, t + 0.5
    if protocol == "naive":
        out = probs_from_thresholds(tA, tB, 0.0, xval, eps)
        return (out + (0.0,)) if report_conv else out
    if protocol != "sne":
        raise ValueError(protocol)
    R, S = xval.shape
    n = t.shape[1]
    c = np.zeros((R, S, n))
    resid = np.inf
    for _ in range(n_iter):
        pp, pm = probs_from_thresholds(tA, tB, c, xval, eps)
        e = pp - pm
        c_new = e.sum(axis=2, keepdims=True) - e          # leave-one-out
        resid = np.abs(c_new - c).max()
        c = (1 - damp) * c + damp * c_new
    out = probs_from_thresholds(tA, tB, c, xval, eps)
    return (out[0], out[1], resid) if report_conv else out


# --------------------------------------------------------------------------- #
# 3b. EXACT pure-strategy Nash correspondence at eps = 0 (clamp included)      #
# --------------------------------------------------------------------------- #
_PROFILES = np.array(list(np.ndindex(3, 3, 3, 3)), dtype=np.int64) - 1   # (81,4)


def nash_pmf_exact(t, xval, cap=CAP, selection="welfare_best"):
    """(R,S,9) degenerate pmf placing mass 1 on the selected Nash aggregate.

    A profile is a pure Nash equilibrium iff no agent strictly prefers another
    unilateral step, evaluated on the ACTUAL clamped post-state.
    selection in {welfare_best, welfare_worst, minimal}.
    """
    R, S = xval.shape
    n = t.shape[1]
    isna = np.zeros((R, S, 2 * n + 1), dtype=bool)
    X = xval[:, :, None] if False else xval
    for prof in _PROFILES:
        D = int(prof.sum())
        y = np.clip(X + D, 0.0, cap)                       # (R,S)
        good = np.ones((R, S), dtype=bool)
        for i in range(n):
            rest = D - prof[i]
            alt = np.stack([np.clip(X + rest + d, 0.0, cap) for d in (-1, 0, 1)],
                           axis=-1)                        # (R,S,3)
            ti = t[:, i][:, None]
            best = np.min((alt - ti[:, :, None]) ** 2, axis=-1)
            good &= (y - ti) ** 2 <= best + 1e-12
        isna[:, :, D + n] |= good
    # welfare at the post-state for each aggregate
    ys = np.stack([np.clip(X + D, 0.0, cap) for D in range(-n, n + 1)], axis=-1)
    pmf = np.zeros((R, S, 2 * n + 1))
    if selection == "minimal":
        score = -np.abs(np.arange(-n, n + 1))[None, None, :] * np.ones_like(ys)
    else:
        # ensemble tracking loss needs mu; caller supplies it via closure-free
        # convention: welfare ordering by (y - mu)^2 is done in nash_pmf_wrap
        raise ValueError("use nash_pmf_wrap")
    score = np.where(isna, score, -np.inf)
    pick = np.argmax(score, axis=-1)
    np.put_along_axis(pmf, pick[:, :, None], 1.0, axis=-1)
    return pmf


def nash_pmf_wrap(t, xval, mu, selection="welfare_best", cap=CAP):
    """As nash_pmf_exact but the welfare selection uses the ensemble target mu."""
    R, S = xval.shape
    n = t.shape[1]
    isna = np.zeros((R, S, 2 * n + 1), dtype=bool)
    for prof in _PROFILES:
        D = int(prof.sum())
        y = np.clip(xval + D, 0.0, cap)
        good = np.ones((R, S), dtype=bool)
        for i in range(n):
            rest = D - prof[i]
            alt = np.stack([np.clip(xval + rest + d, 0.0, cap) for d in (-1, 0, 1)],
                           axis=-1)
            ti = t[:, i][:, None]
            best = np.min((alt - ti[:, :, None]) ** 2, axis=-1)
            good &= (y - ti) ** 2 <= best + 1e-12
        isna[:, :, D + n] |= good
    ys = np.stack([np.clip(xval + D, 0.0, cap) for D in range(-n, n + 1)], axis=-1)
    loss = (ys - mu[:, None, None]) ** 2
    if selection == "welfare_best":
        score = -loss
    elif selection == "welfare_worst":
        score = loss
    elif selection == "minimal":
        score = -np.abs(np.arange(-n, n + 1))[None, None, :] * np.ones_like(ys)
    else:
        raise ValueError(selection)
    score = np.where(isna, score, -np.inf)
    pick = np.argmax(score, axis=-1)
    pmf = np.zeros((R, S, 2 * n + 1))
    np.put_along_axis(pmf, pick[:, :, None], 1.0, axis=-1)
    return pmf, isna


def aggregate_pmf(pp, pm):
    """(R,S,n) three-point pmfs -> (R,S,9) pmf of D = sum_i delta_i over -4..4."""
    R, S, n = pp.shape
    out = np.zeros((R, S, 2 * n + 1))
    out[:, :, n] = 1.0                                # start at D=0
    for i in range(n):
        new = np.zeros_like(out)
        p0 = 1.0 - pp[:, :, i] - pm[:, :, i]
        new[:, :, 1:] += out[:, :, :-1] * pp[:, :, i][:, :, None]
        new[:, :, :] += out * p0[:, :, None]
        new[:, :, :-1] += out[:, :, 1:] * pm[:, :, i][:, :, None]
        out = new
    return out


def transition_matrix(pmf):
    """(R,S,9) -> (R,S,S) row-stochastic transition matrix using TIDX."""
    R, S, A = pmf.shape
    M = np.zeros((R, S, S))
    for j in range(S):
        for a in range(A):
            M[:, j, TIDX[j, a]] += pmf[:, j, a]
    return M


# --------------------------------------------------------------------------- #
# 4.  Exact episode occupancy                                                  #
# --------------------------------------------------------------------------- #
def delegated_occupancy(M, start_idx, T=T_EPISODE):
    """Time-averaged distribution over POST-transition states, exact.

    Returns (R,S) occupancy with rows summing to 1 (average over t=1..T).
    """
    R, S, _ = M.shape
    d = np.zeros((R, S))
    d[np.arange(R), start_idx] = 1.0
    occ = np.zeros((R, S))
    for _ in range(T):
        d = np.einsum("rj,rjk->rk", d, M)
        occ += d
    return occ / T


def planner_occupancy(xval, mu, Wbar, T=T_EPISODE, start_idx=None):
    """Exact finite-horizon backward induction on the SAME 21-state chain.

    Objective: maximise sum_{t=1..T} -( Wbar (x_t - mu)^2 + V ), V constant.
    Returns (value_per_step_excess, occupancy) where the occupancy is over
    post-transition states of the optimal deterministic path.
    """
    R, S = xval.shape
    r = -(xval - mu[:, None]) ** 2                      # drop Wbar and V; add later
    A = len(DELTAS)
    J = np.zeros((R, S))
    policy = np.empty((T, R, S), dtype=np.int64)
    for k in range(T):
        cand = r[:, TIDX] + J[:, TIDX]                  # (R,S,A)
        policy[k] = np.argmax(cand, axis=2)
        J = np.take_along_axis(cand, policy[k][:, :, None], axis=2)[:, :, 0]
    # roll forward: policy[T-1] is used first (T actions remaining)
    idx = start_idx.copy()
    occ = np.zeros((R, S))
    rows = np.arange(R)
    for k in range(T - 1, -1, -1):
        a = policy[k][rows, idx]
        idx = TIDX[idx, a]
        occ[rows, idx] += 1.0
    return occ / T


# --------------------------------------------------------------------------- #
# 5.  Assemble G for a batch of single-resource problems                       #
# --------------------------------------------------------------------------- #
def gap_batch(t, w, x0, eps, protocol="naive", T=T_EPISODE, want_agent=False,
              clamp_aware=True, selection="welfare_best"):
    """Exact signed delegation gap for a batch of one-resource problems.

    t  : (R, n) targets;  w : (R, n) weights;  x0 : (R,) start in [0,10)
    protocol: 'naive' | 'sne' | 'nash' (exact pure Nash, eps must be 0)
    Returns dict with G, and the exact three-term decomposition.
    """
    R, n = t.shape
    Wbar = w.sum(axis=1)
    mu = (w * t).sum(axis=1) / Wbar
    frac = x0 - np.floor(x0)
    xval = state_values(frac)
    sidx = start_index(x0)

    if protocol == "nash":
        if eps != 0:
            raise ValueError("exact Nash correspondence is defined at eps=0")
        pmf, isna = nash_pmf_wrap(t, xval, mu, selection=selection)
        M = transition_matrix(pmf)
    else:
        pp, pm = agent_probs(t, xval, eps, protocol=protocol,
                             clamp_aware=clamp_aware)
        M = transition_matrix(aggregate_pmf(pp, pm))
    occ_d = delegated_occupancy(M, sidx, T=T)
    occ_p = planner_occupancy(xval, mu, Wbar, T=T, start_idx=sidx)

    dev = xval - mu[:, None]
    Td = (occ_d * dev ** 2).sum(axis=1)          # E_pi (x-mu)^2
    Tp = (occ_p * dev ** 2).sum(axis=1)          # planner residual (quantization/boundary)
    G = Wbar * (Td - Tp)

    # --- exact three-term decomposition, CENTRED ON THE PLANNER'S OWN POINT --- #
    #   G/Wbar = Var_pi(x)  +  bdisp^2  +  2 * qres * bdisp
    #   xstar = planner's time-averaged state (its optimum, up to the <=6-step
    #           transient);  qres = xstar - mu is the SIGNED quantization/boundary
    #           residual;  bdisp = E_pi[x] - xstar is the DISPLACEMENT of the
    #           delegated attractor away from the planner's point.
    xbar = (occ_d * xval).sum(axis=1)
    xstar = (occ_p * xval).sum(axis=1)
    var = Td - (xbar - mu) ** 2                  # Var_pi(x)
    qres = xstar - mu
    bdisp = xbar - xstar
    var_p = Tp - qres ** 2                       # planner transient variance (small)
    out = dict(G=G, Wbar=Wbar, mu=mu, Td=Td, Tp=Tp, var=var, xbar=xbar,
               xstar=xstar, qres=qres, bdisp=bdisp, var_p=var_p,
               bias2=(xbar - mu) ** 2, q2=Tp,
               occ_d=occ_d, occ_p=occ_p, xval=xval)
    if want_agent:
        Gi = np.empty((R, n))
        for i in range(n):
            di = xval - t[:, i][:, None]
            Gi[:, i] = w[:, i] * ((occ_d * di ** 2).sum(axis=1)
                                  - (occ_p * di ** 2).sum(axis=1))
        out["Gi"] = Gi
    return out


def draw_problem(rng, R, dgp, knob, sigma, n=N_AGENTS):
    """One-resource draws. Returns t (R,n), w (R,n), x0 (R,)."""
    if dgp == "implemented":
        b = rng.standard_normal(R)
        xi = rng.standard_normal((R, n))
        t = targets_implemented(b, xi, knob)
    elif dgp == "renormalised":
        b = rng.standard_normal(R)
        xi = rng.standard_normal((R, n))
        t = targets_renormalised(b, xi, knob)
    elif dgp == "equicorr":
        z = rng.standard_normal((R, n))
        t = targets_equicorr(z, knob)
    elif dgp == "lam":            # direct: centred targets with dispersion knob
        z = rng.standard_normal((R, n))
        t = knob * z
    else:
        raise ValueError(dgp)
    w = np.clip(sigma + W_DISP * rng.standard_normal((R, n)), 0.0, 1.0)
    x0 = rng.uniform(0.0, CAP, size=R)
    return t, w, x0


# --------------------------------------------------------------------------- #
# 6.  Chain reuse across sigma (the delegated chain is EXACTLY sigma-free, D5)  #
# --------------------------------------------------------------------------- #
def gap_multi_sigma(t, x0, sigmas, eps, rng, protocol="sne", T=T_EPISODE,
                    n_iter=250, damp=0.4, want_agent=False):
    """Compute the delegated chain ONCE, then evaluate G for several sigmas.

    Valid because the delegated transition kernel does not depend on w at all
    (verified as check C2 in a03).  sigma enters only through Wbar and mu, both
    of which are cheap, and through the planner's backward induction.
    """
    R, n = t.shape
    frac = x0 - np.floor(x0)
    xval = state_values(frac)
    sidx = start_index(x0)
    if protocol == "nash":
        raise ValueError("use gap_batch for the exact Nash correspondence")
    pp, pm = agent_probs(t, xval, eps, protocol=protocol, n_iter=n_iter, damp=damp)
    M = transition_matrix(aggregate_pmf(pp, pm))
    occ_d = delegated_occupancy(M, sidx, T=T)
    out = {}
    for sigma in sigmas:
        w = np.clip(sigma + W_DISP * rng.standard_normal((R, n)), 0.0, 1.0)
        Wbar = w.sum(axis=1)
        mu = (w * t).sum(axis=1) / Wbar
        occ_p = planner_occupancy(xval, mu, Wbar, T=T, start_idx=sidx)
        dev = xval - mu[:, None]
        Td = (occ_d * dev ** 2).sum(axis=1)
        Tp = (occ_p * dev ** 2).sum(axis=1)
        G = Wbar * (Td - Tp)
        xbar = (occ_d * xval).sum(axis=1)
        xstar = (occ_p * xval).sum(axis=1)
        d = dict(G=G, Wbar=Wbar, mu=mu, var=Td - (xbar - mu) ** 2,
                 qres=xstar - mu, bdisp=xbar - xstar, xbar=xbar, xstar=xstar)
        if want_agent:
            Gi = np.empty((R, n))
            for i in range(n):
                di = xval - t[:, i][:, None]
                Gi[:, i] = w[:, i] * ((occ_d * di ** 2).sum(axis=1)
                                      - (occ_p * di ** 2).sum(axis=1))
            d["Gi"] = Gi
            d["w"] = w
        out[sigma] = d
    return out
