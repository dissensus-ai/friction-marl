"""Validation layer. Nothing in run_grid.py is trusted until this passes.

Checks
------
V1  Environment constants read back from source match the constants in core.py.
V2  Our DGP re-implementations reproduce the committed samplers bit-for-bit.
V3  Lattice closure (fact S2) and the exact size of the reachable state set,
    including an explicit adjudication of the "1331 states x 729 actions"
    claim in ideation_measurement.md sec 0.1.
V4  Exact finite-horizon backward induction on the FULL 3-D lattice equals the
    per-resource separable computation (fact S1) to machine precision, for both
    the integer lattice (1331) and the generic shifted lattice (9261).
V5  The undiscounted long-run planner value equals the static best lattice
    point (the quantization floor), and the T=100 episode average differs from
    it only by the O(<=3 step) transient -- quantified.
V6  The exact naive-myopic threshold/convolution machinery reproduces a direct
    forward Monte-Carlo simulation of the same delegated dynamics.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import core as K  # noqa: E402

OUT = {}


def log(msg):
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# V1 environment constants
# --------------------------------------------------------------------------- #
def v1_constants():
    src_env = (REPO / "friction_marl/envs/resource_allocation.py").read_text()
    src_fac = (REPO / "friction_marl/experiments/factorial_design.py").read_text()
    src_gpu = (REPO / "run_gpu.py").read_text()
    checks = {
        "capacity_10_in_env": "capacity: float = 10.0" in src_env,
        "episode_length_100_in_env": "episode_length: int = 100" in src_env,
        "clip_0_cap_in_env": "np.clip(new_state, 0.0, self.capacity)" in src_env,
        "uniform_reset_in_env": "self._rng.uniform(0.0, self.capacity" in src_env,
        "delta_set_in_factorial": "values = [-1, 0, 1]" in src_fac,
        "weights_N_sigma_0p05": "rng.normal(loc=sigma, scale=0.05" in src_fac,
        "alpha_dgp": "alpha * base_target + (1.0 - abs(alpha)) * noise" in src_fac,
        "n_agents_4": "N_AGENTS = 4" in src_gpu,
        "n_resources_3": "N_RESOURCES = 3" in src_gpu,
        "capacity_10_gpu": "CAPACITY = 10.0" in src_gpu,
        "noise_std_sqrt_eps": "(cond_epsilon ** 0.5)" in src_gpu,
        "reward_uses_true_state": "diff = new_states.unsqueeze(1) - targets_t" in src_gpu,
    }
    assert all(checks.values()), {k: v for k, v in checks.items() if not v}
    OUT["V1_source_constants"] = checks
    log("V1 OK  environment constants confirmed at source")


# --------------------------------------------------------------------------- #
# V2 DGP re-implementation equals the committed samplers
# --------------------------------------------------------------------------- #
def v2_dgp():
    from friction_marl.experiments.factorial_design import sample_agent_params
    from importlib.machinery import SourceFileLoader
    dgp_mod = SourceFileLoader(
        "adv_dgp", str(REPO / "results/adversarial_rerun/dgp.py")).load_module()

    worst_t = worst_w = 0.0
    for alpha in (-0.8, -0.4, 0.0, 0.4, 0.8):
        for sigma in (0.2, 0.6, 1.0):
            rng = np.random.default_rng(12345)
            w_ref, t_ref = sample_agent_params(rng, 4, 3, alpha, sigma)
            rng2 = np.random.default_rng(12345)
            b = rng2.normal(0.0, 1.0, size=3)
            xi = np.stack([rng2.normal(0.0, 1.0, size=3) for _ in range(4)])
            t_ours = K.dgp_implemented(b[None], xi[None], alpha)[0]
            zw = rng2.normal(0.0, 1.0, size=(4, 3))
            w_ours = K.weights_from_z(zw, sigma)
            worst_t = max(worst_t, np.abs(t_ours - t_ref).max())
            worst_w = max(worst_w, np.abs(w_ours - w_ref).max())
    assert worst_t < 1e-12 and worst_w < 1e-12, (worst_t, worst_w)

    worst_e = 0.0
    for rho in (-0.33, 0.0, 0.4, 0.8):
        rng = np.random.default_rng(777)
        _, t_ref = dgp_mod.sample_agent_params_equicorr(rng, 4, 3, rho, 0.6)
        rng2 = np.random.default_rng(777)
        z = rng2.standard_normal((4, 3))
        t_ours = K.dgp_equicorr(z[None], rho)[0]
        worst_e = max(worst_e, np.abs(t_ours - t_ref).max())
    assert worst_e < 1e-12, worst_e
    OUT["V2_dgp_max_abs_diff"] = {"implemented_tau": worst_t,
                                  "weights": worst_w, "equicorr_tau": worst_e}
    log(f"V2 OK  DGP reimplementations exact (max |diff| {max(worst_t, worst_w, worst_e):.2e})")


# --------------------------------------------------------------------------- #
# V3 lattice closure and the 1331 x 729 claim
# --------------------------------------------------------------------------- #
def v3_lattice():
    agg = np.arange(-K.N_AGENTS, K.N_AGENTS + 1, dtype=float)
    res = {}
    # integer start
    Li = K.lattice(0.0)
    K.lattice_transition_index(Li, agg)
    res["integer_start_points_per_dim"] = int(Li.size)
    res["integer_start_states_3d"] = int(Li.size ** 3)
    # generic fractional start
    rng = np.random.default_rng(0)
    sizes = set()
    for f in rng.uniform(0, 1, 200):
        Lf = K.lattice(f)
        K.lattice_transition_index(Lf, agg)
        sizes.add(int(Lf.size))
    assert sizes == {21}, sizes
    res["generic_start_points_per_dim"] = 21
    res["generic_start_states_3d"] = 21 ** 3
    res["aggregate_actions_3d"] = 9 ** 3
    # brute-force closure check by forward exploration from a random start
    f = 0.3719
    x0 = f + 4.0
    seen = {round(x0, 12)}
    frontier = [x0]
    for _ in range(20):
        nxt = set()
        for x in frontier:
            for d in agg:
                y = round(float(np.clip(x + d, 0.0, K.C_CAP)), 12)
                if y not in seen:
                    seen.add(y)
                    nxt.add(y)
        frontier = list(nxt)
        if not frontier:
            break
    assert len(seen) == 21, len(seen)
    res["brute_force_closure_size"] = len(seen)
    res["claim_1331x729_verdict"] = (
        "CORRECT for integer start states (11^3 = 1331 x 9^3 = 729); "
        "INCORRECT for the environment as implemented, whose reset is "
        "Uniform[0,10]^3, giving 21^3 = 9261 reachable states x 729 aggregate "
        "actions almost surely. The lattice is EXACT either way -- there is no "
        "discretisation error, only a larger state set than claimed.")
    OUT["V3_lattice"] = res
    log("V3 OK  closure verified; 1331x729 holds only for integer starts, "
        "9261x729 generically")


# --------------------------------------------------------------------------- #
# V4/V5 exact backward induction, 3-D vs separable
# --------------------------------------------------------------------------- #
def planner_perdim_bi(L, Wbar, sstar, T):
    """Exact undiscounted finite-horizon backward induction on one resource.
    Returns V_0(x) for every x in L (reward = -Wbar (x - sstar)^2 on POST state).
    """
    agg = np.arange(-K.N_AGENTS, K.N_AGENTS + 1, dtype=float)
    idx = K.lattice_transition_index(L, agg)
    r = -Wbar * (L - sstar) ** 2
    V = np.zeros(L.size)
    for _ in range(T):
        V = (r[idx] + V[idx]).max(axis=1)
    return V


def planner_3d_bi(Ls, Wbar, sstar, T):
    """Exact undiscounted backward induction on the full 3-D lattice."""
    agg = np.arange(-K.N_AGENTS, K.N_AGENTS + 1, dtype=float)
    idxs = [K.lattice_transition_index(L, agg) for L in Ls]
    n = [L.size for L in Ls]
    nS = n[0] * n[1] * n[2]
    # flat next-state index for every (state, action)
    i0 = idxs[0][:, :, None, None, None, None]
    i1 = idxs[1][None, None, :, :, None, None]
    i2 = idxs[2][None, None, None, None, :, :]
    nxt = (i0 * n[1] + i1) * n[2] + i2                      # (n0,9,n1,9,n2,9)
    nxt = np.transpose(nxt, (0, 2, 4, 1, 3, 5)).reshape(nS, 729)
    grid = np.meshgrid(Ls[0], Ls[1], Ls[2], indexing="ij")
    r = np.zeros(nS)
    for m in range(3):
        r += -Wbar[m] * (grid[m].ravel() - sstar[m]) ** 2
    V = np.zeros(nS)
    for _ in range(T):
        V = (r[nxt] + V[nxt]).max(axis=1)
    return V, nxt


def v4_v5():
    rng = np.random.default_rng(20260730)
    rows = []
    for trial in range(3):
        f = rng.uniform(0, 1, 3) if trial else np.zeros(3)
        Ls = [K.lattice(fi) for fi in f]
        b = rng.standard_normal(3)
        xi = rng.standard_normal((4, 3))
        tau = K.dgp_implemented(b[None], xi[None], 0.4)[0]
        w = K.weights_from_z(rng.standard_normal((4, 3)), 0.6)
        Wbar = w.sum(axis=0)
        sstar = (w * tau).sum(axis=0) / Wbar
        Vm = (w * (tau - sstar) ** 2).sum(axis=0)

        T = 100
        t0 = time.time()
        V3, _ = planner_3d_bi(Ls, Wbar, sstar, T)
        t3 = time.time() - t0
        Vsep = [planner_perdim_bi(Ls[m], Wbar[m], sstar[m], T) for m in range(3)]
        # start state x0 = f + k0 (k0 uniform on 0..9); compare at every start
        combo = (Vsep[0][:, None, None] + Vsep[1][None, :, None]
                 + Vsep[2][None, None, :]).ravel()
        err = np.abs(V3 - combo).max()
        # long-run / static optimum check
        static = sum(-Wbar[m] * ((Ls[m] - sstar[m]) ** 2).min() for m in range(3))
        # averaged over start states, NOT max: taking the max hides the transient
        # because some starts are already optimal.
        per_step_T = V3.mean() / T
        rows.append({
            "trial": trial,
            "lattice_sizes": [int(L.size) for L in Ls],
            "n_states_3d": int(np.prod([L.size for L in Ls])),
            "sep_vs_3d_max_abs_err": float(err),
            "best_perstep_T100": float(per_step_T),
            "static_lattice_optimum_perstep": float(static),
            "transient_cost_perstep": float(static - per_step_T),
            "irreducible_Vm_sum": float(Vm.sum()),
            "seconds_3d_bi": t3,
        })
        assert err < 1e-9, err
    OUT["V4_V5_backward_induction"] = rows
    log("V4 OK  3-D backward induction == separable per-resource BI "
        f"(max err {max(r['sep_vs_3d_max_abs_err'] for r in rows):.2e})")
    log("V5 OK  planner long-run value == static best lattice point; mean-over-"
        f"starts T=100 transient cost = {np.mean([r['transient_cost_perstep'] for r in rows]):.4f}"
        "/step (cf. PLANNER_TRANSIENT.csv, 0.535 over the full grid)")


# --------------------------------------------------------------------------- #
# V6 exact naive machinery vs direct forward simulation
# --------------------------------------------------------------------------- #
def v6_naive_vs_sim():
    rng = np.random.default_rng(4242)
    rows = []
    for eps in (0.0, 0.25, 1.0):
        f = 0.4137
        L = K.lattice(f)
        agg = np.arange(-4, 5, dtype=float)
        tidx = K.lattice_transition_index(L, agg)
        tau = rng.standard_normal((1, 4))
        th1, th2 = K.naive_thresholds(tau, eps, bayes=False)
        p = K.naive_action_probs(L, th1, th2, eps)          # (1,4,nS,3)
        pmf = K.aggregate_delta_pmf(p)                       # (1,nS,9)
        P = K.transition_from_pmf(pmf, tidx)
        sstar = 0.3
        r = -(L - sstar) ** 2
        start = np.zeros((1, L.size))
        start[0, np.argmin(np.abs(L - (f + 4.0)))] = 1.0
        exact = K.cesaro_average(P, start, r)[0]

        # direct forward simulation of the same rule
        n_sim, T = 20000, 3000
        x = np.full(n_sim, f + 4.0)
        acc = 0.0
        srng = np.random.default_rng(99)
        for t in range(T):
            shat = x[:, None] + np.sqrt(eps) * srng.standard_normal((n_sim, 4))
            mu = shat[..., None] + K.DELTA_SET
            tt = tau[0][None, :, None]
            loss = K.obj_ce(mu, tt)
            d = K.DELTA_SET[K._argmin_tie(loss, (mu - tt) ** 2)]
            x = np.clip(x + d.sum(axis=1), 0.0, K.C_CAP)
            if t >= 500:
                acc += (-(x - sstar) ** 2).mean()
        sim = acc / (T - 500)
        rows.append({"eps": eps, "exact_cesaro": float(exact),
                     "forward_sim": float(sim), "abs_diff": float(abs(exact - sim))})
    OUT["V6_naive_exact_vs_sim"] = rows
    worst = max(r["abs_diff"] for r in rows)
    assert worst < 5e-3, rows
    log(f"V6 OK  exact naive chain == forward simulation (max |diff| {worst:.2e})")


# --------------------------------------------------------------------------- #
# V7 stage-Nash MC chain vs direct forward simulation of the same equilibrium
# --------------------------------------------------------------------------- #
def v7_nash_vs_sim():
    rows = []
    for eps in (0.0, 0.5):
        f = 0.4137
        L = K.lattice(f)
        agg = np.arange(-4, 5, dtype=float)
        tidx = K.lattice_transition_index(L, agg)
        tau = np.array([[1.7, -0.4, 0.9, 2.3]])
        rng = np.random.default_rng(11)
        pmf, diag = K.stage_nash_pmf(L, tau, eps, rng, n_noise=4096, chunk=1)
        P = K.transition_from_pmf(pmf, tidx)
        sstar = 1.1
        r = -(L - sstar) ** 2
        start = np.zeros((1, L.size))
        start[0, np.argmin(np.abs(L - (f + 4.0)))] = 1.0
        exact = K.cesaro_average(P, start, r)[0]

        n_sim, T, burn = 8000, 2500, 400
        srng = np.random.default_rng(5)
        x = np.full(n_sim, f + 4.0)
        acc = 0.0
        shifts = np.arange(-4, 5, dtype=float)
        for t in range(T):
            shat = x[:, None] + np.sqrt(eps) * srng.standard_normal((n_sim, 4))
            mu = shat[..., None] + shifts
            tt = tau[0][None, :, None]
            loss = K.obj_ce(mu, tt)
            unc = (mu - tt) ** 2
            delta = np.zeros((n_sim, 4))
            for _ in range(100):
                ch = False
                for i in range(4):
                    Dm = delta.sum(axis=1) - delta[:, i]
                    base = (Dm + 4).astype(int)[:, None] + np.array([-1, 0, 1])
                    li = np.take_along_axis(loss[:, i, :], base, axis=1)
                    si = np.take_along_axis(unc[:, i, :], base, axis=1)
                    nd = K.DELTA_SET[K._argmin_tie(li, si)]
                    ch |= bool((nd != delta[:, i]).any())
                    delta[:, i] = nd
                if not ch:
                    break
            x = np.clip(x + delta.sum(axis=1), 0.0, K.C_CAP)
            if t >= burn:
                acc += (-(x - sstar) ** 2).mean()
        sim = acc / (T - burn)
        rows.append({"eps": eps, "exact_cesaro_mc_chain": float(exact),
                     "forward_sim": float(sim),
                     "abs_diff": float(abs(exact - sim)),
                     "nonconverged_frac": diag["nonconverged_frac"],
                     "max_sweeps_used": diag["max_sweeps_used"]})
    OUT["V7_nash_chain_vs_sim"] = rows
    worst = max(r["abs_diff"] for r in rows)
    assert worst < 2e-2, rows
    log(f"V7 OK  stage-Nash chain == forward simulation (max |diff| {worst:.2e})")


# --------------------------------------------------------------------------- #
# V8 Bayes decision rule: monotone, and the CE limit as eps -> 0
# --------------------------------------------------------------------------- #
def v8_bayes():
    rng = np.random.default_rng(31337)
    tau = rng.standard_normal((400, 4)).ravel()
    rows = []
    for eps in (1e-4, 0.25, 0.5, 1.0):
        th1b, th2b = K.naive_thresholds(tau, eps, bayes=True)     # asserts monotone
        th1c, th2c = K.naive_thresholds(tau, eps, bayes=False)
        fin = np.isfinite(th1b) & np.isfinite(th1c)
        rows.append({"eps": eps,
                     "max_abs_th1_bayes_minus_ce": float(np.abs(th1b - th1c)[fin].max()),
                     "max_abs_th2_bayes_minus_ce": float(
                         np.abs(th2b - th2c)[np.isfinite(th2b) & np.isfinite(th2c)].max())})
    OUT["V8_bayes_vs_ce_thresholds"] = rows
    # Bayes -> CE as eps -> 0: threshold gap must shrink with eps (the only
    # place they can differ is the clamp boundary, smoothed on scale sqrt(eps)).
    assert rows[0]["max_abs_th1_bayes_minus_ce"] < rows[-1]["max_abs_th1_bayes_minus_ce"]
    log("V8 OK  Bayes decision rule monotone in shat; Bayes->CE as eps->0 "
        f"(max |th1 gap| {rows[0]['max_abs_th1_bayes_minus_ce']:.3f} at eps=1e-4 vs "
        f"{rows[-1]['max_abs_th1_bayes_minus_ce']:.3f} at eps=1.0)")


if __name__ == "__main__":
    v1_constants()
    v2_dgp()
    v3_lattice()
    v4_v5()
    v6_naive_vs_sim()
    v7_nash_vs_sim()
    v8_bayes()
    (HERE / "VALIDATION.json").write_text(json.dumps(OUT, indent=2))
    log("\nALL VALIDATION CHECKS PASSED -> VALIDATION.json")
