"""X9 -- reproduce ideation_measurement.md sec 0.1's V*_p on the EXACT seed-42
condition draws, by replaying run_gpu.py's own seed construction (:456-460)
and factorial_design.sample_agent_params.

Spec table (sec 0.1):
    alpha=-0.8 sigma=0.6 : V*_p = -0.767 , eval_reward_clean = -1.309 , regret 0.542
    alpha= 0.0 sigma=0.6 : V*_p = -1.014 , eval_reward_clean = -2.287 , regret 1.273
    alpha= 0.8 sigma=1.0 : V*_p = -1.158 , eval_reward_clean = -2.287 , regret 1.129
"""
import json
import numpy as np

CAP, N_AG, N_RES, N_REPS = 10.0, 4, 3, 30
ALPHAS = [-0.8, -0.4, 0.0, 0.4, 0.8]
SIGMAS = [0.2, 0.4, 0.6, 0.8, 1.0]
EPSILONS = [0.0, 0.25, 0.5, 0.75, 1.0]


def condition_seeds():
    """Replay run_gpu.py:456-460 exactly."""
    master = np.random.default_rng(42)
    _probe = master.uniform(0.0, CAP, size=(128, N_RES)).astype(np.float32)
    conds = [(a, s, e) for a in ALPHAS for s in SIGMAS for e in EPSILONS]
    seeds = [int(master.integers(0, 2 ** 62)) for _ in conds]
    return dict(zip(conds, seeds))


def sample_agent_params(rng, alpha, sigma):
    """Verbatim factorial_design.sample_agent_params."""
    base = rng.normal(0.0, 1.0, size=N_RES)
    tg = []
    for _ in range(N_AG):
        noise = rng.normal(0.0, 1.0, size=N_RES)
        tg.append(alpha * base + (1.0 - abs(alpha)) * noise)
    t = np.stack(tg, 0)
    w = np.clip(rng.normal(loc=sigma, scale=0.05, size=(N_AG, N_RES)), 0.0, 1.0)
    return w, t


def planner_values(w, tau, x0):
    """Exact static-lattice-optimum per-step ensemble welfare, and its parts."""
    Wbar = w.sum(0)
    sstar = (w * tau).sum(0) / Wbar
    V = (w * (tau - sstar[None, :]) ** 2).sum(0)
    tot = 0.0
    for m in range(N_RES):
        f = x0[m] % 1.0
        cand = np.array(sorted(set([f + k for k in range(10)] +
                                   [float(k) for k in range(11)])))
        tot += Wbar[m] * np.min((cand - sstar[m]) ** 2) + V[m]
    return -tot, -float(V.sum())


if __name__ == '__main__':
    CS = condition_seeds()
    spec = {(-0.8, 0.6): -0.767, (0.0, 0.6): -1.014, (0.8, 1.0): -1.158}
    out = []
    print("X9  V*_p on the EXACT seed-42 condition draws (30 reps, eps=0.0)")
    print(f"{'alpha':>6}{'sigma':>7} | {'W*/n exact':>11}{'(-V)/n':>10}"
          f"{'W* ens':>10} | {'spec':>8}{'ratio':>8}")
    for (a, s), v in spec.items():
        rng = np.random.default_rng(CS[(a, s, 0.0)])
        rng2 = np.random.default_rng(CS[(a, s, 0.0)] ^ 0xABCDEF)  # x0 for the lattice offset
        ens, vonly = [], []
        for r in range(N_REPS):
            w, t = sample_agent_params(rng, a, s)
            x0 = rng2.uniform(0, CAP, N_RES)
            e, vv = planner_values(w, t, x0)
            ens.append(e); vonly.append(vv)
        ens = float(np.mean(ens)); vonly = float(np.mean(vonly))
        out.append({'alpha': a, 'sigma': s, 'W_star_ensemble': ens,
                    'W_star_per_agent': ens / N_AG,
                    'minus_V_per_agent': vonly / N_AG, 'spec': v,
                    'ratio_spec_over_peragent': v / (ens / N_AG)})
        print(f"{a:6.1f}{s:7.1f} | {ens/N_AG:11.3f}{vonly/N_AG:10.3f}{ens:10.3f} | "
              f"{v:8.3f}{v/(ens/N_AG):8.3f}")
    json.dump(out, open('CROSSCHECK_SPEC_SEEDS.json', 'w'), indent=1)
