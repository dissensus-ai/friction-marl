"""R7. Which delegation concept do the corpus's OWN trained agents resemble?

This is the pivot of both reports' alignment conclusions:
  analyst  (DERIVATION sec 9.4): "IQL is the uncoordinated family -- the one whose
      alignment slope is positive [in G]", i.e. trained agents behave like
      naive-myopic, so the naive sign (+) is the one that matters.
  numericist (sec 3.2): same claim, structurally argued.

Both quote the SPEC's three-row table. The real per-replication results are on
disk (results/gpu_factorial_postfix/replication_results.csv, 3750 rows), so the
claim can be tested on the full alpha grid instead of three rows.

Test: compare the ALPHA PROFILE of realised ensemble welfare. No benchmark
needed -- if trained welfare improves with |alpha| it behaves like stage-Nash;
if it worsens with |alpha| it behaves like naive-myopic.
"""
import numpy as np
import pandas as pd
from r2_bruteforce import (tau_implemented, weights, lattice_grid, choose_delta,
                           CAP, N_AG, N_RES)
from r5_nash_eps0 import ne_mask, br_dynamic, PSUM

T_EP = 100


def realised_welfare(alpha, sigma, eps, rule, D=1200, seed=808, bayes=False):
    """Exact-BR ensemble welfare, T=100 episode average from x0 ~ U[0,10]^3,
    INCLUDING the irreducible V term so it is comparable to eval_reward_clean."""
    rng = np.random.default_rng(seed)
    tau = tau_implemented(rng, D, alpha)
    w = weights(rng, D, sigma)
    s = rng.uniform(0, CAP, (D, N_RES))
    acc = np.zeros(D)
    if rule == "naive":
        sd = np.sqrt(eps)
        for t in range(T_EP):
            shat = (s[:, None, :] + sd * rng.standard_normal((D, N_AG, N_RES))
                    if eps > 0 else np.broadcast_to(s[:, None, :], (D, N_AG, N_RES)))
            dl = choose_delta(shat, tau, eps, bayes, "T1")
            s = np.clip(s + dl.sum(axis=1), 0, CAP)
            acc += -(w * (s[:, None, :] - tau) ** 2).sum(axis=(1, 2))
    elif rule in ("nash_br", "nash_best"):
        assert eps == 0
        for t in range(T_EP):
            Dagg = np.zeros((D, N_RES))
            for m in range(N_RES):
                x = s[:, m][:, None]                       # (D,1) single state
                if rule == "nash_br":
                    Dagg[:, m] = br_dynamic(x, tau[:, :, m])[:, 0]
                else:
                    ok, post = ne_mask(x, tau[:, :, m])
                    Wb = w[:, :, m].sum(axis=1)
                    mu = (w[:, :, m] * tau[:, :, m]).sum(axis=1) / Wb
                    dev = np.abs(post - mu[:, None, None])
                    j = np.where(ok, dev, np.inf).argmin(axis=-1)
                    Dagg[:, m] = PSUM[j][:, 0].astype(float)
            s = np.clip(s + Dagg, 0, CAP)
            acc += -(w * (s[:, None, :] - tau) ** 2).sum(axis=(1, 2))
    return acc / T_EP


if __name__ == "__main__":
    d = pd.read_csv("/home/purrpower/Resurrexi/projects/papers/github-repos/"
                    "friction-marl/results/gpu_factorial_postfix/"
                    "replication_results.csv")
    print("Realised ENSEMBLE welfare (= 4 x per-agent), sigma=0.6, eps=0, T=100 avg")
    print(f"{'alpha':>6} {'IQL x4':>10} {'(se)':>7} {'naive-CE':>10} {'naive-Bayes':>12} "
          f"{'Nash BR-zero':>13} {'Nash best':>10}")
    rows = []
    for a in [-0.8, -0.4, 0.0, 0.4, 0.8]:
        sub = d[(np.isclose(d.alpha, a)) & (np.isclose(d.sigma, 0.6))
                & (np.isclose(d.epsilon, 0.0))]
        iql = 4 * sub.eval_reward_clean.mean()
        se = 4 * sub.eval_reward_clean.std() / np.sqrt(len(sub))
        wn = realised_welfare(a, 0.6, 0.0, "naive").mean()
        wb = realised_welfare(a, 0.6, 0.0, "naive", bayes=True).mean()
        wq = realised_welfare(a, 0.6, 0.0, "nash_br", D=600).mean()
        wk = realised_welfare(a, 0.6, 0.0, "nash_best", D=600).mean()
        rows.append(dict(alpha=a, iql=iql, se=se, naive_ce=wn, naive_bayes=wb,
                         nash_br=wq, nash_best=wk))
        print(f"{a:6.1f} {iql:10.3f} {se:7.3f} {wn:10.3f} {wb:12.3f} {wq:13.3f} "
              f"{wk:10.3f}", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv("R7_iql_profile.csv", index=False)
    print("\nALPHA PROFILE (value at |alpha|=0.8 minus value at alpha=0):")
    for c in ["iql", "naive_ce", "naive_bayes", "nash_br", "nash_best"]:
        v = r.set_index("alpha")[c]
        d08 = 0.5 * (v[-0.8] + v[0.8]) - v[0.0]
        print(f"  {c:12s}  W(|a|=0.8) - W(0) = {d08:+8.3f}   "
              f"{'IMPROVES with alignment' if d08 > 0 else 'WORSENS with alignment'}")
