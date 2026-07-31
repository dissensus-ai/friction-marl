"""X7  validate ideation_measurement.md sec 0.1's planner magnitudes V*_p.
   X8  the stakes-weighted incidence object sum_i s_i G_i / sum_i s_i.

Independent of core.py. Seeded.
"""
import json
import numpy as np

C, N_AG, N_RES = 10.0, 4, 3
OUT = {}


def draw(rng, alpha, sigma, R):
    b = rng.standard_normal((R, N_RES))
    xi = rng.standard_normal((R, N_AG, N_RES))
    tau = alpha * b[:, None, :] + (1 - abs(alpha)) * xi
    w = np.clip(rng.normal(sigma, 0.05, (R, N_AG, N_RES)), 0, 1)
    return w, tau


def planner_static(w, tau, x0):
    """Per-step welfare of the static lattice optimum, and its parts."""
    Wbar = w.sum(1)                              # (R,M)
    sstar = (w * tau).sum(1) / Wbar              # (R,M)
    V = (w * (tau - sstar[:, None, :]) ** 2).sum(1)   # (R,M) irreducible
    R_, M = Wbar.shape
    f = x0 % 1.0
    # exact reachable lattice per (draw, resource): {f+0..9} u {0..10}
    cand = np.concatenate([f[..., None] + np.arange(10)[None, None, :],
                           np.tile(np.arange(11.0)[None, None, :], (R_, M, 1))], axis=2)
    d2 = (cand - sstar[..., None]) ** 2
    floor = d2.min(axis=2)                       # (R,M)
    Wstar = -(Wbar * floor + V).sum(1)           # (R,) ensemble per-step welfare
    return Wstar, Wbar, sstar, V, floor


def x7():
    spec = {(-0.8, 0.6): -0.767, (0.0, 0.6): -1.014, (0.8, 1.0): -1.158}
    rows = []
    print("X7  planner per-step welfare V*_p  vs ideation_measurement.md sec 0.1")
    print(f"{'alpha':>6}{'sigma':>7} | {'ensemble W*':>12}{'W*/n':>10} | "
          f"{'V_m only /n':>12} | {'spec':>8}")
    for (a, s), v in spec.items():
        rng = np.random.default_rng(20260730)
        R = 200000
        w, tau = draw(rng, a, s, R)
        x0 = rng.uniform(0, C, (R, N_RES))
        Wstar, Wbar, sstar, V, floor = planner_static(w, tau, x0)
        ens = float(Wstar.mean())
        rows.append({'alpha': a, 'sigma': s, 'W_star_ensemble': ens,
                     'W_star_per_agent': ens / N_AG,
                     'irreducible_V_per_agent': float(-V.sum(1).mean() / N_AG),
                     'quantization_floor_ensemble': float((Wbar * floor).sum(1).mean()),
                     'spec_value': v})
        print(f"{a:6.1f}{s:7.1f} | {ens:12.3f}{ens/N_AG:10.3f} | "
              f"{-V.sum(1).mean()/N_AG:12.3f} | {v:8.3f}")
    OUT['X7_planner'] = rows
    print("  NOTE: the spec's V*_p are NOT reproduced by the exact static-optimum")
    print("        planner under the DGP mean; see report sec 0.3.")


def _naive_delta_vec(shat, tau_i):
    """Vectorised naive-myopic argmax over delta in {-1,0,1}; tie rule T1."""
    best_d = np.zeros_like(shat)
    best_k1 = np.full(shat.shape, np.inf)
    best_k2 = np.full(shat.shape, np.inf)
    for d in (0, -1, 1):                       # |d| ascending => T1 tie order
        post = np.clip(shat + d, 0., C)
        k1 = (post - tau_i) ** 2
        k2 = (shat + d - tau_i) ** 2
        better = (k1 < best_k1 - 1e-15) | ((np.abs(k1 - best_k1) <= 1e-15) &
                                           (k2 < best_k2 - 1e-15))
        best_d = np.where(better, d, best_d)
        best_k1 = np.where(better, k1, best_k1)
        best_k2 = np.where(better, k2, best_k2)
    return best_d


def x8(alpha=0.0, sigma=0.6, eps=0.25, R=3000, T=2500, burn=800, seed=21):
    """Per-principal G_i and the stakes-weighted incidence sum."""
    rng = np.random.default_rng(seed)
    w, tau = draw(rng, alpha, sigma, R)
    x = rng.uniform(0, C, (R, N_RES))
    Wbar = w.sum(1)
    sstar = (w * tau).sum(1) / Wbar
    f = x % 1.0
    cand = np.concatenate([f[..., None] + np.arange(10)[None, None, :],
                           np.tile(np.arange(11.0)[None, None, :], (R, N_RES, 1))], axis=2)
    xlat = np.take_along_axis(
        cand, np.argmin((cand - sstar[..., None]) ** 2, axis=2)[..., None], 2).squeeze(2)
    acc = np.zeros((R, N_AG, N_RES))
    n = 0
    for t in range(T):
        shat = x[:, None, :] + (rng.normal(0, np.sqrt(eps), (R, N_AG, N_RES))
                                if eps > 0 else 0.)
        d = _naive_delta_vec(shat, tau)
        x = np.clip(x + d.sum(1), 0., C)
        if t >= burn:
            acc += (x[:, None, :] - tau) ** 2
            n += 1
    # per-principal realised loss and benchmark loss (both include agent i's own V_i)
    Li_real = (w * (acc / n)).sum(2)                       # (R,A)
    Li_bench = (w * (xlat[:, None, :] - tau) ** 2).sum(2)  # (R,A)
    Gi = Li_real - Li_bench                                # (R,A)  G_i = W*_i - W_i
    G = Gi.sum(1)
    s_i = w.sum(2)                                         # (R,A) stakes of principal i
    plain = Gi.mean(1)
    weighted = (s_i * Gi).sum(1) / s_i.sum(1)
    OUT['X8_incidence'] = {
        'alpha': alpha, 'sigma': sigma, 'eps': eps, 'R': R,
        'G_mean': float(G.mean()),
        'Gi_mean_over_principals': float(plain.mean()),
        'stakes_weighted_mean': float(weighted.mean()),
        'abs_diff': float(abs(weighted.mean() - plain.mean())),
        'rel_diff_pct': float(100 * (weighted.mean() - plain.mean()) / plain.mean()),
        'Gi_min_mean': float(Gi.min(1).mean()),
        'Gi_max_mean': float(Gi.max(1).mean()),
        'spread_ratio': float(Gi.max(1).mean() / Gi.min(1).mean()),
        'P_any_Gi_neg': float((Gi.min(1) < -1e-9).mean()),
        'corr_si_Gi': float(np.corrcoef(s_i.ravel(), Gi.ravel())[0, 1]),
        'sd_si_over_mean_si': float(s_i.std() / s_i.mean()),
        'Gi_sum_check': float(np.abs(Gi.sum(1) - G).max()),
    }
    d = OUT['X8_incidence']
    print(f"\nX8  stakes-weighted incidence (implemented a={alpha}, sigma={sigma}, eps={eps})")
    print(f"    G (ensemble)                 = {d['G_mean']:.4f}")
    print(f"    plain mean_i G_i             = {d['Gi_mean_over_principals']:.4f}")
    print(f"    stakes-weighted sum_i s_i G_i/sum s_i = {d['stakes_weighted_mean']:.4f}")
    print(f"    difference                   = {d['abs_diff']:.5f}  "
          f"({d['rel_diff_pct']:+.3f}%)")
    print(f"    corr(s_i, G_i) = {d['corr_si_Gi']:+.4f}   CV(s_i) = {d['sd_si_over_mean_si']:.4f}")
    print(f"    worst/best principal G_i     = {d['Gi_max_mean']:.3f} / "
          f"{d['Gi_min_mean']:.3f}  (ratio {d['spread_ratio']:.2f}x)")
    print(f"    P(some principal has G_i<0)  = {d['P_any_Gi_neg']:.3f}")


if __name__ == '__main__':
    x7()
    x8()
    x8(alpha=0.8, sigma=0.2, eps=0.0, seed=22)
    json.dump(OUT, open('CROSSCHECK_PLANNER_INCIDENCE.json', 'w'), indent=1)
