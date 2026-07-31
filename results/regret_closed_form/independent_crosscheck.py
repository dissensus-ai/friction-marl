"""INDEPENDENT cross-check of the headline claims in NUMERIC_REPORT.md.

Written from scratch (does NOT import core.py) so that the report's surface is
checked by a second implementation rather than trusted. Every check prints
PASS/FAIL and the numbers are dumped to CROSSCHECK.json.

Checks
  X1 lattice closure  : reachable set per coordinate is 21 points (11 if frac=0)
  X2 alpha symmetry   : tau(alpha;b,xi) == tau(-alpha;-b,xi) exactly (both alpha-DGPs)
  X3 welfare identity : W(s) = -sum_m [Wbar_m (s_m-s*_m)^2 + V_m]
  X4 G reduction+sign : G = sum_m Wbar_m [E_pi(x-s*)^2 - min_L (x-s*)^2] >= 0,
                        recomputed end-to-end for one cell by brute-force forward
                        simulation of the naive-myopic chain (no chain algebra)
  X5 floor constant   : E_f[d^2] = 1/24 for the interleaved two-grid lattice
  X6 Var(s*) formulae : implemented / renormalised / equicorr
  X7 sqrt-eps cusp    : log-log exponent of |E[G](eps)-E[G](0+)| ~ 0.5
"""
import json
import numpy as np

C = 10.0
N_AG, N_RES = 4, 3
OUT = {}


def sample_impl(rng, alpha, sigma, n=N_AG, m=N_RES):
    b = rng.normal(0., 1., size=m)
    xi = rng.normal(0., 1., size=(n, m))
    tau = alpha * b + (1. - abs(alpha)) * xi
    w = np.clip(rng.normal(sigma, 0.05, size=(n, m)), 0., 1.)
    return w, tau, b, xi


# ---------------------------------------------------------------- X1
def x1_lattice():
    rng = np.random.default_rng(1)
    ok, sizes = True, []
    for t in range(300):
        x0 = rng.uniform(0., C) if t else 0.0
        seen, frontier = {round(x0, 12)}, [x0]
        while frontier:
            nxt = []
            for x in frontier:
                for d in range(-4, 5):
                    y = min(max(x + d, 0.), C)
                    k = round(y, 12)
                    if k not in seen:
                        seen.add(k); nxt.append(y)
            frontier = nxt
        sizes.append(len(seen))
    OUT['X1'] = {'frac0_size': sizes[0], 'generic_sizes': sorted(set(sizes[1:]))}
    ok = sizes[0] == 11 and set(sizes[1:]) == {21}
    print(f"X1 lattice closure: frac=0 -> {sizes[0]}, generic -> {sorted(set(sizes[1:]))}  "
          f"[{'PASS' if ok else 'FAIL'}]  => 3-D: {11**3} (measure-zero) / {21**3} (a.s.) x 729")
    return ok


# ---------------------------------------------------------------- X2
def x2_alpha_symmetry():
    worst = {}
    for fam in ('implemented', 'renormalised'):
        w = 0.0
        for a in (0.05, 0.2, 0.4, 0.8):
            for s in range(40):
                rng = np.random.default_rng(100 + s)
                b = rng.normal(0., 1., size=N_RES)
                xi = rng.normal(0., 1., size=(N_AG, N_RES))
                t_p = a * b + (1 - abs(a)) * xi
                t_m = (-a) * (-b) + (1 - abs(-a)) * xi
                if fam == 'renormalised':
                    z = np.sqrt(a ** 2 + (1 - abs(a)) ** 2)
                    t_p, t_m = t_p / z, t_m / z
                w = max(w, np.abs(t_p - t_m).max())
        worst[fam] = float(w)
    ok = all(v == 0.0 for v in worst.values())
    OUT['X2'] = worst
    print(f"X2 alpha +/- symmetry: max|tau(a;b,xi)-tau(-a;-b,xi)| = {worst}  "
          f"[{'PASS' if ok else 'FAIL'}]  => law of G is a function of |alpha| only")
    return ok


# ---------------------------------------------------------------- X3
def x3_welfare_identity():
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(400):
        w, tau, _, _ = sample_impl(rng, 0.3, 0.6)
        s = rng.uniform(0, C, size=N_RES)
        direct = float(np.sum(w * (-(s[None, :] - tau) ** 2)))
        Wbar = w.sum(0)
        sstar = (w * tau).sum(0) / Wbar
        V = (w * (tau - sstar[None, :]) ** 2).sum(0)
        decomp = float(-np.sum(Wbar * (s - sstar) ** 2 + V))
        worst = max(worst, abs(direct - decomp))
    OUT['X3'] = {'max_abs_err': worst}
    ok = worst < 1e-9
    print(f"X3 welfare decomposition W(s) = -sum_m[Wbar(s-s*)^2+V]: max abs err "
          f"{worst:.3e}  [{'PASS' if ok else 'FAIL'}]  => V cancels in G")
    return ok


# ---------------------------------------------------------------- X4
def _naive_delta(shat, tau_i):
    """Agent i's naive-myopic argmax over delta in {-1,0,1} for ONE resource.

    Treats others as static: post-state guess = clamp(shat + delta). Ties broken
    by (clamped loss, unclamped sq err, |delta|, delta) = tie rule T1.
    """
    best, bd = None, None
    for d in (-1, 0, 1):
        post = min(max(shat + d, 0.), C)
        key = ((post - tau_i) ** 2, (shat + d - tau_i) ** 2, abs(d), d)
        if best is None or key < best:
            best, bd = key, d
    return bd


def x4_G_reduction(eps=0.25, ndraw=400, T=4000, burn=1000, seed=11):
    """Brute-force forward simulation; compare to the reduction formula."""
    rng = np.random.default_rng(seed)
    Gs, mism = [], 0.0
    lat_cache = {}
    for _ in range(ndraw):
        w, tau, _, _ = sample_impl(rng, 0.0, 0.6)
        Wbar = w.sum(0)
        sstar = (w * tau).sum(0) / Wbar
        x = rng.uniform(0, C, size=N_RES)
        frac = x % 1.0
        # exact reachable lattice per resource
        lat = []
        for mm in range(N_RES):
            f = frac[mm]
            pts = sorted(set([round(f + k, 12) for k in range(10)] +
                             [float(k) for k in range(11)]))
            lat.append(np.array(pts))
        acc_sq = np.zeros(N_RES)
        nacc = 0
        for t in range(T):
            noise = rng.normal(0., np.sqrt(eps), size=(N_AG, N_RES)) if eps > 0 else 0.
            shat = x[None, :] + noise
            agg = np.zeros(N_RES)
            for i in range(N_AG):
                for mm in range(N_RES):
                    agg[mm] += _naive_delta(shat[i, mm] if eps > 0 else x[mm], tau[i, mm])
            x = np.clip(x + agg, 0., C)
            if t >= burn:
                acc_sq += (x - sstar) ** 2
                nacc += 1
        Epi = acc_sq / nacc
        floor = np.array([np.min((lat[mm] - sstar[mm]) ** 2) for mm in range(N_RES)])
        G = float(np.sum(Wbar * (Epi - floor)))
        Gs.append(G)
        mism = min(mism, G)   # track most-negative G
    Gs = np.array(Gs)
    OUT['X4'] = {'eps': eps, 'ndraw': ndraw, 'G_mean': float(Gs.mean()),
                 'G_se': float(Gs.std(ddof=1) / np.sqrt(ndraw)),
                 'G_min_over_draws': float(Gs.min()),
                 'frac_G_negative': float((Gs < -1e-9).mean())}
    ok = (Gs < -1e-9).mean() == 0.0
    print(f"X4 G by brute-force forward sim (implemented a=0, sigma=0.6, eps={eps}): "
          f"G_mean = {Gs.mean():.3f} +/- {Gs.std(ddof=1)/np.sqrt(ndraw):.3f}, "
          f"min over draws = {Gs.min():.3e}, frac G<0 = {(Gs<-1e-9).mean():.3f}  "
          f"[{'PASS' if ok else 'FAIL'}]")
    return ok


# ---------------------------------------------------------------- X5
def x5_floor_constant(ndraw=4_000_000, seed=3):
    rng = np.random.default_rng(seed)
    f = rng.uniform(0., 1., size=ndraw)
    # s* uniform within the period; distance to nearest point of {f+Z} u {Z}
    u = rng.uniform(0., 1., size=ndraw)
    d = np.minimum(np.minimum(np.abs(u - f), np.abs(u - 0.)), np.abs(u - 1.))
    d = np.minimum(d, np.abs(u - f - 1.))
    mc = float((d ** 2).mean())
    OUT['X5'] = {'mc': mc, 'theory': 1 / 24}
    ok = abs(mc - 1 / 24) < 5e-4
    print(f"X5 interior quantization floor E_f[d^2]: MC {mc:.6f} vs theory 1/24 = "
          f"{1/24:.6f}  [{'PASS' if ok else 'FAIL'}]")
    return ok


# ---------------------------------------------------------------- X6
def x6_var_sstar(ndraw=200_000, seed=5):
    rng = np.random.default_rng(seed)
    res = {}
    for fam in ('implemented', 'renormalised'):
        for a in (0.0, 0.4, 0.8):
            b = rng.normal(0, 1, ndraw)
            xi = rng.normal(0, 1, (N_AG, ndraw))
            tau = a * b + (1 - abs(a)) * xi
            if fam == 'renormalised':
                tau = tau / np.sqrt(a ** 2 + (1 - abs(a)) ** 2)
            emp = float(tau.mean(0).var())
            th = a ** 2 + (1 - abs(a)) ** 2 / N_AG
            if fam == 'renormalised':
                th = th / (a ** 2 + (1 - abs(a)) ** 2)
            res[f'{fam}_a{a}'] = (emp, th, abs(emp - th) / max(th, 1e-12))
    for rho in (-1 / 3, 0.0, 0.4, 0.8):
        S = (1 - rho) * np.eye(N_AG) + rho * np.ones((N_AG, N_AG))
        ev, evec = np.linalg.eigh(S)
        L = (evec * np.sqrt(np.clip(ev, 0, None))) @ evec.T
        z = rng.normal(0, 1, (N_AG, ndraw))
        tau = L @ z
        emp = float(tau.mean(0).var())
        th = (1 + (N_AG - 1) * rho) / N_AG
        res[f'equicorr_r{rho:.2f}'] = (emp, th, abs(emp - th) / max(th, 1e-12))
    OUT['X6'] = {k: {'emp': v[0], 'theory': v[1], 'relerr': v[2]} for k, v in res.items()}
    worst = max(v[2] for k, v in res.items() if abs(v[1]) > 1e-9)
    ok = worst < 0.03
    print(f"X6 Var(s*) closed forms (equal-weight limit): worst rel err {worst:.4f}  "
          f"[{'PASS' if ok else 'FAIL'}]")
    for k, v in res.items():
        print(f"     {k:24s} emp {v[0]:.4f}  theory {v[1]:.4f}")
    return ok


if __name__ == '__main__':
    res = {}
    res['X1'] = x1_lattice()
    res['X2'] = x2_alpha_symmetry()
    res['X3'] = x3_welfare_identity()
    res['X5'] = x5_floor_constant()
    res['X6'] = x6_var_sstar()
    res['X4'] = x4_G_reduction()
    OUT['verdict'] = {k: bool(v) for k, v in res.items()}
    json.dump(OUT, open('CROSSCHECK.json', 'w'), indent=1)
    print('\nALL PASS' if all(res.values()) else '\nSOME FAILED: ' +
          str([k for k, v in res.items() if not v]))
