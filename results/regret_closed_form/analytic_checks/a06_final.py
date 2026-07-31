"""FINAL numbers for the derivation memo.

Everything is the exact 21-state chain per resource; the only stochastic element
is the Monte Carlo average over preference draws (SEs reported).  3 resources are
i.i.d. replicas, so E[G]_total = 3 * E[G]_per-resource.

Headline delegated protocol = SNE (stage-Nash-in-expectations), evaluated at
eps and at EPS0 = 1e-4 for the eps -> 0+ anchor.  Contrast protocol = naive.
The exact pure-Nash SELECTION BRACKET at eps = 0 is reported once in a05/D0.
"""
from __future__ import annotations

import json
import numpy as np

from a00_lib import (CAP, N_AGENTS, N_RES, gap_multi_sigma, gap_batch,
                     targets_equicorr, targets_implemented, targets_renormalised,
                     rho_of_alpha)

ALPHAS = [-0.8, -0.4, 0.0, 0.4, 0.8]
SIGMAS = [0.2, 0.4, 0.6, 0.8, 1.0]
EPSILONS = [0.0, 0.25, 0.5, 0.75, 1.0]
RHOS = [-1 / 3, -0.25, -0.1, 0.0, 0.1, 0.3077, 0.5, 0.7, 0.9412, 0.99]
R = 8000
EPS0 = 1e-4
SEED = 20260730
OUT = {}


def targets(dgp, knob, rng, R, n=N_AGENTS):
    if dgp == "implemented":
        return targets_implemented(rng.standard_normal(R),
                                   rng.standard_normal((R, n)), knob)
    if dgp == "renormalised":
        return targets_renormalised(rng.standard_normal(R),
                                    rng.standard_normal((R, n)), knob)
    if dgp == "equicorr":
        return targets_equicorr(rng.standard_normal((R, n)), knob)
    raise ValueError(dgp)


def cells(dgp, knob, eps, protocol="sne", R=R, seed=SEED, sigmas=SIGMAS):
    """dict sigma -> (mean, se) of the TOTAL gap over 3 resources."""
    rng = np.random.default_rng(seed)
    t = targets(dgp, knob, rng, R)
    x0 = rng.uniform(0.0, CAP, size=R)
    d = gap_multi_sigma(t, x0, sigmas, max(eps, EPS0), rng, protocol=protocol)
    return {s: (d[s]["G"].mean() * N_RES,
                d[s]["G"].std(ddof=1) / np.sqrt(R) * N_RES) for s in sigmas}


# =========================================================================== #
print("=" * 78)
print("S1.  sigma-HOMOGENEITY (implemented DGP, alpha=0, eps=0)")
for proto in ("sne", "naive"):
    c = cells("implemented", 0.0, 0.0, protocol=proto, R=30000)
    k = c[0.2][0] / 0.2
    print(f"  [{proto:5s}] " + "  ".join(
        f"s={s}: {c[s][0]:7.4f} (ratio {c[s][0]/(s*k):.4f})" for s in SIGMAS))
    OUT[f"homog_{proto}"] = {str(s): c[s] for s in SIGMAS}

# =========================================================================== #
print("\n" + "=" * 78)
print("S2.  ALPHA / RHO PROFILES at sigma = 0.6, eps = 0")
prof = {}
for proto in ("sne", "naive"):
    print(f"  --- {proto} ---")
    print("    implemented (knob alpha):")
    for a in ALPHAS:
        m, se = cells("implemented", a, 0.0, protocol=proto)[0.6]
        prof[f"{proto}|impl|{a}"] = (m, se)
        print(f"      alpha={a:+.1f}  rho={rho_of_alpha(a):.4f}  E[G]={m:8.4f} +- {se:.4f}")
    print("    renormalised (knob alpha, unit marginal variance):")
    for a in ALPHAS:
        m, se = cells("renormalised", a, 0.0, protocol=proto)[0.6]
        prof[f"{proto}|renorm|{a}"] = (m, se)
        print(f"      alpha={a:+.1f}  rho={rho_of_alpha(a):.4f}  E[G]={m:8.4f} +- {se:.4f}")
    print("    equicorr (knob = SIGNED pairwise correlation rho):")
    for r in RHOS:
        m, se = cells("equicorr", r, 0.0, protocol=proto)[0.6]
        prof[f"{proto}|equi|{r:.4f}"] = (m, se)
        print(f"      rho={r:+.4f}          E[G]={m:8.4f} +- {se:.4f}")
OUT["profiles"] = prof

# =========================================================================== #
print("\n" + "=" * 78)
print("S3.  EPS PROFILES at sigma = 0.6")
epsg = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
eprof = {}
for proto in ("sne", "naive"):
    for a in (0.0, 0.8):
        vs = []
        for e in epsg:
            m, se = cells("implemented", a, e, protocol=proto)[0.6]
            vs.append((e, m, se))
        eprof[f"{proto}|{a}"] = vs
        print(f"  [{proto:5s} alpha={a:+.1f}] " + " ".join(
            f"{e}:{m:.3f}" for e, m, _ in vs))
OUT["eps_profiles"] = eprof

# =========================================================================== #
print("\n" + "=" * 78)
print("S4.  FIRST-ORDER EXPANSION AT (alpha, eps) = (0, 0), sigma = 0.6")
RB = 60000
for proto in ("sne", "naive"):
    print(f"  --- {proto} ---")
    b0 = cells("implemented", 0.0, 0.0, protocol=proto, R=RB)[0.6]
    print(f"    E[G](0,0) = {b0[0]:.4f} +- {b0[1]:.4f}")
    for h in (0.02, 0.05):
        for dgp in ("implemented", "renormalised"):
            lo = cells(dgp, -h, 0.0, protocol=proto, R=RB)[0.6]
            hi = cells(dgp, +h, 0.0, protocol=proto, R=RB)[0.6]
            dr, dl = (hi[0] - b0[0]) / h, (b0[0] - lo[0]) / h
            sd = 3 * np.sqrt(b0[1] ** 2 + hi[1] ** 2) / h
            print(f"    h={h}  [{dgp:12s}] right dG/dalpha={dr:+8.4f}"
                  f"  left={dl:+8.4f}  (3SE~{sd:.3f})  "
                  f"{'KINK' if abs(dr-dl) > sd else 'smooth'}")
        lo = cells("equicorr", -h, 0.0, protocol=proto, R=RB)[0.6]
        hi = cells("equicorr", +h, 0.0, protocol=proto, R=RB)[0.6]
        print(f"    h={h}  [equicorr   ] dG/drho central ="
              f" {(hi[0]-lo[0])/(2*h):+8.4f}   (right {(hi[0]-b0[0])/h:+.4f},"
              f" left {(b0[0]-lo[0])/h:+.4f})")
    for e in (0.0025, 0.01, 0.04, 0.09):
        v = cells("implemented", 0.0, e, protocol=proto, R=RB)[0.6]
        print(f"    eps={e:6.4f}: E[G]={v[0]:7.4f}  dG/deps={(v[0]-b0[0])/e:+9.3f}"
              f"   dG/d(sqrt eps)={(v[0]-b0[0])/np.sqrt(e):+8.4f}")

# =========================================================================== #
print("\n" + "=" * 78)
print("S5.  FULL 5x5x5 GRID (implemented DGP) and the canonical-F head-to-head")
grid = {}
for proto in ("sne", "naive"):
    for a in ALPHAS:
        for e in EPSILONS:
            c = cells("implemented", a, e, protocol=proto)
            for s in SIGMAS:
                grid[f"{proto}|{a}|{s}|{e}"] = c[s]
OUT["grid"] = grid

for proto in ("sne", "naive"):
    y, F, cols = [], [], []
    for a in ALPHAS:
        for s in SIGMAS:
            for e in EPSILONS:
                y.append(grid[f"{proto}|{a}|{s}|{e}"][0])
                F.append(s * (1 + e) / (1 + a))
                cols.append((a, s, e))
    y, F = np.array(y), np.array(F)
    A = np.column_stack([np.ones_like(F), F])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    R2 = 1 - (y - A @ beta).var() / y.var()
    rho_s = np.corrcoef(np.argsort(np.argsort(y)), np.argsort(np.argsort(F)))[0, 1]
    print(f"  [{proto}] best affine a+b*F : a={beta[0]:+.4f} b={beta[1]:+.4f}"
          f"   R^2={R2:.4f}   Spearman={rho_s:+.4f}")
    for nm, i in (("alpha", 0), ("sigma", 1), ("eps", 2)):
        lv = sorted(set(c[i] for c in cols))
        gm = [np.mean([y[k] for k in range(len(y)) if cols[k][i] == v]) for v in lv]
        fm = [np.mean([F[k] for k in range(len(y)) if cols[k][i] == v]) for v in lv]
        print(f"     {nm:6s} levels {lv}")
        print(f"       E[G] {['%.3f' % v for v in gm]}")
        print(f"       F    {['%.3f' % v for v in fm]}")
    OUT[f"affine_{proto}"] = dict(a=float(beta[0]), b=float(beta[1]), R2=float(R2),
                                  spearman=float(rho_s))

# =========================================================================== #
print("\n" + "=" * 78)
print("S6.  ISO-F FAN TEST")
for proto in ("sne", "naive"):
    buckets = {}
    for a in ALPHAS:
        for s in SIGMAS:
            for e in EPSILONS:
                Fv = round(s * (1 + e) / (1 + a), 6)
                buckets.setdefault(Fv, []).append(grid[f"{proto}|{a}|{s}|{e}"][0])
    big = {k: np.array(v) for k, v in buckets.items() if len(v) >= 3}
    within = np.mean([v.std(ddof=1) for v in big.values()])
    tot = np.concatenate(list(big.values())).std(ddof=1)
    print(f"  [{proto}] {len(big)} iso-F levels with n>=3 ; mean within-F SD ="
          f" {within:.4f} vs pooled SD = {tot:.4f}   ratio = {within/tot:.3f}")
    for k in sorted(big):
        v = big[k]
        print(f"      F={k:6.3f} n={len(v)}  range [{v.min():7.4f}, {v.max():7.4f}]"
              f"  spread factor {v.max()/max(v.min(),1e-9):6.2f}x")
    OUT[f"isoF_{proto}"] = dict(within=float(within), total=float(tot))

# =========================================================================== #
print("\n" + "=" * 78)
print("S7.  PER-PRINCIPAL INCIDENCE  G_i  (sum_i G_i = G exactly)")
inc = {}
for proto in ("sne", "naive"):
    for a in (0.0, 0.8):
        for e in (0.0, 1.0):
            rng = np.random.default_rng(SEED)
            t = targets("implemented", a, rng, R)
            x0 = rng.uniform(0, CAP, R)
            d = gap_multi_sigma(t, x0, [0.6], max(e, EPS0), rng, protocol=proto,
                                want_agent=True)[0.6]
            Gi, G = d["Gi"], d["G"]
            chk = np.abs(Gi.sum(1) - G).max()
            neg = (Gi < -1e-9).mean()
            sh = np.sort(Gi, axis=1)
            # stakes-weighted incidence and concentration at fixed total
            wgt = d["w"]
            sw = (wgt * Gi).sum(1) / wgt.sum(1)
            pos = np.clip(Gi, 0, None)
            hhi = (pos ** 2).sum(1) / np.maximum(pos.sum(1), 1e-12) ** 2
            print(f"  [{proto:5s} a={a:+.1f} e={e:.1f}] sum_i G_i == G : {chk:.2e}"
                  f" | P(G_i<0) = {neg:.4f} | E[min_i G_i] = {sh[:,0].mean():+.4f}"
                  f" | E[max_i G_i] = {sh[:,-1].mean():+.4f}"
                  f" | E[HHI of borne loss] = {hhi.mean():.4f}"
                  f" | stakes-wtd = {sw.mean():.4f}")
            inc[f"{proto}|{a}|{e}"] = dict(neg=float(neg), hhi=float(hhi.mean()),
                                           mn=float(sh[:, 0].mean()),
                                           mx=float(sh[:, -1].mean()))
OUT["incidence"] = inc

with open("a06_results.json", "w") as f:
    json.dump(OUT, f, indent=1, default=float)
print("\nwrote a06_results.json")
