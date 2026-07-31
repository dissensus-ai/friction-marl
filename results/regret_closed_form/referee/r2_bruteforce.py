"""R2. REFEREE CHECK: brute-force forward simulation of the FULL 3-D environment,
written from scratch (imports neither core.py nor a00_lib.py, and shares no code
with independent_crosscheck.py).

Recomputes cells of G_grid_naive_{ce,bayes}_cesaro.csv by literally running the
environment: draw preferences, reset x0 ~ U[0,10]^3, give each agent a noisy
observation, let it pick its delta by exact best response with the others treated
as static, step s' = clip(s + sum delta, 0, 10), pay W(s'), time-average.

Benchmark = the static ensemble-optimal point of the reachable lattice L(f_m)
(this is what run_grid.py::cell_metrics uses for the Cesaro column; the planner
reaches it in <= 3 of T steps and stays, so its Cesaro value IS the static
optimum).

Tie rules implemented BOTH ways so the T1 convention can be priced:
  T1  = numericist/analyst rule: min clamped loss, then min UNCLAMPED sq error,
        then min |d|, then min d
  T0  = naive rule: min clamped loss, then min |d|, then min d
"""
import numpy as np
import pandas as pd

CAP = 10.0
N_AG = 4
N_RES = 3
DELTAS = np.array([-1.0, 0.0, 1.0])


# ---------------------------------------------------------------- DGPs (from source)
def tau_implemented(rng, D, alpha):
    b = rng.standard_normal((D, 1, N_RES))
    xi = rng.standard_normal((D, N_AG, N_RES))
    return alpha * b + (1.0 - abs(alpha)) * xi


def tau_renormalised(rng, D, alpha):
    return tau_implemented(rng, D, alpha) / np.sqrt(alpha ** 2 + (1 - abs(alpha)) ** 2)


def tau_equicorr(rng, D, rho):
    S = (1 - rho) * np.eye(N_AG) + rho * np.ones((N_AG, N_AG))
    ev, evec = np.linalg.eigh(S)
    L = (evec * np.sqrt(np.clip(ev, 0, None))) @ evec.T
    z = rng.standard_normal((D, N_AG, N_RES))
    return np.einsum("ij,djm->dim", L, z)


def weights(rng, D, sigma):
    return np.clip(rng.normal(sigma, 0.05, size=(D, N_AG, N_RES)), 0.0, 1.0)


# ---------------------------------------------------------------- agent decision
def bayes_loss(mu, t, eps):
    """E[(clip(Z,0,CAP)-t)^2], Z ~ N(mu, eps).  Independent derivation:
       = E[Z^2 1_{0<Z<C}] - 2t E[Z 1_{0<Z<C}] + t^2 P(0<Z<C)
         + t^2 P(Z<=0) + (C-t)^2 P(Z>=C).
    """
    if eps <= 0:
        return (np.clip(mu, 0.0, CAP) - t) ** 2
    from scipy.stats import norm
    sd = np.sqrt(eps)
    a = (0.0 - mu) / sd
    b = (CAP - mu) / sd
    Fa, Fb = norm.cdf(a), norm.cdf(b)
    fa, fb = norm.pdf(a), norm.pdf(b)
    P = Fb - Fa
    # E[Z 1] and E[Z^2 1] for the truncated normal
    EZ = mu * P + sd * (fa - fb)
    EZ2 = (mu ** 2 + eps) * P + sd * ((mu + 0.0) * fa - (mu + CAP) * fb)
    return EZ2 - 2 * t * EZ + t ** 2 * P + t ** 2 * Fa + (CAP - t) ** 2 * (1 - Fb)


def choose_delta(shat, tau, eps, bayes, tie="T1"):
    """shat, tau: (..., ) broadcastable. returns delta in {-1,0,1} same shape."""
    mu = shat[..., None] + DELTAS
    t = tau[..., None]
    loss = bayes_loss(mu, t, eps) if bayes else (np.clip(mu, 0, CAP) - t) ** 2
    unc = (mu - t) ** 2
    tol = 1e-12
    m = loss.min(axis=-1, keepdims=True)
    ok = loss <= m + tol
    if tie == "T1":
        sec = np.where(ok, unc, np.inf)
        m2 = sec.min(axis=-1, keepdims=True)
        ok = ok & (sec <= m2 + tol)
    # remaining tie-break: min |d| then min d -> candidate order 0(d=0),-1,+1
    order = np.array([1, 0, 2])
    okd = ok[..., order]
    j = okd.argmax(axis=-1)
    return DELTAS[order[j]]


# ---------------------------------------------------------------- lattice / benchmark
def lattice_grid(f):
    """(D, 21) for f shape (D,)"""
    D = f.shape[0]
    L = np.empty((D, 21))
    for k in range(11):
        L[:, 2 * k] = k
    for k in range(10):
        L[:, 2 * k + 1] = f + k
    return L


def run_cell(fam, ctrl, sigma, eps, bayes=False, D=2000, T=3000, burn=500,
             seed=777, tie="T1"):
    rng = np.random.default_rng(seed)
    if fam == "implemented":
        tau = tau_implemented(rng, D, ctrl)
    elif fam == "renormalised":
        tau = tau_renormalised(rng, D, ctrl)
    elif fam == "equicorr":
        tau = tau_equicorr(rng, D, ctrl)
    else:
        raise ValueError(fam)
    w = weights(rng, D, sigma)                                  # (D,A,M)
    Wbar = w.sum(axis=1)                                        # (D,M)
    sstar = (w * tau).sum(axis=1) / Wbar                        # (D,M)

    k0 = rng.integers(0, 10, size=(D, N_RES)).astype(float)
    f = rng.uniform(1e-9, 1 - 1e-9, size=(D, N_RES))
    x0 = k0 + f

    # static planner benchmark on the reachable lattice, per resource
    Wstar = np.zeros(D)
    for m in range(N_RES):
        L = lattice_grid(f[:, m])                               # (D,21)
        d2 = (L - sstar[:, m][:, None]) ** 2
        Wstar += -(Wbar[:, m] * d2.min(axis=1))
    # (V_m dropped from BOTH sides: we subtract it from the realised side too)
    Vsum = (w * (tau - sstar[:, None, :]) ** 2).sum(axis=(1, 2))

    # forward simulation
    s = x0.copy()
    acc = np.zeros(D)
    nacc = 0
    sd = np.sqrt(eps)
    for t in range(T):
        if eps > 0:
            shat = s[:, None, :] + sd * rng.standard_normal((D, N_AG, N_RES))
        else:
            shat = np.broadcast_to(s[:, None, :], (D, N_AG, N_RES))
        dl = choose_delta(shat, tau, eps, bayes, tie)            # (D,A,M)
        s = np.clip(s + dl.sum(axis=1), 0.0, CAP)
        if t >= burn:
            # realised welfare with V removed (cancels in G anyway)
            acc += -(Wbar * (s - sstar) ** 2).sum(axis=1)
            nacc += 1
    Wreal = acc / nacc
    G = Wstar - Wreal
    return dict(family=fam, control=ctrl, sigma=sigma, eps=eps,
                belief="bayes" if bayes else "ce", tie=tie, D=D, T=T,
                G_mean=float(G.mean()), G_se=float(G.std(ddof=1) / np.sqrt(D)),
                G_min=float(G.min()), frac_neg=float((G < 0).mean()),
                Vsum_mean=float(Vsum.mean()))


CELLS = [
    ("implemented", 0.0, 0.6, 0.0),
    ("implemented", 0.0, 0.6, 0.25),
    ("implemented", 0.0, 0.6, 1.0),
    ("implemented", 0.8, 0.6, 0.0),
    ("implemented", -0.8, 0.6, 0.25),
    ("renormalised", 0.8, 0.6, 0.25),
    ("equicorr", -0.33, 0.6, 0.0),
    ("equicorr", 0.8, 0.6, 1.0),
    ("equicorr", 0.4, 0.6, 0.5),
    ("implemented", 0.0, 0.2, 0.25),
    ("implemented", 0.0, 1.0, 0.25),
]

if __name__ == "__main__":
    import sys
    ref = pd.read_csv("../G_grid_naive_ce_cesaro.csv")
    rows = []
    for fam, ctrl, sg, ep in CELLS:
        r = run_cell(fam, ctrl, sg, ep, bayes=False, D=1500, T=2200, burn=400)
        sel = ref[(ref.family == fam) & (np.isclose(ref.control, ctrl))
                  & (np.isclose(ref.sigma, sg)) & (np.isclose(ref.eps, ep))]
        r["core_G"] = float(sel.G_mean.iloc[0])
        r["core_se"] = float(sel.G_se.iloc[0])
        r["ratio"] = r["G_mean"] / r["core_G"]
        rows.append(r)
        print(f"{fam:13s} c={ctrl:+.2f} sig={sg:.1f} eps={ep:.2f} | "
              f"referee {r['G_mean']:8.4f}+-{r['G_se']:.4f}  "
              f"core {r['core_G']:8.4f}+-{r['core_se']:.4f}  ratio {r['ratio']:.4f}",
              flush=True)
    pd.DataFrame(rows).to_csv("R2_bruteforce.csv", index=False)
    print("\n--- Bayes arm ---")
    rows2 = []
    refb = pd.read_csv("../G_grid_naive_bayes_cesaro.csv")
    for fam, ctrl, sg, ep in [("implemented", 0.0, 0.6, 0.25),
                              ("implemented", 0.0, 0.6, 1.0),
                              ("equicorr", -0.33, 0.6, 1.0)]:
        r = run_cell(fam, ctrl, sg, ep, bayes=True, D=1500, T=2200, burn=400)
        sel = refb[(refb.family == fam) & (np.isclose(refb.control, ctrl))
                   & (np.isclose(refb.sigma, sg)) & (np.isclose(refb.eps, ep))]
        r["core_G"] = float(sel.G_mean.iloc[0])
        r["ratio"] = r["G_mean"] / r["core_G"]
        rows2.append(r)
        print(f"{fam:13s} c={ctrl:+.2f} eps={ep:.2f} | referee {r['G_mean']:8.4f} "
              f" core {r['core_G']:8.4f}  ratio {r['ratio']:.4f}", flush=True)
    pd.DataFrame(rows2).to_csv("R2_bruteforce_bayes.csv", index=False)
