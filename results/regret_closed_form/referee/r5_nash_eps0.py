"""R5. REFEREE CHECK of the ONE arm nobody independently replicated: exact
pure-strategy stage-Nash at eps = 0.

The two builders disagree about how much equilibrium selection matters:
  numericist (NUMERIC_REPORT sec 3.1): "selection sensitivity -1.7% to +5.3%,
      median 0.5%; a second-order perturbation, not a driver of any result"
  analyst    (DERIVATION sec 6.4):    E[G] spans 0.265 (welfare-best) ->
      72.91 (minimal-move) -> 140.5 (welfare-worst): three orders of magnitude

Both cannot be the headline. This script enumerates all 3^4 = 81 profiles from
scratch, finds the exact pure-NE set at every reachable state, and evaluates
FOUR selection rules on the same draws, plus the numericist's Gauss-Seidel BR
dynamic from the all-zero profile.
"""
import numpy as np
import pandas as pd

CAP = 10.0
N_AG = 4
PROFILES = np.array(list(np.ndindex(3, 3, 3, 3))) - 1          # (81,4)
PSUM = PROFILES.sum(axis=1)                                     # (81,)


def lattice_grid(f):
    L = np.empty((f.shape[0], 21))
    for k in range(11):
        L[:, 2 * k] = k
    for k in range(10):
        L[:, 2 * k + 1] = f + k
    return L


def ne_mask(x, tau):
    """x: (D,21) states, tau: (D,4). -> ok (D,21,81) bool, post (D,21,81)."""
    D, S = x.shape
    post = np.clip(x[:, :, None] + PSUM[None, None, :], 0, CAP)     # (D,S,81)
    ok = np.ones((D, S, 81), dtype=bool)
    for i in range(N_AG):
        rest = PSUM - PROFILES[:, i]                                 # (81,)
        alts = np.stack([np.clip(x[:, :, None] + (rest + d)[None, None, :], 0, CAP)
                         for d in (-1, 0, 1)], axis=-1)              # (D,S,81,3)
        ti = tau[:, i][:, None, None, None]
        best = ((alts - ti) ** 2).min(axis=-1)                       # (D,S,81)
        ok &= ((post - tau[:, i][:, None, None]) ** 2 <= best + 1e-12)
    return ok, post


def br_dynamic(x, tau, max_sweeps=100):
    """Gauss-Seidel iterated best response from the all-zero profile, order 0..3.
    Returns the selected aggregate D (D,21)."""
    Dn, S = x.shape
    dl = np.zeros((Dn, S, N_AG))
    for _ in range(max_sweeps):
        changed = np.zeros((Dn, S), dtype=bool)
        for i in range(N_AG):
            rest = dl.sum(axis=-1) - dl[..., i]                      # (D,S)
            cand = np.stack([np.clip(x + rest + d, 0, CAP) for d in (-1., 0., 1.)],
                            axis=-1)                                  # (D,S,3)
            loss = (cand - tau[:, i][:, None, None]) ** 2
            # tie rule T1: min loss, then min unclamped sq err, then min |d|, then min d
            unc = np.stack([(x + rest + d - tau[:, i][:, None]) ** 2
                            for d in (-1., 0., 1.)], axis=-1)
            m = loss.min(axis=-1, keepdims=True)
            okm = loss <= m + 1e-12
            sec = np.where(okm, unc, np.inf)
            m2 = sec.min(axis=-1, keepdims=True)
            okm &= sec <= m2 + 1e-12
            order = np.array([1, 0, 2])
            j = okm[..., order].argmax(axis=-1)
            new = np.array([-1., 0., 1.])[order[j]]
            changed |= new != dl[..., i]
            dl[..., i] = new
        if not changed.any():
            break
    return dl.sum(axis=-1)


def select(ok, post, mu, rule):
    """-> chosen aggregate D per (draw, state), as an index shift."""
    Dn, S, _ = ok.shape
    dev = np.abs(post - mu[:, None, None])
    if rule == "welfare_best":
        score = np.where(ok, dev, np.inf)
        j = score.argmin(axis=-1)
    elif rule == "welfare_worst":
        score = np.where(ok, dev, -np.inf)
        j = score.argmax(axis=-1)
    elif rule == "minimal":
        score = np.where(ok, np.abs(PSUM)[None, None, :].astype(float), np.inf)
        j = score.argmin(axis=-1)
    elif rule == "maximal":
        score = np.where(ok, np.abs(PSUM)[None, None, :].astype(float), -np.inf)
        j = score.argmax(axis=-1)
    else:
        raise ValueError(rule)
    return PSUM[j].astype(float)


def cesaro_G(x, Dsel, mu, Wbar, x0idx, T=600, burn=200):
    """Deterministic map on the 21-point lattice; time-average (x-mu)^2."""
    Dn, S = x.shape
    # transition index
    nxt = np.clip(x + Dsel, 0, CAP)
    idx = np.abs(nxt[:, :, None] - x[:, None, :]).argmin(axis=-1)    # (D,S)
    cur = x0idx.copy()
    rows = np.arange(Dn)
    acc = np.zeros(Dn)
    n = 0
    for t in range(T):
        cur = idx[rows, cur]
        if t >= burn:
            acc += (x[rows, cur] - mu) ** 2
            n += 1
    track = acc / n
    floor = ((x - mu[:, None]) ** 2).min(axis=1)
    return Wbar * (track - floor)


def run(fam, ctrl, sigma=0.6, D=4000, seed=99):
    rng = np.random.default_rng(seed)
    tot = {r: 0.0 for r in ["welfare_best", "welfare_worst", "minimal", "maximal",
                            "br_zero"]}
    nne = []
    for m in range(3):                                    # 3 independent resources
        if fam == "implemented":
            b = rng.standard_normal(D)
            xi = rng.standard_normal((D, N_AG))
            tau = ctrl * b[:, None] + (1 - abs(ctrl)) * xi
        elif fam == "equicorr":
            S = (1 - ctrl) * np.eye(N_AG) + ctrl * np.ones((N_AG, N_AG))
            ev, U = np.linalg.eigh(S)
            L = (U * np.sqrt(np.clip(ev, 0, None))) @ U.T
            tau = rng.standard_normal((D, N_AG)) @ L.T
        w = np.clip(rng.normal(sigma, 0.05, (D, N_AG)), 0, 1)
        Wbar = w.sum(axis=1)
        mu = (w * tau).sum(axis=1) / Wbar
        f = rng.uniform(1e-9, 1 - 1e-9, D)
        k0 = rng.integers(0, 10, D)
        x = lattice_grid(f)
        x0idx = 2 * k0 + 1
        ok, post = ne_mask(x, tau)
        nne.append(ok.sum(axis=-1).ravel())
        for r in ["welfare_best", "welfare_worst", "minimal", "maximal"]:
            Dsel = select(ok, post, mu, r)
            tot[r] += cesaro_G(x, Dsel, mu, Wbar, x0idx).mean()
        Dbr = br_dynamic(x, tau)
        # is the BR fixed point actually a NE?
        tot["br_zero"] += cesaro_G(x, Dbr, mu, Wbar, x0idx).mean()
    cnt = np.concatenate(nne)
    return tot, dict(mean_n_pure_ne=float(cnt.mean()), frac_zero=float((cnt == 0).mean()),
                     frac_multi=float((cnt > 1).mean()), max_n=int(cnt.max()))


if __name__ == "__main__":
    rows = []
    for fam, ctrl in [("implemented", 0.0), ("implemented", 0.8),
                      ("equicorr", -0.33), ("equicorr", 0.8)]:
        tot, cen = run(fam, ctrl)
        r = dict(family=fam, control=ctrl, **tot, **cen)
        rows.append(r)
        print(f"{fam:12s} c={ctrl:+.2f} | best {tot['welfare_best']:8.4f}  "
              f"BR-zero {tot['br_zero']:8.4f}  minimal {tot['minimal']:8.3f}  "
              f"maximal {tot['maximal']:8.3f}  worst {tot['welfare_worst']:8.3f} "
              f"| nNE mean {cen['mean_n_pure_ne']:.2f} max {cen['max_n']} "
              f"multi {cen['frac_multi']:.3f} zero {cen['frac_zero']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv("R5_nash_eps0.csv", index=False)
