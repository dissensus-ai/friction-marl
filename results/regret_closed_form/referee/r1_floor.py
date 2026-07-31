"""R1. REFEREE CHECK: the quantization-floor constant dispute.

Numericist (NUMERIC_REPORT.md sec 1.2):  E[d^2] ~= Var(mu)/2 + 1/24
Analyst    (DERIVATION.md sec 4)      :  E[d^2] ~= Var(mu)/2 + 1/48   (large v)
                                          E[d^2] ~= v - (4/3)sqrt(2/pi) v^{3/2} (small v)

d = distance from mu to the nearest point of L(f) = {0..10} u {f,...,f+9},
f ~ U(0,1), mu ~ N(0, v)  (mu = s* is symmetric about 0 under every DGP here).

Written from scratch. Imports nothing from core.py or a00_lib.py.
"""
import numpy as np

RNG = np.random.default_rng(11_07_2026)


def lattice(f, cap=10.0):
    ints = np.arange(0.0, cap + 0.5, 1.0)
    sh = f + np.arange(0.0, 10.0)
    return np.sort(np.concatenate([ints, sh]))


def exact_d2(mu, f, cap=10.0):
    """min_{x in L(f)} (x-mu)^2, vectorised over (mu, f) pairs."""
    out = np.empty(mu.shape)
    # candidates: 0, cap, floor/ceil of mu on the integer grid, and on the shifted grid
    cand = []
    cand.append(np.zeros_like(mu))
    cand.append(np.full_like(mu, cap))
    ki = np.clip(np.floor(mu), 0, cap)
    cand.append(ki)
    cand.append(np.clip(ki + 1, 0, cap))
    ks = np.clip(np.floor(mu - f), 0, 9)
    cand.append(f + ks)
    cand.append(f + np.clip(ks + 1, 0, 9))
    C = np.stack(cand, axis=-1)
    out = ((C - mu[..., None]) ** 2).min(axis=-1)
    return out


def check(v, n=8_000_000):
    mu = RNG.normal(0.0, np.sqrt(v), size=n)
    f = RNG.uniform(1e-9, 1 - 1e-9, size=n)
    d2 = exact_d2(mu, f)
    ex = d2.mean()
    se = d2.std(ddof=1) / np.sqrt(n)
    num = v / 2 + 1 / 24
    ana = v / 2 + 1 / 48
    small = v - (4 / 3) * np.sqrt(2 / np.pi) * v ** 1.5
    # interior residual: E[d^2] - v/2  (should be ~1/48 when v is large enough)
    return dict(v=v, exact=ex, se=se,
                numericist=num, num_err_pct=100 * (num - ex) / ex,
                analyst=ana, ana_err_pct=100 * (ana - ex) / ex,
                small_v=small, small_err_pct=100 * (small - ex) / ex,
                residual=ex - v / 2)


if __name__ == "__main__":
    print(f"{'v':>8} {'exact':>10} {'v/2+1/24':>10} {'err%':>8} "
          f"{'v/2+1/48':>10} {'err%':>8} {'smallv':>10} {'err%':>9} {'resid':>9}")
    for v in [0.01, 0.0625, 0.25, 0.55, 0.625, 0.65, 1.0, 1.3333]:
        r = check(v)
        print(f"{r['v']:8.4f} {r['exact']:10.6f} {r['numericist']:10.6f} "
              f"{r['num_err_pct']:+8.2f} {r['analyst']:10.6f} {r['ana_err_pct']:+8.2f} "
              f"{r['small_v']:10.6f} {r['small_err_pct']:+9.2f} {r['residual']:9.6f}")
    print("\n1/24 =", 1 / 24, "  1/48 =", 1 / 48)
    # interior constant, restricted to mu strictly interior (mu in (2,8)) -- should be 1/24
    n = 8_000_000
    mu = RNG.uniform(2.0, 8.0, size=n)
    f = RNG.uniform(1e-9, 1 - 1e-9, size=n)
    d2 = exact_d2(mu, f)
    print(f"interior-only E[d^2] (mu~U(2,8)) = {d2.mean():.6f} "
          f"+/- {d2.std(ddof=1)/np.sqrt(n):.6f}   vs 1/24 = {1/24:.6f}")
