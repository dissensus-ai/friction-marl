"""Is E[G] CONTINUOUS at eps = 0?  And what is the correct lambda-law fit?

The naive-myopic protocol is computed EXACTLY at eps = 0 (deterministic chain,
no fixed-point iteration), so its eps -> 0+ behaviour is decisive.
"""
import numpy as np
from a00_lib import CAP, N_AGENTS, gap_batch

def draw(rng, R, lam, gamma, sigma=0.6, n=N_AGENTS):
    z = rng.standard_normal((R, n))
    e = lam * (z - z.mean(1, keepdims=True)) / np.sqrt((n - 1) / n)
    t = gamma * rng.standard_normal((R, 1)) + e
    w = np.clip(sigma + 0.05 * rng.standard_normal((R, n)), 0, 1)
    return t, w, rng.uniform(0, CAP, R)

R = 20000
print("=" * 74)
print("E1  eps -> 0+ for the EXACTLY computable naive protocol (lam=0.87 gamma=0.5)")
rng = np.random.default_rng(101)
t, w, x0 = draw(rng, R, 0.87, 0.5)
prev = None
for eps in (0.0, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0):
    g = gap_batch(t, w, x0, eps, protocol="naive")
    G = g["G"].mean() * 3
    print(f"    eps={eps:9.1e}  sd={np.sqrt(eps):8.2e}  E[G]={G:8.5f}"
          f"   Var-term={(g['Wbar']*g['var']).mean()*3:7.4f}"
          f"   disp-term={(g['Wbar']*g['bdisp']**2).mean()*3:7.4f}")
    if eps == 0.0:
        prev = G
print(f"    JUMP at eps -> 0+ : E[G](0) = {prev:.5f} vs E[G](1e-8) = "
      f"{gap_batch(t,w,x0,1e-8,protocol='naive')['G'].mean()*3:.5f}")

print("=" * 74)
print("E2  the lambda law, refit (SNE at eps=1e-4; lam=0 is a knife-edge, excluded)")
lams = [0.1, 0.2, 0.3, 0.45, 0.6, 0.87, 1.2, 1.6, 2.2, 3.0]
ys = []
for lam in lams:
    r = np.random.default_rng(202)
    t, w, x0 = draw(r, R, lam, 0.5)
    ys.append(gap_batch(t, w, x0, 1e-4, protocol="sne")["G"].mean() * 3)
    print(f"    lam={lam:4.2f}  E[G]={ys[-1]:8.5f}")
L = np.array(lams); Y = np.array(ys)
m = L >= 0.45
A = np.column_stack([np.ones(m.sum()), L[m] ** 2])
b, *_ = np.linalg.lstsq(A, Y[m], rcond=None)
pred = A @ b
print(f"    fit E[G] = a + b*lam^2 on lam>=0.45 :  a={b[0]:.4f}  b={b[1]:.4f}"
      f"   R^2={1-((Y[m]-pred).var()/Y[m].var()):.5f}")
print(f"    per-resource contention coefficient kappa_c = b/(3*E[Wbar])"
      f" = {b[1]/(3*4*0.6):.4f}")
print(f"    theory bracket for kappa_c: [Var(med-mean), E[(tau_(2)-mean)^2]]/(3/4)"
      f" = [{0.0482/0.75:.4f}, {0.1986/0.75:.4f}]")
p = np.polyfit(np.log(L[m]), np.log(Y[m] - b[0]), 1)
print(f"    log-log exponent of the offset-removed excess: {p[0]:.3f}  (theory 2)")

print("=" * 74)
print("E3  the same law under the exact NASH correspondence at eps = 0 (welfare-best)")
for lam in (0.3, 0.6, 0.87, 1.2, 1.6, 2.2):
    r = np.random.default_rng(202)
    t, w, x0 = draw(r, R, lam, 0.5)
    g = gap_batch(t, w, x0, 0.0, protocol="nash", selection="welfare_best")
    print(f"    lam={lam:4.2f}  E[G]={g['G'].mean()*3:8.5f}")
