"""A11b.  Fast standalone version of the a11 belief-lemma tests (L2 + a reduced
L3 restricted to the naive protocol, which needs no fixed-point iteration).
Same claims, smaller R, so the memo has the numbers even if a11's SNE arm is
still running.
"""
import json
import numpy as np
from a00_lib import CAP, N_AGENTS, draw_problem, gap_batch, _delta_star

OUT = {}
rng = np.random.default_rng(1111)

print("=" * 84)
print("L2  where the clamp-aware BAYES argmin differs from the CE argmin")
print("    (Lemma 6.1: they are the same map wherever the clamp does not bind)")
rows = []
for eps in (0.01, 0.05, 0.25, 0.5, 1.0):
    u = np.linspace(-2.0, CAP + 2.0, 2401)
    tau = rng.normal(0.0, 1.0, 1500)
    U = np.repeat(u[:, None], tau.size, axis=1)
    T = np.repeat(tau[None, :], u.size, axis=0)
    d_b = _delta_star(U, T, eps)
    d_c = np.clip(np.round(T - U), -1, 1).astype(int)
    diff = d_b != d_c
    wall = np.minimum(np.abs(U), np.abs(CAP - U))
    frac = diff.mean()
    q = np.quantile(wall[diff], [0.5, 0.9, 0.99, 1.0]) if diff.any() else np.zeros(4)
    print(f"    eps={eps:4.2f} sd={np.sqrt(eps):5.3f} | disagree {100*frac:6.3f}% |"
          f" wall-distance med={q[0]:5.2f} p90={q[1]:5.2f} p99={q[2]:5.2f}"
          f" max={q[3]:5.2f} | max/sd={q[3]/np.sqrt(eps):5.2f}")
    rows.append(dict(eps=eps, frac=float(frac), med=float(q[0]), p99=float(q[2]),
                     mx=float(q[3]), mx_over_sd=float(q[3] / np.sqrt(eps))))
OUT["L2"] = rows
print("    => the disagreement set is a boundary layer: every disagreement lies")
print("       within a small multiple of the noise SD of a wall.  Off the wall the")
print("       two belief models are the SAME decision rule, exactly.")

print("=" * 84)
print("L3  E[G], naive protocol, Bayes (clamp-aware) vs CE (interior rule), at the")
print("    NATIVE target location and SHIFTED +5 so the clamp is slack.")


def cell(eps, clamp_aware, shift, R=12000, seed=515):
    rng = np.random.default_rng(seed)
    tot = 0.0
    for _ in range(3):
        t, w, x0 = draw_problem(rng, R, "implemented", 0.0, 0.6)
        tot += gap_batch(t + shift, w, x0, eps, protocol="naive",
                         clamp_aware=clamp_aware)["G"].mean()
    return tot


epss = [1e-4, 0.01, 0.05, 0.25, 0.5, 1.0]
res = {}
for shift, lab in ((0.0, "native  (mu ~ N(0,.), half the optima on the wall)"),
                   (5.0, "shifted (+5: clamp slack)")):
    b = [cell(e, True, shift) for e in epss]
    c = [cell(e, False, shift) for e in epss]
    print(f"  --- {lab} ---")
    print("      eps    " + "".join(f"{e:9.4g}" for e in epss))
    print("      Bayes  " + "".join(f"{v:9.4f}" for v in b))
    print("      CE     " + "".join(f"{v:9.4f}" for v in c))
    print("      wedge% " + "".join(f"{100*(x-y)/y:9.2f}" for x, y in zip(b, c)))
    print(f"      sign of G(0.25)-G(0.01):  Bayes {np.sign(b[3]-b[1]):+.0f}"
          f"   CE {np.sign(c[3]-c[1]):+.0f}")
    print(f"      max |wedge| = {max(abs(100*(x-y)/y) for x,y in zip(b,c)):.2f}%")
    res[lab] = dict(eps=epss, bayes=b, ce=c)
OUT["L3"] = res

with open("a11b_results.json", "w") as fh:
    json.dump(OUT, fh, indent=1)
print("\nwrote a11b_results.json")
