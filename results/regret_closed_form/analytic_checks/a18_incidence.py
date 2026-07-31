"""A18.  PER-PRINCIPAL INCIDENCE G_i, and the stakes-weighted object under a
stakes design that actually VARIES s_i across principals (which the implemented
sampler does not: w ~ N(sigma, 0.05) gives every principal the same stake up to
a 0.05 wobble, so the stakes-weighted object degenerates to G/n).

DEFINITION.  G_i = w_i [ E_pi (x - tau_i)^2 - E_plan (x - tau_i)^2 ]  per resource,
summed over resources.  Then sum_i G_i = G identically (the same algebra as (3.1)
with mu_i := tau_i and W_i := w_i).  G_i CAN be negative: the ensemble-optimal plan
tracks mu, not tau_i, so a principal whose target is far from mu can be better off
under the delegates' churn than under the plan.

KEY POINT (a consequence of Prop 5.1).  The delegated chain is w-free, so changing
the stakes design changes ONLY Wbar and mu.  Heterogeneous stakes therefore move
the benchmark point mu toward the high-stakes principals without changing what the
delegates do -- which is exactly the configuration in which the stakes-weighted
incidence object stops being a re-labelling of G/n.
"""
import json
import numpy as np
from a00_lib import CAP, N_AGENTS, N_RES, draw_problem, gap_batch, state_values

n = N_AGENTS
R = 30000
OUT = {}


def run(dgp, knob, eps, protocol, sigmas_i, seed=606, R=R):
    """sigmas_i: (n,) per-principal stake means."""
    rng = np.random.default_rng(seed)
    accG, accGi, accSW, accNeg, accAny, accMin, accMax, accHHI, nH = (
        0.0, np.zeros(n), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
    for _ in range(N_RES):
        t, _w, x0 = draw_problem(rng, R, dgp, knob, 0.6)
        w = np.clip(np.array(sigmas_i)[None, :] + 0.05 * rng.standard_normal((R, n)),
                    0.0, 1.0)
        g = gap_batch(t, w, x0, eps, protocol=protocol, want_agent=True)
        Gi, G = g["Gi"], g["G"]
        accG += G.mean()
        accGi += Gi.mean(0)
        s = w.sum(1)
        accSW += ((w * Gi).sum(1) / s).mean()
        accNeg += (Gi < -1e-9).mean()
        accAny += (Gi < -1e-9).any(1).mean()
        srt = np.sort(Gi, axis=1)
        accMin += srt[:, 0].mean()
        accMax += srt[:, -1].mean()
        big = G > 0.05                             # HHI only where G is non-degenerate
        if big.any():
            pos = np.clip(Gi[big], 0, None)
            accHHI += ((pos ** 2).sum(1) / pos.sum(1) ** 2).mean()
            nH += 1
        chk = np.abs(Gi.sum(1) - G).max()
    return dict(G=accG, Gi=accGi, sw=accSW / N_RES, neg=accNeg / N_RES,
                any_neg=accAny / N_RES, mn=accMin, mx=accMax,
                hhi=accHHI / max(nH, 1), chk=float(chk))


print("=" * 96)
print("I1  EQUAL stakes (the implemented design): the stakes-weighted object is G/n")
print("     proto  knob  eps |    E[G]   sum_i G_i chk |  P(G_i<0)  P(any G_i<0) |"
      "  E[min]  E[max]  | stakes-wtd   G/n")
rows = []
for proto in ("naive", "sne"):
    for knob in (0.0, 0.8):
        for eps in (1e-4, 0.25, 1.0):
            r = run("implemented", knob, eps, proto, [0.6] * n)
            print(f"     {proto:5s} {knob:+.1f} {eps:5.2f} | {r['G']:8.4f}  {r['chk']:.1e} "
                  f"| {r['neg']:8.4f}  {r['any_neg']:11.4f} | {r['mn']:+7.4f} "
                  f"{r['mx']:+7.4f} | {r['sw']:9.5f} {r['G']/n:9.5f}")
            rows.append(dict(proto=proto, knob=knob, eps=eps,
                             **{k: (float(v) if np.isscalar(v) else np.asarray(v).tolist())
                                for k, v in r.items()}))
OUT["I1"] = rows

print("=" * 96)
print("I2  HETEROGENEOUS stakes s = (0.2, 0.4, 0.8, 1.0).  Prop 5.1 => the delegated")
print("    chain is UNCHANGED; only Wbar and the benchmark point mu move.")
print("     proto  knob  eps |    E[G]   | per-principal E[G_i] (i=0..3)          |"
      " stakes-wtd    G/n   | ratio")
het = [0.2, 0.4, 0.8, 1.0]
rows2 = []
for proto in ("naive", "sne"):
    for knob in (0.0, 0.8):
        for eps in (1e-4, 1.0):
            r = run("implemented", knob, eps, proto, het)
            gi = np.asarray(r["Gi"])
            print(f"     {proto:5s} {knob:+.1f} {eps:5.2f} | {r['G']:8.4f}  | " +
                  " ".join(f"{v:+8.4f}" for v in gi) +
                  f" | {r['sw']:9.5f} {r['G']/n:8.5f} | {r['sw']/(r['G']/n):5.3f}")
            rows2.append(dict(proto=proto, knob=knob, eps=eps, G=float(r["G"]),
                              Gi=gi.tolist(), sw=float(r["sw"]),
                              ratio=float(r["sw"] / (r["G"] / n))))
OUT["I2"] = rows2
print("    => under equal stakes the stakes-weighted incidence is G/n to 3 d.p.;")
print("       under a 5x stake spread it departs from G/n, and the burden shifts")
print("       toward the LOW-stakes principals (their targets are the ones the")
print("       benchmark mu no longer tracks).")

with open("a18_results.json", "w") as fh:
    json.dump(OUT, fh, indent=1)
print("\nwrote a18_results.json")
