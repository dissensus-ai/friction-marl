"""A17.  The quantization/boundary floor: a CORRECTED closed form.

a14 F2 shows the commonly quoted  E[d^2] ~ Var(mu)/2 + 1/24  over-predicts by
+4% (Var(mu)=1) to +58% (Var(mu)=0.0625) and by infinity at Var(mu)=0.  The
error is a double count: the 1/24 is the *interior* constant, and the interior
branch only carries the mass mu > 0.  Since mu is symmetric about 0 and the
lattice lives on [0, C], the branch mu <= 0 is absorbed by the point 0 exactly.

  LARGE-v BRANCH (Var(mu) >~ 0.25):
      E[d^2]  =  Var(mu)/2  +  1/48  +  err          1/48 = 0.0208333

  SMALL-v BRANCH (Var(mu) <~ 0.1): mu | mu>0 sits inside the first unit cell,
  where the nearest lattice point is 0 or f, so with d = min(mu, |f-mu|),
      E[d^2 | mu] = mu^2 - (4/3) mu^3         (f ~ U(0,1), 0 < mu < 1/2)
      E[mu^3 1{mu>0}] = v^{3/2} * 2/sqrt(2 pi)
  =>  E[d^2]  =  v/2 + v/2 - (4/3)(2/sqrt(2 pi)) v^{3/2}
              =  v  -  1.063846 * v^{3/2}  +  O(v^2)
"""
import json
import numpy as np

C = 10.0
OUT = {}


def d_lat(mu, f):
    mu = np.atleast_1d(np.asarray(mu, float))
    f = np.atleast_1d(np.asarray(f, float)) * np.ones_like(mu)
    d1 = np.min(np.abs(mu[:, None] - np.arange(11.0)[None, :]), axis=1)
    d2 = np.min(np.abs(mu[:, None] - (f[:, None] + np.arange(10.0)[None, :])), axis=1)
    return np.minimum(d1, d2)


def mc(v, N=8_000_000, chunk=1_000_000, seed=17):
    rng = np.random.default_rng(seed)
    tot, cnt, pos = 0.0, 0, 0.0
    for i in range(0, N, chunk):
        mu = rng.normal(0, np.sqrt(v), chunk)
        f = rng.uniform(0, 1, chunk)
        d = d_lat(mu, f)
        tot += float((d ** 2).sum())
        pos += float((d[mu > 0] ** 2).sum())
        cnt += chunk
    return tot / cnt, pos / cnt


K_SMALL = (4 / 3) * (2 / np.sqrt(2 * np.pi))
print("=" * 92)
print(f"    small-v coefficient (4/3)*2/sqrt(2 pi) = {K_SMALL:.6f}")
print("=" * 92)
print("  Var(mu) |   exact    | v/2 + 1/48  err%  | v - 1.0638 v^1.5  err%  | v/2+1/24  err%"
      " | interior part")
rows = []
for v in (0.01, 0.04, 0.0625, 0.1, 0.25, 0.4375, 0.625, 0.8125, 0.955, 1.0, 1.3333):
    ex, interior = mc(v)
    big = v / 2 + 1 / 48
    small = v - K_SMALL * v ** 1.5
    old = v / 2 + 1 / 24
    print(f"   {v:7.4f} | {ex:9.6f}  | {big:9.6f} {100*(big-ex)/ex:+7.2f}% |"
          f" {small:9.6f} {100*(small-ex)/ex:+7.2f}% | {old:8.5f} {100*(old-ex)/ex:+8.2f}%"
          f" | {interior:.6f}")
    rows.append(dict(v=v, exact=ex, interior=interior, big=big, small=small, old=old))
OUT["table"] = rows

print("\n  => use  E[d^2] = Var(mu)/2 + 1/48   for Var(mu) >= 0.25   (|err| <= 0.6%)")
print("     use  E[d^2] = Var(mu) - 1.0638 Var(mu)^{3/2}  for Var(mu) <= 0.1 (|err| <= 1%)")
print("     the interior residual E[d^2 1{mu>0}] is 0.0203-0.0212 = 1/48 +- 2% over")
print("     Var(mu) in [0.25, 1.33], which is what pins the constant.")

print("=" * 92)
print("THE FLOOR PER DGP KNOB.  Floor = m * E[Wbar] * E[d^2],  E[Wbar] = 4 sigma_eff.")
print("  Var(mu) = v_tau - (3/4) lambda^2.   sigma = 0.6 => E[Wbar] = 2.4, m = 3.")


def floor_of(v_tau, lam):
    v = v_tau - 0.75 * lam ** 2
    e = v / 2 + 1 / 48 if v >= 0.15 else v - K_SMALL * v ** 1.5
    return 3 * 2.4 * e, v


print("   DGP            knob    | v_tau  lambda^2  Var(mu) |  E[d^2]  |  floor (welfare units)")
fl = []
for name, knob, vt, l2 in (
        ("implemented", -0.8, 0.8 ** 2 + 0.2 ** 2, 0.2 ** 2),
        ("implemented", 0.0, 1.0, 1.0),
        ("implemented", +0.8, 0.8 ** 2 + 0.2 ** 2, 0.2 ** 2),
        ("renormalised", 0.0, 1.0, 1.0),
        ("renormalised", 0.8, 1.0, 0.2 ** 2 / (0.8 ** 2 + 0.2 ** 2)),
        ("equicorr", -1 / 3, 1.0, 4 / 3),
        ("equicorr", 0.0, 1.0, 1.0),
        ("equicorr", 0.5, 1.0, 0.5),
        ("equicorr", 0.94, 1.0, 0.06)):
    f_, v = floor_of(vt, np.sqrt(l2))
    print(f"   {name:13s}{knob:+7.3f} | {vt:6.4f} {l2:8.4f} {v:8.4f} |"
          f" {f_/(3*2.4):8.5f} | {f_:8.4f}")
    fl.append(dict(dgp=name, knob=float(knob), v_tau=vt, lam2=l2, vmu=float(v),
                   floor=float(f_)))
OUT["floor_by_dgp"] = fl
print("   note equicorr rho = -1/3 : Var(mu) = (1+3 rho)/4 = 0 EXACTLY -- the four")
print("   targets sum to a constant, mu is deterministic and lands on the lattice")
print("   point 0, so the principal has NO quantization loss at the PSD boundary.")

with open("a17_results.json", "w") as fh:
    json.dump(OUT, fh, indent=1)
print("\nwrote a17_results.json")
