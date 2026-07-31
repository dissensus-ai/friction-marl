"""A16b.  The contention-branch refit of Psi~(lambda) = a0(eps) + kc(eps) lambda^2,
using the naive-protocol rows a16 W1 computed (a16's SNE arm was retired for cost;
see a16_out.txt).  Restricting the fit to lambda > lambda*(eps) is the point: below
lambda* the herding branch dominates and no quadratic law holds there.

The a16 W1 rows are transcribed here verbatim so this script is self-contained and
the fit is reproducible without re-running the 25 000-draw chains.
"""
import json
import numpy as np

LAMS = [0.05, 0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.87, 1.00, 1.1547,
        1.35, 1.60, 2.00, 2.50, 3.00]
PSI = {  # protocol = naive, sigma = 0.6; Psi~ = E[G] / (3 E[Wbar])
    0.0:  [0.604, 0.580, 0.566, 0.557, 0.511, 0.436, 0.366, 0.321, 0.287,
           0.261, 0.242, 0.241, 0.294, 0.434, 0.668],
    0.01: [0.609, 0.591, 0.537, 0.453, 0.343, 0.267, 0.221, 0.198, 0.183,
           0.173, 0.170, 0.182, 0.234, 0.369, 0.585],
    0.25: [0.305, 0.293, 0.256, 0.216, 0.171, 0.145, 0.131, 0.125, 0.123,
           0.123, 0.128, 0.139, 0.171, 0.249, 0.386],
    1.0:  [0.120, 0.118, 0.112, 0.106, 0.100, 0.097, 0.096, 0.097, 0.098,
           0.101, 0.106, 0.115, 0.135, 0.179, 0.253],
}
CAP = np.sqrt(4 / 3)          # equicorrelated PSD cap on lambda, n = 4

L = np.array(LAMS)
out = {}
print("=" * 96)
print("W2  contention-branch refit  Psi~(lambda) = a0(eps) + kappa_c(eps) lambda^2,")
print("    fitted on lambda > lambda*(eps) only.  Order-statistic bracket for kappa_c")
print("    (a07 K1): [Var(med-mean), E[(tau_(2)-mean)^2]] = [0.0482, 0.1986].")
print("   eps   lambda*  in design range? |    a0       kappa_c     R^2    log-log exp"
      "  n   kappa_c in bracket?")
for eps, y in PSI.items():
    Y = np.array(y)
    j = int(np.argmin(Y))
    m = L > L[j] + 1e-9
    A = np.column_stack([np.ones(m.sum()), L[m] ** 2])
    b, *_ = np.linalg.lstsq(A, Y[m], rcond=None)
    pred = A @ b
    r2 = 1 - ((Y[m] - pred) ** 2).sum() / ((Y[m] - Y[m].mean()) ** 2).sum()
    e = np.polyfit(np.log(L[m]), np.log(np.maximum(Y[m] - b[0], 1e-12)), 1)[0]
    inb = 0.0482 <= b[1] <= 0.1986
    print(f"  {eps:5g}  {L[j]:7.4f}  {'INSIDE ' if L[j] < CAP else 'above  ':>9s}       |"
          f" {b[0]:+7.4f}  {b[1]:+7.4f}  {r2:7.5f}   {e:6.3f}      {m.sum()}"
          f"   {'YES' if inb else 'no'}")
    out[str(eps)] = dict(lam_star=float(L[j]), a0=float(b[0]), kappa_c=float(b[1]),
                         r2=float(r2), loglog_exponent=float(e), npts=int(m.sum()),
                         in_bracket=bool(inb))
print()
print("  READINGS")
print("  * lambda* falls monotonically in eps (1.60 -> 1.35 -> 1.00 -> 0.75): noise")
print("    shrinks the herding branch, the same mechanism as kappa falling (eq. 6.3).")
print(f"  * at eps <= 0.01, lambda* > {CAP:.4f} = the equicorrelated PSD cap, so the whole")
print("    design sits on the DECREASING branch and dG/drho > 0 everywhere.")
print("  * kappa_c(naive, eps=0) = 0.0752 sits INSIDE the order-statistic bracket, near")
print("    its median-attractor end (0.0482) -- i.e. for naive delegation the contention")
print("    branch really is the median-vs-mean displacement of eq. (5.2).")
print("  * kappa_c also falls in eps (0.075 -> 0.019): the same de-synchronisation.")
print("  * the log-log exponent drifts 1.94 -> 1.65 as eps grows because the fit window")
print("    extends further down toward lambda*, where the quadratic law is not yet")
print("    asymptotic; at eps = 0 (window furthest from lambda*) it is 1.94 vs theory 2.")

with open("a16b_results.json", "w") as fh:
    json.dump(out, fh, indent=1)
print("\nwrote a16b_results.json")
