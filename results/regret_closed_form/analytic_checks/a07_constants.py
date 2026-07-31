"""Asymptotic constants for the closed form.

K1  The large-lambda contention constant: the noiseless stage-Nash attractor sits
    in the MEDIAN interval [tau_(2), tau_(3)] (n=4) while the planner tracks the
    weighted MEAN.  So E[bdisp^2] -> lambda^2 * Var(median - mean)/lambda^2.
    For n i.i.d. normals, Cov(median, mean) = Var(mean) (the mean is the
    efficient equivariant estimator), so
        Var(med - mean) = Var(med) - Var(mean) = (V_med(n) - 1/n) * lambda^2.
K2  The small-eps flip rate: for a threshold rule with threshold density p(0) at
    the switch point, E[flip] = sqrt(eps) * sqrt(2/pi) * p(0) + O(eps).
K3  sigma_eff(sigma) = E[clip(N(sigma,0.05^2),0,1)] -- the exact stakes scale.
"""
import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(3)
N = 4_000_000
print("=" * 74)
print("K1  order-statistic constants for n = 4 standard normals")
X = rng.standard_normal((N, 4))
Xs = np.sort(X, axis=1)
med = 0.5 * (Xs[:, 1] + Xs[:, 2])
mean = X.mean(axis=1)
Vmed = med.var()
print(f"    Var(mean)                    = {mean.var():.6f}   (exact 1/4 = 0.25)")
print(f"    Var(median = (X_(2)+X_(3))/2)= {Vmed:.6f}")
print(f"    Cov(median, mean)            = {np.cov(med, mean)[0,1]:.6f}"
      f"   (theory: = Var(mean) = 0.25)")
print(f"    Var(median - mean)           = {(med-mean).var():.6f}"
      f"   = V_med - 1/4 = {Vmed-0.25:.6f}")
print(f"    E[(tau_(2)-mean)^2]          = {((Xs[:,1]-mean)**2).mean():.6f}")
print(f"    E[(tau_(3)-mean)^2]          = {((Xs[:,2]-mean)**2).mean():.6f}")
print("    => the contention coefficient kappa_c in E[G] ~ Wbar * kappa_c * lambda^2")
print(f"       lies in [{(med-mean).var():.4f}, "
      f"{((Xs[:,1]-mean)**2).mean():.4f}] depending on where in the median")
print("       interval the dynamics rests.")

print("=" * 74)
print("K2  threshold-crossing rate  E[1{flip}] for noise sd = sqrt(eps)")
print("    P(flip | distance d to threshold) = Phi(-|d|/sqrt(eps))")
print("    E_d[.] with density p near 0  =  sqrt(eps) * 2 * E[Z_+] * p(0) + O(eps)")
print(f"    2 E[Z_+] = 2/sqrt(2 pi) = {2/np.sqrt(2*np.pi):.6f}")
print("    => the leading eps-response is O(sqrt(eps)) = O(noise SD), NOT O(eps).")

print("=" * 74)
print("K3  sigma_eff(sigma) = E[clip(N(sigma, 0.05^2), 0, 1)]")
for s in (0.2, 0.4, 0.6, 0.8, 1.0):
    a, b = (0 - s) / 0.05, (1 - s) / 0.05
    e = (s * (norm.cdf(b) - norm.cdf(a)) - 0.05 * (norm.pdf(b) - norm.pdf(a))
         + 1.0 * (1 - norm.cdf(b)))
    print(f"    sigma={s:.1f}: sigma_eff={e:.6f}   deficit {100*(e/s-1):+.3f}%"
          f"   E[Wbar]=n*sigma_eff={4*e:.4f}")
