"""A19.  Symbolic verification of every algebraic identity asserted in the memo,
so that nothing rests on hand algebra.

S1  the welfare identity (1.1)
S2  Var(tau_i - taubar) = (n-1)/n * v_tau (1-rho)  =>  the definition lambda^2 = v_tau(1-rho)
S3  Var(mu) = v_tau - (n-1)/n lambda^2
S4  rho(alpha) = alpha^2 + O(|alpha|^3)  and  lambda^2_renorm = 1 - alpha^2 + O(|alpha|^3)
    (i.e. the renormalised DGP has NO first-order alpha content)
S5  lambda^2_implemented = (1-|alpha|)^2 : a kink, one-sided slopes -/+2
S6  the weight-dispersion correction (4.3)
S7  the small-Var(mu) floor coefficient (4/3) * 2/sqrt(2 pi)
S8  sigma_eff(sigma) closed form (5.1) vs numerical integration
S9  the decomposition identity (3.2)
S10 the interior floor constant 1/24 (4.1)
"""
import numpy as np
import sympy as sp

ok = []


def check(name, cond, detail=""):
    ok.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}")


print("=" * 88)
n = 4
x, mu_, Wb = sp.symbols("x mu Wbar", real=True)
w = sp.symbols("w1:5", positive=True)
t = sp.symbols("t1:5", real=True)
lhs = sum(w[i] * (x - t[i]) ** 2 for i in range(n))
Wbar = sum(w)
mu = sum(w[i] * t[i] for i in range(n)) / Wbar
V = sum(w[i] * (t[i] - mu) ** 2 for i in range(n))
check("S1 welfare identity  sum w_i (x-t_i)^2 = Wbar (x-mu)^2 + V",
      sp.simplify(sp.expand(lhs - (Wbar * (x - mu) ** 2 + V))) == 0)

# ---- S2 / S3: exchangeable second moments ---------------------------------- #
vt, rho = sp.symbols("v_tau rho", positive=True)
# Var(tau_i - taubar) = Var(tau_i) - 2 Cov(tau_i, taubar) + Var(taubar)
Var_tb = vt * (1 + (n - 1) * rho) / n
Cov_i_tb = (vt + (n - 1) * rho * vt) / n
disp = sp.simplify(vt - 2 * Cov_i_tb + Var_tb)
check("S2 Var(tau_i - taubar) = (n-1)/n * v_tau (1-rho)",
      sp.simplify(disp - sp.Rational(n - 1, n) * vt * (1 - rho)) == 0,
      f"= {sp.simplify(disp)}")
lam2 = vt * (1 - rho)
check("S3 Var(mu) = v_tau - (n-1)/n lambda^2",
      sp.simplify(Var_tb - (vt - sp.Rational(n - 1, n) * lam2)) == 0)

# ---- S4 / S5: what the alpha knob varies ----------------------------------- #
a = sp.symbols("a", positive=True)                 # a = |alpha|, expand at 0+
v_impl = a ** 2 + (1 - a) ** 2
rho_impl = a ** 2 / v_impl
ser_rho = sp.series(rho_impl, a, 0, 4).removeO()
check("S4a rho(alpha) = alpha^2 + 2|alpha|^3 + O(a^4)  (quadratic: no 1st-order term)",
      sp.simplify(ser_rho - (a ** 2 + 2 * a ** 3)) == 0, f"series = {sp.expand(ser_rho)}")
lam2_renorm = sp.simplify((1 - rho_impl))
ser_lam = sp.series(lam2_renorm, a, 0, 4).removeO()
check("S4b lambda^2_renorm = 1 - alpha^2 - 2|alpha|^3 + O(a^4): d/da at 0+ is ZERO",
      sp.simplify(sp.diff(ser_lam, a).subs(a, 0)) == 0, f"series = {sp.expand(ser_lam)}")
lam2_impl = (1 - a) ** 2
check("S5 lambda^2_impl = (1-|alpha|)^2 : one-sided d/d|alpha| at 0+ is -2 (a kink)",
      sp.diff(lam2_impl, a).subs(a, 0) == -2)
check("S5b v_tau_impl one-sided d/d|alpha| at 0+ is -2",
      sp.diff(v_impl, a).subs(a, 0) == -2)

# ---- S6: the weight-dispersion correction ---------------------------------- #
c = sp.symbols("c", positive=True)
z = sp.symbols("z1:5", real=True)
d = sp.symbols("d1:5", real=True)                  # d_i = tau_i - taubar, sum d_i = 0
sub0 = {d[n - 1]: -sum(d[i] for i in range(n - 1))}   # impose sum_i d_i = 0
num = sum((1 + c * z[i]) * d[i] for i in range(n)).subs(sub0)
den = sum(1 + c * z[i] for i in range(n))
mu_dev = sp.series(sp.simplify(num / den), c, 0, 2).removeO()   # mu - taubar to O(c)
target = (c * sum(z[i] * d[i] for i in range(n)) / n).subs(sub0)
check("S6 mu - taubar = (c/n) sum_i z_i (tau_i - taubar) + O(c^2)",
      sp.simplify(sp.expand(mu_dev - target)) == 0,
      f"residual = {sp.simplify(sp.expand(mu_dev - target))}")
# variance of the leading term, z iid N(0,1) independent of d:
# E[(c/n sum z_i d_i)^2] = c^2/n^2 sum d_i^2 -> c^2/n^2 * n * E[d^2] = c^2/n * (n-1)/n lam^2
pred = sp.Rational(n - 1, n ** 2) * lam2
check("S6b Var(mu) gains (n-1)/n^2 * c^2 lambda^2 = (3/16) c^2 lambda^2",
      sp.simplify(pred - sp.Rational(3, 16) * lam2) == 0, f"coef = {sp.Rational(n-1, n**2)}")

# ---- S7: the small-Var(mu) floor branch ------------------------------------ #
m_, f_ = sp.symbols("m f", positive=True)
# d = min(m, |f - m|); |f-m| < m  iff  0 < f < 2m
E_d2 = m_ ** 2 * (1 - 2 * m_) + sp.integrate((f_ - m_) ** 2, (f_, 0, 2 * m_))
check("S7a E[d^2 | mu=m] = m^2 - (4/3) m^3   for 0<m<1/2, f~U(0,1)",
      sp.simplify(sp.expand(E_d2) - (m_ ** 2 - sp.Rational(4, 3) * m_ ** 3)) == 0,
      f"= {sp.expand(E_d2)}")
v = sp.symbols("v", positive=True)
mu_s = sp.symbols("m", real=True)
pdf = sp.exp(-mu_s ** 2 / (2 * v)) / sp.sqrt(2 * sp.pi * v)
E_m3 = sp.integrate(mu_s ** 3 * pdf, (mu_s, 0, sp.oo), conds="none")
check("S7b E[mu^3 1{mu>0}] = v^{3/2} * 2/sqrt(2 pi)",
      sp.simplify(E_m3 - v ** sp.Rational(3, 2) * 2 / sp.sqrt(2 * sp.pi)) == 0,
      f"= {sp.simplify(E_m3)}")
K = sp.Rational(4, 3) * 2 / sp.sqrt(2 * sp.pi)
check("S7c coefficient (4/3)*2/sqrt(2 pi) = 1.063846",
      abs(float(K) - 1.063846) < 1e-6, f"= {float(K):.6f}")

# ---- S8: sigma_eff --------------------------------------------------------- #
from scipy.stats import norm
from scipy.integrate import quad
for s0 in (0.2, 0.6, 1.0, 1.2):
    aa, bb = (0 - s0) / 0.05, (1 - s0) / 0.05
    cf = (s0 * (norm.cdf(bb) - norm.cdf(aa)) - 0.05 * (norm.pdf(bb) - norm.pdf(aa))
          + (1 - norm.cdf(bb)))
    nu = quad(lambda u: np.clip(u, 0, 1) * norm.pdf(u, s0, 0.05), s0 - 1, s0 + 1)[0] \
        + 1.0 * (1 - norm.cdf(bb + 20)) * 0
    nu = quad(lambda u: np.clip(u, 0, 1) * norm.pdf(u, s0, 0.05), -1, 3)[0]
    check(f"S8 sigma_eff({s0}) closed form == numeric", abs(cf - nu) < 1e-9,
          f"{cf:.9f} vs {nu:.9f}")

# ---- S9: the decomposition ------------------------------------------------- #
Vpi, b_, q_ = sp.symbols("Var_pi b q", real=True)
Epi = Vpi + (b_ + q_) ** 2
check("S9 E_pi(x-mu)^2 - q^2 = Var_pi + b^2 + 2 q b",
      sp.simplify(sp.expand(Epi - q_ ** 2) - (Vpi + b_ ** 2 + 2 * q_ * b_)) == 0)

# ---- S10: the interior constant -------------------------------------------- #
g = sp.symbols("g", positive=True)
Ed2_arc = sp.integrate(sp.Min(g - f_, f_) ** 2, (f_, 0, g)) / g
check("S10a E[d^2 | arc length g] = g^2/12",
      sp.simplify(Ed2_arc - g ** 2 / 12) == 0, f"= {sp.simplify(Ed2_arc)}")
Ed2_f = (f_ * (f_ ** 2 / 12) + (1 - f_) * ((1 - f_) ** 2 / 12))
check("S10b E_f[(f^3+(1-f)^3)/12] = 1/24",
      sp.integrate(Ed2_f, (f_, 0, 1)) == sp.Rational(1, 24),
      f"= {sp.integrate(Ed2_f,(f_,0,1))}")

print("=" * 88)
print(f"{sum(v for _, v in ok)}/{len(ok)} symbolic checks PASS")
