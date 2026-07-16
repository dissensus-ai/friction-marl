"""Decompose WHY the linear rho*w summary fails: intercept-shape vs slope-shape."""
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
REPO = Path(__file__).resolve().parents[2]
PS = REPO/"results"/"partial_sharing"
def load(p):
    d=pd.read_csv(p); d["gap_norm"]=-d["mean_reward"]/d["sigma"]; return d
for learner,fn in (("IQL","partial_iql.csv"),("VDN","partial_vdn.csv")):
    df=load(PS/fn); wl=sorted(df.w.unique())
    m_lin  = smf.ols("gap_norm ~ rho * w", data=df).fit()          # both linear in w
    m_slin = smf.ols("gap_norm ~ C(w) + rho + rho:w", data=df).fit() # cat intercept, linear slope
    m_full = smf.ols("gap_norm ~ C(w) * rho", data=df).fit()        # cat intercept, cat slope
    # intercept-shape: lin vs slin (adds 2 df = intercept curvature), slope held linear
    a_int = anova_lm(m_lin, m_slin)
    # slope-shape: slin vs full (adds 2 df = slope curvature), intercept held categorical
    a_slp = anova_lm(m_slin, m_full)
    print(f"[{learner}]")
    print(f"  INTERCEPT-shape (linear->categorical intercept, slope linear): "
          f"F({a_int['df_diff'].iloc[-1]:.0f})={a_int['F'].iloc[-1]:.2f} p={a_int['Pr(>F)'].iloc[-1]:.3e}")
    print(f"  SLOPE-shape     (linear->categorical slope, intercept categ.): "
          f"F({a_slp['df_diff'].iloc[-1]:.0f})={a_slp['F'].iloc[-1]:.2f} p={a_slp['Pr(>F)'].iloc[-1]:.3e}")
    print(f"  linear-ramp slope coef (rho:w | cat intercepts) = {m_slin.params['rho:w']:+.3f} "
          f"p={m_slin.pvalues['rho:w']:.2e}")
    sl=[smf.ols('gap_norm~rho',data=df[df.w==w]).fit() for w in wl]
    slopes=np.array([m.params['rho'] for m in sl])
    sp=stats.spearmanr(wl,slopes)
    print(f"  per-w slopes {np.round(slopes,3)}  Spearman(w,slope)={sp.statistic:+.3f} p={sp.pvalue:.3f}")
