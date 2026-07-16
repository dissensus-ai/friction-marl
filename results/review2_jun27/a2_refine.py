"""A2 refinements: isolate SLOPE-shape misspecification + adjacent-level slope contrasts."""
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

REPO = Path(__file__).resolve().parents[2]
PS = REPO / "results" / "partial_sharing"

def load(path):
    df = pd.read_csv(path); df["gap_norm"] = -df["mean_reward"]/df["sigma"]; return df

def _lab(nm):
    try: return float(nm.split("[T.")[1].split("]")[0])
    except Exception: return None

def slope_contrast(model, wA, wB, w_levels, ref_w):
    """t_test slope(wA) - slope(wB) = 0 with NAMED ContrastResults attrs."""
    names = list(model.model.exog_names); idx={n:i for i,n in enumerate(names)}
    c = np.zeros(len(names))
    for w, sgn in ((wA, 1.0), (wB, -1.0)):
        c[idx["rho"]] += sgn
        if w != ref_w:
            m=[n for n in names if n.endswith(":rho") and _lab(n) is not None and abs(_lab(n)-w)<1e-9]
            c[idx[m[0]]] += sgn
    tt = model.t_test(c); ci = tt.conf_int(alpha=0.05)
    return float(np.ravel(tt.effect)[0]), float(ci[0,0]), float(ci[0,1]), float(np.ravel(tt.pvalue)[0])

for learner, fn in (("IQL","partial_iql.csv"),("VDN","partial_vdn.csv")):
    df = load(PS/fn); wl=sorted(df.w.unique()); ref=wl[0]
    m_cat = smf.ols("gap_norm ~ C(w)*rho", data=df).fit(cov_type="HC3")
    print("="*78); print(f"[{learner}]"); print("="*78)
    # adjacent + key slope-difference contrasts
    print("slope-difference contrasts (HC3):")
    for a,b,tag in [(wl[1],wl[0],"w=1/3 vs w=0  (low-region flatness)"),
                    (wl[2],wl[1],"w=2/3 vs w=1/3 (ENGAGEMENT step)"),
                    (wl[3],wl[2],"w=1   vs w=2/3 (high-region)"),
                    (wl[3],wl[0],"w=1   vs w=0   (endpoint span)")]:
        d,lo,hi,pv = slope_contrast(m_cat,a,b,wl,ref)
        print(f"   {tag:34} dslope={d:+.3f} [{lo:+.3f},{hi:+.3f}] p={pv:.3e}")
    # isolate SLOPE-shape misspecification: categorical intercepts, slope LINEAR-in-w vs full categorical slope
    m_slin = smf.ols("gap_norm ~ C(w) + rho + rho:w", data=df).fit()   # flexible intercepts, slope linear in w
    m_full = smf.ols("gap_norm ~ C(w)*rho", data=df).fit()             # flexible intercepts AND per-w slopes
    aov = anova_lm(m_slin, m_full)
    F=float(aov["F"].iloc[-1]); pv=float(aov["Pr(>F)"].iloc[-1]); ddf=float(aov["df_diff"].iloc[-1])
    print(f"\nSLOPE-shape-only misspecification test (intercepts categorical in both):")
    print(f"   slope linear-in-w  vs  per-w slopes :  F({ddf:.0f},{float(aov['df_resid'].iloc[-1]):.0f})={F:.2f}  p={pv:.3e}")
    print(f"   AIC slope-linear={m_slin.aic:.1f}  per-w-slope={m_full.aic:.1f} (D={m_slin.aic-m_full.aic:+.1f})")
