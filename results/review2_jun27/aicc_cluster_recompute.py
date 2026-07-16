import numpy as np, pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "adversarial_rerun"
N_AGENTS, N_RESOURCES, CAPACITY = 4, 3, 10.0
RHOS = [-0.33,-0.2,-0.1,0.0,0.2,0.4,0.6,0.8]; SIGMAS=[0.6,1.0]; N=16

print("#"*80); print("# A3(i)  AICc for the signed functional-form comparison (M1,M2,M4; n=16)"); print("#"*80)

def condition_means(csv):
    d=pd.read_csv(csv); d["gap"]=-d["mean_reward"]
    return (d.groupby(["rho","sigma"],as_index=False)
              .agg(gap=("gap","mean"),eps=("epsilon","first")))

def fit(y,X1,label):
    X=sm.add_constant(X1,has_constant="add")
    m=sm.OLS(np.asarray(y,float),np.asarray(X,float)).fit()
    kmean=X.shape[1]                                  # regressors incl intercept
    # back out the parameter count statsmodels used in its AIC: aic = -2 llf + 2*k_used
    k_used=(m.aic+2*m.llf)/2.0
    return dict(model=label,kmean=kmean,llf=m.llf,aic=m.aic,k_used_in_sm_aic=k_used,r2=m.rsquared)

def aicc_from(aic_sm, kmean, n=N, count_sigma2=True):
    """AICc. If count_sigma2: K=kmean+1 (Burnham&Anderson strict) and the base AIC is
    corrected to that convention (+2). Else K=kmean (statsmodels base param count)."""
    if count_sigma2:
        K=kmean+1; base=aic_sm+2.0            # add the sigma^2 param to the 2K penalty
    else:
        K=kmean;   base=aic_sm
    corr=2*K*(K+1)/(n-K-1)
    return base+corr, K, corr

for name,fn in [("IQL","equicorr_iql.csv"),("VDN","equicorr_vdn.csv")]:
    g=condition_means(D/fn); assert (g["eps"]==0).all()
    rho,sig,y=g["rho"].values,g["sigma"].values,g["gap"].values
    F1=sig*(1.0)/(1.0+rho)                 # M1 sigma/(1+rho)
    C2=sig-rho                             # M2 sigma-rho
    X4=np.column_stack([rho,sig])          # M4 rho+sigma
    r1=fit(y,F1.reshape(-1,1),"M1");  r2=fit(y,C2.reshape(-1,1),"M2");  r4=fit(y,X4,"M4")
    print(f"\n=== {name} ===")
    for r in (r1,r2,r4):
        print(f"  {r['model']}: kmean(incl intercept)={r['kmean']}  llf={r['llf']:.4f}  "
              f"AIC_sm={r['aic']:.4f}  k_used_in_sm_AIC={r['k_used_in_sm_aic']:.2f}  R2={r['r2']:.4f}")
    # Convention I: B&A strict, K = kmean + 1 (sigma^2 counted)
    a1_I,K1,c1=aicc_from(r1['aic'],r1['kmean'],count_sigma2=True)
    a4_I,K4,c4=aicc_from(r4['aic'],r4['kmean'],count_sigma2=True)
    a2_I,_,_  =aicc_from(r2['aic'],r2['kmean'],count_sigma2=True)
    # Convention II: k = kmean only (statsmodels base param count)
    a1_II,k1,d1=aicc_from(r1['aic'],r1['kmean'],count_sigma2=False)
    a4_II,k4,d4=aicc_from(r4['aic'],r4['kmean'],count_sigma2=False)
    a2_II,_,_  =aicc_from(r2['aic'],r2['kmean'],count_sigma2=False)
    print(f"  -- Convention I (B&A strict, K=kmean+1 incl sigma^2): M1 K={K1} corr={c1:.4f} AICc={a1_I:.4f} | "
          f"M2 AICc={a2_I:.4f} | M4 K={K4} corr={c4:.4f} AICc={a4_I:.4f}")
    print(f"     dAICc(M4-M1)={a4_I-a1_I:+.4f}   dAICc(M4-M2)={a4_I-a2_I:+.4f}")
    print(f"  -- Convention II (k=kmean, sigma^2 not counted): M1 k={k1} corr={d1:.4f} AICc={a1_II:.4f} | "
          f"M2 AICc={a2_II:.4f} | M4 k={k4} corr={d4:.4f} AICc={a4_II:.4f}")
    print(f"     dAICc(M4-M1)={a4_II-a1_II:+.4f}   dAICc(M4-M2)={a4_II-a2_II:+.4f}")
    print(f"  -- plain dAIC(M4-M1)={r4['aic']-r1['aic']:+.4f}  (CSV dAIC_M1_minus_M4={r1['aic']-r4['aic']:.4f})")

print()
print("#"*80); print("# A3(ii)  Cluster-robust SE for the n=1920 rho x w interaction (gap/sigma ~ rho*w)"); print("#"*80)

def recon_w(seed=42,n_reps=30):
    m=np.random.default_rng(seed); m.uniform(0,CAPACITY,size=(128,N_RESOURCES)).astype(np.float32)
    ck=[(a,s) for a in RHOS for s in SIGMAS]; cs={k:int(m.integers(0,2**62)) for k in ck}
    rows=[]
    for (rho,sig) in ck:
        rng=np.random.default_rng(cs[(rho,sig)])
        for r in range(n_reps):
            rng.standard_normal((N_AGENTS,N_RESOURCES))
            w=np.clip(rng.normal(sig,0.05,(N_AGENTS,N_RESOURCES)),0,1)
            for i in range(N_AGENTS):
                rows.append(dict(rho=rho,sigma=sig,replication=r,agent=f"agent_{i}_reward",w=float(w[i].mean())))
    return pd.DataFrame(rows)
W=recon_w()

def load(csv):
    dd=pd.read_csv(csv)
    L=dd.melt(id_vars=["mode","sigma","replication","rho"],
              value_vars=[f"agent_{i}_reward" for i in range(N_AGENTS)],var_name="agent",value_name="reward")
    L["gap"]=-L["reward"]; L["gap_norm"]=L["gap"]/L["sigma"]
    L=L.merge(W,on=["rho","sigma","replication","agent"]); assert L["w"].notna().all()
    L["cell"]=L["rho"].astype(str)+"|"+L["sigma"].astype(str)             # condition cell (rho,sigma): G=16
    L["seed"]=L["cell"]+"|"+L["replication"].astype(str)                  # independent run (rho,sigma,rep): G=480
    return L

for name,fn in [("IQL","equicorr_iql.csv"),("VDN","equicorr_vdn.csv")]:
    d=load(D/fn)
    f="gap_norm ~ rho*w"
    base=smf.ols(f,data=d).fit()
    it=[k for k in base.params.index if ":" in k][0]
    coef=base.params[it]; se_n=base.bse[it]; p_n=base.pvalues[it]
    cl=smf.ols(f,data=d).fit(cov_type="cluster",cov_kwds={"groups":d["cell"]})
    se_c16,p_c16=cl.bse[it],cl.pvalues[it]
    sd=smf.ols(f,data=d).fit(cov_type="cluster",cov_kwds={"groups":d["seed"]})
    se_c480,p_c480=sd.bse[it],sd.pvalues[it]
    nG16=d["cell"].nunique(); nG480=d["seed"].nunique()
    print(f"\n=== {name}  (n={int(base.nobs)}, interaction term '{it}') ===")
    print(f"  coef(rho:w)            = {coef:+.4f}")
    print(f"  naive (i.i.d.)         SE={se_n:.4f}  p={p_n:.3e}")
    print(f"  cluster by cell G={nG16:<3d}   SE={se_c16:.4f}  p={p_c16:.3e}")
    print(f"  cluster by seed G={nG480:<3d}  SE={se_c480:.4f}  p={p_c480:.3e}")
    # also raw-gap spec for completeness
    fr="gap ~ rho*w"
    br=smf.ols(fr,data=d).fit(); itr=[k for k in br.params.index if ":" in k][0]
    pr_n=br.pvalues[itr]
    pr_c16=smf.ols(fr,data=d).fit(cov_type="cluster",cov_kwds={"groups":d["cell"]}).pvalues[itr]
    pr_c480=smf.ols(fr,data=d).fit(cov_type="cluster",cov_kwds={"groups":d["seed"]}).pvalues[itr]
    print(f"  [raw-gap spec gap~rho*w] coef={br.params[itr]:+.4f} naive p={pr_n:.3e} "
          f"clustG16 p={pr_c16:.3e} clustG480 p={pr_c480:.3e}")
