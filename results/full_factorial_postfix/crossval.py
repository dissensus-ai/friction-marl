"""GPU/CPU cross-validation, recomputed after the CPU arm's defects were fixed.

Written 6 Aug 2026; extended the same day after an adversarial check found two
errors in the manuscript paragraph built on the first version of this script.
Source of record for every number in the cross-validation subsection.

BACKGROUND. The CPU arm carried three defects the GPU arm did not:
  1. action encoding -- build_action_map emits {-1,0,+1} but the env decoder
     expects MultiDiscrete {0,1,2} and subtracts one, giving deltas {-2,-1,0}.
     The shared resource state could only ever FALL.
  2. observation noise scaled by eps rather than sqrt(eps) (covariance eps^2 I
     where Definition 3.3 specifies eps I).
  3. replay stored a CLEAN next observation beside a noisy current one.
Only (1) is unique to the CPU path; (2) and (3) were GPU defects first, fixed
there in July and never propagated to the CPU runner. All three are fixed as of
2c0bfd5 and the arm was re-run in full (3,750 runs, 32.8 h).

WHAT DEFECT (1) ACTUALLY DID -- and the sign error worth recording. The first
draft of the manuscript paragraph reasoned that "a state that can only fall sits
further from every agent's target". That is backwards. sample_agent_params draws
targets about ZERO and the state is clipped to [0, capacity], so a monotonically
descending state moves TOWARD every target: the broken environment was a free
ride, which is why the defective arm scored HIGHER. `pinned_state_check` below
is the identification that survives -- with the state pinned at 0 the reward has
a closed form, -sum_j w_ij t_ij^2, and the defective arm's 125 condition means
sit on it.

WHAT THE RE-RUN DOES AND DOES NOT IDENTIFY. Three defects were repaired in one
re-run, so the collapse of the grid-mean offset is jointly attributable. Defects
(2) and (3) sit behind an `if cond_epsilon > 0` guard in run_parallel.py and are
inert at eps = 0, so the eps = 0 stratum is a clean single-defect counterfactual
for the decoder -- and there the repair does not zero the offset, it overshoots.
`epsilon_strata` reports this rather than hiding it in a grid mean.

EQUIVALENCE. The post-fix grid-mean offset is paired by factorial condition. The
script therefore reports a paired one-sample TOST and its corresponding 90% CI
at both the prespecified +/-0.10 margin and the stricter +/-0.05 margin. The
machine-readable output is `crossval_equivalence.csv`.

Run:  python3 results/full_factorial_postfix/crossval.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from friction_marl.experiments.factorial_design import sample_agent_params  # noqa: E402

K = ["alpha", "sigma", "epsilon"]
N_AGENTS, N_RESOURCES = 4, 3
PINNED_DRAWS = 4000


def condition_means(df, col="mean_reward", name="v"):
    return df.groupby(K, as_index=False)[col].mean().rename(columns={col: name})


def iccs(a, b):
    """Shrout & Fleiss two-way, single measure: consistency (3,1) and absolute
    agreement (A,1). The distinction is load-bearing here -- (3,1) is invariant
    to a pure column shift, so it CANNOT support a claim about levels. It reads
    0.907 on an arm carrying a +0.499 offset. (A,1) reads 0.769 on the same arm."""
    x = np.vstack([a, b]).T
    n, k = x.shape
    gm = x.mean()
    ms_r = k * ((x.mean(1) - gm) ** 2).sum() / (n - 1)
    ms_c = n * ((x.mean(0) - gm) ** 2).sum() / (k - 1)
    ms_e = ((x - x.mean(1, keepdims=True) - x.mean(0) + gm) ** 2).sum() / ((n - 1) * (k - 1))
    consistency = (ms_r - ms_e) / (ms_r + (k - 1) * ms_e)
    absolute = (ms_r - ms_e) / (ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n)
    return float(consistency), float(absolute)


def pinned_state_reward(alpha, sigma, seed):
    """Closed-form mean reward when the state is pinned at 0: -sum_j w_j t_j^2,
    averaged over agents, under the actual parameter sampler."""
    rng = np.random.default_rng(seed)
    return float(np.mean([
        (-(w * t ** 2).sum(axis=1)).mean()
        for w, t in (sample_agent_params(rng, N_AGENTS, N_RESOURCES, alpha, sigma)
                     for _ in range(PINNED_DRAWS))
    ]))


def load():
    new = condition_means(
        pd.read_csv(REPO / "results/full_factorial_postfix/replication_results.csv"),
        name="cpu_new")
    gpu_new = condition_means(
        pd.read_csv(REPO / "results/gpu_factorial_postfix/replication_results.csv"),
        name="gpu_new")
    old = pd.read_csv(REPO / "results/full_factorial/analysis/metrics.csv")
    old["cpu_old"] = -old["reward_gap"]                     # gap = -mean reward
    gold = pd.read_csv(REPO / "results/gpu_factorial/analysis/metrics.csv")
    gold["gpu_old"] = -gold["reward_gap"]
    m = (new.merge(gpu_new, on=K)
            .merge(old[K + ["cpu_old"]], on=K)
            .merge(gold[K + ["gpu_old"]], on=K))
    assert len(m) == 125, f"expected 125 matched conditions, got {len(m)}"
    return m


def agreement_table(m):
    """Both rows scored against the SAME (post-fix) GPU arm of record, so only the
    CPU arm varies. The 'before' row therefore differs from the +0.478 / 0.948 /
    0.945 / 0.913 printed in earlier versions, which paired the defective CPU arm
    against the PRE-fix GPU arm. That older pairing is printed here too."""
    print(f"{'comparison':44}{'offset':>9}{'rho':>8}{'r':>8}{'ICC(3,1)':>10}{'ICC(A,1)':>10}")
    for label, a, b in [
            ("cpu_old vs gpu_old  (earlier versions)", "cpu_old", "gpu_old"),
            ("cpu_old vs gpu_new  (before, re-scored)", "cpu_old", "gpu_new"),
            ("cpu_new vs gpu_new  (after all three)", "cpu_new", "gpu_new")]:
        c31, ca1 = iccs(m[a].values, m[b].values)
        print(f"{label:44}{(m[a] - m[b]).mean():+9.3f}"
              f"{stats.spearmanr(m[a], m[b]).statistic:8.3f}"
              f"{stats.pearsonr(m[a], m[b]).statistic:8.3f}{c31:10.3f}{ca1:10.3f}")
    print(f"{'gpu_new vs gpu_old  (GPU arm movement)':44}"
          f"{(m.gpu_new - m.gpu_old).mean():+9.3f}")


def pinned_state_check(m):
    """Identification for defect (1) that does not depend on the other two."""
    m = m.assign(pinned=[pinned_state_reward(a, s, seed=i)
                         for i, (a, s) in enumerate(zip(m.alpha, m.sigma))])
    print(f"\n{'arm':10}{'MAE vs pinned':>15}{'mean diff':>12}{'Pearson r':>12}")
    for c in ["cpu_old", "cpu_new", "gpu_new"]:
        print(f"{c:10}{(m[c] - m.pinned).abs().mean():15.3f}{(m[c] - m.pinned).mean():+12.3f}"
              f"{stats.pearsonr(m[c], m.pinned).statistic:12.3f}")
    print(f"defective arm scored higher than GPU in {(m.cpu_old > m.gpu_new).sum()}/125 cells; "
          f"repair moved CPU reward {(m.cpu_new - m.cpu_old).mean():+.3f}")
    return m


def epsilon_strata(m):
    """eps = 0 isolates the decoder: defects (2) and (3) are behind an
    `if cond_epsilon > 0` guard, and the else-branches are byte-identical either
    side of commit 2c0bfd5, so the +1 action shift is the only live change there."""
    pre, post = m.cpu_old - m.gpu_new, m.cpu_new - m.gpu_new
    print(f"\n{'eps':>6}{'pre-fix':>10}{'p':>10}{'post-fix':>10}{'p':>10}")
    for e, g in m.groupby("epsilon"):
        a, b = g.cpu_old - g.gpu_new, g.cpu_new - g.gpu_new
        print(f"{e:>6.2f}{a.mean():+10.3f}{stats.ttest_1samp(a, 0).pvalue:10.1e}"
              f"{b.mean():+10.3f}{stats.ttest_1samp(b, 0).pvalue:10.1e}")
    z, nz = m.epsilon == 0, m.epsilon > 0
    print(f"pre-fix  eps=0 {pre[z].mean():+.3f} vs eps>0 {pre[nz].mean():+.3f}  "
          f"Welch p={stats.ttest_ind(pre[z], pre[nz], equal_var=False).pvalue:.1e}")
    print(f"post-fix eps=0 {post[z].mean():+.3f} vs eps>0 {post[nz].mean():+.3f} "
          f"(eps>0 pooled p={stats.ttest_1samp(post[nz], 0).pvalue:.2f})")
    for lab, d in [("pre", pre), ("post", post)]:
        f = stats.f_oneway(*[v.values for _, v in d.groupby(m.epsilon)])
        print(f"one-way ANOVA over eps, {lab:4}-fix residual: F={f.statistic:.2f} p={f.pvalue:.4f}")


def paired_tost(d, margin, alpha=0.05):
    """Paired one-sample TOST for mean(d) inside (-margin, +margin).

    At alpha=.05, equivalence is established when both one-sided tests reject;
    this is identical to the 90% CI lying wholly inside the equivalence bounds.
    """
    d = np.asarray(d, dtype=float)
    n = len(d)
    mean = float(d.mean())
    se = float(stats.sem(d))
    df = n - 1
    ci_lo, ci_hi = stats.t.interval(1 - 2 * alpha, df, loc=mean, scale=se)
    lower_t = (mean + margin) / se   # H0: mean <= -margin
    upper_t = (mean - margin) / se   # H0: mean >= +margin
    lower_p = float(stats.t.sf(lower_t, df))
    upper_p = float(stats.t.cdf(upper_t, df))
    tost_p = max(lower_p, upper_p)
    return dict(
        margin=float(margin), n=n, mean_offset=mean, se=se,
        ci_level=1 - 2 * alpha, ci_lo=float(ci_lo), ci_hi=float(ci_hi),
        lower_t=float(lower_t), lower_p=lower_p,
        upper_t=float(upper_t), upper_p=upper_p,
        tost_p=tost_p, equivalent_at_05=bool(tost_p < alpha),
    )


def equivalence_table(d):
    rows = pd.DataFrame([paired_tost(d, margin) for margin in (0.10, 0.05)])
    ci = rows.iloc[0]
    print(f"\npaired offset: {ci.mean_offset:+.6f}; "
          f"90% CI [{ci.ci_lo:+.6f}, {ci.ci_hi:+.6f}]")
    print(f"{'margin':>8}{'lower p':>12}{'upper p':>12}{'TOST p':>12}{'equivalent':>13}")
    for row in rows.itertuples(index=False):
        print(f"{row.margin:8.2f}{row.lower_p:12.4g}{row.upper_p:12.4g}"
              f"{row.tost_p:12.4g}{str(row.equivalent_at_05):>13}")
    return rows


def main():
    m = load()
    print(f"matched conditions: {len(m)}\n")
    agreement_table(m)
    m = pinned_state_check(m)
    epsilon_strata(m)

    d = m.cpu_new - m.gpu_new
    equivalence = equivalence_table(d)
    lo, hi = d.mean() - 1.96 * d.std(), d.mean() + 1.96 * d.std()
    print(f"\nnew arm Bland-Altman: bias {d.mean():+.3f}, limits [{lo:.3f}, {hi:.3f}], "
          f"{100 * ((d > lo) & (d < hi)).mean():.1f}% inside (nominal 95%)")
    print(f"residual offset vs zero: t p={stats.ttest_1samp(d, 0).pvalue:.2f}, "
          f"Wilcoxon p={stats.wilcoxon(d).pvalue:.2f}")

    out = REPO / "results/full_factorial_postfix/crossval_conditions.csv"
    m.to_csv(out, index=False)
    equivalence_out = REPO / "results/full_factorial_postfix/crossval_equivalence.csv"
    equivalence.to_csv(equivalence_out, index=False)
    print(f"\n-> {out}\n-> {equivalence_out}")


if __name__ == "__main__":
    main()
