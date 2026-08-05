"""Regenerate the three appendix figures from the post-fix factorial.

Written 5 Aug 2026. Three problems this fixes at once:

 1. PROVENANCE. figures/marl_*.pdf were rendered 28 Jun from the PRE-fix run,
    while every table in the paper is now post-fix. No generation script for them
    was ever committed, so they could not be checked or reproduced -- the same
    gap between a stated quantity and its code that the paper is about.

 2. A WITHDRAWN PROXY ON THE PAGE. The old main-effects figure carried a second
    row plotting convergence time, including the impossible 1,200-episode values
    inside a 1,000-episode budget. Section "Dependent variables" withdraws that
    measure; the figure kept showing it. The row is dropped here.

 3. TWO INCOMPATIBLE GINI DEFINITIONS. dynamics_analysis.py used a min-shifted
    Gini (lowest at alpha=0); generate_appendix_latex.py and regenerate_tables.py
    use Gini over |reward| (peaks at alpha=0). They disagree in SHAPE, so the
    appendix caption asserted the opposite of what its own panel showed. This
    script uses the |reward| definition throughout, matching the tables.

Also prints the Appendix A.2 quantities so they can be checked against the text.

Run:  python3 results/gpu_factorial_postfix/make_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "results/gpu_factorial_postfix/replication_results.csv"
FIGS = REPO.parent.parent / "papers-official/marl-coordination/figures"
AGENTS = [f"agent_{i}_reward" for i in range(4)]
BURGUNDY = "#7B2D3B"


def gini(row):
    """Gini over |agent reward| -- the definition used by regenerate_tables.py
    and therefore by tab:interactions. Rewards are negative, hence magnitudes."""
    x = np.sort(np.abs(np.asarray(row, dtype=float)))
    if x.sum() == 0:
        return 0.0
    n = x.size
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def main():
    d = pd.read_csv(DATA)
    d["gini"] = d[AGENTS].apply(lambda r: gini(r.values), axis=1)
    d["rvar"] = d[AGENTS].var(axis=1, ddof=0)
    FIGS.mkdir(parents=True, exist_ok=True)

    # ---- main effects: ONE row (mean reward). The convergence-time row is gone.
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, (f, lab) in zip(axes, [("alpha", r"$\alpha$ (Alignment)"),
                                   ("sigma", r"$\sigma$ (Stakes)"),
                                   ("epsilon", r"$\varepsilon$ (Entropy / Obs. Noise)")]):
        g = d.groupby(f)["mean_reward"].agg(["mean", "sem"])
        ax.errorbar(g.index, g["mean"], yerr=1.96 * g["sem"], marker="o",
                    color=BURGUNDY, capsize=3, lw=1.6)
        ax.set_xlabel(lab)
        ax.set_ylabel("Mean reward")
        ax.set_title(f"range = {g['mean'].max() - g['mean'].min():.4f}", fontsize=9)
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIGS / "marl_main_effects.pdf")
    plt.close(fig)

    # ---- alpha x sigma heatmap (marginalized over epsilon)
    piv = d.pivot_table(index="alpha", columns="sigma", values="mean_reward")
    fig, ax = plt.subplots(figsize=(5.2, 4))
    im = ax.imshow(piv.values, cmap="RdBu_r", aspect="auto", origin="lower")
    ax.set_xticks(range(len(piv.columns)), [f"{c:g}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), [f"{i:+g}" for i in piv.index])
    ax.set_xlabel(r"$\sigma$ (Stakes)")
    ax.set_ylabel(r"$\alpha$ (Alignment)")
    ax.set_title("Mean reward, marginalized over $\\varepsilon$", fontsize=10)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(FIGS / "marl_heatmap_alpha_sigma.pdf")
    plt.close(fig)

    # ---- agent-level dispersion vs alpha: variance (scale-dependent) + Gini
    gv = d.groupby("alpha")[["rvar", "gini"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].plot(gv.index, gv.rvar, marker="o", color=BURGUNDY, lw=1.6)
    axes[0].set_xlabel(r"$\alpha$ (Alignment)")
    axes[0].set_ylabel("Agent reward variance")
    axes[1].plot(gv.index, gv.gini, marker="s", color=BURGUNDY, lw=1.6)
    axes[1].set_xlabel(r"$\alpha$ (Alignment)")
    axes[1].set_ylabel("Gini over $|$reward$|$")
    for ax in axes:
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIGS / "marl_agent_specialization.pdf")
    plt.close(fig)

    # ---- Appendix A.2 quantities, for checking against the prose
    print("regenerated 3 figures ->", FIGS)
    print("\nAppendix A.2 quantities (post-fix):")
    print("  agent reward variance by alpha:")
    for a, v in gv.rvar.items():
        print(f"    alpha={a:+.1f}: {v:.3f}")
    print(f"  ratio (alpha=0) / (mean of |alpha|=0.8): "
          f"{gv.rvar.loc[0.0] / gv.rvar.loc[[-0.8, 0.8]].mean():.1f}x")
    sv = d.groupby("sigma")["rvar"].mean()
    print(f"  variance by sigma: {sv.iloc[0]:.3f} -> {sv.iloc[-1]:.3f}")
    print(f"  Gini by alpha: " + ", ".join(f"{a:+.1f}={v:.3f}" for a, v in gv.gini.items()))
    cg = d.groupby(["alpha", "sigma", "epsilon"])["gini"].mean()
    print(f"  Gini range across conditions: {cg.min():.3f} - {cg.max():.3f}")


if __name__ == "__main__":
    main()
