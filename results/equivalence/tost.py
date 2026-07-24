"""Two One-Sided Tests (TOST) for statistical EQUIVALENCE (ADDITIVE).

A non-significant difference test is NOT evidence of equivalence. To claim that
switching the learner (IQL -> VDN / MAPPO) does not change the reward -- i.e. that
the cooperation finding is algorithm-invariant -- we need an equivalence test with
an a-priori margin. TOST (Schuirmann 1987) rejects the null of a difference at
least as large as the margin via two one-sided t-tests; the equivalence p-value is
the LARGER of the two one-sided p-values.

This module is a thin, dependency-light wrapper over
`statsmodels.stats.weightstats.ttost_ind`, plus a helper that turns a raw effect
size into a symmetric margin (margin = fraction * anchor_effect).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
from statsmodels.stats.weightstats import ttost_ind


@dataclass
class TOSTResult:
    n_x: int
    n_y: int
    mean_x: float
    mean_y: float
    mean_diff: float          # mean_x - mean_y
    margin: float             # symmetric equivalence bound (+/- margin)
    p_tost: float             # equivalence p-value = max of the two one-sided p-values
    p_lower: float            # H0: diff <= -margin
    p_upper: float            # H0: diff >= +margin
    equivalent: bool          # p_tost < alpha
    alpha: float

    def as_dict(self) -> dict:
        return asdict(self)


def margin_from_effect(anchor_effect: float, fraction: float) -> float:
    """Symmetric equivalence margin = fraction * |anchor_effect| (a-priori)."""
    return float(abs(fraction) * abs(anchor_effect))


def tost_equivalence(x, y, margin: float, alpha: float = 0.05,
                     usevar: str = "unequal") -> TOSTResult:
    """TOST that mean(x) - mean(y) lies within +/- `margin`.

    x, y: 1-D arrays of independent per-run scores (NaNs dropped).
    usevar: 'unequal' (Welch, default, safest) or 'pooled'.
    Equivalence is declared (equivalent=True) when p_tost < alpha.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        raise ValueError("need >=2 finite observations in each group for TOST")
    margin = float(abs(margin))
    p_tost, res_low, res_upp = ttost_ind(x, y, -margin, margin, usevar=usevar)
    return TOSTResult(
        n_x=len(x), n_y=len(y),
        mean_x=float(x.mean()), mean_y=float(y.mean()),
        mean_diff=float(x.mean() - y.mean()),
        margin=margin,
        p_tost=float(p_tost),
        p_lower=float(res_low[1]),
        p_upper=float(res_upp[1]),
        equivalent=bool(p_tost < alpha),
        alpha=float(alpha),
    )
