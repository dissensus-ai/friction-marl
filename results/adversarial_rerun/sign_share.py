#!/usr/bin/env python3
"""
sign_share.py
=============
Provenance for the sign-share figures 99.3% (IQL) / 94.6% (VDN) quoted in the
manuscript's "The sign now carries the variance" paragraph (Sec. equicorr-results,
"Coordination friction tracks signed correlation"), previously unscripted.

On the stake-normalized replication-level gap g/sigma = -mean_reward/sigma from the
signed equicorrelation re-run (n = 480 per learner), fit OLS of g/sigma on rho
(signed-linear) and on rho + rho^2 (quadratic). The sign-share is the fraction of
the quadratic model's explained variance attributable to the signed-linear term,
R^2_lin / R^2_quad. Contrast: the first-stage sign-blind design realizes an
|alpha|-only correlation, so its signed-linear share is ~0% by construction.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent


def r2(X: np.ndarray, y: np.ndarray) -> float:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return 1.0 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))


def main() -> None:
    for mode in ("iql", "vdn"):
        df = pd.read_csv(HERE / f"equicorr_{mode}.csv")
        g_norm = -df["mean_reward"].to_numpy() / df["sigma"].to_numpy()
        rho = df["rho"].to_numpy()
        ones = np.ones_like(rho)
        r2_lin = r2(np.column_stack([ones, rho]), g_norm)
        r2_quad = r2(np.column_stack([ones, rho, rho**2]), g_norm)
        share = 100.0 * r2_lin / r2_quad
        print(
            f"{mode.upper()}: n={len(df)}  R2_lin={r2_lin:.4f}  "
            f"R2_quad={r2_quad:.4f}  sign-share={share:.1f}%"
        )


if __name__ == "__main__":
    main()
