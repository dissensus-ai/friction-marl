"""R3. REFEREE ADJUDICATION: the eps -> 0+ ladder.

FLAT CONTRADICTION between the two builders, same DGP / sigma / protocol:

  implemented alpha=0, sigma=0.6, naive-myopic, CE beliefs
                      eps=0      1e-10     1e-2      1e-1
  numericist EPS_CUSP  2.3269    1.8805    2.0827    2.1674   -> INCREASING
  analyst    a13 P1    2.3095    1.4647    1.4127    1.1091   -> DECREASING

They agree at eps = 0 and diverge by 28% at 0+ and by a FACTOR OF 2 at 0.1,
with opposite signs of dG/deps. This script settles it by brute-force forward
simulation of the actual environment (same code path as r2, which reproduced 11
cells of the numericist's grid to 1-8%).
"""
import numpy as np
import pandas as pd
from r2_bruteforce import run_cell

LADDER = [0.0, 1e-10, 1e-6, 1e-4, 1e-3, 4e-3, 1e-2, 4e-2, 0.1, 0.25]

if __name__ == "__main__":
    rows = []
    for fam, ctrl, label in [("implemented", 0.0, "implemented a=0 (lambda=1)"),
                             ("equicorr", 0.8, "equicorr rho=0.8 (lambda=0.447)"),
                             ("equicorr", 0.0, "equicorr rho=0 (lambda=1)")]:
        print(f"\n=== {label} ===", flush=True)
        for ep in LADDER:
            r = run_cell(fam, ctrl, 0.6, ep, bayes=False, D=1500, T=1600, burn=300,
                         seed=4242)
            r["label"] = label
            rows.append(r)
            print(f"  eps={ep:<9.1e} G={r['G_mean']:8.4f} +- {r['G_se']:.4f}",
                  flush=True)
    pd.DataFrame(rows).to_csv("R3_epsladder.csv", index=False)
