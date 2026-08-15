"""Assemble the paired partial-sharing sweep from exact endpoint runners.

The endpoint gate proves that ``partial_runner`` at k=0 and k=m is numerically
identical to the standalone separable and shared runners under the same seed.
Accordingly the expensive follow-up runs only k=1,2 and reuses the already scored
frozen endpoints without duplicating training.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shared", nargs="+", required=True)
    ap.add_argument("--separable", nargs="+", required=True)
    ap.add_argument("--middle", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    shared = pd.concat([pd.read_csv(path) for path in args.shared], ignore_index=True)
    separable = pd.concat([pd.read_csv(path) for path in args.separable], ignore_index=True)
    middle = pd.concat([pd.read_csv(path) for path in args.middle], ignore_index=True)
    shared = shared.assign(overlap_k=3, w=1.0, env="partial_endpoint_shared")
    separable = separable.assign(overlap_k=0, w=0.0, env="partial_endpoint_separable")
    required_middle = set(middle["overlap_k"].unique())
    if required_middle != {1, 2}:
        raise ValueError(f"middle files must contain exactly k=1,2; got {required_middle}")

    combined = pd.concat([separable, middle, shared], ignore_index=True, sort=False)
    key = ["target_support", "mode", "overlap_k", "rho", "sigma", "replication"]
    duplicates = combined.duplicated(key, keep=False)
    if duplicates.any():
        raise ValueError("duplicate partial-sharing blocks after endpoint assembly")
    expected = (
        combined["target_support"].nunique()
        * combined["mode"].nunique() * 4 * combined["rho"].nunique()
        * combined["sigma"].nunique() * combined["replication"].nunique()
    )
    if len(combined) != expected:
        raise ValueError(f"incomplete partial grid: {len(combined)} rows, expected {expected}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    combined.sort_values(key).to_csv(args.out, index=False)
    print(f"assembled {len(combined)} rows -> {args.out}")


if __name__ == "__main__":
    main()
