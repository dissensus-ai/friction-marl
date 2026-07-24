"""Equivalence headline: is the cooperation finding algorithm-invariant? (ADDITIVE)

Runs TOST (results.equivalence.tost) on the runner CSVs to test whether switching
the learner (IQL -> VDN / MAPPO) leaves the episode reward EQUIVALENT within an
a-priori margin. The margin is anchored to the COOPERATION EFFECT itself: the
absolute reward difference between the structured-preference cells (|alpha| or |rho|
at its maximum) and the unstructured cell (knob = 0), measured on a reference mode.
The default margin is `--margin-frac` (0.5) of that effect, so "equivalent" means
"any residual algorithm effect is less than half the effect we care about".

Emits `equivalence_headline.csv`: one row per algorithm pair with the mean
difference, the margin, the TOST equivalence p-value, and the verdict.

USAGE
-----
    python -m results.equivalence.run_equivalence \
        --inputs results/review_upgrade/vdn_vs_iql.csv \
        --out results/equivalence/equivalence_headline.csv
"""
from __future__ import annotations

import argparse
import glob as globmod
import os
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from results.equivalence.tost import margin_from_effect, tost_equivalence


def _infer_mode(df: pd.DataFrame, path: str) -> pd.Series:
    if "mode" in df.columns:
        return df["mode"].astype(str)
    base = os.path.basename(path).lower()
    for m in ("mappo", "vdn", "iql"):
        if base.endswith(f"_{m}.csv") or f"_{m}" in base:
            return pd.Series([m] * len(df), index=df.index)
    return pd.Series(["unknown"] * len(df), index=df.index)


def load(paths: list[str], metric: str) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        if metric not in df.columns:
            raise SystemExit(f"metric '{metric}' not in {p}")
        df = df.copy()
        df["mode"] = _infer_mode(df, p)
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    return full[full["mode"] != "unknown"].copy()


def cooperation_anchor(df: pd.DataFrame, metric: str, ref_mode: str | None) -> tuple[float, str]:
    """|mean(metric | structured) - mean(metric | unstructured)| on a reference mode.

    'structured' = the design knob (alpha, else rho) at max |value|; 'unstructured'
    = knob closest to 0. Returns (anchor_effect, description).
    """
    knob = "alpha" if "alpha" in df.columns else ("rho" if "rho" in df.columns else None)
    if knob is None:
        raise SystemExit("no 'alpha' or 'rho' column: cannot anchor the margin to the "
                         "cooperation effect (pass --anchor explicitly)")
    ref = df if (ref_mode in (None, "all")) else df[df["mode"] == ref_mode]
    if ref.empty:
        ref = df
    absknob = ref[knob].abs()
    unstructured = ref[absknob == absknob.min()][metric]
    structured = ref[absknob == absknob.max()][metric]
    if len(unstructured) == 0 or len(structured) == 0:
        raise SystemExit("could not locate structured/unstructured cells for the anchor")
    eff = float(structured.mean() - unstructured.mean())
    desc = (f"{knob}: |{knob}|max={absknob.max():.3g} vs |{knob}|min={absknob.min():.3g} "
            f"on mode={ref_mode or 'all'} ({metric}); "
            f"structured_mean={structured.mean():.4f} unstructured_mean={unstructured.mean():.4f}")
    return eff, desc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="*", default=[], help="explicit CSV files")
    ap.add_argument("--input-dir", type=str, default=None)
    ap.add_argument("--glob", type=str, default="*_iql.csv *_vdn.csv *_mappo.csv")
    ap.add_argument("--metric", type=str, default="mean_reward")
    ap.add_argument("--margin-frac", type=float, default=0.5,
                    help="margin = margin_frac * |cooperation effect| (a-priori)")
    ap.add_argument("--anchor", type=float, default=None,
                    help="override the cooperation-effect anchor with an explicit value")
    ap.add_argument("--ref-mode", type=str, default=None,
                    help="mode used to measure the anchor effect (default: all pooled)")
    ap.add_argument("--baseline", type=str, default="iql",
                    help="reference algorithm each other mode is tested against")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--usevar", type=str, default="unequal", choices=["unequal", "pooled"])
    ap.add_argument("--out", type=str, default="results/equivalence/equivalence_headline.csv")
    args = ap.parse_args()

    paths = list(args.inputs)
    if args.input_dir:
        for pat in args.glob.split():
            paths.extend(sorted(globmod.glob(os.path.join(args.input_dir, pat))))
    paths = sorted(set(paths))
    if not paths:
        raise SystemExit("no input CSVs (pass --inputs and/or --input-dir/--glob)")

    df = load(paths, args.metric)
    modes = sorted(df["mode"].unique())
    print(f"[equivalence] modes={modes} rows={len(df)} metric={args.metric}")

    if args.anchor is not None:
        anchor, anchor_desc = float(args.anchor), f"explicit anchor={args.anchor}"
    else:
        anchor, anchor_desc = cooperation_anchor(df, args.metric, args.ref_mode)
    margin = margin_from_effect(anchor, args.margin_frac)
    print(f"[equivalence] cooperation effect anchor = {anchor:.4f}  ->  margin = "
          f"+/- {margin:.4f}  ({args.margin_frac} x |anchor|)")
    print(f"              anchor basis: {anchor_desc}")

    # pairs: every non-baseline mode vs the baseline, plus all remaining pairs
    pairs: list[tuple[str, str]] = []
    if args.baseline in modes:
        pairs += [(m, args.baseline) for m in modes if m != args.baseline]
    for a, b in combinations(modes, 2):
        if (a, b) not in pairs and (b, a) not in pairs:
            pairs.append((a, b))

    rows = []
    for a, b in pairs:
        xa = df.loc[df["mode"] == a, args.metric].to_numpy(float)
        xb = df.loc[df["mode"] == b, args.metric].to_numpy(float)
        try:
            res = tost_equivalence(xa, xb, margin, alpha=args.alpha, usevar=args.usevar)
        except ValueError as exc:
            print(f"[equivalence] skip {a} vs {b}: {exc}")
            continue
        rec = {"algo_x": a, "algo_y": b, "anchor_effect": anchor, "margin_frac": args.margin_frac,
               **res.as_dict()}
        rows.append(rec)
        verdict = "EQUIVALENT" if res.equivalent else "not shown"
        print(f"  {a:>6} vs {b:<6} diff={res.mean_diff:+.4f} margin=+/-{margin:.4f} "
              f"p_tost={res.p_tost:.4g} -> {verdict}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[equivalence] wrote {out}")


if __name__ == "__main__":
    main()
