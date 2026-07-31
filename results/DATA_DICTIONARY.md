# Data dictionary — column semantics across results/

Written 31 Jul 2026 for the Zenodo v2 deposit, after a review caught the `eval_gap`
name collision. **Two runner families define `eval_gap` incompatibly.** Committed CSVs
are not rewritten; this file is the disambiguation, and new runs of the b_vdn family
additionally write the unambiguous `eval_reward`.

## `eval_gap` — THE COLLISION

| producer | files | definition |
|---|---|---|
| `run_gpu.py` | `gpu_factorial_postfix/replication_results.csv` (and any future run_gpu output) | **true gap**: `mean(final training reward) − mean(eval_reward_clean)` — positive when training reward overstates held-out performance |
| `results/review_upgrade/b_vdn_runner.py` + everything built on it (`adversarial_rerun/adv_runner.py`, `mode_comparison_postfix/*.csv`, `b1_budget_sweep/`) | `adv_equicorr.csv`, `adv_team_opp.csv`, `modes_iql_vdn_mappo.csv`, `b1_budget_sweep.csv` | **negated held-out eval reward**: `−mean(eval_reward)`. NOT a gap. Use `eval_reward = −eval_gap` (newer files carry `eval_reward` directly). |

Readers should prefer `eval_reward` when present, else negate the b_vdn-family
`eval_gap` (this is what `mode_comparison_postfix/cluster_robust.py` does).

## Other shared columns

- `mean_reward` — mean over agents of final-window **training** reward (window =
  last `LEARNING_CURVE_WINDOW` episodes). The paper's training-gap DV is its negation.
- `eval_reward_clean` (run_gpu) — frozen-policy held-out eval, **noise-free** obs.
- `eval_reward_matched` (run_gpu) — frozen-policy held-out eval under matched
  observation noise at the condition's ε.
- `alpha` — control knob of the DGP the file was produced under. In the factorial and
  modes files this is the **|α|-magnitude** family (sign-blind by design; see paper
  artifact #2). In `adv_equicorr.csv` the knob column is `rho` (genuinely signed
  pairwise correlation); in `adv_team_opp.csv` it is `opp` (opposition strength in
  [0,1], enters as ±opp²; sign meaningless).
- `replication` — replication index. Seeds are drawn **per condition**
  (`cond_seeds`), so `replication` carries no shared component across conditions
  (measured ICC ≈ 0); do not cluster on it expecting a correction.
- `convergence_time` — episodes to convergence heuristic; near-degenerate at long
  horizons (mostly pinned at ceiling), treat as descriptive only.

## Per-directory provenance

- `gpu_factorial_postfix/` — 5×5×5 factorial re-run at HEAD after the obs-noise-units
  and replay-leak fixes (29–30 Jul); the paper's post-fix η² numbers derive here.
- `mode_comparison_postfix/` — IQL/VDN/MAPPO on the magnitude DGP + the two
  adversarial arms (equicorr signed ρ; team opp ∈ [0,1] re-run on the correct grid).
- `b1_budget_sweep/` — committed replacement (31 Jul, fresh seeds) for the
  session-scratchpad MAPPO budget check; supersedes the 29–30 Jul session-record
  table, whose CSVs were never committed and whose DGP label was unrecorded.
- `iso_f_fan/` — analysis-only: equal-F ties in the post-fix factorial.
- `equivalence/` — TOST bound on the ε effect (run_tost.py) over the post-fix factorial.
- `regret_closed_form/` — the 30 Jul exact-benchmark derivation (analyst /
  numericist / referee); see `REFEREE_VERDICT.md` there for scope conditions before
  quoting anything from it.
