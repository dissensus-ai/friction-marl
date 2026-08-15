# Zero-bias residual diagnostic (epsilon = 0)

**Frozen before inspecting zero-bias outcomes:** 8 August 2026

## Question

The repaired CPU arm retains a grid-mean offset of -0.166 relative to the GPU arm
when `epsilon = 0`. At that stratum the decoder is the only repaired defect that
is active. The leading remaining implementation difference is bias
initialization: all GPU-network biases are initialized to zero, whereas the CPU
arm of record uses PyTorch's default `nn.Linear` bias initialization.

This diagnostic changes only that initialization by passing `--zero-bias` to the
CPU runner.

## Design

- Grid: the 25 `(alpha, sigma)` cells at `epsilon = 0` from the original
  5 x 5 x 5 factorial.
- Training: 1,000 episodes, four agents, three resources, otherwise the repaired
  CPU arm of record.
- Seeds: seed 42, assigned on the original 125-condition x 30-replication
  full-grid schedule. Restricted runs must use `--seed-grid-replications 30` so
  replication `r` in each epsilon-zero cell matches replication `r` in
  `results/full_factorial_postfix/replication_results.csv`.
- Initial stage: 5 replications per cell (125 runs).
- Extensions if inconclusive: 10, then 20, then at most 30 replications per cell.
  Each extension resumes the same output directory and adds the next matched
  replications; it does not replace the earlier stage.

## Estimands and decision rule

The primary estimand is the grid-mean condition-level **mean-reward** offset
(higher is better) between the zero-bias CPU arm and the post-fix GPU arm over
the 25 epsilon-zero conditions. This is the scale on which the existing offset
is -0.166; the corresponding reward-gap offset is +0.166.
The secondary estimand is the paired change from the repaired default-bias CPU
arm to the zero-bias CPU arm on the matched replication rows.

At each planned stage, report the point estimates and 95% bootstrap confidence
intervals over conditions. The zero-bias explanation is supported only if:

1. the new CPU-minus-GPU grid-mean offset is contained within the manuscript's
   existing +/-0.10 equivalence margin; and
2. the paired initialization change moves the CPU arm toward the GPU arm rather
   than merely adding variance.

It is refuted if the new offset remains outside +/-0.10 with a confidence
interval excluding the margin, or if the paired change moves in the wrong
direction. Otherwise the run extends to the next planned stage. No other
hyperparameter or analysis choice changes between stages.

`analyze.py` writes the latest result to `analysis/summary.json` and also preserves
stage-numbered summaries and paired tables (for example, `summary_stage_05.json`) so
the sequential stopping trail remains auditable after an extension.

**Post-stage-5 implementation note:** after the frozen rule returned `extend`, the
analyzer's output serialization was amended to retain those stage-numbered copies.
No estimand, bootstrap setting, threshold, condition, or decision branch changed;
the stage-5 result was rerun unchanged from the still-complete 125-row file and saved
as `summary_stage_05.json` before the extension reached its first checkpoint.

## Command

```bash
python run_parallel.py \
  --n-replications 5 \
  --n-episodes 1000 \
  --epsilons 0 \
  --zero-bias \
  --seed-grid-replications 30 \
  --workers 22 \
  --output-dir results/zero_bias_eps0
```

## Outcome

The frozen rule was evaluated after each complete planned stage. Estimates are
CPU-minus-GPU mean reward for the zero-bias arm (primary), and zero-bias minus
default-bias CPU mean reward on matched rows (paired change; positive is an
improvement because higher reward is better).

| Reps/cell | Runs | Primary offset (95% CI) | Paired change (95% CI) | Matched default offset | Rule output |
|---:|---:|---:|---:|---:|:---|
| 5 | 125 | -0.1566 [-0.3464, 0.0342] | 0.0192 [-0.0099, 0.0483] | -0.1757 | extend |
| 10 | 250 | -0.0818 [-0.1723, 0.0082] | 0.0190 [0.0012, 0.0368] | -0.1008 | extend |
| 20 | 500 | -0.1228 [-0.2167, -0.0417] | 0.0333 [0.0234, 0.0437] | -0.1562 | extend |
| 30 | 750 | -0.1246 [-0.2322, -0.0323] | 0.0412 [0.0315, 0.0514] | -0.1658 | extend |

At every stage the initialization change moved the CPU arm toward the GPU arm,
and from 10 replications onward the paired improvement excluded zero. At the
30-replication cap, however, the primary interval was neither contained within
the +/-0.10 equivalence margin nor wholly outside it. The operational verdict is
therefore **inconclusive at the planned maximum**: zero initialization is a
real, directionally helpful implementation difference, but this diagnostic does
not establish that it fully explains the epsilon-zero residual. No run beyond
30 replications per cell is authorized by this protocol.
