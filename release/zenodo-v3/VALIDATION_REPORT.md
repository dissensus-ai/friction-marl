# Zenodo v3 local validation report

**Status:** READY FOR AUTHOR REVIEW AND ZENODO DRAFT CREATION. This is not a
publication claim. No Zenodo API call, upload, draft creation, or publication was
performed during packaging.

- Source commit: `556e27cd7cdd37e320a6a2718495c94a201091fb`
- Prepared and checked: 2026-08-19
- Durable cold-extraction root:
  `/home/purrpower/Resurrexi/projects/papers/papers-official/marl-coordination/release-qa/zenodo-v3-2026-08-19-556e27c/`
- Public state during validation: Zenodo v2 remains the latest public record.

## Candidate artifacts

| File | Bytes | SHA-256 |
|---|---:|---|
| `friction-marl-code-v3.0.0.tar.gz` | 180,893 | `f6c2afeff1aaefd4f33cb22ee20f01f1b63f53ed6110fe907ce696d091263f9f` |
| `friction-marl-data-v3.0.0.tar.gz` | 10,702,334 | `9f881774c2b21f7f85ae07c828bf30db2022dc803970fb76114311c72a034cb5` |
| `CODE_INVENTORY.txt` | 2,657 | `d1496830f114990e6672e9f6be2bfd3ad58993adbe88adcacff44d96ec0896cf` |
| `DATA_INVENTORY.txt` | 55,127 | `594bd1d1c2081a10f909b68a22d48cff38a40d8a09563a1304f7dea143efd57f` |
| `RELEASE_INFO.txt` | 1,477 | `421d6b39ea4eda857bc38e4419563c7f03312a0aa6c0d71cedd64a4e2fa60379` |

`SHA256SUMS` authenticates the files above together with this report. Its own hash
is reported in the external preparation ledger rather than recursively within itself.

## Archive and inventory gates

- Both gzip streams pass `gzip -t`.
- Both tar streams contain only relative paths and contain no symlinks, hard links,
  devices, or `..` traversal components.
- `git get-tar-commit-id` recovers the exact source commit from the code archive.
- Code inventory: 26 regular files, 614,544 uncompressed bytes; every extracted
  file matches its inventory size and SHA-256.
- Data inventory: 442 regular files, 27,259,103 uncompressed bytes, including
  `RELEASE_INFO.txt`; every extracted file matches its inventory size and SHA-256.
- The 441 extracted `results/` files match the live evidence closure byte for byte.
- No `__pycache__`, `.pyc`, or path containing `smoke` occurs in the data archive.
- Required post-fix GPU/CPU factorials, cross-validation and equivalence tables,
  signed-DGP refits, Reviewer3 frozen/feasible/oracle/noise/partial-sharing data and
  scripts, zero-bias completion record, and interrupted-predecessor warning are present.

## Cold-extraction execution gates

The pristine extraction was retained unchanged. The following were run in a second
extraction using the repository's newer local verification environment (Python
3.13.11), not as a claim that this was the archived training environment. The code
README names `requirements.txt` as the training environment of record and identifies
`uv.lock` as a newer verification/development lock.

- Python compilation of the core package, CPU entrypoint, dynamic oracle and
  Reviewer3 analysis/verification scripts: PASS.
- Independent dynamic oracle: 54/54 two-step brute-force cases PASS.
- Partial-sharing endpoint gate: IQL and VDN, `k=0 == separable` and `k=m == shared`: PASS.
- Signed functional-form refit: PASS. Equal-condition OLS reproduces
  `Delta AICc(M1-M4)=22.458719` IQL and `17.704074` VDN; WLS and
  replication-row sensitivity outputs regenerate.
- Frozen/feasible/oracle and paired partial-sharing analysis: PASS, including
  provenance, one-to-one pairing and decomposition assertions.
- Signed-noise analysis: PASS; clean mean penalties regenerate as 2.231679 IQL and
  0.714208 VDN, with rho-by-noise terms 1.559084 and 0.962022.
- Zero-bias stopping analysis: PASS at 30 replications per condition; decision
  remains `extend`, interpreted as inconclusive at the planned maximum.
- Repaired CPU/GPU cross-validation: PASS; paired offset -0.026070, 90% CI
  [-0.065210, 0.013069], equivalent at +/-0.10 but not +/-0.05.
- README quickstart: PASS and produced the documented output family. Four expected
  `RankWarning` messages arise because a one-condition smoke cannot identify a
  plotting regression; they do not affect execution or the archived results.
- The 17 regenerated CSVs that differ bytewise under the newer numerical stack
  match the canonical tables in shape, labels and values (`max_abs_delta =
  1.22e-12`, tolerance `rtol=atol=1e-12`). The pristine checksummed copies were not
  replaced.
- Structured-file read gate: 156 non-empty CSVs, 20 JSON files and 8 NPZ files
  parse successfully; all 5 PDFs pass `qpdf --check`.

Two one-byte CSV placeholders are intentionally empty:
`results/reviewer3_followup/partial_target_support_interactions.csv` and
`partial_target_support_paired_slope_changes.csv`. Only legacy-support partial-sharing
was run, so no paired partial target-support contrast exists. Neither file supports a
reader-paper claim; the applicable paired partial outputs are
`partial_frozen_interactions.csv`, `partial_frozen_shape_tests.csv`,
`partial_outcome_slopes.csv`, and `partial_legacy_paired.csv`.

## Disposition

The package is suitable for creating a new, unpublished Zenodo draft under concept
DOI `10.5281/zenodo.20636467`. Upload and publication remain separate actions. The
metadata must continue to describe the controls as exploratory, the n=2 result as
feasible-correlation and agent-count qualified, separable VDN as a non-detection
rather than equivalence, and the zero-bias decision as inconclusive.
