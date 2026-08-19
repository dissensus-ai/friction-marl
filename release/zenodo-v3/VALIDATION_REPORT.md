# Zenodo v3 local validation report

**Status:** READY FOR AUTHOR REVIEW AND ZENODO DRAFT CREATION. This is not a
publication claim. No Zenodo API call, upload, draft creation, or publication was
performed during packaging.

- Source commit: `026594350a7821b143a5be874a0dca9a2651da4d`
- Prepared and checked: 2026-08-19
- Durable cold-extraction root:
  `/home/purrpower/Resurrexi/projects/papers/papers-official/marl-coordination/release-qa/zenodo-v3-2026-08-19-0265943/`
- Public state during validation: Zenodo v2 remains the latest public record.

## Candidate artifacts

| File | Bytes | SHA-256 |
|---|---:|---|
| `friction-marl-code-v3.0.0.tar.gz` | 178,773 | `a40553b22c463a0412f8b2d2235d3a7f011ec46230e32af6ce1d894384f848b8` |
| `friction-marl-data-v3.0.0.tar.gz` | 10,604,781 | `5a9be6c56757751a892f54fd6cc560bb1b09eeed37296cf46086d53f4ba82c8e` |
| `CODE_INVENTORY.txt` | 2,657 | `dfe60296b0397746d9360dc9f0cbeeb0df0a2386587804c0b0a972b5db3f30d9` |
| `DATA_INVENTORY.txt` | 55,215 | `96f8f14cdb26b08c3753f2544715d90d6ca1f24f12b860a805b505569af5887e` |
| `RELEASE_INFO.txt` | 1,477 | `5c407e70b991e44a98c5706f33953a8841b313671cf3b867e7f1386b09a3feef` |

`SHA256SUMS` authenticates the files above together with this report. Its own hash
is reported in the external preparation ledger rather than recursively within itself.

## Archive and inventory gates

- Both gzip streams pass `gzip -t`.
- Both tar streams contain only paths below their declared archive root and contain no symlinks,
  hard links, devices, absolute paths, or `..` traversal components.
- `git get-tar-commit-id` recovers the exact source commit from the code archive.
- A second deterministic build of each archive is byte-identical to the candidate.
- Code inventory: 26 regular files, 614,590 uncompressed bytes; every extracted
  file matches its inventory size and SHA-256.
- Data inventory: 443 regular files, 27,266,975 uncompressed bytes, including
  `RELEASE_INFO.txt`; every extracted file matches its inventory size and SHA-256.
- The 442 extracted `results/` files match the live evidence closure byte for byte.
- No `__pycache__`, `.pyc`, or path containing `smoke` occurs in the data archive.
- Required post-fix GPU/CPU factorials, cross-validation and equivalence tables,
  signed-DGP refits, Reviewer3 frozen/feasible/oracle/noise/partial-sharing data and
  scripts, zero-bias completion record, and interrupted-predecessor warning are present.

## Final text-only refresh gates

The full numerical cold-extraction gate was completed at source boundary
`556e27cd7cdd37e320a6a2718495c94a201091fb` and is retained at
`marl-coordination/release-qa/zenodo-v3-2026-08-19-556e27c/`. Relative to that
validated extraction, the final source boundary changes only the root `README.md`,
adds `results/README.md`, and changes 16 result-side prose, licence, LaTeX-caption,
or Python-module-docstring files plus `RELEASE_INFO.txt`. All other code and data
files are byte-identical. No numerical CSV, JSON, NPZ, PDF, or executable statement
changed, so heavy numerical gates were not rerun.

The final pristine extraction was retained unchanged. Targeted checks at the new
source boundary found:

- Python compilation of the four result scripts receiving historical-scope
  docstrings: PASS. Their abstract syntax trees, after removing the module
  docstrings, are identical to the previously validated versions.
- Repository framing and data-licence wording: PASS. The archive names the work as
  an artifact-control battery, identifies the candidate ratio as the original test
  object, and uses the current `Dissensus` attribution.
- Historical-provenance guard gate: PASS for all 15 deliberately retained
  interpretive reports, generators, and LaTeX surfaces receiving immediate scope
  warnings. The new `results/README.md` provides the archive-level authority map.
- README cross-archive relative links: PASS.
- Required evidence path gate: PASS for 17 post-fix, functional-form, Reviewer3,
  zero-bias, and interrupted-predecessor records.

The prior numerical gates therefore remain applicable to the byte-identical
computational evidence: independent dynamic-oracle 54/54; partial-sharing endpoints;
functional-form refit; frozen/feasible/oracle and paired analyses; signed noise;
zero-bias stopping; repaired CPU/GPU cross-validation; README quickstart; structured
files; and PDF integrity. The prior 17 regenerated CSV comparisons agreed with the
frozen tables to `max_abs_delta = 1.212e-12` under the newer development numerical
stack; the frozen checksummed files remain unchanged.

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
