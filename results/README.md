# Results archive: authority and historical provenance

This directory deliberately preserves both the evidence used by the current paper and the
earlier artifacts that motivated its control battery. Presence in the archive does **not** make
every generated interpretation current.

For current claims, start with the repository `README.md`, `MARL_FINDINGS.md`, and these result
surfaces:

- `adversarial_rerun/` for the genuinely signed target-coordinate design and functional-form refit;
- `partial_sharing/FINDINGS.md` together with `reviewer3_followup/` for overlap, frozen-policy,
  feasible-target, dynamic-oracle, signed-noise, and paired analyses; the older
  `partial_sharing/ANALYSIS.txt` is not the current paired verdict;
- `reviewer3_controls/` for the separable and two-agent controls, subject to the qualifications in
  its `FINDINGS.md`;
- `gpu_factorial_postfix/`, `full_factorial_postfix/`, and `zero_bias_eps0/` for the repaired
  factorial, cross-platform comparison, and completed bias-initialization diagnostic.

The following are retained as **historical failure provenance**, not as current inferential
reports: the top-level `CROSS_VALIDATION_REPORT.md`; `gpu_factorial/`; `full_factorial/`;
`cross_validation/`; and the U-shape output and generators in `review_upgrade/B_SUMMARY.txt`,
`review_upgrade/b_analyze.py`, and `review_upgrade/b_vdn_runner.py`. In those artifacts, the
legacy sampler did not deliver negative target correlation, the raw sigma effect includes
degree-one reward scaling, and several construct labels and theory-validation conclusions were
later withdrawn. The legacy CPU/GPU reports also predate the repaired implementation and the
current paired equivalence analysis.

Within `review2_jun27/`, `RJOINT_FINDINGS.md` and `rjoint_recompute.py` use a static single-state
benchmark and cannot identify a pure learning shortfall; the current finite-horizon decomposition
is in `reviewer3_followup/`. `CATEGORICAL_FINDINGS.md` is an older unpaired, legacy-support
partial-sharing analysis. `AICC_CLUSTER_FINDINGS.md` remains a usable legacy-support robustness
record subject to its stated scope.

Do not cite a historical report's prose as the current verdict. Its numerical files are retained
so the failure and subsequent corrections remain auditable.
