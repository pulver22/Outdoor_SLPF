# Evidence-First ICRA Submission Report

## Provenance

Canonical evidence manifest: `results/icra_submission/evidence/manifest.json`. Git commit: `3b0a8dc9ce16b1b56f3b11200c39d90442997174`. Configuration: `configs/icra/alpha_huber3_cap50.yaml`.
Evidence status: `controlled_matrix_fallback`; fresh rerun: `False`.

## Baseline Evidence

| traversal | method | ape_raw_rmse | ape_align_rmse | inrow_cross_track | inrow_wrong_sec |
| --- | --- | --- | --- | --- | --- |
| rh_run1 | baseline_alpha_huber3_cap50 | 1.010 | 0.970 | 1.039 | 214.667 |
| rh_run2 | baseline_alpha_huber3_cap50 | 0.949 | 0.875 | 1.182 | 253.333 |
| rh_run1 | delayed_correction | 0.956 | 0.914 | 1.078 | 213.333 |
| rh_run2 | delayed_correction | 0.995 | 0.905 | 1.138 | 246.667 |
| rh_run1 | row_mixture | 1.159 | 1.118 | 1.153 | 245.333 |
| rh_run2 | row_mixture | 1.029 | 0.929 | 1.114 | 216.000 |
| rh_run1 | row_mixture_delayed | 0.996 | 0.959 | 1.103 | 208.000 |
| rh_run2 | row_mixture_delayed | 0.979 | 0.883 | 1.066 | 200.000 |
| rh_run1 | row_mixture_delayed_gnss_gating | 1.031 | 0.980 | 1.030 | 220.000 |
| rh_run2 | row_mixture_delayed_gnss_gating | 0.952 | 0.885 | 1.084 | 205.333 |

Row-identity acceptance uses in-row frames. Headland frames are evaluated with cross-track and transition-recovery metrics because nearest-row identity is ambiguous outside a corridor. Total wrong-row duration is retained only as an all-frame diagnostic.

## Claim Boundary

Detector accuracy is not claimed without a supplied SemanticBLT validation YAML. GTSAM and row-mixture/delayed-correction variants are not promoted as paper-facing methods.

## Source Hashes

| path | sha256 |
| --- | --- |
| results/localization_improvement_row_identity/followup_metrics_per_seed.csv | 2eb5409dfc52d9263c0c1005ee9f56803e20ee8f7938eea24c5fc9c3bc3c85d3 |
