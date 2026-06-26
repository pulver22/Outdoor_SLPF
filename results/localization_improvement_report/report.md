# Outdoor SLPF Localisation Improvement Report

## Summary
Branch `iros_revision_plan` at commit `fc03230`. This report captures the current localisation improvement state, the diagnostic artifacts generated for the rh_run1 seed-11 failure, and the follow-up experiment decision tree for external review.

## Changes Implemented
- Added diagnostics that compare traversal/seed metrics with frame-level GNSS innovation, ESS, max weight, semantic-hit, and smoother-status signals.
- Added a fixed candidate matrix runner for the accepted alpha setting, robust-loss/cap variants, no-smoothing, fixed-lag, and optional GTSAM smoke.
- Added this Markdown plus JSON report for sharing with ChatGPT or parsing programmatically.

## Current Metric Summary
| traversal | mean_aligned_ape_m | seed11 | seed22 | seed33 |
| --- | --- | --- | --- | --- |
| rh_run2 | 0.843 | 0.793 | 0.796 | 0.942 |
| rh_run1 | 1.300 | 1.993 | 1.057 | 0.849 |

## Diagnostics
| traversal | candidate_id | seed | ape_align_rmse | cross_track_mean | row_correct_fraction | wrong_row_duration_sec |
| --- | --- | --- | --- | --- | --- | --- |
| rh_run2 | alpha_huber_cap | 11 | 0.793 | 1.433 | 0.715 | 448.000 |
| rh_run2 | alpha_huber_cap | 22 | 0.796 | 1.483 | 0.697 | 476.000 |
| rh_run2 | alpha_huber_cap | 33 | 0.942 | 1.396 | 0.700 | 472.000 |
| rh_run1 | alpha_huber_cap | 11 | 1.993 | 1.641 | 0.657 | 548.000 |
| rh_run1 | alpha_huber_cap | 22 | 1.057 | 1.457 | 0.682 | 508.000 |
| rh_run1 | alpha_huber_cap | 33 | 0.849 | 1.327 | 0.774 | 360.000 |

## Follow-Up Experiments
| stage | traversal | candidate_id | seed | ape_align_rmse | cross_track_mean | row_correct_fraction | wrong_row_duration_sec |
| --- | --- | --- | --- | --- | --- | --- | --- |
| screen | rh_run1 | accepted_alpha_huber3_cap20 | 11 | 1.031 | 1.298 | 0.742 | 412.000 |
| screen | rh_run1 | alpha_huber5_nocap | 11 | 1.072 | 1.272 | 0.707 | 468.000 |
| screen | rh_run1 | alpha_huber3_cap50 | 11 | 0.832 | 1.336 | 0.714 | 456.000 |
| screen | rh_run1 | alpha_cauchy3_cap20 | 11 | 0.921 | 1.349 | 0.704 | 472.000 |
| screen | rh_run1 | alpha_huber3_cap20_no_smoothing | 11 | 0.919 | 1.431 | 0.704 | 472.000 |
| screen | rh_run1 | fixedlag_huber3_cap20 | 11 | 1.717 | 1.628 | 0.624 | 600.000 |
| full | rh_run1 | alpha_huber3_cap50 | 11 | 0.965 | 1.307 | 0.757 | 388.000 |
| full | rh_run1 | alpha_huber3_cap50 | 22 | 1.046 | 1.358 | 0.722 | 444.000 |
| full | rh_run1 | alpha_huber3_cap50 | 33 | 0.905 | 1.395 | 0.712 | 460.000 |
| full | rh_run1 | alpha_huber3_cap20_no_smoothing | 11 | 0.843 | 1.344 | 0.729 | 432.000 |
| full | rh_run1 | alpha_huber3_cap20_no_smoothing | 22 | 1.212 | 1.457 | 0.687 | 500.000 |
| full | rh_run1 | alpha_huber3_cap20_no_smoothing | 33 | 0.924 | 1.468 | 0.724 | 440.000 |
| full | rh_run2 | alpha_huber3_cap50 | 11 | 0.908 | 1.494 | 0.669 | 520.000 |
| full | rh_run2 | alpha_huber3_cap50 | 22 | 0.893 | 1.551 | 0.649 | 552.000 |
| full | rh_run2 | alpha_huber3_cap50 | 33 | 0.919 | 1.498 | 0.705 | 464.000 |
| full | rh_run2 | alpha_huber3_cap20_no_smoothing | 11 | 0.924 | 1.541 | 0.651 | 548.000 |
| full | rh_run2 | alpha_huber3_cap20_no_smoothing | 22 | 0.904 | 1.428 | 0.715 | 448.000 |
| full | rh_run2 | alpha_huber3_cap20_no_smoothing | 33 | 0.838 | 1.446 | 0.710 | 456.000 |

### Follow-Up Aggregates
| stage | traversal | candidate_id | seed_count | ape_align_rmse_mean | cross_track_mean_mean | row_correct_fraction_mean | wrong_row_duration_sec_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full | rh_run1 | alpha_huber3_cap20_no_smoothing | 3 | 0.993 | 1.423 | 0.713 | 457.333 |
| full | rh_run1 | alpha_huber3_cap50 | 3 | 0.972 | 1.353 | 0.730 | 430.667 |
| full | rh_run2 | alpha_huber3_cap20_no_smoothing | 3 | 0.889 | 1.471 | 0.692 | 484.000 |
| full | rh_run2 | alpha_huber3_cap50 | 3 | 0.907 | 1.514 | 0.674 | 512.000 |
| screen | rh_run1 | accepted_alpha_huber3_cap20 | 1 | 1.031 | 1.298 | 0.742 | 412.000 |
| screen | rh_run1 | alpha_cauchy3_cap20 | 1 | 0.921 | 1.349 | 0.704 | 472.000 |
| screen | rh_run1 | alpha_huber3_cap20_no_smoothing | 1 | 0.919 | 1.431 | 0.704 | 472.000 |
| screen | rh_run1 | alpha_huber3_cap50 | 1 | 0.832 | 1.336 | 0.714 | 456.000 |
| screen | rh_run1 | alpha_huber5_nocap | 1 | 1.072 | 1.272 | 0.707 | 468.000 |
| screen | rh_run1 | fixedlag_huber3_cap20 | 1 | 1.717 | 1.628 | 0.624 | 600.000 |

## Commands
- `MPLCONFIGDIR=/home/pulver/projects/vineyard_pf/Outdoor_SLPF_iros_revision_plan/.tmp_mpl python3 scripts/analyze_localization_followup.py --output-dir results/localization_improvement_report/diagnostics`
- `MPLCONFIGDIR=/home/pulver/projects/vineyard_pf/Outdoor_SLPF_iros_revision_plan/.tmp_mpl CUDA_VISIBLE_DEVICES=0 python3 scripts/run_localization_followup_experiments.py --stage screen --traversals rh_run1 --seeds 11 --max-frames 40 --include-gtsam --output-root results/localization_improvement_followup_smoke_rh1_seed11_40_v3 --cuda-visible-devices 0 --require-cuda`
- `MPLCONFIGDIR=/home/pulver/projects/vineyard_pf/Outdoor_SLPF_iros_revision_plan/.tmp_mpl CUDA_VISIBLE_DEVICES=0 python3 scripts/run_localization_followup_experiments.py --stage screen --traversals rh_run1 --seeds 11 --output-root results/localization_improvement_followup --cuda-visible-devices 0 --require-cuda`
- `MPLCONFIGDIR=/home/pulver/projects/vineyard_pf/Outdoor_SLPF_iros_revision_plan/.tmp_mpl CUDA_VISIBLE_DEVICES=0 python3 scripts/run_localization_followup_experiments.py --stage full --traversals rh_run1,rh_run2 --seeds 11,22,33 --promote-from results/localization_improvement_followup/followup_metrics_per_seed.csv --top-n 2 --output-root results/localization_improvement_followup --cuda-visible-devices 0 --require-cuda`
- `MPLCONFIGDIR=/home/pulver/projects/vineyard_pf/Outdoor_SLPF_iros_revision_plan/.tmp_mpl python3 scripts/generate_localization_improvement_report.py`

## Artifact Paths
- Markdown report: `/home/pulver/projects/vineyard_pf/Outdoor_SLPF_iros_revision_plan/results/localization_improvement_report/report.md`
- JSON data: `/home/pulver/projects/vineyard_pf/Outdoor_SLPF_iros_revision_plan/results/localization_improvement_report/report_data.json`
- Diagnostics CSV: `/home/pulver/projects/vineyard_pf/Outdoor_SLPF_iros_revision_plan/results/localization_improvement_report/diagnostics/diagnostic_metrics.csv`
- Follow-up CSV: `/home/pulver/projects/vineyard_pf/Outdoor_SLPF_iros_revision_plan/results/localization_improvement_followup/followup_metrics_per_seed.csv`
- Follow-up aggregate CSV: `/home/pulver/projects/vineyard_pf/Outdoor_SLPF_iros_revision_plan/results/localization_improvement_followup/followup_metrics_aggregate.csv`

## Failures
- rh_run1 seed 11 is the current generalisation concern unless follow-up results show it below the configured thresholds.
- GTSAM remains smoke-only unless its screened aligned APE beats the alpha baseline.

## Successes
- rh_run2 accepted alpha+Huber+cap result remains the current positive result.
- The workflow now produces repeatable diagnostics, candidate commands, metrics, and report artifacts.

## Interpretation Notes
- Current accepted rh_run2 configuration mean aligned APE is 0.843 m.
- Current rh_run1 mean aligned APE is 1.300 m, with seed 11 at 1.993 m.
- GTSAM smoke completed at 1.108 m aligned APE; it is only promoted if it beats alpha in the screen.
- Best generated follow-up row is alpha_huber3_cap50 on rh_run1 seed 11 at 0.832 m aligned APE.
- Best promoted full configuration by weighted mean is alpha_huber3_cap50 at 0.939 m overall (rh_run1 0.972 m, rh_run2 0.907 m).
- Outcome branch: success.

## Next Experiments
- Failure branch: if rh_run1 seed 11 remains above 1.5 m, diagnose map/GNSS frame consistency, early particle collapse, headland-only error, and wrong-row duration before changing paper claims.
- Success branch: if rh_run1 seed 11 drops below 1.2 m and full rh_run1 mean is below 1.0 m, freeze one config and regenerate full ablation, runtime, trajectory plots, and manuscript tables.
- Partial branch: if rh_run1 improves but remains 1.0-1.5 m, keep rh_run2 accepted, report rh_run1 as a robustness limitation, and run one additional seed-stability check.
