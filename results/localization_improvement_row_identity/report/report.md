# Outdoor SLPF Localisation Improvement Report

## Summary
Branch `iros_revision_plan` at commit `3899694`. This report captures the current localisation improvement state, the diagnostic artifacts generated for the rh_run1 seed-11 failure, and the follow-up experiment decision tree for external review.

## Changes Implemented
- Added diagnostics that compare traversal/seed metrics with frame-level GNSS innovation, ESS, max weight, semantic-hit, smoother-status, row-hypothesis, row-switch, and GNSS-row-gating signals.
- Added opt-in SPF switches for top-k row-mixture likelihood, lightweight delayed row correction, and adaptive GNSS row gating while freezing `alpha_huber3_cap50` as the reference baseline.
- Added a controlled row-identity ablation runner matrix for baseline, row mixture, delayed correction, row mixture plus delayed correction, and row mixture plus delayed correction plus GNSS gating.
- Added this Markdown plus JSON report for sharing with ChatGPT or parsing programmatically.

## Current Metric Summary
| traversal | mean_aligned_ape_m | seed11 | seed22 | seed33 |
| --- | --- | --- | --- | --- |
| rh_run2 | 0.843 | 0.793 | 0.796 | 0.942 |
| rh_run1 | 1.300 | 1.993 | 1.057 | 0.849 |

## Diagnostics
| traversal | candidate_id | seed | ape_align_rmse | cross_track_mean | row_correct_fraction | wrong_row_duration_sec |
| --- | --- | --- | --- | --- | --- | --- |
| rh_run1 | baseline_alpha_huber3_cap50 | 11 | 0.980 | 1.311 | 0.754 | 392.000 |
| rh_run1 | baseline_alpha_huber3_cap50 | 22 | 1.130 | 1.275 | 0.719 | 448.000 |
| rh_run1 | baseline_alpha_huber3_cap50 | 33 | 0.800 | 1.360 | 0.737 | 420.000 |
| rh_run1 | row_mixture | 11 | 1.187 | 1.324 | 0.732 | 428.000 |
| rh_run1 | row_mixture | 22 | 1.308 | 1.462 | 0.604 | 632.000 |
| rh_run1 | row_mixture | 33 | 0.859 | 1.366 | 0.774 | 360.000 |
| rh_run1 | delayed_correction | 11 | 0.896 | 1.245 | 0.722 | 444.000 |
| rh_run1 | delayed_correction | 22 | 0.965 | 1.295 | 0.689 | 496.000 |
| rh_run1 | delayed_correction | 33 | 0.881 | 1.296 | 0.732 | 428.000 |
| rh_run1 | row_mixture_delayed | 11 | 1.116 | 1.196 | 0.719 | 448.000 |
| rh_run1 | row_mixture_delayed | 22 | 0.870 | 1.263 | 0.727 | 436.000 |
| rh_run1 | row_mixture_delayed | 33 | 0.890 | 1.397 | 0.724 | 440.000 |
| rh_run1 | row_mixture_delayed_gnss_gating | 11 | 0.938 | 1.235 | 0.722 | 444.000 |
| rh_run1 | row_mixture_delayed_gnss_gating | 22 | 1.041 | 1.230 | 0.714 | 456.000 |
| rh_run1 | row_mixture_delayed_gnss_gating | 33 | 0.963 | 1.266 | 0.694 | 488.000 |
| rh_run2 | baseline_alpha_huber3_cap50 | 11 | 0.846 | 1.519 | 0.639 | 568.000 |
| rh_run2 | baseline_alpha_huber3_cap50 | 22 | 0.850 | 1.439 | 0.707 | 460.000 |
| rh_run2 | baseline_alpha_huber3_cap50 | 33 | 0.930 | 1.433 | 0.674 | 512.000 |
| rh_run2 | row_mixture | 11 | 0.953 | 1.512 | 0.654 | 544.000 |
| rh_run2 | row_mixture | 22 | 0.868 | 1.367 | 0.748 | 396.000 |
| rh_run2 | row_mixture | 33 | 0.966 | 1.412 | 0.707 | 460.000 |
| rh_run2 | delayed_correction | 11 | 0.884 | 1.393 | 0.659 | 536.000 |
| rh_run2 | delayed_correction | 22 | 0.941 | 1.346 | 0.674 | 512.000 |
| rh_run2 | delayed_correction | 33 | 0.888 | 1.363 | 0.687 | 492.000 |
| rh_run2 | row_mixture_delayed | 11 | 0.842 | 1.325 | 0.707 | 460.000 |
| rh_run2 | row_mixture_delayed | 22 | 0.826 | 1.336 | 0.751 | 392.000 |
| rh_run2 | row_mixture_delayed | 33 | 0.981 | 1.290 | 0.712 | 452.000 |
| rh_run2 | row_mixture_delayed_gnss_gating | 11 | 0.881 | 1.332 | 0.700 | 472.000 |
| rh_run2 | row_mixture_delayed_gnss_gating | 22 | 0.830 | 1.307 | 0.756 | 384.000 |
| rh_run2 | row_mixture_delayed_gnss_gating | 33 | 0.943 | 1.359 | 0.695 | 480.000 |

## Row-Identity Diagnostics
| traversal | variant | seed | ape_align_rmse | row_switch_count | row_entropy_mean | row_gap_median | gnss_row_switch_corr | gnss_gate_scale_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rh_run1 | baseline_alpha_huber3_cap50 | 11 | 0.980 | 127 | 0.718 | 0.603 | 0.103 | 1.000 |
| rh_run1 | baseline_alpha_huber3_cap50 | 22 | 1.130 | 153 | 0.730 | 0.530 | 0.135 | 1.000 |
| rh_run1 | baseline_alpha_huber3_cap50 | 33 | 0.800 | 144 | 0.693 | 0.586 | 0.066 | 1.000 |
| rh_run1 | row_mixture | 11 | 1.187 | 127 | 0.752 | 0.383 | 0.045 | 1.000 |
| rh_run1 | row_mixture | 22 | 1.308 | 141 | 0.746 | 0.457 | 0.172 | 1.000 |
| rh_run1 | row_mixture | 33 | 0.859 | 146 | 0.709 | 0.419 | 0.031 | 1.000 |
| rh_run1 | delayed_correction | 11 | 0.896 | 126 | 0.687 | 0.659 | 0.143 | 1.000 |
| rh_run1 | delayed_correction | 22 | 0.965 | 141 | 0.696 | 0.701 | 0.155 | 1.000 |
| rh_run1 | delayed_correction | 33 | 0.881 | 143 | 0.691 | 0.602 | 0.103 | 1.000 |
| rh_run1 | row_mixture_delayed | 11 | 1.116 | 154 | 0.758 | 0.414 | 0.027 | 1.000 |
| rh_run1 | row_mixture_delayed | 22 | 0.870 | 115 | 0.702 | 0.456 | 0.097 | 1.000 |
| rh_run1 | row_mixture_delayed | 33 | 0.890 | 135 | 0.686 | 0.426 | 0.037 | 1.000 |
| rh_run1 | row_mixture_delayed_gnss_gating | 11 | 0.938 | 133 | 0.709 | 0.623 | 0.058 | 1.350 |
| rh_run1 | row_mixture_delayed_gnss_gating | 22 | 1.041 | 129 | 0.691 | 0.688 | 0.133 | 1.350 |
| rh_run1 | row_mixture_delayed_gnss_gating | 33 | 0.963 | 135 | 0.700 | 0.570 | 0.034 | 1.350 |
| rh_run2 | baseline_alpha_huber3_cap50 | 11 | 0.846 | 151 | 0.710 | 0.555 | 0.066 | 1.000 |
| rh_run2 | baseline_alpha_huber3_cap50 | 22 | 0.850 | 152 | 0.705 | 0.627 | 0.110 | 1.000 |
| rh_run2 | baseline_alpha_huber3_cap50 | 33 | 0.930 | 142 | 0.727 | 0.582 | 0.022 | 1.000 |
| rh_run2 | row_mixture | 11 | 0.953 | 146 | 0.727 | 0.417 | 0.022 | 1.000 |
| rh_run2 | row_mixture | 22 | 0.868 | 137 | 0.741 | 0.469 | -0.004 | 1.000 |
| rh_run2 | row_mixture | 33 | 0.966 | 137 | 0.735 | 0.463 | 0.023 | 1.000 |
| rh_run2 | delayed_correction | 11 | 0.884 | 129 | 0.702 | 0.595 | 0.067 | 1.000 |
| rh_run2 | delayed_correction | 22 | 0.941 | 145 | 0.719 | 0.628 | 0.116 | 1.000 |
| rh_run2 | delayed_correction | 33 | 0.888 | 139 | 0.718 | 0.640 | 0.026 | 1.000 |
| rh_run2 | row_mixture_delayed | 11 | 0.842 | 152 | 0.727 | 0.448 | 0.068 | 1.000 |
| rh_run2 | row_mixture_delayed | 22 | 0.826 | 129 | 0.717 | 0.528 | 0.051 | 1.000 |
| rh_run2 | row_mixture_delayed | 33 | 0.981 | 130 | 0.742 | 0.441 | 0.010 | 1.000 |
| rh_run2 | row_mixture_delayed_gnss_gating | 11 | 0.881 | 140 | 0.714 | 0.528 | 0.120 | 1.350 |
| rh_run2 | row_mixture_delayed_gnss_gating | 22 | 0.830 | 148 | 0.703 | 0.576 | 0.080 | 1.350 |
| rh_run2 | row_mixture_delayed_gnss_gating | 33 | 0.943 | 145 | 0.713 | 0.632 | 0.040 | 1.350 |

## Follow-Up Experiments
| stage | traversal | candidate_id | seed | ape_align_rmse | cross_track_mean | row_correct_fraction | wrong_row_duration_sec |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full | rh_run1 | baseline_alpha_huber3_cap50 | 11 | 0.980 | 1.311 | 0.754 | 392.000 |
| full | rh_run1 | baseline_alpha_huber3_cap50 | 22 | 1.130 | 1.275 | 0.719 | 448.000 |
| full | rh_run1 | baseline_alpha_huber3_cap50 | 33 | 0.800 | 1.360 | 0.737 | 420.000 |
| full | rh_run1 | row_mixture | 11 | 1.187 | 1.324 | 0.732 | 428.000 |
| full | rh_run1 | row_mixture | 22 | 1.308 | 1.462 | 0.604 | 632.000 |
| full | rh_run1 | row_mixture | 33 | 0.859 | 1.366 | 0.774 | 360.000 |
| full | rh_run1 | delayed_correction | 11 | 0.896 | 1.245 | 0.722 | 444.000 |
| full | rh_run1 | delayed_correction | 22 | 0.965 | 1.295 | 0.689 | 496.000 |
| full | rh_run1 | delayed_correction | 33 | 0.881 | 1.296 | 0.732 | 428.000 |
| full | rh_run1 | row_mixture_delayed | 11 | 1.116 | 1.196 | 0.719 | 448.000 |
| full | rh_run1 | row_mixture_delayed | 22 | 0.870 | 1.263 | 0.727 | 436.000 |
| full | rh_run1 | row_mixture_delayed | 33 | 0.890 | 1.397 | 0.724 | 440.000 |
| full | rh_run1 | row_mixture_delayed_gnss_gating | 11 | 0.938 | 1.235 | 0.722 | 444.000 |
| full | rh_run1 | row_mixture_delayed_gnss_gating | 22 | 1.041 | 1.230 | 0.714 | 456.000 |
| full | rh_run1 | row_mixture_delayed_gnss_gating | 33 | 0.963 | 1.266 | 0.694 | 488.000 |
| full | rh_run2 | baseline_alpha_huber3_cap50 | 11 | 0.846 | 1.519 | 0.639 | 568.000 |
| full | rh_run2 | baseline_alpha_huber3_cap50 | 22 | 0.850 | 1.439 | 0.707 | 460.000 |
| full | rh_run2 | baseline_alpha_huber3_cap50 | 33 | 0.930 | 1.433 | 0.674 | 512.000 |
| full | rh_run2 | row_mixture | 11 | 0.953 | 1.512 | 0.654 | 544.000 |
| full | rh_run2 | row_mixture | 22 | 0.868 | 1.367 | 0.748 | 396.000 |
| full | rh_run2 | row_mixture | 33 | 0.966 | 1.412 | 0.707 | 460.000 |
| full | rh_run2 | delayed_correction | 11 | 0.884 | 1.393 | 0.659 | 536.000 |
| full | rh_run2 | delayed_correction | 22 | 0.941 | 1.346 | 0.674 | 512.000 |
| full | rh_run2 | delayed_correction | 33 | 0.888 | 1.363 | 0.687 | 492.000 |
| full | rh_run2 | row_mixture_delayed | 11 | 0.842 | 1.325 | 0.707 | 460.000 |
| full | rh_run2 | row_mixture_delayed | 22 | 0.826 | 1.336 | 0.751 | 392.000 |
| full | rh_run2 | row_mixture_delayed | 33 | 0.981 | 1.290 | 0.712 | 452.000 |
| full | rh_run2 | row_mixture_delayed_gnss_gating | 11 | 0.881 | 1.332 | 0.700 | 472.000 |
| full | rh_run2 | row_mixture_delayed_gnss_gating | 22 | 0.830 | 1.307 | 0.756 | 384.000 |
| full | rh_run2 | row_mixture_delayed_gnss_gating | 33 | 0.943 | 1.359 | 0.695 | 480.000 |

### Follow-Up Aggregates
| stage | traversal | candidate_id | seed_count | ape_align_rmse_mean | cross_track_mean_mean | row_correct_fraction_mean | wrong_row_duration_sec_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full | rh_run1 | baseline_alpha_huber3_cap50 | 3 | 0.970 | 1.315 | 0.737 | 420.000 |
| full | rh_run1 | delayed_correction | 3 | 0.914 | 1.278 | 0.714 | 456.000 |
| full | rh_run1 | row_mixture | 3 | 1.118 | 1.384 | 0.703 | 473.333 |
| full | rh_run1 | row_mixture_delayed | 3 | 0.959 | 1.285 | 0.723 | 441.333 |
| full | rh_run1 | row_mixture_delayed_gnss_gating | 3 | 0.980 | 1.244 | 0.710 | 462.667 |
| full | rh_run2 | baseline_alpha_huber3_cap50 | 3 | 0.875 | 1.464 | 0.673 | 513.333 |
| full | rh_run2 | delayed_correction | 3 | 0.905 | 1.367 | 0.673 | 513.333 |
| full | rh_run2 | row_mixture | 3 | 0.929 | 1.430 | 0.703 | 466.667 |
| full | rh_run2 | row_mixture_delayed | 3 | 0.883 | 1.317 | 0.723 | 434.667 |
| full | rh_run2 | row_mixture_delayed_gnss_gating | 3 | 0.885 | 1.333 | 0.717 | 445.333 |

## Controlled Row-Identity Ablation
| stage | traversal | candidate_id | seed_count | ape_align_rmse_mean | cross_track_mean_mean | wrong_row_duration_sec_mean |
| --- | --- | --- | --- | --- | --- | --- |
| full | rh_run1 | baseline_alpha_huber3_cap50 | 3 | 0.970 | 1.315 | 420.000 |
| full | rh_run1 | delayed_correction | 3 | 0.914 | 1.278 | 456.000 |
| full | rh_run1 | row_mixture | 3 | 1.118 | 1.384 | 473.333 |
| full | rh_run1 | row_mixture_delayed | 3 | 0.959 | 1.285 | 441.333 |
| full | rh_run1 | row_mixture_delayed_gnss_gating | 3 | 0.980 | 1.244 | 462.667 |
| full | rh_run2 | baseline_alpha_huber3_cap50 | 3 | 0.875 | 1.464 | 513.333 |
| full | rh_run2 | delayed_correction | 3 | 0.905 | 1.367 | 513.333 |
| full | rh_run2 | row_mixture | 3 | 0.929 | 1.430 | 466.667 |
| full | rh_run2 | row_mixture_delayed | 3 | 0.883 | 1.317 | 434.667 |
| full | rh_run2 | row_mixture_delayed_gnss_gating | 3 | 0.885 | 1.333 | 445.333 |

## Acceptance Check
Outcome: `partial`. Accepted candidate: `none`.

| candidate_id | accepted | rh1_seed11 | rh1_mean | rh2_mean | rh1_wrong_s | rh2_wrong_s | rh1_ct | rh2_ct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| delayed_correction | False | 0.896 | 0.914 | 0.905 | 456.000 | 513.333 | 1.278 | 1.367 |
| row_mixture | False | 1.187 | 1.118 | 0.929 | 473.333 | 466.667 | 1.384 | 1.430 |
| row_mixture_delayed | False | 1.116 | 0.959 | 0.883 | 441.333 | 434.667 | 1.285 | 1.317 |
| row_mixture_delayed_gnss_gating | False | 0.938 | 0.980 | 0.885 | 462.667 | 445.333 | 1.244 | 1.333 |

## Commands
- `/home/pulver/projects/vineyard_pf/Outdoor_SLPF/.venv/bin/python scripts/run_localization_followup_experiments.py --matrix row-identity --stage full --traversals rh_run1,rh_run2 --seeds 11,22,33 --cuda-visible-devices 0 --output-root results/localization_improvement_row_identity`
- `python3 scripts/analyze_localization_followup.py --output-dir results/localization_improvement_row_identity/diagnostics [10 row-identity run specs]`
- `python3 scripts/generate_localization_improvement_report.py --output-dir results/localization_improvement_row_identity/report ...`

## Artifact Paths
- Markdown report: `results/localization_improvement_row_identity/report/report.md`
- JSON data: `results/localization_improvement_row_identity/report/report_data.json`
- Diagnostics CSV: `results/localization_improvement_row_identity/diagnostics/diagnostic_metrics.csv`
- Follow-up CSV: `results/localization_improvement_row_identity/followup_metrics_per_seed.csv`
- Follow-up aggregate CSV: `results/localization_improvement_row_identity/followup_metrics_aggregate.csv`

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
- Best generated follow-up row is baseline_alpha_huber3_cap50 on rh_run1 seed 33 at 0.800 m aligned APE.
- Best promoted full configuration by weighted mean is delayed_correction at 0.909 m overall (rh_run1 0.914 m, rh_run2 0.905 m).
- Outcome branch: success.
- Scientific hypothesis under test: multi-hypothesis row reasoning should reduce worst-case wrong-row recovery errors beyond scalar robust-loss tuning.
- No row-identity candidate passed the full acceptance gate; the current result is partial because APE improvements trade off against rh_run2 regression or wrong-row duration.

## Next Experiments
- Failure branch: if rh_run1 seed 11 remains above 1.5 m, diagnose map/GNSS frame consistency, early particle collapse, headland-only error, and wrong-row duration before changing paper claims.
- Success branch: if rh_run1 seed 11 drops below 1.2 m and full rh_run1 mean is below 1.0 m, freeze one config and regenerate full ablation, runtime, trajectory plots, and manuscript tables.
- Partial branch: if rh_run1 improves but remains 1.0-1.5 m, keep rh_run2 accepted, report rh_run1 as a robustness limitation, and run one additional seed-stability check.
