# ICRA 2027 Scientific Paths Design

## Decision

Outdoor SLPF has completed the IROS-revision engineering work and a controlled first pass at row-identity reasoning. The repository is ready for one of two scientifically defensible paths:

1. An evidence-first ICRA submission built around the validated GNSS-aided semantic-wall particle filter.
2. A higher-risk algorithmic submission built around persistent row/corridor hypotheses and delayed discrete selection.

The evidence-first path is the default recommendation because the ICRA 2027 paper deadline is 15 September 2026 and the current manuscript is already nine pages. The algorithmic path is viable only if it passes a hard go/no-go gate by 4 September 2026. A failed algorithmic gate immediately returns the project to the evidence-first path; it does not trigger another parameter sweep.

## Current Evidence

The authoritative controlled matrix is `results/localization_improvement_row_identity/followup_metrics_per_seed.csv` and its aggregate companion. It contains both traversals, seeds 11, 22, and 33, and five variants.

| Result | rh_run1 | rh_run2 | Interpretation |
| --- | ---: | ---: | --- |
| Baseline aligned APE mean | 0.970 m | 0.875 m | `alpha_huber3_cap50` remains the reference. |
| Delayed-correction aligned APE mean | 0.914 m | 0.905 m | Better rh_run1 APE, but no clean robustness win. |
| Row-mixture+delayed aligned APE mean | 0.959 m | 0.883 m | Near-baseline rh_run2 APE and improved row metrics, but seed-dependent rh_run1 effects. |
| Baseline in-row wrong duration mean | 214.7 s | 253.3 s | Primary row-identity reference after metric correction. |
| Row-mixture+delayed in-row wrong duration mean | 208.0 s | 200.0 s | Small rh_run1 and substantial rh_run2 improvement. |
| Baseline in-row cross-track mean | 1.039 m | 1.182 m | Primary corridor-adherence reference. |
| Row-mixture+delayed in-row cross-track mean | 1.103 m | 1.066 m | Regresses on rh_run1 and improves rh_run2. |

No existing candidate is accepted as the new method. The previous report rejected all candidates using total wrong-row duration. That metric includes headland frames, even though the evaluator assigns each headland pose to the nearest row and row identity is not physically unique there. The corrected scientific interpretation is:

- Use `inrow_wrong_row_duration_sec`, `inrow_row_correct_fraction`, and `inrow_cross_track_mean` for row-identity claims.
- Use `headland_cross_track_mean`, transition recovery time, and recovery distance for headland claims.
- Retain total wrong-row duration only as a diagnostic and label its headland ambiguity.

The current top-k likelihood is not a persistent multi-hypothesis estimator. It marginalises row scores within one measurement update. The current delayed-correction buffer does not re-score history; it nudges the current pose toward the selected row and stores diagnostic history. A stronger scientific claim therefore requires a discrete persistent row-state model, not further tuning of the existing top-k temperature or correction gain.

## Shared Scientific Invariants

- The semantic-wall map is fixed during localisation.
- The baseline is `alpha_huber3_cap50`: alpha pose backend, Huber GNSS threshold 3.0, semantic penalty cap 50.
- Both traversals and seeds `11,22,33` are required for final comparisons.
- Raw map-frame APE is reported alongside aligned APE and RPE. Alignment must not hide map-frame failure.
- RTAB RGB and RGBD use start-pose translation and yaw anchoring without scale or whole-trajectory fitting before raw metrics.
- GTSAM remains a negative/smoke-only result and is not promoted without new evidence.
- Detector accuracy is not claimed unless a SemanticBLT validation YAML is supplied and evaluated. Detector limitations remain explicit otherwise.
- No result is promoted from a short smoke run or a single seed.
- Hyperparameters are registered before a full run; the full run is not used as a tuning set.
- Generated reports must name the exact branch, commit, configuration, dataset paths, seeds, commands, and source metric files.

## Path A: Evidence-First Submission

### Scientific claim

Surveyed semantic-wall constraints improve GNSS-aided row-level map localisation and corridor adherence in a repetitive vineyard, while retaining explicit dependence on map quality, landmark visibility, and a globally bounding GNSS signal.

This path does not claim that the current system solves row identity in all headland transitions. It does not promote the experimental row-mixture code. The row-mixture results can appear only as a compact negative or partial ablation if space permits.

### Required evidence

- One canonical six-seed baseline rerun at the frozen commit and configuration.
- Reconciled baseline, GNSS-degradation, RTAB+NoisyGNSS, ablation, runtime, and trajectory artifacts.
- Corrected in-row and headland metric interpretation.
- A claim-to-evidence matrix with no unsupported deployment or guaranteed-recovery language.
- An anonymous, complete paper of at most eight pages, including references, as required by the official ICRA 2027 call.

### Stop condition

This path is complete when every numeric claim is generated from the canonical evidence bundle, the paper compiles to at most eight pages without fatal or material layout errors, anonymisation checks pass, and an independent reviewer can reproduce the tables from recorded commands.

## Path B: Persistent Row-Hypothesis Research

### Scientific claim under test

Persistent discrete row/corridor hypotheses improve worst-case vineyard localisation by preventing premature loss of plausible row modes and allowing a bounded-latency decision once headland evidence becomes informative.

### Estimator design

Use a labelled multi-model particle filter rather than a scalar mixture score:

- Each particle has a `row_id` label.
- Particle weights are evaluated against the labelled row's semantic walls.
- Mode probability is the sum of weights for each row label after applying row-transition and GNSS mode priors.
- Stratified resampling reserves a minimum quota for the top `K=3` row modes so one high instantaneous score cannot delete all alternatives.
- Row transitions are strongly penalised in-row and permitted only near mapped row endpoints or when semantic confidence is low.
- A lightweight fixed-lag discrete decoder retains per-mode pose estimates and transition back-pointers for `2.0 s`; it emits both causal and lagged trajectories.
- GNSS affects row-mode priors. It does not directly overwrite the continuous pose or bypass semantic evidence.

The map topology is derived once from the fixed semantic-wall map. It contains row IDs, polylines, endpoints, nearest neighbours, and endpoint transition candidates. No online map update is introduced.

### Pre-registered acceptance gate

The candidate is promotable only if all conditions hold on the full six-seed run:

- `rh_run1` seed 11 aligned APE is below 1.20 m.
- `rh_run1` mean aligned APE is below 1.00 m.
- `rh_run2` mean aligned APE is no more than 0.01 m above its paired baseline; the 0.01 m allowance is numerical tolerance, not a performance margin.
- Mean in-row wrong-row duration decreases on both traversals and by at least 10% when the two traversals are pooled.
- Mean in-row cross-track error does not regress on either traversal and decreases by at least 5% when pooled.
- No individual seed has aligned APE more than 0.10 m worse than its paired baseline.
- The lagged estimate uses at most 2.0 s of latency, and causal metrics are reported separately.
- Runtime p95 is at most 1.5 times the baseline and the effective processing rate remains at or above the input rate.

### Go/no-go schedule

- By 2 September: unit tests and a synthetic aliasing test pass.
- By 3 September: `rh_run1` seed 11 targeted run passes the APE and in-row row-metric gates.
- By 4 September: a two-traversal, one-seed screen passes with no severe regression.
- If either targeted gate fails, stop Path B and execute Path A.
- If both gates pass, complete the six-seed matrix by 6 September and freeze the method. No second architecture iteration occurs before submission.

## Publication Constraints

The official ICRA 2027 call states that the complete submission, including references, is limited to eight pages, uses double-anonymous review, and is due 15 September 2026 at 11:59 PST. The existing `paper/paper.pdf` is nine pages and the current source contains author identities, so page reduction and anonymisation are mandatory in both paths.

Official source: https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/

## Decision Record

The repository should preserve both plans, but only one path may own the paper-facing result at submission time. Path B is an experiment with a deadline-bound fallback. Path A is the reliable submission route and remains executable even if Path B produces a scientifically useful negative result.
