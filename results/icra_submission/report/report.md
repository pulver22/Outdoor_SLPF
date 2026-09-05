# Evidence-First ICRA Submission Report

## Provenance

Canonical evidence manifest: `results/icra_submission/evidence/manifest.json`. Git commit: `fb76bafc82399bb5b10e6a60a1d4ad16659b2676`. Configuration: `configs/icra/alpha_huber3_cap50.yaml`.
Evidence status: `canonical`; fresh rerun: `True`.

## Baseline Evidence

| traversal | method | ape_raw_rmse | ape_align_rmse | inrow_cross_track | inrow_wrong_sec |
| --- | --- | --- | --- | --- | --- |
| rh_run1 | AMCL | 1.366 | 1.016 | 1.404 | n/a |
| rh_run2 | AMCL | 3.500 | 2.043 | 1.551 | n/a |
| rh_run1 | AMCL + NoisyGNSS | 1.329 | 0.990 | 1.389 | n/a |
| rh_run2 | AMCL + NoisyGNSS | 3.381 | 1.976 | 1.512 | n/a |
| rh_run1 | NoisyGNSS | 3.036 | 2.978 | 1.771 | n/a |
| rh_run2 | NoisyGNSS | 3.162 | 3.088 | 1.990 | n/a |
| rh_run1 | RTAB (RGB) + NoisyGNSS | 37.777 | 6.279 | 5.350 | n/a |
| rh_run2 | RTAB (RGB) + NoisyGNSS | 41.963 | 8.334 | 6.159 | n/a |
| rh_run1 | RTAB (RGB-D) + NoisyGNSS | 39.205 | 6.316 | 5.374 | n/a |
| rh_run2 | RTAB (RGB-D) + NoisyGNSS | 59.911 | 8.367 | 6.605 | n/a |
| rh_run1 | RTAB-Map (RGB) | 59.616 | 6.682 | 6.207 | n/a |
| rh_run2 | RTAB-Map (RGB) | 85.948 | 9.118 | 6.806 | n/a |
| rh_run1 | RTAB-Map (RGB-D) | 61.333 | 10.025 | 6.780 | n/a |
| rh_run2 | RTAB-Map (RGB-D) | 87.169 | 9.063 | 7.247 | n/a |
| rh_run1 | SLPF (ours) | 0.942 | 0.892 | 1.082 | 224.000 |
| rh_run2 | SLPF (ours) | 0.984 | 0.911 | 1.173 | 250.667 |

Row-identity acceptance uses in-row frames. Headland frames are evaluated with cross-track and transition-recovery metrics because nearest-row identity is ambiguous outside a corridor. Total wrong-row duration is retained only as an all-frame diagnostic.

## Claim Boundary

Detector accuracy is not claimed without a supplied SemanticBLT validation YAML. GTSAM and row-mixture/delayed-correction variants are not promoted as paper-facing methods.

## Statistical Summary

The machine-readable summary uses seed-level paired effects only for genuinely rerun methods. Dedicated AMCL/RTAB-Map rows are treated as fixed references without p-values. Paired comparison records: `586`; fixed-reference records: `328`; all `n=3` results are descriptive.

## Source Hashes

| path | sha256 |
| --- | --- |
| results/icra_submission/reruns/localization/followup_metrics_per_seed.csv | e80122a6cd4f93983e3a73dc92685465b9fffd3e19ea0dc526e0e0483ee5f633 |
| results/icra_submission/evidence/canonical_baseline_metrics_per_seed.csv | 36ba9dd10660818fe1bf47dccf94e0ccc9fff683ae7e1d63fe0b83d0e6209a40 |
| results/icra_submission/reruns/gnss_stress_full_v2/localization_metrics_aggregate.csv | cb7df24be4f698e99e78fde35546d4708e2610ef958d92b47b3367500e0bf040 |
| results/icra_submission/reruns/ablation/rh_run1/20260902_125857_ablation/ablation_metrics_aggregate.csv | ef11a24fe5ce867fdaf8f88d8dbfb3ca66a2d2ce183436a7f1320970937827df |
| results/icra_submission/reruns/robustness/robustness_metrics_aggregate.csv | 6b46e9a21c4b9e35e3aad2cf1ac8fa126755d5df2b3610f51cceb9ccbe6e13b0 |
| results/icra_submission/reruns/localization/followup_protocol_full.json | 96c9a11443913a8564874365f71d75e87e9aaf71144e559724634db23be37243 |
| results/icra_submission/reruns/ablation/rh_run1/20260902_125857_ablation/ablation_protocol.json | 1a4eaf3e64ff2a980d96dfc2359a62472b4ed58f17825a697ac009ab402d7447 |
| results/icra_submission/reruns/ablation/rh_run2/20260902_131924_ablation/ablation_protocol.json | b762e6bee9e1c103e79c6a37171e6062cb24922da3588cfeb9cf89d27e37201a |
| results/icra_submission/reruns/robustness/rh_run1/20260902_134037_run1_robustness_gpu/run_protocol.json | 93296fb0ef153a2d1b98e14e1c91318c862b4e02ab5017550d86464a3471f1f5 |
| results/icra_submission/reruns/robustness/rh_run2/20260902_135644_run1_robustness_gpu/run_protocol.json | e9ba4c228ea5733cf2e01d9bd3c35e05c8578b3b64ffb04f7c125110c166865e |
| results/icra_submission/evidence/semanticblt_roboflow_metadata.json | bb3ae8c9c8b50c811e8b18e1cc26cceaa5dff5a14bdb1a872a1904a1d8415259 |
| results/icra_submission/reruns/statistics/icra_statistics_summary.json | 36b144169b18bbb98fb5fdbac55162c110dee5758dbbc6099181400711ed7dc2 |
