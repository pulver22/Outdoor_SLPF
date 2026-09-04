# Evidence-First ICRA Submission Report

## Provenance

Canonical evidence manifest: `results/icra_submission/evidence/manifest.json`. Git commit: `fb574fdeee9b401e54c16deeb8a0b57dfa59c6b9`. Configuration: `configs/icra/alpha_huber3_cap50.yaml`.
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
| results/icra_submission/reruns/gnss_stress_full_v2/localization_metrics_aggregate.csv | aa24b49ee6a28171165f931284d0167cf67c9bcf857ff35b0e92076e37dd3d59 |
| results/icra_submission/reruns/ablation/ablation_metrics_aggregate.csv | e50e7d49db5146da483dd3a0ec24144f9df0e960e323199ddef5e25545856632 |
| results/icra_submission/reruns/robustness/robustness_metrics_aggregate.csv | 6b46e9a21c4b9e35e3aad2cf1ac8fa126755d5df2b3610f51cceb9ccbe6e13b0 |
| results/icra_submission/reruns/localization/followup_protocol_full.json | 96c9a11443913a8564874365f71d75e87e9aaf71144e559724634db23be37243 |
| results/icra_submission/reruns/gnss_stress_full_v2/run_protocol.json | 903f8ff163e29427b6f6e4abf68af30146f33b872332e453f66bae6afb163294 |
| results/icra_submission/reruns/ablation/rh_run1/20260902_125857_ablation/ablation_protocol.json | 1a4eaf3e64ff2a980d96dfc2359a62472b4ed58f17825a697ac009ab402d7447 |
| results/icra_submission/reruns/ablation/rh_run2/20260902_131924_ablation/ablation_protocol.json | b762e6bee9e1c103e79c6a37171e6062cb24922da3588cfeb9cf89d27e37201a |
| results/icra_submission/reruns/robustness/rh_run1/20260902_134037_run1_robustness_gpu/run_protocol.json | 93296fb0ef153a2d1b98e14e1c91318c862b4e02ab5017550d86464a3471f1f5 |
| results/icra_submission/reruns/robustness/rh_run2/20260902_135644_run1_robustness_gpu/run_protocol.json | e9ba4c228ea5733cf2e01d9bd3c35e05c8578b3b64ffb04f7c125110c166865e |
| configs/icra/alpha_huber3_cap50.yaml | 5162b24cb671d506ea112a1c83c1456b6b09429c4eb3902f241b4fbbcd06fdd7 |
| scripts/build_icra_evidence_bundle.py | 058e006cd7d9aea0f16d26e9b3195d4daf088e9249a7924a80095f165f8f7ae2 |
| scripts/merge_icra_matrix_metrics.py | 71ee1e68bb1b13d2af934d87dbafdb73745d9cd78461e4aab0f722201d4b6c09 |
| scripts/summarize_icra_statistics.py | b4856236fdc657dbc1f4d8b49947ae42efd27ee086e799bb5b599fc9c3d726ce |
| scripts/run_localization_followup_experiments.py | a41f9bd8efd9d3acf63165a02c90c97babcc0534e93587d268c0d7a2541a15fe |
| scripts/run_gnss_degradation_localization_eval.py | 81b9be009b220235df26792885a677d9b50ab2bf06e61202db2af2776a8609a3 |
| scripts/run_spfpp_ablation.py | 3e8404e7880989c9a2a6c74c8796594817dc6cca1b13b69393ea97c1229229e4 |
| scripts/run_run1_robustness_experiments.py | 304a0304dd4f6a81b23dd8f12707c438df09226adba5511d88a82e61c0e72b10 |
| scripts/experiment_runtime.py | 326bd877bae7ffed2d10bf424546ed40853cb0becfa7229065b6ba42a2e5fa7e |
| paper/icra_claims_matrix.md | ccdffd1ce2c1e28478c7e7e3b845c122c10317b683dd9f300169bed252ad0bd0 |
| results/icra_submission/reruns/localization/followup_metrics_aggregate.csv | befa45ae38d6333aa749b7afb7f1be4f2ceefcac7682ecb85db9cc19cdec98bb |
| results/icra_submission/reruns/localization/followup_metrics_per_seed.csv | e80122a6cd4f93983e3a73dc92685465b9fffd3e19ea0dc526e0e0483ee5f633 |
| results/icra_submission/reruns/gnss_stress_full_v2/compact_summary.csv | ec4873ef11ea9132b2f7ea3edadf798fae9785c1854e44d6d59467ef8524031e |
| results/icra_submission/reruns/gnss_stress_full_v2/localization_metrics_aggregate.csv | aa24b49ee6a28171165f931284d0167cf67c9bcf857ff35b0e92076e37dd3d59 |
| results/icra_submission/reruns/gnss_stress_full_v2/localization_metrics_per_seed.csv | 87a469c24b5611fa2ef5927c1d77fa105379ee7339539a59f33319ceec7565a4 |
| results/icra_submission/reruns/gnss_stress_full_v2/run_protocol.json | 903f8ff163e29427b6f6e4abf68af30146f33b872332e453f66bae6afb163294 |
| results/icra_submission/reruns/ablation/ablation_metrics_per_seed.csv | 711cea1bed515274a130fb4507f4c63dca8a1be57b352f912010a952f95c1c83 |
| results/icra_submission/reruns/ablation/ablation_metrics_aggregate.csv | e50e7d49db5146da483dd3a0ec24144f9df0e960e323199ddef5e25545856632 |
| results/icra_submission/reruns/ablation/rh_run1/20260902_125857_ablation/ablation_protocol.json | 1a4eaf3e64ff2a980d96dfc2359a62472b4ed58f17825a697ac009ab402d7447 |
| results/icra_submission/reruns/ablation/rh_run2/20260902_131924_ablation/ablation_protocol.json | b762e6bee9e1c103e79c6a37171e6062cb24922da3588cfeb9cf89d27e37201a |
| results/icra_submission/reruns/robustness/robustness_metrics_per_seed.csv | 43f68612043a7485525638a5454bea937c6e53ffef1f7bbc024b0c990da4f764 |
| results/icra_submission/reruns/robustness/robustness_metrics_aggregate.csv | 6b46e9a21c4b9e35e3aad2cf1ac8fa126755d5df2b3610f51cceb9ccbe6e13b0 |
| results/icra_submission/reruns/robustness/option_b_recovery_aggregate.csv | 24deb24f8d073c477f2e425d2952b6e772df928a6ccc724613007184f6874133 |
| results/icra_submission/reruns/robustness/rh_run1/20260902_134037_run1_robustness_gpu/run1_robustness_per_seed.csv | 72a0e5742d0e60526aef016ef672b79f632b5bd058739cf81c725e9f62f5cb08 |
| results/icra_submission/reruns/robustness/rh_run1/20260902_134037_run1_robustness_gpu/run1_robustness_aggregate.csv | 690d93088a7457ff251e76fab9a15ba0a1ff8b3f5c77581ee2813a5632e76421 |
| results/icra_submission/reruns/robustness/rh_run1/20260902_134037_run1_robustness_gpu/option_b_recovery_per_seed.csv | 0c97ad1d1f1095adfa5deff61dd37aa7e28428b4f848a45ea7195f3ac1e4bbd3 |
| results/icra_submission/reruns/robustness/rh_run1/20260902_134037_run1_robustness_gpu/run_protocol.json | 93296fb0ef153a2d1b98e14e1c91318c862b4e02ab5017550d86464a3471f1f5 |
| results/icra_submission/reruns/robustness/rh_run2/20260902_135644_run1_robustness_gpu/run1_robustness_per_seed.csv | 4378e3d50f1281632e235d79de79bcbaf68781b9f615540fec586dc2dc388fdc |
| results/icra_submission/reruns/robustness/rh_run2/20260902_135644_run1_robustness_gpu/run1_robustness_aggregate.csv | 73127eb2deb28cbf5eaef4011422dbfb0ebd10ea9837dd3aa8ef4e9bf9120ca3 |
| results/icra_submission/reruns/robustness/rh_run2/20260902_135644_run1_robustness_gpu/option_b_recovery_per_seed.csv | 9d1ef7ac2dd4908d2e9c7de362beade391381fdb0556df9e169bf64affabdaa3 |
| results/icra_submission/reruns/robustness/rh_run2/20260902_135644_run1_robustness_gpu/run_protocol.json | e9ba4c228ea5733cf2e01d9bd3c35e05c8578b3b64ffb04f7c125110c166865e |
| results/icra_submission/evidence/canonical_baseline_metrics_per_seed.csv | 36ba9dd10660818fe1bf47dccf94e0ccc9fff683ae7e1d63fe0b83d0e6209a40 |
| data/2025/amcl/amcl_3runs_metrics/amcl_metrics_3runs_per_run.csv | 700767df1a7100b22e1974973a8dc136de763766744031900065c6997fee913c |
| data/2025/amcl/amcl_3runs_metrics/amcl_metrics_3runs_aggregate.csv | 914a528e6a228e250b3ef98969050cb82a3927e77a8c233cd19655a45720d1a8 |
| data/2025/amcl/amcl_3runs_metrics/amcl_pose_clean.tum | 384e8d7b4ca1e0122ead7d5fd2d0056f56422347ce799f04e2d64e7a7a2e59be |
| data/2025/amcl/amcl_3runs_metrics/amcl_pose1_2_clean.tum | 9d934d2776e85056f4dc4a2ac9949efdfcaaef065432e1d50db68da369d0d10f |
| data/2025/amcl/amcl_3runs_metrics/amcl_pose1_3_clean.tum | 88e15218d698f74593ab6a67f3a11e6c8a0f90b7f23db96f454ca3c1de35d2be |
| data/2025/amcl/amcl_3runs_metrics/amcl_ngps_fused/amcl_pose_ngps_s11.tum | fa12022a62e405767e056de97f742aaa59efd64c8282b035a5f77145923eade1 |
| data/2025/amcl/amcl_3runs_metrics/amcl_ngps_fused/amcl_pose1_2_ngps_s22.tum | da21578a6cbfb8cdeda251ba4ec5dacdfedd7178bdfdea0301506b1f004ad010 |
| data/2025/amcl/amcl_3runs_metrics/amcl_ngps_fused/amcl_pose1_3_ngps_s33.tum | 73eb9505877e17aae194a7177cc4b1d5a28f53a1ac0038aa342d620c43d42119 |
| results/rtabmap/rgb_run1_3runs/run1/rtabmap/rgb/tum1/rtabmap_rgb_filtered.tum | 0cfc422801ed47ee39d0584630b78b11c5b6bc99eee9163d7b4b291dacb19b39 |
| results/rtabmap/rgb_run1_3runs/run2/rtabmap/rgb/tum1/rtabmap_rgb_filtered.tum | 57286b961d9e6795522357d2b92561632ca8675aefa95621720d25cbd5674c6e |
| results/rtabmap/rgb_run1_3runs/run3/rtabmap/rgb/tum1/rtabmap_rgb_filtered.tum | 7b12129d29038955aef25ed76ee3dad75f554c05ccecdb4f8965784522737599 |
| results/rtabmap/rgbd_run1_3runs/run1/rtabmap/rgbd/tum1/rtabmap_rgbd_filtered.tum | 785691097f63d4d547d23d24e1f9786a10772287facf1a7ee2848a5a73dacaa9 |
| results/rtabmap/rgbd_run1_3runs/run2/rtabmap/rgbd/tum1/rtabmap_rgbd_filtered.tum | 4ba7fb43f9a6c1e0fb1a68476d79e091ed6b995b3f00a4f54bb46da6faa7904a |
| results/rtabmap/rgbd_run1_3runs/run3/rtabmap/rgbd/tum1/rtabmap_rgbd_filtered.tum | bdfc2facaf6f1b4ba6d221cf022ed555f2c094b61d443b4f2a8f9b196ce068fa |
| results/icra_submission/evidence/semanticblt_roboflow_metadata.json | bb3ae8c9c8b50c811e8b18e1cc26cceaa5dff5a14bdb1a872a1904a1d8415259 |
| paper/2603.10847v1.pdf | 1a672b40c383ae1633936b579311825a4262a76b5eb405364328e9f7416f15c4 |
| results/icra_submission/reruns/statistics/icra_statistics_summary.json | 36b144169b18bbb98fb5fdbac55162c110dee5758dbbc6099181400711ed7dc2 |
