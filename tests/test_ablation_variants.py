from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from scripts import run_spfpp_ablation


def test_ablation_variants_include_new_localisation_improvement_suite() -> None:
    expected = [
        "full",
        "no_semantic_walls",
        "point_only_semantics",
        "trunks_only",
        "poles_only",
        "no_background",
        "no_corridor",
        "no_gnss",
        "static_gnss_weight",
        "no_pose_smoothing",
        "pf_only",
        "pf_fixed_lag_smoother",
    ]

    assert run_spfpp_ablation.VARIANT_ORDER == expected
    assert run_spfpp_ablation.VARIANT_ARGS["pf_fixed_lag_smoother"] == ["--pose-backend", "fixed-lag"]


def test_ablation_command_routes_data_and_robust_localisation_options() -> None:
    args = SimpleNamespace(
        miss_penalty=4.0,
        wrong_hit_penalty=4.0,
        gps_weight=0.5,
        data_path=Path("/data/rh_run2"),
        geojson=Path("/maps/rows.geojson"),
        frame_stride=4,
        semantic_sigma=0.05,
        gps_sigma=1.1,
        corridor_weight=0.3,
        corridor_dist_sigma=1.5,
        corridor_heading_sigma=0.35,
        background_class_weight=0.2,
        max_background_obs=120,
        expected_obs_count=150.0,
        pose_smooth_alpha_pos=0.55,
        pose_smooth_alpha_theta=0.5,
        odom_yaw_filter_alpha=0.9,
        particle_count=100,
        point_ang_sigma=0.08,
        point_range_sigma=0.35,
        point_ang_gate=0.2,
        point_max_range_diff=1.5,
        pose_backend="alpha",
        fixed_lag_window=8,
        gnss_robust_mode="huber",
        gnss_outlier_threshold=3.0,
        semantic_penalty_cap=20.0,
        log_particle_cloud_every=0,
        diagnostics_level="standard",
        max_frames=40,
        require_cuda=True,
    )

    cmd = run_spfpp_ablation.build_spf_command(
        args,
        python_exec=Path("/venv/bin/python"),
        spf_script=Path("/repo/scripts/spf_lidar.py"),
        seed_dir=Path("/runs/full/seed_22"),
        seed=22,
        variant="pf_fixed_lag_smoother",
    )

    assert cmd[0] == "/venv/bin/python"
    assert "--data-path" in cmd
    assert cmd[cmd.index("--data-path") + 1] == "/data/rh_run2"
    assert "--geojson-path" in cmd
    assert cmd[cmd.index("--geojson-path") + 1] == "/maps/rows.geojson"
    assert cmd[cmd.index("--gnss-robust-mode") + 1] == "huber"
    assert cmd[cmd.index("--gnss-outlier-threshold") + 1] == "3.0"
    assert cmd[cmd.index("--semantic-penalty-cap") + 1] == "20.0"
    assert cmd[-2:] == ["--pose-backend", "fixed-lag"]
    assert "--require-cuda" in cmd
