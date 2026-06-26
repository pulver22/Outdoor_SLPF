from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from scripts import tune_spfpp_optuna


def test_tuning_grid_starts_with_accepted_alpha_huber_cap_candidate() -> None:
    first = tune_spfpp_optuna.grid_candidates(limit=1)[0]

    assert first == {
        "pose_backend": "alpha",
        "fixed_lag_window": 8,
        "gnss_robust_mode": "huber",
        "gnss_outlier_threshold": 3.0,
        "semantic_penalty_cap": 20.0,
    }


def test_tuning_grid_can_include_gtsam_without_making_it_default() -> None:
    without_gtsam = tune_spfpp_optuna.grid_candidates(limit=20, include_gtsam=False)
    with_gtsam = tune_spfpp_optuna.grid_candidates(limit=20, include_gtsam=True)

    assert all(candidate["pose_backend"] != "gtsam" for candidate in without_gtsam)
    assert any(candidate["pose_backend"] == "gtsam" for candidate in with_gtsam)


def test_tuning_command_routes_candidate_options() -> None:
    args = SimpleNamespace(
        python_exec=Path("/venv/bin/python"),
        data_path=Path("/data/rh_run2"),
        geojson=Path("/maps/rows.geojson"),
        frame_stride=4,
        max_frames=40,
        particle_count=100,
        miss_penalty=4.0,
        wrong_hit_penalty=4.0,
        gps_weight=0.5,
        require_cuda=True,
    )
    candidate = {
        "pose_backend": "gtsam",
        "fixed_lag_window": 4,
        "gnss_robust_mode": "huber",
        "gnss_outlier_threshold": 3.0,
        "semantic_penalty_cap": 20.0,
    }

    cmd = tune_spfpp_optuna.build_trial_command(args, candidate, seed=22, seed_dir=Path("/runs/seed_22"))

    assert cmd[cmd.index("--pose-backend") + 1] == "gtsam"
    assert cmd[cmd.index("--fixed-lag-window") + 1] == "4"
    assert cmd[cmd.index("--gnss-robust-mode") + 1] == "huber"
    assert cmd[cmd.index("--semantic-penalty-cap") + 1] == "20.0"
    assert "--require-cuda" in cmd
