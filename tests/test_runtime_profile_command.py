from __future__ import annotations

from pathlib import Path

from scripts.run_runtime_profile_experiment import _build_trial_command


def test_runtime_profile_command_routes_robust_localisation_options() -> None:
    cmd = _build_trial_command(
        python_exec=Path("/venv/bin/python"),
        trial_dir=Path("/runs/trial"),
        data_path=Path("/data/rh_run2"),
        geojson_path=Path("/maps/rows.geojson"),
        seed=22,
        frame_stride=4,
        max_frames=45,
        warmup_frames=5,
        particle_count=100,
        segment_chunk=4096,
        require_cuda=True,
        miss_penalty=4.0,
        wrong_hit_penalty=4.0,
        gps_weight=0.5,
        pose_backend="alpha",
        fixed_lag_window=8,
        gnss_robust_mode="huber",
        gnss_outlier_threshold=3.0,
        semantic_penalty_cap=20.0,
        diagnostics_level="standard",
    )

    assert cmd[0] == "/venv/bin/python"
    assert cmd[cmd.index("--data-path") + 1] == "/data/rh_run2"
    assert cmd[cmd.index("--geojson-path") + 1] == "/maps/rows.geojson"
    assert cmd[cmd.index("--gnss-robust-mode") + 1] == "huber"
    assert cmd[cmd.index("--gnss-outlier-threshold") + 1] == "3.0"
    assert cmd[cmd.index("--semantic-penalty-cap") + 1] == "20.0"
    assert "--require-cuda" in cmd
