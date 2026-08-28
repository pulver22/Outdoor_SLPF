from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from scripts import analyze_localization_followup as diagnostics
from scripts import generate_localization_improvement_report as report
from scripts import run_localization_followup_experiments as followup


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_tum(path: Path, xy: list[tuple[float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for idx, (x, y) in enumerate(xy):
            f.write(f"{float(idx)} {x} {y} 0 0 0 0 1\n")


def test_diagnostic_summary_reads_metrics_and_frame_stats(tmp_path: Path) -> None:
    run_root = tmp_path / "alpha_huber_cap_rh1"
    _write_csv(
        run_root / "trajectory_metrics_per_seed.csv",
        [
            {
                "run_name": "alpha_huber_cap_rh1_seed_11",
                "est_tum": str(run_root / "seed_11" / "trajectory_0.5.tum"),
                "gt_tum": str(run_root / "seed_11" / "gps_pose.tum"),
                "ape_align_rmse": 1.9,
                "cross_track_mean": 1.6,
                "row_correct_fraction": 0.65,
                "wrong_row_duration_sec": 20.0,
                "headland_cross_track_mean": 2.4,
                "inrow_cross_track_mean": 1.1,
                "seed": 11,
            }
        ],
    )
    _write_csv(
        run_root / "seed_11" / "stats.csv",
        [
            {
                "frame_idx": 0,
                "gnss_innovation": 4.0,
                "gnss_robust_scale": 0.75,
                "ess": 25.0,
                "max_weight": 0.15,
                "correct_hits": 10,
                "incorrect_hits": 1,
                "no_hits": 30,
                "row_hypothesis_id": "row_a",
                "row_hypothesis_second_id": "row_b",
                "row_hypothesis_gap": 0.2,
                "row_hypothesis_entropy": 0.68,
                "row_switch_event": 0,
                "gnss_row_gate_scale": 1.0,
                "smoother_status": "alpha",
            },
            {
                "frame_idx": 4,
                "gnss_innovation": 8.0,
                "gnss_robust_scale": 0.375,
                "ess": 50.0,
                "max_weight": 0.08,
                "correct_hits": 20,
                "incorrect_hits": 2,
                "no_hits": 10,
                "row_hypothesis_id": "row_b",
                "row_hypothesis_second_id": "row_a",
                "row_hypothesis_gap": 0.5,
                "row_hypothesis_entropy": 0.52,
                "row_switch_event": 1,
                "gnss_row_gate_scale": 0.55,
                "smoother_status": "alpha",
            },
        ],
    )
    _write_tum(run_root / "seed_11" / "trajectory_0.5.tum", [(0.0, 0.0), (1.0, 0.0)])
    _write_tum(run_root / "seed_11" / "gps_pose.tum", [(0.0, 0.0), (1.0, 0.0)])

    rows, frame_rows = diagnostics.collect_diagnostics(
        [
            diagnostics.RunSpec(
                traversal="rh_run1",
                variant="alpha_huber_cap",
                root=run_root,
            )
        ]
    )

    assert rows[0]["traversal"] == "rh_run1"
    assert rows[0]["seed"] == 11
    assert rows[0]["ape_align_rmse"] == 1.9
    assert rows[0]["frame_count"] == 2
    assert rows[0]["gnss_innovation_median"] == 6.0
    assert rows[0]["ess_min_first_50"] == 25.0
    assert rows[0]["semantic_hit_ratio_mean"] == np.mean([10 / 41, 20 / 32])
    assert rows[0]["row_switch_count"] == 1
    assert rows[0]["row_hypothesis_entropy_mean"] == np.mean([0.68, 0.52])
    assert rows[0]["row_hypothesis_gap_median"] == 0.35
    assert rows[0]["dominant_row_hypothesis_id"] == "row_a"
    assert rows[0]["dominant_row_hypothesis_fraction"] == 0.5
    assert np.isclose(rows[0]["gnss_row_switch_corr"], 1.0)
    assert rows[0]["gnss_row_gate_scale_median"] == 0.775
    assert frame_rows[0]["traversal"] == "rh_run1"
    assert frame_rows[0]["frame_idx"] == 0


def test_followup_candidate_matrix_and_promotion_rules() -> None:
    candidates = followup.candidate_matrix(include_gtsam=True)

    assert [candidate["id"] for candidate in candidates] == [
        "accepted_alpha_huber3_cap20",
        "alpha_huber5_nocap",
        "alpha_huber3_cap50",
        "alpha_cauchy3_cap20",
        "alpha_huber3_cap20_no_smoothing",
        "fixedlag_huber3_cap20",
        "gtsam_huber3_cap20",
    ]
    assert candidates[0]["args"] == [
        "--pose-backend",
        "alpha",
        "--gnss-robust-mode",
        "huber",
        "--gnss-outlier-threshold",
        "3.0",
        "--semantic-penalty-cap",
        "20",
    ]

    promoted = followup.select_promoted_candidates(
        [
            {"candidate_id": "gtsam_huber3_cap20", "ape_align_rmse": 0.80},
            {"candidate_id": "accepted_alpha_huber3_cap20", "ape_align_rmse": 0.90},
            {"candidate_id": "alpha_huber5_nocap", "ape_align_rmse": 0.95},
        ],
        candidates,
        top_n=2,
    )
    assert [candidate["id"] for candidate in promoted] == ["gtsam_huber3_cap20", "accepted_alpha_huber3_cap20"]

    promoted = followup.select_promoted_candidates(
        [
            {"candidate_id": "gtsam_huber3_cap20", "ape_align_rmse": 1.20},
            {"candidate_id": "accepted_alpha_huber3_cap20", "ape_align_rmse": 0.90},
            {"candidate_id": "alpha_huber5_nocap", "ape_align_rmse": 0.95},
        ],
        candidates,
        top_n=2,
    )
    assert [candidate["id"] for candidate in promoted] == ["accepted_alpha_huber3_cap20", "alpha_huber5_nocap"]


def test_next_step_candidate_matrix_freezes_baseline_and_adds_row_identity_ablations() -> None:
    candidates = followup.next_step_candidate_matrix()

    assert [candidate["id"] for candidate in candidates] == [
        "baseline_alpha_huber3_cap50",
        "row_mixture",
        "delayed_correction",
        "row_mixture_delayed",
        "row_mixture_delayed_gnss_gating",
    ]
    assert candidates[0]["args"] == [
        "--config-yaml",
        str(followup.BASE_DIR / "configs/icra/alpha_huber3_cap50.yaml"),
    ]
    assert "--row-likelihood-mode" in candidates[1]["args"]
    assert "--row-mixture-top-k" in candidates[1]["args"]
    assert "--delayed-row-correction" in candidates[2]["args"]
    assert "--gnss-row-gating" in candidates[4]["args"]
    assert all("gtsam" not in candidate["id"] for candidate in candidates)


def test_followup_command_contains_dataset_candidate_and_output_paths(tmp_path: Path) -> None:
    candidate = followup.candidate_matrix(include_gtsam=False)[4]
    cmd = followup.build_spf_command(
        python_exec=Path("/venv/bin/python"),
        data_root=Path("/data/2025"),
        geojson=Path("/maps/rows.geojson"),
        output_root=tmp_path / "runs",
        traversal="rh_run1",
        seed=11,
        candidate=candidate,
        max_frames=40,
        require_cuda=True,
    )

    assert cmd[0] == "/venv/bin/python"
    assert cmd[cmd.index("--data-path") + 1] == "/data/2025/rh_run1"
    assert cmd[cmd.index("--output-folder") + 1].endswith("rh_run1/alpha_huber3_cap20_no_smoothing/seed_11")
    assert "--disable-pose-smoothing" in cmd
    assert "--require-cuda" in cmd


def test_report_generation_includes_required_sections_and_artifacts(tmp_path: Path) -> None:
    current_rh2 = tmp_path / "rh2_summary.json"
    current_rh1 = tmp_path / "rh1_summary.json"
    diagnostics_csv = tmp_path / "diagnostics.csv"
    followup_csv = tmp_path / "followup.csv"
    current_rh2.write_text(json.dumps({"ape_align_rmse_mean": 0.843, "per_seed_ape_align_rmse": {"11": 0.79}}))
    current_rh1.write_text(json.dumps({"ape_align_rmse_mean": 1.300, "per_seed_ape_align_rmse": {"11": 1.99}}))
    _write_csv(
        diagnostics_csv,
        [
            {
                "traversal": "rh_run1",
                "seed": 11,
                "ape_align_rmse": 1.99,
                "row_switch_count": 7,
                "row_hypothesis_entropy_mean": 0.61,
                "row_hypothesis_gap_median": 0.12,
                "gnss_row_switch_corr": 0.4,
            }
        ],
    )
    _write_csv(followup_csv, [{"stage": "screen", "candidate_id": "accepted_alpha_huber3_cap20", "ape_align_rmse": 1.2}])

    out_md = tmp_path / "report.md"
    out_json = tmp_path / "report_data.json"
    data = report.generate_report(
        output_markdown=out_md,
        output_json=out_json,
        current_rh2_summary=current_rh2,
        current_rh1_summary=current_rh1,
        diagnostics_csv=diagnostics_csv,
        followup_csv=followup_csv,
        branch="iros_revision_plan",
        commit="6bd2fb9",
        commands=["python scripts/analyze_localization_followup.py"],
    )

    text = out_md.read_text()
    assert "# Outdoor SLPF Localisation Improvement Report" in text
    assert "## Changes Implemented" in text
    assert "## Current Metric Summary" in text
    assert "## Follow-Up Experiments" in text
    assert "## Row-Identity Diagnostics" in text
    assert "## Controlled Row-Identity Ablation" in text
    assert "## Acceptance Check" in text
    assert "multi-hypothesis row reasoning" in text
    assert "## Next Experiments" in text
    assert "6bd2fb9" in text
    assert data["git"]["branch"] == "iros_revision_plan"
    assert out_json.exists()
