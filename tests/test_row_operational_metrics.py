from __future__ import annotations

import math

import numpy as np

from scripts.run_ab_validation import classify_headland_mask, compute_row_metrics


def _parallel_rows() -> dict[str, np.ndarray]:
    return {
        "row_a": np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float64),
        "row_b": np.array([[0.0, 2.0], [10.0, 2.0]], dtype=np.float64),
    }


def test_wrong_row_duration_distance_and_recovery_metrics():
    rows = _parallel_rows()
    timestamps = np.arange(6, dtype=np.float64)
    gt = np.column_stack(
        [
            np.arange(6, dtype=np.float64),
            np.zeros(6, dtype=np.float64),
            np.zeros(6, dtype=np.float64),
        ]
    )
    est = gt.copy()
    est[2:5, 1] = 2.0

    metrics = compute_row_metrics(est, gt, rows, timestamps=timestamps)

    assert metrics["row_correct_fraction"] == 0.5
    assert metrics["failure_rate"] == 0.5
    assert metrics["row_switch_events"] == 1.0
    assert metrics["wrong_row_duration_sec"] == 3.0
    assert metrics["wrong_row_distance_m"] == 3.0
    assert metrics["max_wrong_row_duration_sec"] == 3.0
    assert metrics["max_wrong_row_distance_m"] == 3.0
    assert metrics["mean_recovery_distance_m"] == 3.0
    assert metrics["xt_below_0p25_fraction"] == 0.5
    assert metrics["xt_below_0p5_fraction"] == 0.5
    assert metrics["xt_below_1p0_fraction"] == 0.5
    assert metrics["inrow_failure_rate"] == 0.5
    assert math.isnan(metrics["headland_failure_rate"])


def test_headland_and_inrow_breakdown_uses_start_sample_interval_region():
    rows = _parallel_rows()
    timestamps = np.arange(6, dtype=np.float64)
    gt = np.column_stack(
        [
            np.arange(6, dtype=np.float64),
            np.zeros(6, dtype=np.float64),
            np.zeros(6, dtype=np.float64),
        ]
    )
    est = gt.copy()
    est[2:5, 1] = 2.0
    headland_mask = np.array([True, True, False, False, False, True], dtype=bool)

    metrics = compute_row_metrics(est, gt, rows, timestamps=timestamps, headland_mask=headland_mask)

    assert metrics["headland_row_correct_fraction"] == 1.0
    assert metrics["headland_failure_rate"] == 0.0
    assert metrics["headland_wrong_row_duration_sec"] == 0.0
    assert metrics["headland_xt_below_0p25_fraction"] == 1.0
    assert metrics["inrow_row_correct_fraction"] == 0.0
    assert metrics["inrow_failure_rate"] == 1.0
    assert metrics["inrow_wrong_row_duration_sec"] == 3.0
    assert metrics["inrow_max_wrong_row_distance_m"] == 3.0


def test_classify_headland_mask_uses_row_endpoints_and_cross_track_threshold():
    rows = _parallel_rows()
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [5.0, 5.0, 0.0],
        ],
        dtype=np.float64,
    )

    mask = classify_headland_mask(points, rows, endpoint_radius_m=0.5, row_distance_threshold_m=2.5)

    assert mask.tolist() == [True, False, True, True]
