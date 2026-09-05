from __future__ import annotations

import math

import numpy as np

from outdoor_slpf.smoothing import AlphaPoseSmoother, FixedLagSmoother


def test_alpha_pose_smoother_matches_existing_prediction_blend() -> None:
    smoother = AlphaPoseSmoother(alpha_pos=0.55, alpha_theta=0.50)

    first = smoother.update(np.asarray([0.0, 0.0, 0.0]), delta_distance=0.0, delta_theta=0.0, had_prev_odom=False)
    second = smoother.update(np.asarray([2.0, 0.0, math.pi / 2.0]), delta_distance=1.0, delta_theta=0.0, had_prev_odom=True)

    assert np.allclose(first, [0.0, 0.0, 0.0])
    assert np.allclose(second[:2], [1.55, 0.0])
    assert np.isclose(second[2], math.pi / 4.0)


def test_fixed_lag_smoother_reduces_latest_straight_line_outlier() -> None:
    smoother = FixedLagSmoother(window=5, pf_std=0.8, odom_std=0.1, gnss_std=0.2)

    raw_outputs = []
    smooth_outputs = []
    for idx in range(6):
        gt = np.asarray([float(idx), 0.0, 0.0])
        raw = gt.copy()
        if idx == 5:
            raw[1] = 2.0
        raw_outputs.append(raw)
        smooth_outputs.append(
            smoother.update(
                timestamp=float(idx),
                raw_pose=raw,
                delta_distance=1.0 if idx > 0 else 0.0,
                delta_theta=0.0,
                had_prev_odom=idx > 0,
                gnss_xy=gt[:2],
            )
        )

    raw_error = np.linalg.norm(raw_outputs[-1][:2] - np.asarray([5.0, 0.0]))
    smooth_error = np.linalg.norm(smooth_outputs[-1][:2] - np.asarray([5.0, 0.0]))

    assert smooth_error < raw_error
    assert smooth_outputs[-1][1] < 1.0


def test_fixed_lag_smoother_handles_turning_yaw_without_scale_change() -> None:
    smoother = FixedLagSmoother(window=4, pf_std=0.5, odom_std=0.2, gnss_std=0.5)

    latest = None
    for idx, yaw in enumerate([0.0, 0.25, 0.50, 0.75]):
        latest = smoother.update(
            timestamp=float(idx),
            raw_pose=np.asarray([float(idx), 0.0, yaw]),
            delta_distance=1.0 if idx > 0 else 0.0,
            delta_theta=0.25 if idx > 0 else 0.0,
            had_prev_odom=idx > 0,
            gnss_xy=(float(idx), 0.0),
        )

    assert latest is not None
    assert latest[0] > 2.5
    assert 0.4 < latest[2] < 0.9
