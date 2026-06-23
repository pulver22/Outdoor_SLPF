from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.run_ab_validation import (
    prepare_eval_trajectory,
    quaternion_to_yaw_xyzw,
    start_pose_anchor_positions,
    start_pose_anchored_trajectory,
    yaw_to_quaternion_xyzw,
)
from scripts.run_iros_multiseed import method_uses_start_pose_anchor


def _write_tum(path: Path, rows: list[tuple[float, float, float, float, float]]) -> None:
    path.write_text(
        "# timestamp tx ty tz qx qy qz qw\n"
        + "\n".join(
            f"{t} {x} {y} {z} 0.0 0.0 {np.sin(yaw * 0.5)} {np.cos(yaw * 0.5)}"
            for t, x, y, z, yaw in rows
        )
        + "\n",
        encoding="utf-8",
    )


def test_start_pose_anchor_translates_rotates_and_does_not_scale() -> None:
    est_pos = np.asarray([[10.0, 20.0, 2.0], [12.0, 20.0, 5.0]], dtype=np.float64)
    est_quat = np.asarray([yaw_to_quaternion_xyzw(np.pi / 2), yaw_to_quaternion_xyzw(np.pi / 2)])
    gt_pos = np.asarray([[1.0, 2.0, 0.5], [1.0, 4.0, 0.5]], dtype=np.float64)
    gt_quat = np.asarray([yaw_to_quaternion_xyzw(0.0), yaw_to_quaternion_xyzw(0.0)])

    anchored = start_pose_anchor_positions(est_pos, est_quat, gt_pos, gt_quat)

    assert np.allclose(anchored[0], gt_pos[0])
    assert np.allclose(anchored[1], [1.0, 0.0, 3.5])
    assert np.isclose(np.linalg.norm(anchored[1, :2] - anchored[0, :2]), 2.0)


def test_start_pose_anchored_tum_first_pose_matches_ground_truth(tmp_path: Path) -> None:
    est = tmp_path / "est.tum"
    gt = tmp_path / "gt.tum"
    _write_tum(est, [(0.0, 10.0, 20.0, 0.0, np.pi / 2), (1.0, 12.0, 20.0, 0.0, np.pi / 2)])
    _write_tum(gt, [(0.0, 1.0, 2.0, 0.0, 0.0), (1.0, 1.0, 4.0, 0.0, 0.0)])

    anchored = start_pose_anchored_trajectory(est, gt)
    prepared = prepare_eval_trajectory(est, gt, tmp_path / "eval", start_pose_anchor=True)

    assert prepared.name == "start_pose_anchored_estimate.tum"
    assert np.allclose(anchored.positions[0], [1.0, 2.0, 0.0])
    assert np.isclose(quaternion_to_yaw_xyzw(anchored.quaternions[0]), 0.0)


def test_only_rtab_baselines_route_through_start_pose_anchor() -> None:
    assert method_uses_start_pose_anchor("rtab_rgb")
    assert method_uses_start_pose_anchor("rtab_rgbd")
    assert not method_uses_start_pose_anchor("slpf")
    assert not method_uses_start_pose_anchor("AMCL+NGPS")
