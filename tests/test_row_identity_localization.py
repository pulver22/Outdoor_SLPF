from __future__ import annotations

import math

import numpy as np

from tests.test_spf_lidar_external_gnss import _import_spf_lidar


def _tiny_row_map() -> dict[str, list[dict[str, object]]]:
    return {
        "row_a": [
            {"coords": np.asarray([0.0, 0.0]), "class": 2},
            {"coords": np.asarray([0.0, 10.0]), "class": 2},
        ],
        "row_b": [
            {"coords": np.asarray([4.0, 0.0]), "class": 4},
            {"coords": np.asarray([4.0, 10.0]), "class": 4},
        ],
    }


def test_row_segment_index_preserves_segment_to_row_identity() -> None:
    spf_lidar = _import_spf_lidar()

    index = spf_lidar.build_row_segment_index(_tiny_row_map())

    assert index["row_ids"] == ["row_a", "row_b"]
    assert index["segment_row_indices"].tolist() == [0, 1]
    assert index["segment_classes"].tolist() == [2, 4]
    assert np.allclose(index["segment_p1"], [[0.0, 0.0], [4.0, 0.0]])
    assert np.allclose(index["segment_p2"], [[0.0, 10.0], [4.0, 10.0]])


def test_row_hypothesis_summary_reports_top2_probability_gap_and_entropy() -> None:
    spf_lidar = _import_spf_lidar()

    summary = spf_lidar.summarize_row_hypotheses(
        {"row_a": -0.2, "row_b": -0.7, "row_c": -5.0},
        top_k=2,
        temperature=1.0,
    )

    assert summary["row_hypothesis_id"] == "row_a"
    assert summary["row_hypothesis_second_id"] == "row_b"
    assert math.isclose(summary["row_hypothesis_gap"], 0.5, abs_tol=1e-9)
    assert 0.0 < summary["row_hypothesis_second_prob"] < summary["row_hypothesis_prob"] < 1.0
    assert summary["row_hypothesis_entropy"] > 0.0


def test_topk_logsumexp_marginalisation_keeps_ambiguous_modes_alive() -> None:
    spf_lidar = _import_spf_lidar()
    scores = np.asarray([-0.2, -0.3, -6.0], dtype=np.float64)

    top1 = spf_lidar.marginalize_topk_row_scores(scores, top_k=1, temperature=1.0)
    top2 = spf_lidar.marginalize_topk_row_scores(scores, top_k=2, temperature=1.0)

    assert top2 > top1
    assert top2 < top1 + math.log(2.0) + 1e-9


def test_gnss_row_gate_downweights_strong_semantic_and_rejects_jumps() -> None:
    spf_lidar = _import_spf_lidar()

    strong = spf_lidar.resolve_gnss_row_gate(
        base_weight=0.5,
        row_hypothesis_gap=0.8,
        num_observations=80,
        gnss_innovation=1.5,
        enabled=True,
    )
    sparse = spf_lidar.resolve_gnss_row_gate(
        base_weight=0.5,
        row_hypothesis_gap=0.1,
        num_observations=3,
        gnss_innovation=1.5,
        enabled=True,
    )
    jump = spf_lidar.resolve_gnss_row_gate(
        base_weight=0.5,
        row_hypothesis_gap=0.1,
        num_observations=3,
        gnss_innovation=12.0,
        enabled=True,
    )

    assert strong.weight < 0.5
    assert strong.reason == "semantic_strong"
    assert sparse.weight > 0.5
    assert sparse.reason == "semantic_sparse"
    assert jump.weight < strong.weight
    assert jump.reason == "gnss_jump"


def test_delayed_row_correction_nudges_ambiguous_pose_toward_confirmed_row() -> None:
    spf_lidar = _import_spf_lidar()
    correction = spf_lidar.DelayedRowCorrectionBuffer(
        window=3,
        gain=0.5,
        min_gap=0.3,
        max_step=1.0,
    )

    pose = np.asarray([2.0, 5.0, 0.0], dtype=np.float64)
    corrected, info = correction.correct_pose(
        pose,
        row_id="row_a",
        row_gap=0.7,
        grouped_map_points=_tiny_row_map(),
    )

    assert info["delayed_correction_status"] == "corrected"
    assert 0.0 < info["delayed_correction_m"] <= 1.0
    assert abs(corrected[0]) < abs(pose[0])
    assert math.isclose(corrected[1], pose[1], abs_tol=1e-9)


def test_delayed_row_correction_does_not_move_low_confidence_pose() -> None:
    spf_lidar = _import_spf_lidar()
    correction = spf_lidar.DelayedRowCorrectionBuffer(window=3, gain=0.5, min_gap=0.3)
    pose = np.asarray([2.0, 5.0, 0.0], dtype=np.float64)

    corrected, info = correction.correct_pose(
        pose,
        row_id="row_a",
        row_gap=0.05,
        grouped_map_points=_tiny_row_map(),
    )

    assert info["delayed_correction_status"] == "low_confidence"
    assert info["delayed_correction_m"] == 0.0
    assert np.allclose(corrected, pose)
