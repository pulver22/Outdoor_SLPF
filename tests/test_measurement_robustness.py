from __future__ import annotations

import numpy as np

from outdoor_slpf.measurement import (
    GnssRobustResult,
    compute_gnss_robust_loss,
    resolve_robust_gnss_weight,
    semantic_penalty_cap,
)


def test_gnss_robust_losses_match_expected_ordering() -> None:
    distances = np.asarray([0.0, 1.0, 4.0], dtype=np.float64)

    off = compute_gnss_robust_loss(distances, mode="off", threshold=2.0)
    huber = compute_gnss_robust_loss(distances, mode="huber", threshold=2.0)
    cauchy = compute_gnss_robust_loss(distances, mode="cauchy", threshold=2.0)

    assert np.allclose(off, [0.0, 1.0, 16.0])
    assert np.allclose(huber, [0.0, 1.0, 12.0])
    assert cauchy[-1] < huber[-1] < off[-1]


def test_gnss_gate_turns_weight_off_for_large_innovation() -> None:
    result = resolve_robust_gnss_weight(
        base_weight=0.5,
        distances=np.asarray([8.0, 9.0, 10.0], dtype=np.float64),
        mode="gate",
        threshold=3.0,
    )

    assert isinstance(result, GnssRobustResult)
    assert result.weight == 0.0
    assert result.scale == 0.0
    assert result.innovation > 3.0


def test_semantic_penalty_cap_only_limits_negative_scores() -> None:
    values = np.asarray([-100.0, -3.0, 0.5], dtype=np.float64)

    capped = semantic_penalty_cap(values, cap=5.0)

    assert np.allclose(capped, [-5.0, -3.0, 0.5])
