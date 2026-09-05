from __future__ import annotations

import numpy as np

from scripts.run_gnss_degradation_eval import apply_fixed_outage, build_profile_matrix


def test_gnss_degradation_profile_matrix_names_and_outage_lengths():
    profiles = build_profile_matrix()
    names = [profile.name for profile in profiles]

    assert names == [
        "nominal",
        "gaussian_high",
        "drift_bias",
        "multipath_bursts",
        "outage_5s",
        "outage_10s",
        "outage_20s",
        "headland_only_degradation",
    ]
    assert [p.outage_seconds for p in profiles if p.name.startswith("outage_")] == [5.0, 10.0, 20.0]
    assert profiles[-1].region == "headland"


def test_apply_fixed_outage_is_deterministic_and_preserves_quaternion_columns():
    data = np.zeros((11, 8), dtype=np.float64)
    data[:, 0] = np.arange(11, dtype=np.float64)
    data[:, 1] = np.arange(11, dtype=np.float64)
    data[:, 4:8] = np.array([0.0, 0.0, 0.0, 1.0])

    degraded, mask = apply_fixed_outage(data, outage_seconds=5.0, start_time=3.0, mode="nan")

    assert mask.tolist() == [False, False, False, True, True, True, True, True, False, False, False]
    assert np.isnan(degraded[mask, 1:4]).all()
    assert np.array_equal(degraded[:, 4:8], data[:, 4:8])


def test_apply_fixed_outage_remove_mode_drops_only_outage_samples():
    data = np.zeros((6, 8), dtype=np.float64)
    data[:, 0] = np.arange(6, dtype=np.float64)

    degraded, mask = apply_fixed_outage(data, outage_seconds=2.0, start_time=2.0, mode="remove")

    assert mask.tolist() == [False, False, True, True, False, False]
    assert degraded[:, 0].tolist() == [0.0, 1.0, 4.0, 5.0]
