from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

from outdoor_slpf.pipeline import build_stats_fieldnames, maybe_log_particle_cloud


def test_stats_fieldnames_append_diagnostics_without_renaming_existing_columns() -> None:
    existing = ["frame_idx", "gps_dist", "log_gps", "weight"]

    fieldnames = build_stats_fieldnames(existing, diagnostics_level="standard")

    assert fieldnames[: len(existing)] == existing
    for name in [
        "raw_best_x",
        "raw_weighted_x",
        "smoothed_x",
        "ess",
        "max_weight",
        "particle_count",
        "gnss_innovation",
        "gnss_robust_scale",
        "smoother_status",
    ]:
        assert name in fieldnames


def test_particle_cloud_logger_writes_gzip_csv_at_requested_interval(tmp_path: Path) -> None:
    particles = np.asarray([[1.0, 2.0, 0.1], [3.0, 4.0, 0.2]], dtype=np.float64)
    weights = np.asarray([0.25, 0.75], dtype=np.float64)

    maybe_log_particle_cloud(
        output_folder=tmp_path,
        processed_idx=2,
        frame_idx=8,
        timestamp=123.5,
        particles=particles,
        weights=weights,
        interval=2,
    )

    path = tmp_path / "diagnostics" / "particle_cloud.csv.gz"
    assert path.exists()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        lines = handle.read().strip().splitlines()

    assert lines[0] == "frame_idx,processed_idx,timestamp,particle_idx,x,y,theta,weight"
    assert lines[1].startswith("8,2,123.5,0,1.0,2.0,0.1,0.25")
    assert lines[2].startswith("8,2,123.5,1,3.0,4.0,0.2,0.75")
