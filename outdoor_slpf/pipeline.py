from __future__ import annotations

import csv
import gzip
from pathlib import Path

import numpy as np


DIAGNOSTIC_STATS_FIELDS = [
    "raw_best_x",
    "raw_best_y",
    "raw_best_theta",
    "raw_weighted_x",
    "raw_weighted_y",
    "raw_weighted_theta",
    "smoothed_x",
    "smoothed_y",
    "smoothed_theta",
    "ess",
    "max_weight",
    "particle_count",
    "gnss_innovation",
    "gnss_robust_scale",
    "semantic_penalty_cap",
    "smoother_status",
]


def build_stats_fieldnames(existing: list[str], *, diagnostics_level: str = "standard") -> list[str]:
    if diagnostics_level not in {"minimal", "standard", "full"}:
        raise ValueError(f"Unsupported diagnostics level: {diagnostics_level}")
    fieldnames = list(existing)
    for name in DIAGNOSTIC_STATS_FIELDS:
        if name not in fieldnames:
            fieldnames.append(name)
    return fieldnames


def effective_sample_size(weights) -> float:
    w = np.asarray(weights, dtype=np.float64)
    total = float(w.sum())
    if total <= 0.0 or not np.isfinite(total):
        return 0.0
    w = w / total
    denom = float(np.sum(w * w))
    return float(1.0 / denom) if denom > 0.0 else 0.0


def maybe_log_particle_cloud(
    *,
    output_folder,
    processed_idx: int,
    frame_idx: int,
    timestamp: float,
    particles,
    weights,
    interval: int,
) -> Path | None:
    interval = int(interval)
    if interval <= 0 or int(processed_idx) % interval != 0:
        return None

    out_dir = Path(output_folder) / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "particle_cloud.csv.gz"
    write_header = not path.exists()

    particles = np.asarray(particles, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    with gzip.open(path, "at", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(["frame_idx", "processed_idx", "timestamp", "particle_idx", "x", "y", "theta", "weight"])
        for idx, (particle, weight) in enumerate(zip(particles, weights)):
            writer.writerow(
                [
                    int(frame_idx),
                    int(processed_idx),
                    float(timestamp),
                    int(idx),
                    float(particle[0]),
                    float(particle[1]),
                    float(particle[2]),
                    float(weight),
                ]
            )
    return path
