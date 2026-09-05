from __future__ import annotations

from dataclasses import dataclass

import numpy as np


VALID_GNSS_ROBUST_MODES = {"off", "huber", "cauchy", "gate"}


@dataclass(frozen=True)
class GnssRobustResult:
    weight: float
    innovation: float
    scale: float
    mode: str


def _validate_mode(mode: str) -> str:
    mode = str(mode)
    if mode not in VALID_GNSS_ROBUST_MODES:
        raise ValueError(f"Unsupported GNSS robust mode: {mode}")
    return mode


def compute_gnss_robust_loss(distances, *, mode: str, threshold: float) -> np.ndarray:
    mode = _validate_mode(mode)
    d = np.asarray(distances, dtype=np.float64)
    t = max(float(threshold), 1e-9)

    if mode in {"off", "gate"}:
        return d * d
    if mode == "huber":
        abs_d = np.abs(d)
        return np.where(abs_d <= t, d * d, 2.0 * t * abs_d - t * t)
    if mode == "cauchy":
        return t * t * np.log1p((d / t) ** 2)
    raise AssertionError(mode)


def resolve_robust_gnss_weight(
    *,
    base_weight: float,
    distances,
    mode: str,
    threshold: float,
) -> GnssRobustResult:
    mode = _validate_mode(mode)
    finite = np.asarray(distances, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    innovation = float(np.median(finite)) if finite.size else float("inf")

    if mode == "gate" and innovation > float(threshold):
        return GnssRobustResult(weight=0.0, innovation=innovation, scale=0.0, mode=mode)
    if mode == "off":
        scale = 1.0
    else:
        scale = float(min(1.0, max(float(threshold), 1e-9) / max(innovation, 1e-9)))
    return GnssRobustResult(weight=float(base_weight), innovation=innovation, scale=scale, mode=mode)


def semantic_penalty_cap(values, *, cap: float | None):
    arr = np.asarray(values)
    if cap is None or not np.isfinite(float(cap)) or float(cap) <= 0.0:
        return arr
    return np.maximum(arr, -float(cap))
