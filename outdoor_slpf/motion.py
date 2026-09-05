from __future__ import annotations

import math

import numpy as np


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(t3, t4))


def wrap_to_pi(angle):
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def angle_diff(target, source):
    return wrap_to_pi(np.asarray(target) - np.asarray(source))


def circular_lerp(source: float, target: float, alpha: float) -> float:
    return float(wrap_to_pi(float(source) + float(alpha) * float(angle_diff(target, source))))


def yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    cy = math.cos(float(yaw) * 0.5)
    sy = math.sin(float(yaw) * 0.5)
    return 0.0, 0.0, sy, cy


def motion_update(
    particles: np.ndarray,
    delta_distance: float,
    delta_theta: float,
    *,
    angle_std: float,
    min_distance: float = 0.03,
    min_angle: float = np.deg2rad(1.0),
    noise_std: tuple[float, float] = (0.1, 0.1),
) -> np.ndarray:
    if delta_distance < min_distance and abs(delta_theta) < min_angle:
        return particles

    n = len(particles)
    noise_x = np.random.normal(0.0, noise_std[0], size=n)
    noise_y = np.random.normal(0.0, noise_std[1], size=n)
    noise_theta = np.random.normal(0.0, angle_std, size=n)

    theta = particles[:, 2]
    particles[:, 0] += delta_distance * np.cos(theta) + noise_x
    particles[:, 1] += delta_distance * np.sin(theta) + noise_y
    particles[:, 2] = wrap_to_pi(particles[:, 2] + delta_theta + noise_theta)
    return particles


def estimate_pose_from_particles(
    particles: np.ndarray,
    weights: np.ndarray,
    *,
    fallback_map_if_multimodal: bool = True,
) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    wsum = float(w.sum())
    if wsum <= 0.0 or not np.isfinite(wsum):
        w = np.ones(len(particles), dtype=np.float64) / len(particles)
    else:
        w = w / wsum

    x_mean = float(np.sum(w * particles[:, 0]))
    y_mean = float(np.sum(w * particles[:, 1]))
    s = float(np.sum(w * np.sin(particles[:, 2])))
    c = float(np.sum(w * np.cos(particles[:, 2])))
    theta_mean = float(np.arctan2(s, c))

    if fallback_map_if_multimodal and (1.0 - np.hypot(s, c)) > 0.4:
        idx = int(np.argmax(w))
        return particles[idx].astype(np.float64, copy=True)
    return np.asarray([x_mean, y_mean, theta_mean], dtype=np.float64)
