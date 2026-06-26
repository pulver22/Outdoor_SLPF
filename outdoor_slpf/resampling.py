from __future__ import annotations

import math

import numpy as np


def adaptive_resample(
    particles,
    weights,
    *,
    min_particles: int = 80,
    max_particles: int = 250,
    kld_err: float = 0.07,
    kld_z: float = 0.99,
    bin_sizes=(0.5, 0.5, np.deg2rad(10.0)),
    jitter_std=(0.02, 0.02, np.deg2rad(1.0)),
):
    particles = np.asarray(particles, dtype=np.float64)
    n = particles.shape[0]
    if n == 0:
        return particles

    w = np.asarray(weights, dtype=np.float64)
    total = float(w.sum())
    if total <= 0.0 or not np.isfinite(total):
        w = np.ones(n, dtype=np.float64) / n
    else:
        w = w / total

    def norm_ppf(p: float) -> float:
        a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.3577518672690, -30.66479806614716, 2.506628277459239]
        b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572]
        c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783]
        d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416]
        plow = 0.02425
        phigh = 1 - plow
        if p < plow:
            q = math.sqrt(-2 * math.log(p))
            return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        if p > phigh:
            q = math.sqrt(-2 * math.log(1 - p))
            return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)

    z = norm_ppf(1.0 - kld_z)
    bx, by, bt = bin_sizes

    def bin_key(p):
        theta = (p[2] + np.pi) % (2.0 * np.pi) - np.pi
        return (int(np.floor(p[0] / bx)), int(np.floor(p[1] / by)), int(np.floor(theta / bt)))

    cdf = np.cumsum(w)
    rng = np.random.default_rng()
    new_particles = []
    seen = set()
    required_n = min_particles
    while len(new_particles) < required_n and len(new_particles) < max_particles:
        idx = int(np.searchsorted(cdf, rng.random(), side="right"))
        p = particles[min(idx, n - 1)].copy()
        key = bin_key(p)
        if key not in seen:
            seen.add(key)
            k = len(seen)
            if k > 1:
                km1 = k - 1.0
                term = 1.0 - 2.0 / (9.0 * km1) + math.sqrt(2.0 / (9.0 * km1)) * z
                required_n = max(min_particles, int(math.ceil((km1 / (2.0 * kld_err)) * (term**3))))
        new_particles.append(p)

    out = np.asarray(new_particles, dtype=np.float64)
    if jitter_std is not None and len(out):
        jx, jy, jt = jitter_std
        out[:, 0] += rng.normal(0.0, jx, size=len(out))
        out[:, 1] += rng.normal(0.0, jy, size=len(out))
        out[:, 2] = (out[:, 2] + rng.normal(0.0, jt, size=len(out)) + np.pi) % (2.0 * np.pi) - np.pi
    return out
