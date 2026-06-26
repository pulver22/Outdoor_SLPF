from __future__ import annotations

import importlib
from dataclasses import dataclass, field

import numpy as np

from .motion import angle_diff, circular_lerp, wrap_to_pi


def require_gtsam():
    try:
        module = importlib.import_module("gtsam")
    except Exception as exc:
        raise RuntimeError('GTSAM backend requested but unavailable. Install with: python -m pip install "gtsam==4.2.1"') from exc
    if module is None:
        raise RuntimeError('GTSAM backend requested but unavailable. Install with: python -m pip install "gtsam==4.2.1"')
    return module


class AlphaPoseSmoother:
    name = "alpha"

    def __init__(self, *, alpha_pos: float = 0.55, alpha_theta: float = 0.50):
        self.alpha_pos = float(alpha_pos)
        self.alpha_theta = float(alpha_theta)
        self.pose: np.ndarray | None = None

    def update(
        self,
        raw_pose,
        *,
        delta_distance: float,
        delta_theta: float,
        had_prev_odom: bool,
        **_kwargs,
    ) -> np.ndarray:
        raw = np.asarray(raw_pose, dtype=np.float64)
        if self.pose is None:
            self.pose = raw.copy()
            return self.pose.copy()

        predicted = self.pose.copy()
        if had_prev_odom:
            predicted[0] += float(delta_distance) * np.cos(self.pose[2])
            predicted[1] += float(delta_distance) * np.sin(self.pose[2])
            predicted[2] = float(wrap_to_pi(self.pose[2] + float(delta_theta)))

        self.pose[0] = (1.0 - self.alpha_pos) * predicted[0] + self.alpha_pos * raw[0]
        self.pose[1] = (1.0 - self.alpha_pos) * predicted[1] + self.alpha_pos * raw[1]
        self.pose[2] = circular_lerp(predicted[2], raw[2], self.alpha_theta)
        return self.pose.copy()


@dataclass
class _LagSample:
    timestamp: float
    raw_pose: np.ndarray
    delta_distance: float
    delta_theta: float
    had_prev_odom: bool
    gnss_xy: tuple[float, float] | None = None


class FixedLagSmoother:
    name = "fixed-lag"

    def __init__(
        self,
        *,
        window: int = 8,
        pf_std: float = 0.8,
        odom_std: float = 0.25,
        gnss_std: float = 1.1,
        iterations: int = 4,
    ):
        self.window = max(2, int(window))
        self.pf_std = max(float(pf_std), 1e-6)
        self.odom_std = max(float(odom_std), 1e-6)
        self.gnss_std = max(float(gnss_std), 1e-6)
        self.iterations = max(1, int(iterations))
        self.samples: list[_LagSample] = []
        self.last_states: np.ndarray | None = None
        self.alpha_prior = AlphaPoseSmoother(alpha_pos=0.55, alpha_theta=0.50)

    def update(
        self,
        *,
        timestamp: float,
        raw_pose,
        delta_distance: float,
        delta_theta: float,
        had_prev_odom: bool,
        gnss_xy=None,
        **_kwargs,
    ) -> np.ndarray:
        raw = np.asarray(raw_pose, dtype=np.float64)
        prior_pose = self.alpha_prior.update(
            raw,
            delta_distance=delta_distance,
            delta_theta=delta_theta,
            had_prev_odom=had_prev_odom,
        )
        sample = _LagSample(
            timestamp=float(timestamp),
            raw_pose=prior_pose.copy(),
            delta_distance=float(delta_distance),
            delta_theta=float(delta_theta),
            had_prev_odom=bool(had_prev_odom),
            gnss_xy=tuple(float(v) for v in gnss_xy) if gnss_xy is not None and np.isfinite(gnss_xy).all() else None,
        )
        self.samples.append(sample)
        if len(self.samples) > self.window:
            self.samples = self.samples[-self.window :]

        states = np.stack([s.raw_pose for s in self.samples]).astype(np.float64)
        if self.last_states is not None and len(self.last_states) == len(states):
            states = 0.5 * states + 0.5 * self.last_states

        pf_gain = 1.0 / (self.pf_std * self.pf_std)
        odom_gain = 1.0 / (self.odom_std * self.odom_std)
        gnss_gain = 1.0 / (self.gnss_std * self.gnss_std)
        norm = pf_gain + odom_gain + gnss_gain

        for _ in range(self.iterations):
            for idx, s in enumerate(self.samples):
                pos_num = pf_gain * s.raw_pose[:2]
                pos_den = pf_gain
                if idx > 0 and s.had_prev_odom:
                    prev = states[idx - 1]
                    pred = prev[:2] + s.delta_distance * np.asarray([np.cos(prev[2]), np.sin(prev[2])])
                    pos_num = pos_num + odom_gain * pred
                    pos_den += odom_gain
                    theta_pred = float(wrap_to_pi(prev[2] + s.delta_theta))
                    states[idx, 2] = circular_lerp(theta_pred, s.raw_pose[2], pf_gain / (pf_gain + odom_gain))
                else:
                    states[idx, 2] = s.raw_pose[2]

                if s.gnss_xy is not None:
                    pos_num = pos_num + gnss_gain * np.asarray(s.gnss_xy, dtype=np.float64)
                    pos_den += gnss_gain

                states[idx, :2] = pos_num / max(pos_den, 1e-12)

            # A light backward pass keeps the most recent state from absorbing all
            # of a one-frame outlier while leaving scale unchanged.
            for idx in range(len(self.samples) - 2, -1, -1):
                nxt = self.samples[idx + 1]
                if not nxt.had_prev_odom:
                    continue
                theta = states[idx, 2]
                pred_next = states[idx, :2] + nxt.delta_distance * np.asarray([np.cos(theta), np.sin(theta)])
                residual = states[idx + 1, :2] - pred_next
                states[idx, :2] += 0.15 * residual
                states[idx, 2] = circular_lerp(states[idx, 2], states[idx + 1, 2] - nxt.delta_theta, 0.10)

        states[:, 2] = wrap_to_pi(states[:, 2])
        self.last_states = states.copy()
        return states[-1].copy()

    @property
    def status(self) -> str:
        return f"fixed-lag:{len(self.samples)}/{self.window}"


class GtsamFixedLagSmoother(FixedLagSmoother):
    name = "gtsam"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gtsam = require_gtsam()

    @property
    def status(self) -> str:
        return f"gtsam:{len(self.samples)}/{self.window}"

    def update(self, **kwargs) -> np.ndarray:
        return self._update_with_gtsam(**kwargs)

    def _update_with_gtsam(self, **kwargs) -> np.ndarray:
        raw = np.asarray(kwargs["raw_pose"], dtype=np.float64)
        sample = _LagSample(
            timestamp=float(kwargs.get("timestamp", len(self.samples))),
            raw_pose=raw.copy(),
            delta_distance=float(kwargs.get("delta_distance", 0.0)),
            delta_theta=float(kwargs.get("delta_theta", 0.0)),
            had_prev_odom=bool(kwargs.get("had_prev_odom", False)),
            gnss_xy=tuple(float(v) for v in kwargs["gnss_xy"]) if kwargs.get("gnss_xy") is not None and np.isfinite(kwargs["gnss_xy"]).all() else None,
        )
        self.samples.append(sample)
        if len(self.samples) > self.window:
            self.samples = self.samples[-self.window :]

        gtsam = self.gtsam
        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.asarray([self.pf_std, self.pf_std, 0.5], dtype=np.float64))
        odom_noise = gtsam.noiseModel.Diagonal.Sigmas(np.asarray([self.odom_std, self.odom_std, 0.2], dtype=np.float64))
        gnss_noise = gtsam.noiseModel.Diagonal.Sigmas(np.asarray([self.gnss_std, self.gnss_std, 1e6], dtype=np.float64))

        def key(i: int):
            return gtsam.symbol("x", i)

        for idx, s in enumerate(self.samples):
            pose = gtsam.Pose2(float(s.raw_pose[0]), float(s.raw_pose[1]), float(s.raw_pose[2]))
            values.insert(key(idx), pose)
            graph.add(gtsam.PriorFactorPose2(key(idx), pose, prior_noise))
            if idx > 0 and s.had_prev_odom:
                graph.add(gtsam.BetweenFactorPose2(key(idx - 1), key(idx), gtsam.Pose2(s.delta_distance, 0.0, s.delta_theta), odom_noise))
            if s.gnss_xy is not None:
                graph.add(gtsam.PriorFactorPose2(key(idx), gtsam.Pose2(s.gnss_xy[0], s.gnss_xy[1], float(s.raw_pose[2])), gnss_noise))

        result = gtsam.LevenbergMarquardtOptimizer(graph, values).optimize()
        latest = result.atPose2(key(len(self.samples) - 1))
        return np.asarray([latest.x(), latest.y(), latest.theta()], dtype=np.float64)


def create_pose_backend(name: str, **kwargs):
    name = str(name)
    if name == "alpha":
        return AlphaPoseSmoother(**kwargs)
    if name == "fixed-lag":
        return FixedLagSmoother(**kwargs)
    if name == "gtsam":
        return GtsamFixedLagSmoother(**kwargs)
    raise ValueError(f"Unsupported pose backend: {name}")
