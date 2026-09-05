from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PoseBackendConfig:
    pose_backend: str = "alpha"
    fixed_lag_window: int = 8


@dataclass(frozen=True)
class RobustnessConfig:
    gnss_robust_mode: str = "off"
    gnss_outlier_threshold: float = 5.0
    semantic_penalty_cap: float | None = None


@dataclass(frozen=True)
class DiagnosticsConfig:
    diagnostics_level: str = "standard"
    log_particle_cloud_every: int = 0


def _coerce_scalar(value: str) -> Any:
    text = value.strip()
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    if text.lower() in {"none", "null"}:
        return None
    try:
        if any(ch in text for ch in [".", "e", "E"]):
            return float(text)
        return int(text)
    except ValueError:
        return text.strip("\"'")


def load_yaml_overlay(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        import yaml  # type: ignore
    except Exception:
        yaml = None
    if yaml is not None:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Config YAML must contain a mapping: {path}")
        return dict(data)

    values: dict[str, Any] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise ValueError(f"Unsupported YAML line {line_no} in {path}: {line}")
        key, value = stripped.split(":", 1)
        values[key.strip().replace("-", "_")] = _coerce_scalar(value)
    return values


def apply_config_defaults(parser: argparse.ArgumentParser, config_path: str | Path | None) -> None:
    if config_path is None:
        return
    overlay = load_yaml_overlay(config_path)
    valid_dests = {action.dest for action in parser._actions}
    unknown = sorted(set(overlay) - valid_dests)
    if unknown:
        raise ValueError(f"Unknown config YAML option(s): {unknown}")
    parser.set_defaults(**overlay)
