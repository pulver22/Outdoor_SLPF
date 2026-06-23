#!/usr/bin/env python3
"""Generate GNSS degradation profiles for the IROS revision experiments."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from degrade_gps_vineyard import NoiseParams, apply_noise, read_tum_with_comments, write_tum
except ModuleNotFoundError:
    from scripts.degrade_gps_vineyard import NoiseParams, apply_noise, read_tum_with_comments, write_tum


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = BASE_DIR / "results" / "iros_revision" / "gnss_degradation"


@dataclass(frozen=True)
class DegradationProfile:
    name: str
    noise: NoiseParams
    outage_seconds: float | None = None
    outage_start_fraction: float = 0.5
    region: str = "whole"


def build_profile_matrix() -> list[DegradationProfile]:
    return [
        DegradationProfile(
            name="nominal",
            noise=NoiseParams(
                sigma_w_xy=0.6,
                sigma_w_z=0.0,
                sigma_b_xy=0.6,
                sigma_b_z=0.0,
                tau=30.0,
                outlier_prob=0.002,
                outlier_scale_xy=3.0,
                outlier_scale_z=0.0,
                dropout_rate=0.0,
                dropout_mode="nan",
            ),
        ),
        DegradationProfile(
            name="gaussian_high",
            noise=NoiseParams(
                sigma_w_xy=2.0,
                sigma_w_z=0.0,
                sigma_b_xy=1.0,
                sigma_b_z=0.0,
                tau=30.0,
                outlier_prob=0.0,
                dropout_rate=0.0,
                dropout_mode="nan",
            ),
        ),
        DegradationProfile(
            name="drift_bias",
            noise=NoiseParams(
                sigma_w_xy=0.4,
                sigma_w_z=0.0,
                sigma_b_xy=3.0,
                sigma_b_z=0.0,
                tau=90.0,
                outlier_prob=0.0,
                dropout_rate=0.0,
                dropout_mode="nan",
            ),
        ),
        DegradationProfile(
            name="multipath_bursts",
            noise=NoiseParams(
                sigma_w_xy=0.8,
                sigma_w_z=0.0,
                sigma_b_xy=1.0,
                sigma_b_z=0.0,
                tau=30.0,
                outlier_prob=0.03,
                outlier_scale_xy=8.0,
                outlier_scale_z=0.0,
                outlier_dof=3.0,
                dropout_rate=0.0,
                dropout_mode="nan",
            ),
        ),
        DegradationProfile(
            name="outage_5s",
            noise=NoiseParams(sigma_w_xy=0.6, sigma_w_z=0.0, sigma_b_xy=0.6, sigma_b_z=0.0, dropout_mode="nan"),
            outage_seconds=5.0,
        ),
        DegradationProfile(
            name="outage_10s",
            noise=NoiseParams(sigma_w_xy=0.6, sigma_w_z=0.0, sigma_b_xy=0.6, sigma_b_z=0.0, dropout_mode="nan"),
            outage_seconds=10.0,
        ),
        DegradationProfile(
            name="outage_20s",
            noise=NoiseParams(sigma_w_xy=0.6, sigma_w_z=0.0, sigma_b_xy=0.6, sigma_b_z=0.0, dropout_mode="nan"),
            outage_seconds=20.0,
        ),
        DegradationProfile(
            name="headland_only_degradation",
            noise=NoiseParams(
                sigma_w_xy=2.0,
                sigma_w_z=0.0,
                sigma_b_xy=2.0,
                sigma_b_z=0.0,
                tau=60.0,
                outlier_prob=0.02,
                outlier_scale_xy=6.0,
                outlier_scale_z=0.0,
                dropout_rate=0.0,
                dropout_mode="nan",
            ),
            region="headland",
        ),
    ]


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def selected_profiles(names: str | None) -> list[DegradationProfile]:
    profiles = build_profile_matrix()
    if not names:
        return profiles
    wanted = {name.strip() for name in names.split(",") if name.strip()}
    by_name = {profile.name: profile for profile in profiles}
    missing = sorted(wanted - set(by_name))
    if missing:
        raise ValueError(f"Unknown degradation profile(s): {', '.join(missing)}")
    return [by_name[name] for name in wanted]


def fixed_outage_start_time(timestamps: np.ndarray, outage_seconds: float, start_fraction: float) -> float:
    if timestamps.size == 0:
        return 0.0
    start = float(timestamps[0])
    end = float(timestamps[-1])
    span = max(0.0, end - start - outage_seconds)
    return start + float(np.clip(start_fraction, 0.0, 1.0)) * span


def apply_fixed_outage(
    data: np.ndarray,
    *,
    outage_seconds: float,
    start_time: float,
    mode: str = "nan",
) -> tuple[np.ndarray, np.ndarray]:
    if mode not in {"nan", "hold", "remove"}:
        raise ValueError("mode must be one of: nan | hold | remove")
    if outage_seconds <= 0.0:
        return data.copy(), np.zeros(len(data), dtype=bool)

    out = data.copy()
    timestamps = out[:, 0]
    mask = (timestamps >= start_time) & (timestamps < start_time + outage_seconds)
    if not np.any(mask):
        return out, mask

    if mode == "nan":
        out[mask, 1:4] = np.nan
    elif mode == "hold":
        held = None
        for idx in range(len(out)):
            if mask[idx]:
                if held is not None:
                    out[idx, 1:4] = held
            else:
                held = out[idx, 1:4].copy()
    elif mode == "remove":
        out = out[~mask].copy()
    return out, mask


def headland_mask_by_fraction(timestamps: np.ndarray, fraction: float) -> np.ndarray:
    if timestamps.size == 0:
        return np.zeros(0, dtype=bool)
    fraction = float(np.clip(fraction, 0.0, 0.5))
    start = float(timestamps[0])
    end = float(timestamps[-1])
    duration = max(0.0, end - start)
    margin = duration * fraction
    return (timestamps <= start + margin) | (timestamps >= end - margin)


def apply_profile(
    data: np.ndarray,
    profile: DegradationProfile,
    seed: int,
    *,
    dropout_mode: str,
    headland_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    timestamps = data[:, 0].astype(float)
    original_pos = data[:, 1:4].copy()
    pos_noisy, stochastic_dropout, err = apply_noise(timestamps, original_pos, profile.noise, rng)

    out = data.copy()
    if profile.region == "headland":
        affected = headland_mask_by_fraction(timestamps, headland_fraction)
        out[affected, 1:4] = pos_noisy[affected]
        full_error = np.zeros_like(original_pos)
        full_error[affected] = out[affected, 1:4] - original_pos[affected]
    else:
        affected = np.ones(len(out), dtype=bool)
        out[:, 1:4] = pos_noisy
        full_error = err

    outage_mask = stochastic_dropout.copy()
    if profile.outage_seconds is not None:
        start_time = fixed_outage_start_time(timestamps, profile.outage_seconds, profile.outage_start_fraction)
        out, outage_mask = apply_fixed_outage(
            out,
            outage_seconds=profile.outage_seconds,
            start_time=start_time,
            mode=dropout_mode,
        )

    return out, affected | outage_mask, full_error


def summarize_error(error: np.ndarray, affected_mask: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(error).all(axis=1)
    mask = affected_mask & finite
    if not np.any(mask):
        return {
            "affected_fraction": 0.0,
            "xy_error_rmse_m": float("nan"),
            "xy_error_mean_m": float("nan"),
            "xy_error_max_m": float("nan"),
        }
    xy_error = np.linalg.norm(error[mask, :2], axis=1)
    return {
        "affected_fraction": float(np.mean(affected_mask)),
        "xy_error_rmse_m": float(np.sqrt(np.mean(xy_error * xy_error))),
        "xy_error_mean_m": float(np.mean(xy_error)),
        "xy_error_max_m": float(np.max(xy_error)),
    }


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_tum", type=Path, help="RTK/ground-truth TUM file to degrade.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seeds", default="11,22,33")
    parser.add_argument("--profiles", default=None, help="Comma-separated subset of profile names.")
    parser.add_argument("--dropout-mode", choices=["nan", "hold", "remove"], default="nan")
    parser.add_argument("--headland-fraction", type=float, default=0.15)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.input_tum.exists():
        raise FileNotFoundError(f"Missing input TUM: {args.input_tum}")
    seeds = parse_int_list(args.seeds)
    if not seeds:
        raise ValueError("At least one seed is required.")

    profiles = selected_profiles(args.profiles)
    data = read_tum_with_comments(str(args.input_tum))
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    protocol = {
        "input_tum": str(args.input_tum.resolve()),
        "output_root": str(output_root),
        "seeds": seeds,
        "dropout_mode": args.dropout_mode,
        "headland_fraction": args.headland_fraction,
        "profiles": [],
    }

    for profile in profiles:
        protocol["profiles"].append(
            {
                "name": profile.name,
                "region": profile.region,
                "outage_seconds": profile.outage_seconds,
                "noise": asdict(profile.noise),
            }
        )
        for seed in seeds:
            out_dir = output_root / profile.name / f"seed_{seed}"
            out_dir.mkdir(parents=True, exist_ok=True)
            degraded, affected_mask, error = apply_profile(
                data,
                profile,
                seed,
                dropout_mode=args.dropout_mode,
                headland_fraction=args.headland_fraction,
            )
            out_tum = out_dir / "noisy_gnss.tum"
            write_tum(
                str(out_tum),
                degraded,
                header=f"generated by run_gnss_degradation_eval.py profile={profile.name} seed={seed}",
            )
            row = {
                "profile": profile.name,
                "seed": seed,
                "out_tum": str(out_tum),
                "samples_in": len(data),
                "samples_out": len(degraded),
                "region": profile.region,
                "outage_seconds": profile.outage_seconds if profile.outage_seconds is not None else "",
                **summarize_error(error, affected_mask),
            }
            summary_rows.append(row)

    write_csv(output_root / "profile_summary.csv", summary_rows)
    (output_root / "run_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"[INFO] Wrote GNSS degradation profiles to {output_root}")


if __name__ == "__main__":
    main()
