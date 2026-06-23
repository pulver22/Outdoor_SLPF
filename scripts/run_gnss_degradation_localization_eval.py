#!/usr/bin/env python3
"""Run localisation baselines under GNSS degradation profiles for the IROS revision."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".tmp_mpl") if "BASE_DIR" in globals() else str(Path(__file__).resolve().parents[1] / ".tmp_mpl"))

try:
    from run_ab_validation import (
        BASE_DIR,
        compute_row_metrics,
        compute_smoothness_metrics,
        interpolate_positions,
        load_rows_from_geojson,
        read_tum_file,
        start_pose_anchored_trajectory,
        umeyama_alignment,
        apply_transform,
    )
    from run_gnss_degradation_eval import apply_profile, parse_int_list, selected_profiles, write_tum
    from run_iros_multiseed import (
        DEFAULT_AMCL_NGPS_AMCL_STD,
        DEFAULT_AMCL_NGPS_GPS_STD,
        DEFAULT_AMCL_NGPS_PROCESS_STD,
        DEFAULT_RTAB_NGPS_GPS_STD,
        DEFAULT_RTAB_NGPS_PROCESS_STD,
        DEFAULT_RTAB_NGPS_RTAB_STD,
        build_kalman_ngps_fused_tum,
        write_csv,
    )
except ModuleNotFoundError:
    from scripts.run_ab_validation import (
        BASE_DIR,
        compute_row_metrics,
        compute_smoothness_metrics,
        interpolate_positions,
        load_rows_from_geojson,
        read_tum_file,
        start_pose_anchored_trajectory,
        umeyama_alignment,
        apply_transform,
    )
    from scripts.run_gnss_degradation_eval import apply_profile, parse_int_list, selected_profiles, write_tum
    from scripts.run_iros_multiseed import (
        DEFAULT_AMCL_NGPS_AMCL_STD,
        DEFAULT_AMCL_NGPS_GPS_STD,
        DEFAULT_AMCL_NGPS_PROCESS_STD,
        DEFAULT_RTAB_NGPS_GPS_STD,
        DEFAULT_RTAB_NGPS_PROCESS_STD,
        DEFAULT_RTAB_NGPS_RTAB_STD,
        build_kalman_ngps_fused_tum,
        write_csv,
    )


DEFAULT_OUTPUT_ROOT = BASE_DIR / "results" / "iros_revision" / "gnss_degradation"
DEFAULT_GEOJSON = BASE_DIR / "data" / "riseholme_poles_trunk.geojson"
DEFAULT_SEEDS = "11,22,33"


@dataclass(frozen=True)
class TraversalConfig:
    name: str
    data_path: Path
    baseline_root: Path


TRAVERSALS = {
    "rh1": TraversalConfig(
        name="rh1",
        data_path=BASE_DIR / "data" / "2025" / "rh_run1",
        baseline_root=BASE_DIR / "results" / "iros_rh1" / "20260217_172656_multiseed_main",
    ),
    "rh2": TraversalConfig(
        name="rh2",
        data_path=BASE_DIR / "data" / "2025" / "rh_run2",
        baseline_root=BASE_DIR / "results" / "iros_rh2" / "20260225_105822_multiseed_all_methods",
    ),
}


METHOD_LABELS = {
    "slpf": "SLPF",
    "ngps": "NoisyGNSS",
    "amcl_ngps": "AMCL+NoisyGNSS",
    "rtab_rgb_ngps": "RTAB RGB+NoisyGNSS",
    "rtab_rgbd_ngps": "RTAB RGBD+NoisyGNSS",
}


def parse_name_list(text: str | None, available: Iterable[str]) -> list[str]:
    choices = list(available)
    if not text:
        return choices
    wanted = [item.strip() for item in text.split(",") if item.strip()]
    missing = sorted(set(wanted) - set(choices))
    if missing:
        raise ValueError(f"Unknown name(s): {', '.join(missing)}")
    return wanted


def run_cmd(cmd: list[str], log_path: Path, cwd: Path, env: dict[str, str]) -> float:
    start = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    duration = time.time() - start
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "$ " + " ".join(cmd) + "\n\n" + (proc.stdout or "") + f"\n[exit_code={proc.returncode} duration_sec={duration:.2f}]\n",
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)} (see {log_path})")
    return duration


def tum_for_method(config: TraversalConfig, method: str, seed: int) -> tuple[Path, Path]:
    root = config.baseline_root
    seed_dir = root / method / f"seed_{seed}"
    gt = seed_dir / "gps_pose.tum"
    if method == "rtab_rgb":
        est = seed_dir / "rtabmap_rgb_filtered.tum"
        if not est.exists():
            est = seed_dir / "trajectory_0.5.tum"
    elif method == "rtab_rgbd":
        est = seed_dir / "rtabmap_rgbd_filtered.tum"
        if not est.exists():
            est = seed_dir / "trajectory_0.5.tum"
    else:
        est = seed_dir / "trajectory_0.5.tum"
    return est, gt


def build_slpf_degraded_cmd(
    python_exec: Path,
    out_dir: Path,
    seed: int,
    data_path: Path,
    degraded_gnss_tum: Path,
    max_frames: int | None,
    require_cuda: bool,
) -> list[str]:
    cmd = [
        str(python_exec),
        str(BASE_DIR / "scripts" / "spf_lidar.py"),
        "--miss-penalty", "4.0",
        "--wrong-hit-penalty", "4.0",
        "--gps-weight", "0.5",
        "--seed", str(seed),
        "--output-folder", str(out_dir),
        "--frame-stride", "4",
        "--semantic-sigma", "0.05",
        "--gps-sigma", "1.1",
        "--corridor-weight", "0.30",
        "--corridor-dist-sigma", "1.50",
        "--corridor-heading-sigma", "0.35",
        "--background-class-weight", "0.20",
        "--max-background-obs", "120",
        "--expected-obs-count", "150",
        "--pose-smooth-alpha-pos", "0.55",
        "--pose-smooth-alpha-theta", "0.50",
        "--odom-yaw-filter-alpha", "0.90",
        "--particle-count", "100",
        "--data-path", str(data_path),
        "--external-noisy-gnss-tum", str(degraded_gnss_tum),
        "--no-visualization",
    ]
    if max_frames is not None:
        cmd.extend(["--max-frames", str(max_frames)])
    if require_cuda:
        cmd.append("--require-cuda")
    return cmd


def valid_rows(est_pos: np.ndarray, gt_pos: np.ndarray) -> np.ndarray:
    return np.isfinite(est_pos).all(axis=1) & np.isfinite(gt_pos).all(axis=1)


def ape_stats(errors: np.ndarray, prefix: str) -> dict[str, float]:
    errors = np.asarray(errors, dtype=np.float64)
    errors = errors[np.isfinite(errors)]
    if errors.size == 0:
        return {
            f"{prefix}_rmse": float("nan"),
            f"{prefix}_mean": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    return {
        f"{prefix}_rmse": float(np.sqrt(np.mean(errors * errors))),
        f"{prefix}_mean": float(np.mean(errors)),
        f"{prefix}_median": float(np.median(errors)),
        f"{prefix}_max": float(np.max(errors)),
    }


def evaluate_python_metrics(est_tum: Path, gt_tum: Path, rows_map: dict[str, np.ndarray]) -> dict[str, float | str]:
    est = read_tum_file(est_tum)
    gt = read_tum_file(gt_tum)
    gt_interp = interpolate_positions(gt.timestamps, gt.positions, est.timestamps)
    finite = valid_rows(est.positions, gt_interp)
    if not np.any(finite):
        raise ValueError(f"No finite overlapping poses for {est_tum}")

    est_pos = est.positions[finite]
    gt_pos = gt_interp[finite]
    ts = est.timestamps[finite]
    raw_errors = np.linalg.norm(est_pos - gt_pos, axis=1)
    scale, rot, trans = umeyama_alignment(est_pos, gt_pos, with_scaling=False)
    est_aligned = apply_transform(est_pos, scale, rot, trans)
    aligned_errors = np.linalg.norm(est_aligned - gt_pos, axis=1)

    out: dict[str, float | str] = {
        "eval_est_tum": str(est_tum),
        "gt_tum": str(gt_tum),
        "finite_samples": int(np.sum(finite)),
        "dropped_samples": int(len(finite) - np.sum(finite)),
        "alignment_scale": float(scale),
    }
    out.update(ape_stats(raw_errors, "ape_raw"))
    out.update(ape_stats(aligned_errors, "ape_align"))
    out.update(compute_row_metrics(est_aligned, gt_pos, rows_map, timestamps=ts))
    out.update(compute_smoothness_metrics(ts, est_aligned))
    return out


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = [
        "ape_raw_rmse",
        "ape_align_rmse",
        "row_correct_fraction",
        "failure_rate",
        "wrong_row_duration_sec",
        "cross_track_mean",
        "speed_mean",
        "accel_rms",
    ]
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["traversal"]), str(row["profile"]), str(row["method"])), []).append(row)
    out = []
    for (traversal, profile, method), group in sorted(groups.items()):
        item: dict[str, object] = {"traversal": traversal, "profile": profile, "method": method, "n_runs": len(group)}
        for key in keys:
            vals = np.asarray([float(r.get(key, float("nan"))) for r in group], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            item[f"{key}_mean"] = float(np.mean(vals)) if vals.size else float("nan")
            item[f"{key}_std"] = float(np.std(vals)) if vals.size else float("nan")
        out.append(item)
    return out


def compact_summary(agg_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in agg_rows:
        groups.setdefault((str(row["profile"]), str(row["method"])), []).append(row)
    out = []
    for (profile, method), group in sorted(groups.items()):
        item: dict[str, object] = {"profile": profile, "method": method, "n_traversals": len(group)}
        for key in ("ape_align_rmse_mean", "row_correct_fraction_mean", "failure_rate_mean"):
            vals = np.asarray([float(r.get(key, float("nan"))) for r in group], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            item[key.replace("_mean", "_across_traversals_mean")] = float(np.mean(vals)) if vals.size else float("nan")
        out.append(item)
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python-exec", type=Path, default=Path("python3"))
    parser.add_argument("--geojson", type=Path, default=DEFAULT_GEOJSON)
    parser.add_argument("--traversals", default="rh1,rh2")
    parser.add_argument("--rh1-data-path", type=Path, default=None)
    parser.add_argument("--rh2-data-path", type=Path, default=None)
    parser.add_argument("--profiles", default=None)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--methods", default="slpf,ngps,amcl_ngps,rtab_rgb_ngps,rtab_rgbd_ngps")
    parser.add_argument("--dropout-mode", choices=["nan", "hold", "remove"], default="nan")
    parser.add_argument("--headland-fraction", type=float, default=0.15)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--allow-existing-slpf", action="store_true", help="Use existing SLPF TUM if a live degraded-GNSS SLPF run cannot be executed.")
    parser.add_argument("--amcl-ngps-amcl-std", type=float, default=DEFAULT_AMCL_NGPS_AMCL_STD)
    parser.add_argument("--amcl-ngps-gps-std", type=float, default=DEFAULT_AMCL_NGPS_GPS_STD)
    parser.add_argument("--amcl-ngps-process-std", type=float, default=DEFAULT_AMCL_NGPS_PROCESS_STD)
    parser.add_argument("--rtab-ngps-rtab-std", type=float, default=DEFAULT_RTAB_NGPS_RTAB_STD)
    parser.add_argument("--rtab-ngps-gps-std", type=float, default=DEFAULT_RTAB_NGPS_GPS_STD)
    parser.add_argument("--rtab-ngps-process-std", type=float, default=DEFAULT_RTAB_NGPS_PROCESS_STD)
    return parser.parse_args(argv)


def traversal_config_from_args(name: str, args: argparse.Namespace) -> TraversalConfig:
    config = TRAVERSALS[name]
    override = getattr(args, f"{name}_data_path")
    if override is None:
        return config
    override = override.expanduser()
    if not override.is_absolute():
        override = (BASE_DIR / override).resolve()
    return TraversalConfig(name=config.name, data_path=override, baseline_root=config.baseline_root)


def validate_slpf_inputs(config: TraversalConfig) -> None:
    missing = []
    for path in [config.data_path / "data.csv", config.data_path / "rgb", config.data_path / "depth", config.data_path / "lidar"]:
        if not path.exists():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing SLPF dataset inputs: " + ", ".join(missing))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seeds = parse_int_list(args.seeds)
    traversals = parse_name_list(args.traversals, TRAVERSALS.keys())
    profiles = selected_profiles(args.profiles)
    methods = parse_name_list(args.methods, METHOD_LABELS.keys())
    rows_map = load_rows_from_geojson(args.geojson.resolve())
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(BASE_DIR / ".tmp_mpl")
    (BASE_DIR / ".tmp_mpl").mkdir(parents=True, exist_ok=True)

    per_seed_rows: list[dict[str, object]] = []
    protocol: dict[str, object] = {
        "output_root": str(output_root),
        "seeds": seeds,
        "traversals": traversals,
        "methods": methods,
        "dropout_mode": args.dropout_mode,
        "headland_fraction": args.headland_fraction,
        "profiles": [
            {
                "name": profile.name,
                "region": profile.region,
                "outage_seconds": profile.outage_seconds,
                "noise": asdict(profile.noise),
            }
            for profile in profiles
        ],
        "commands": [],
    }

    for traversal_name in traversals:
        config = traversal_config_from_args(traversal_name, args)
        if "slpf" in methods and not args.allow_existing_slpf:
            validate_slpf_inputs(config)
        _, gt_source = tum_for_method(config, "slpf", seeds[0])
        gt_data = read_tum_file(gt_source)
        gt_array = np.column_stack([gt_data.timestamps, gt_data.positions, gt_data.quaternions])

        for profile in profiles:
            for seed in seeds:
                work_dir = output_root / traversal_name / profile.name / f"seed_{seed}"
                work_dir.mkdir(parents=True, exist_ok=True)
                degraded, affected_mask, error = apply_profile(
                    gt_array,
                    profile,
                    seed,
                    dropout_mode=args.dropout_mode,
                    headland_fraction=args.headland_fraction,
                )
                degraded_tum = work_dir / "degraded_noisy_gnss.tum"
                write_tum(
                    str(degraded_tum),
                    degraded,
                    header=f"generated by run_gnss_degradation_localization_eval.py traversal={traversal_name} profile={profile.name} seed={seed}",
                )

                method_inputs: dict[str, tuple[Path, Path, str]] = {}
                if "ngps" in methods:
                    method_inputs["ngps"] = (degraded_tum, gt_source, "generated_degraded_gnss")

                if "slpf" in methods:
                    slpf_dir = work_dir / "slpf"
                    slpf_est = slpf_dir / "trajectory_0.5.tum"
                    slpf_gt = slpf_dir / "gps_pose.tum"
                    slpf_source = "live_external_degraded_gnss"
                    cmd = build_slpf_degraded_cmd(
                        args.python_exec,
                        slpf_dir,
                        seed,
                        config.data_path,
                        degraded_tum,
                        args.max_frames,
                        bool(args.require_cuda),
                    )
                    try:
                        runtime = run_cmd(cmd, slpf_dir / "run_slpf_degraded.log", cwd=BASE_DIR, env=env)
                        protocol["commands"].append({"method": "SLPF", "traversal": traversal_name, "profile": profile.name, "seed": seed, "command": cmd, "runtime_sec": runtime})
                    except Exception:
                        if not args.allow_existing_slpf:
                            raise
                        existing_est, existing_gt = tum_for_method(config, "slpf", seed)
                        slpf_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(existing_est, slpf_est)
                        shutil.copy2(existing_gt, slpf_gt)
                        slpf_source = "existing_clean_gnss_fallback"
                    method_inputs["slpf"] = (slpf_est, slpf_gt if slpf_gt.exists() else gt_source, slpf_source)

                if "amcl_ngps" in methods:
                    amcl_est, amcl_gt = tum_for_method(config, "amcl", seed)
                    out_dir = work_dir / "amcl_ngps"
                    out_est = out_dir / "trajectory_0.5.tum"
                    out_gt = out_dir / "gps_pose.tum"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(amcl_gt, out_gt)
                    build_kalman_ngps_fused_tum(
                        primary_est=amcl_est,
                        ngps_est=degraded_tum,
                        out_est=out_est,
                        primary_pos_std=args.amcl_ngps_amcl_std,
                        gps_pos_std=args.amcl_ngps_gps_std,
                        process_accel_std=args.amcl_ngps_process_std,
                    )
                    method_inputs["amcl_ngps"] = (out_est, out_gt, "kalman_amcl_degraded_gnss")

                for rtab_key, primary_method in (("rtab_rgb_ngps", "rtab_rgb"), ("rtab_rgbd_ngps", "rtab_rgbd")):
                    if rtab_key not in methods:
                        continue
                    rtab_est, rtab_gt = tum_for_method(config, primary_method, seed)
                    out_dir = work_dir / rtab_key
                    out_est = out_dir / "trajectory_0.5.tum"
                    out_gt = out_dir / "gps_pose.tum"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(rtab_gt, out_gt)
                    build_kalman_ngps_fused_tum(
                        primary_est=rtab_est,
                        ngps_est=degraded_tum,
                        out_est=out_est,
                        primary_pos_std=args.rtab_ngps_rtab_std,
                        gps_pos_std=args.rtab_ngps_gps_std,
                        process_accel_std=args.rtab_ngps_process_std,
                        primary_gt=rtab_gt,
                        anchor_primary_start_pose=True,
                    )
                    method_inputs[rtab_key] = (out_est, out_gt, "kalman_start_pose_anchored_rtab_degraded_gnss")

                for method_key, (est_tum, gt_tum, source) in method_inputs.items():
                    metrics = evaluate_python_metrics(est_tum, gt_tum, rows_map)
                    metrics.update(
                        {
                            "traversal": traversal_name,
                            "profile": profile.name,
                            "seed": seed,
                            "method": METHOD_LABELS[method_key],
                            "method_key": method_key,
                            "source": source,
                            "degraded_gnss_tum": str(degraded_tum),
                            "affected_fraction": float(np.mean(affected_mask)),
                        }
                    )
                    per_seed_rows.append(metrics)

    write_csv(output_root / "localization_metrics_per_seed.csv", per_seed_rows)
    agg_rows = aggregate(per_seed_rows)
    write_csv(output_root / "localization_metrics_aggregate.csv", agg_rows)
    write_csv(output_root / "compact_summary.csv", compact_summary(agg_rows))
    (output_root / "run_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"[INFO] Wrote GNSS degradation localisation outputs to {output_root}")


if __name__ == "__main__":
    main()
