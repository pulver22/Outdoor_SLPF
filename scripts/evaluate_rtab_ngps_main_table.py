#!/usr/bin/env python3
"""Evaluate RTAB-Map+NoisyGNSS rows for the IROS main table."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".tmp_mpl"))
os.environ.setdefault("HOME", str(Path(__file__).resolve().parents[1]))

try:
    from run_ab_validation import BASE_DIR, evaluate_run, load_rows_from_geojson
    from run_iros_multiseed import (
        DEFAULT_RTAB_NGPS_GPS_STD,
        DEFAULT_RTAB_NGPS_PROCESS_STD,
        DEFAULT_RTAB_NGPS_RTAB_STD,
        aggregate_by_method,
        build_kalman_ngps_fused_tum,
        parse_int_list,
        write_csv,
    )
except ModuleNotFoundError:
    from scripts.run_ab_validation import BASE_DIR, evaluate_run, load_rows_from_geojson
    from scripts.run_iros_multiseed import (
        DEFAULT_RTAB_NGPS_GPS_STD,
        DEFAULT_RTAB_NGPS_PROCESS_STD,
        DEFAULT_RTAB_NGPS_RTAB_STD,
        aggregate_by_method,
        build_kalman_ngps_fused_tum,
        parse_int_list,
        write_csv,
    )


DEFAULT_OUTPUT_ROOT = BASE_DIR / "results" / "iros_revision" / "rtab_ngps_main_table"
DEFAULT_GEOJSON = BASE_DIR / "data" / "riseholme_poles_trunk.geojson"


TRAVERSAL_ROOTS = {
    "exp1": BASE_DIR / "results" / "iros_rh1" / "20260217_172656_multiseed_main",
    "exp2": BASE_DIR / "results" / "iros_rh2" / "20260225_105822_multiseed_all_methods",
}


def rtab_est_path(root: Path, method: str, seed: int) -> Path:
    seed_dir = root / method / f"seed_{seed}"
    if method == "rtab_rgb":
        preferred = seed_dir / "rtabmap_rgb_filtered.tum"
    elif method == "rtab_rgbd":
        preferred = seed_dir / "rtabmap_rgbd_filtered.tum"
    else:
        raise ValueError(f"Unsupported RTAB method: {method}")
    return preferred if preferred.exists() else seed_dir / "trajectory_0.5.tum"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--geojson", type=Path, default=DEFAULT_GEOJSON)
    parser.add_argument("--seeds", default="11,22,33")
    parser.add_argument("--evo-bin-dir", type=Path, default=BASE_DIR / ".venv" / "bin")
    parser.add_argument("--rtab-std", type=float, default=DEFAULT_RTAB_NGPS_RTAB_STD)
    parser.add_argument("--gps-std", type=float, default=DEFAULT_RTAB_NGPS_GPS_STD)
    parser.add_argument("--process-std", type=float, default=DEFAULT_RTAB_NGPS_PROCESS_STD)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seeds = parse_int_list(args.seeds)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows_map = load_rows_from_geojson(args.geojson.resolve())
    evo_ape = args.evo_bin_dir / "evo_ape"
    evo_rpe = args.evo_bin_dir / "evo_rpe"
    if not evo_ape.exists() or not evo_rpe.exists():
        raise FileNotFoundError(f"Missing evo binaries under {args.evo_bin_dir}")

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(BASE_DIR / ".tmp_mpl")
    env["HOME"] = str(BASE_DIR)
    (BASE_DIR / ".tmp_mpl").mkdir(parents=True, exist_ok=True)

    per_seed_rows = []
    protocol = {
        "output_root": str(output_root),
        "seeds": seeds,
        "rtab_ngps": {
            "rtab_std_m": float(args.rtab_std),
            "gps_std_m": float(args.gps_std),
            "process_accel_std_mps2": float(args.process_std),
            "anchor_primary_start_pose": True,
        },
    }

    for traversal, root in TRAVERSAL_ROOTS.items():
        for seed in seeds:
            ngps = root / "ngps" / f"seed_{seed}" / "trajectory_0.5.tum"
            for rtab_method, label in (
                ("rtab_rgb", "RTAB RGB+NoisyGNSS"),
                ("rtab_rgbd", "RTAB RGBD+NoisyGNSS"),
            ):
                seed_dir = root / rtab_method / f"seed_{seed}"
                rtab_est = rtab_est_path(root, rtab_method, seed)
                rtab_gt = seed_dir / "gps_pose.tum"
                out_dir = output_root / traversal / rtab_method / f"seed_{seed}"
                out_est = out_dir / "trajectory_0.5.tum"
                out_gt = out_dir / "gps_pose.tum"
                out_gt.parent.mkdir(parents=True, exist_ok=True)
                out_gt.write_text(rtab_gt.read_text(encoding="utf-8"), encoding="utf-8")
                build_kalman_ngps_fused_tum(
                    primary_est=rtab_est,
                    ngps_est=ngps,
                    out_est=out_est,
                    primary_pos_std=float(args.rtab_std),
                    gps_pos_std=float(args.gps_std),
                    process_accel_std=float(args.process_std),
                    primary_gt=rtab_gt,
                    anchor_primary_start_pose=True,
                )
                metrics = evaluate_run(
                    name=f"{traversal}_{rtab_method}_ngps_seed_{seed}",
                    est_tum=out_est,
                    gt_tum=out_gt,
                    out_dir=out_dir / "eval",
                    rows=rows_map,
                    evo_ape_bin=evo_ape,
                    evo_rpe_bin=evo_rpe,
                    env=env,
                )
                metrics.update({"method": label, "method_key": f"{rtab_method}_ngps", "traversal": traversal, "seed": seed, "runtime_sec": 0.0})
                per_seed_rows.append(metrics)

    per_seed_rows.sort(key=lambda r: (str(r["traversal"]), str(r["method"]), int(r["seed"])))
    write_csv(output_root / "rtab_ngps_main_table_per_seed.csv", per_seed_rows)

    for traversal in sorted(TRAVERSAL_ROOTS):
        traversal_rows = [row for row in per_seed_rows if row["traversal"] == traversal]
        write_csv(output_root / f"{traversal}_rtab_ngps_main_table_aggregate.csv", aggregate_by_method(traversal_rows))

    output_root.joinpath("run_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"[INFO] Wrote RTAB+NoisyGNSS main-table metrics to {output_root}")


if __name__ == "__main__":
    main()
