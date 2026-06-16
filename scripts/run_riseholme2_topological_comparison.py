#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

from run_ab_validation import (
    aligned_estimate,
    compute_row_metrics,
    compute_smoothness_metrics,
    evaluate_run,
    interpolate_positions,
    load_rows_from_geojson,
    read_tum_file,
    run_cmd,
)

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = BASE_DIR / "results" / "iros_rh2_topological"
DEFAULT_DATA_PATH = BASE_DIR / "data" / "2025" / "rh_run2"
DEFAULT_GEOJSON = BASE_DIR / "data" / "riseholme_poles_trunk.geojson"
DEFAULT_GRAPH_JSON = BASE_DIR / "data" / "topological" / "riseholme_midrow_outer_graph.json"
PUBLISHED_SEED22 = (
    BASE_DIR
    / "results"
    / "iros_rh2"
    / "20260225_105822_multiseed_all_methods"
    / "slpf"
    / "seed_22"
    / "trajectory_0.5.tum"
)


def parse_int_list(text: str) -> List[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def safe_float(value, default=float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
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
        writer.writerows(rows)


def aggregate_by_method(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    metric_keys = [
        "ape_raw_rmse",
        "ape_align_rmse",
        "rpe_2m_align_rmse",
        "rpe_5m_align_rmse",
        "rpe_10m_align_rmse",
        "cross_track_mean",
        "row_correct_fraction",
        "row_switch_events",
        "speed_mean",
        "accel_rms",
        "jerk_rms",
        "heading_rate_rms",
        "heading_accel_rms",
        "runtime_sec",
    ]
    out = []
    for method in sorted({str(row["method"]) for row in rows}):
        group = [row for row in rows if str(row["method"]) == method]
        agg: Dict[str, object] = {"method": method, "n_runs": len(group)}
        for metric in metric_keys:
            values = np.asarray([safe_float(row.get(metric)) for row in group], dtype=np.float64)
            values = values[np.isfinite(values)]
            agg[f"{metric}_mean"] = float(np.mean(values)) if values.size else float("nan")
            agg[f"{metric}_std"] = float(np.std(values)) if values.size else float("nan")
            agg[f"{metric}_median"] = float(np.median(values)) if values.size else float("nan")
        out.append(agg)
    return out


def rmse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(values * values)))


def distance_rpe_rmse(est_aligned: np.ndarray, gt_interp: np.ndarray, delta_m: float) -> float:
    gt_xy = np.asarray(gt_interp[:, :2], dtype=np.float64)
    est_xy = np.asarray(est_aligned[:, :2], dtype=np.float64)
    if len(gt_xy) < 2:
        return float("nan")

    step = np.linalg.norm(np.diff(gt_xy, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(step)])
    errors: list[float] = []
    for i, start_dist in enumerate(cumulative[:-1]):
        j = int(np.searchsorted(cumulative, start_dist + delta_m, side="left"))
        if j >= len(cumulative):
            continue
        est_delta = est_xy[j] - est_xy[i]
        gt_delta = gt_xy[j] - gt_xy[i]
        errors.append(float(np.linalg.norm(est_delta - gt_delta)))
    return rmse(np.asarray(errors, dtype=np.float64))


def local_fallback_evaluate_run(
    *,
    name: str,
    est_tum: Path,
    gt_tum: Path,
    rows: Dict[str, np.ndarray],
    warning: str,
) -> Dict[str, float | str]:
    est = read_tum_file(est_tum)
    gt = read_tum_file(gt_tum)
    gt_interp_raw = interpolate_positions(gt.timestamps, gt.positions, est.timestamps)
    raw_errors = np.linalg.norm(est.positions - gt_interp_raw, axis=1)

    aligned = aligned_estimate(est_tum, gt_tum)
    align_errors = np.asarray(aligned["errors"], dtype=np.float64)
    row_metrics = compute_row_metrics(aligned["est_aligned"], aligned["gt_interp"], rows)
    smooth = compute_smoothness_metrics(aligned["timestamps"], aligned["est_aligned"])

    return {
        "run_name": name,
        "est_tum": str(est_tum),
        "gt_tum": str(gt_tum),
        "ape_raw_rmse": rmse(raw_errors),
        "ape_align_rmse": rmse(align_errors),
        "rpe_2m_align_rmse": distance_rpe_rmse(aligned["est_aligned"], aligned["gt_interp"], 2.0),
        "rpe_5m_align_rmse": distance_rpe_rmse(aligned["est_aligned"], aligned["gt_interp"], 5.0),
        "rpe_10m_align_rmse": distance_rpe_rmse(aligned["est_aligned"], aligned["gt_interp"], 10.0),
        "evaluation_backend": "local_fallback",
        "evaluation_warning": warning,
        **row_metrics,
        **smooth,
    }


def evaluate_run_with_fallback(
    *,
    name: str,
    est_tum: Path,
    gt_tum: Path,
    out_dir: Path,
    rows: Dict[str, np.ndarray],
    evo_ape_bin: Path,
    evo_rpe_bin: Path,
    env: Dict[str, str],
) -> Dict[str, float | str]:
    try:
        metrics = evaluate_run(
            name=name,
            est_tum=est_tum,
            gt_tum=gt_tum,
            out_dir=out_dir,
            rows=rows,
            evo_ape_bin=evo_ape_bin,
            evo_rpe_bin=evo_rpe_bin,
            env=env,
        )
        metrics["evaluation_backend"] = "evo"
        metrics["evaluation_warning"] = ""
        return metrics
    except RuntimeError as exc:
        return local_fallback_evaluate_run(
            name=name,
            est_tum=est_tum,
            gt_tum=gt_tum,
            rows=rows,
            warning=str(exc),
        )


def resolve(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else (BASE_DIR / path).resolve()


def check_cuda_available(python_exec: Path, env: Dict[str, str]) -> bool:
    proc = subprocess.run(
        [str(python_exec), "-c", "import torch; print('1' if torch.cuda.is_available() else '0')"],
        cwd=str(BASE_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    lines = (proc.stdout or "").strip().splitlines()
    return proc.returncode == 0 and bool(lines) and lines[-1].strip() == "1"


def common_slpf_args(
    *,
    seed: int,
    out_dir: Path,
    data_path: Path,
    geojson_path: Path,
    max_frames: int | None,
    require_cuda: bool,
) -> list[str]:
    cmd = [
        "--miss-penalty",
        "4.0",
        "--wrong-hit-penalty",
        "4.0",
        "--gps-weight",
        "0.5",
        "--seed",
        str(seed),
        "--output-folder",
        str(out_dir),
        "--frame-stride",
        "4",
        "--semantic-sigma",
        "0.05",
        "--gps-sigma",
        "1.1",
        "--corridor-weight",
        "0.30",
        "--corridor-dist-sigma",
        "1.50",
        "--corridor-heading-sigma",
        "0.35",
        "--background-class-weight",
        "0.20",
        "--max-background-obs",
        "120",
        "--expected-obs-count",
        "150",
        "--pose-smooth-alpha-pos",
        "0.55",
        "--pose-smooth-alpha-theta",
        "0.50",
        "--odom-yaw-filter-alpha",
        "0.90",
        "--particle-count",
        "100",
        "--segment-chunk",
        "4096",
        "--no-visualization",
        "--data-path",
        str(data_path),
        "--geojson-path",
        str(geojson_path),
    ]
    if max_frames is not None:
        cmd.extend(["--max-frames", str(max_frames)])
    if require_cuda:
        cmd.append("--require-cuda")
    return cmd


def build_standard_cmd(
    python_exec: Path,
    seed: int,
    out_dir: Path,
    data_path: Path,
    geojson_path: Path,
    max_frames: int | None,
    require_cuda: bool,
) -> list[str]:
    return [
        str(python_exec),
        str(BASE_DIR / "scripts" / "spf_lidar.py"),
        *common_slpf_args(
            seed=seed,
            out_dir=out_dir,
            data_path=data_path,
            geojson_path=geojson_path,
            max_frames=max_frames,
            require_cuda=require_cuda,
        ),
    ]


def build_topological_cmd(
    python_exec: Path,
    seed: int,
    out_dir: Path,
    data_path: Path,
    geojson_path: Path,
    graph_json: Path,
    topological_pf_root: Path,
    max_frames: int | None,
    require_cuda: bool,
) -> list[str]:
    return [
        str(python_exec),
        str(BASE_DIR / "scripts" / "run_topological_slpf.py"),
        *common_slpf_args(
            seed=seed,
            out_dir=out_dir,
            data_path=data_path,
            geojson_path=geojson_path,
            max_frames=max_frames,
            require_cuda=require_cuda,
        ),
        "--graph-json",
        str(graph_json),
        "--topological-pf-root",
        str(topological_pf_root),
    ]


def read_tum_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    stamps = []
    xy = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            stamps.append(float(parts[0]))
            xy.append([float(parts[1]), float(parts[2])])
    return np.asarray(stamps, dtype=np.float64), np.asarray(xy, dtype=np.float64)


def assert_topological_trajectory_on_graph(tum_path: Path, graph_json: Path, tolerance: float = 1e-9) -> None:
    payload = json.loads(graph_json.read_text(encoding="utf-8"))
    graph_xy = np.asarray([[float(node["x"]), float(node["y"])] for node in payload["nodes"]], dtype=np.float64)
    _, trajectory_xy = read_tum_xy(tum_path)
    if trajectory_xy.size == 0:
        raise ValueError(f"No trajectory poses found in {tum_path}")
    diffs = trajectory_xy[:, None, :] - graph_xy[None, :, :]
    nearest = np.sqrt(np.sum(diffs * diffs, axis=2)).min(axis=1)
    max_distance = float(np.max(nearest))
    if max_distance > tolerance:
        raise AssertionError(f"Topological trajectory leaves graph nodes: max nearest-node distance {max_distance:.6g} m")


def check_standard_seed22_timestamps(tum_path: Path) -> dict:
    stamps, _ = read_tum_xy(tum_path)
    out = {"checked": False}
    if stamps.size >= 3:
        out.update(
            {
                "checked": True,
                "first_timestamps": [float(x) for x in stamps[:3]],
                "matches_published_prefix": bool(np.allclose(stamps[:3], np.asarray([0.0, 4.0, 8.0]))),
            }
        )
    if PUBLISHED_SEED22.exists() and stamps.size:
        published_stamps, _ = read_tum_xy(PUBLISHED_SEED22)
        n = min(8, len(stamps), len(published_stamps))
        out["matches_committed_seed22_timestamps"] = bool(np.allclose(stamps[:n], published_stamps[:n]))
    return out


def run_plot_cmd(
    python_exec: Path,
    geojson_path: Path,
    graph_json: Path,
    trajectory_tum: Path,
    reference_tum: Path,
    map_out: Path,
    trajectory_out: Path,
    title: str,
    log_path: Path,
    env: Dict[str, str],
) -> None:
    cmd = [
        str(python_exec),
        str(BASE_DIR / "scripts" / "plot_topological_outputs.py"),
        "--geojson-path",
        str(geojson_path),
        "--graph-json",
        str(graph_json),
        "--trajectory-tum",
        str(trajectory_tum),
        "--reference-tum",
        str(reference_tum),
        "--map-out",
        str(map_out),
        "--trajectory-out",
        str(trajectory_out),
        "--trajectory-title",
        title,
    ]
    run_cmd(cmd, log_path, cwd=BASE_DIR, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Riseholme-2 SLPF vs graph-constrained SLPF comparison.")
    parser.add_argument("--python-exec", type=Path, default=BASE_DIR / ".venv310" / "bin" / "python")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--geojson-path", type=Path, default=DEFAULT_GEOJSON)
    parser.add_argument("--graph-json", type=Path, default=DEFAULT_GRAPH_JSON)
    parser.add_argument("--topological-pf-root", type=Path, default=WORKSPACE_DIR_DEFAULT())
    parser.add_argument("--seeds", type=str, default="11,22,33")
    parser.add_argument("--representative-seed", type=int, default=22)
    parser.add_argument("--cuda-visible-devices", type=str, default="0")
    parser.add_argument("--require-cuda", dest="require_cuda", action="store_true", default=True)
    parser.add_argument("--allow-cpu", dest="require_cuda", action="store_false")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--skip-standard", action="store_true")
    parser.add_argument("--skip-topological", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    python_exec = resolve(args.python_exec)
    data_path = resolve(args.data_path)
    geojson_path = resolve(args.geojson_path)
    graph_json = resolve(args.graph_json)
    topological_pf_root = resolve(args.topological_pf_root)
    output_root = resolve(args.output_root)
    seeds = parse_int_list(args.seeds)
    if not seeds:
        raise ValueError("At least one seed is required.")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(BASE_DIR / ".tmp_mpl")
    (BASE_DIR / ".tmp_mpl").mkdir(parents=True, exist_ok=True)

    if args.require_cuda and not check_cuda_available(python_exec, env):
        raise RuntimeError("CUDA is required but not visible in the selected Python runtime/environment.")

    evo_ape_bin = python_exec.parent / "evo_ape"
    evo_rpe_bin = python_exec.parent / "evo_rpe"
    if not evo_ape_bin.exists() or not evo_rpe_bin.exists():
        raise FileNotFoundError("evo_ape/evo_rpe not found in virtualenv bin directory.")

    for required in (data_path / "data.csv", geojson_path, graph_json, topological_pf_root):
        if not required.exists():
            raise FileNotFoundError(required)

    rows_map = load_rows_from_geojson(geojson_path)
    run_dir = output_root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_slpf_vs_topological"
    run_dir.mkdir(parents=True, exist_ok=True)

    per_seed_rows: list[dict] = []
    protocol: dict = {
        "run_dir": str(run_dir),
        "seeds": seeds,
        "representative_seed": int(args.representative_seed),
        "data_path": str(data_path),
        "geojson_path": str(geojson_path),
        "graph_json": str(graph_json),
        "topological_pf_root": str(topological_pf_root),
        "published_reference": str(PUBLISHED_SEED22),
        "cuda_visible_devices": str(args.cuda_visible_devices),
        "require_cuda": bool(args.require_cuda),
        "commands": [],
        "validation": {},
    }

    for seed in seeds:
        if not args.skip_standard:
            out_dir = run_dir / "slpf" / f"seed_{seed}"
            cmd = build_standard_cmd(python_exec, seed, out_dir, data_path, geojson_path, args.max_frames, args.require_cuda)
            runtime = run_cmd(cmd, out_dir / "run_slpf.log", cwd=BASE_DIR, env=env)
            metrics = evaluate_run_with_fallback(
                name=f"slpf_seed_{seed}",
                est_tum=out_dir / "trajectory_0.5.tum",
                gt_tum=out_dir / "gps_pose.tum",
                out_dir=out_dir / "eval",
                rows=rows_map,
                evo_ape_bin=evo_ape_bin,
                evo_rpe_bin=evo_rpe_bin,
                env=env,
            )
            metrics.update({"method": "SLPF(ours)", "seed": seed, "runtime_sec": runtime})
            per_seed_rows.append(metrics)
            protocol["commands"].append({"method": "SLPF(ours)", "seed": seed, "command": cmd, "runtime_sec": runtime})
            if seed == 22:
                protocol["validation"]["seed22_standard_timestamps"] = check_standard_seed22_timestamps(out_dir / "trajectory_0.5.tum")

        if not args.skip_topological:
            out_dir = run_dir / "slpf_topological" / f"seed_{seed}"
            cmd = build_topological_cmd(
                python_exec,
                seed,
                out_dir,
                data_path,
                geojson_path,
                graph_json,
                topological_pf_root,
                args.max_frames,
                args.require_cuda,
            )
            runtime = run_cmd(cmd, out_dir / "run_topological_slpf.log", cwd=BASE_DIR, env=env)
            assert_topological_trajectory_on_graph(out_dir / "trajectory_0.5.tum", graph_json)
            metrics = evaluate_run_with_fallback(
                name=f"slpf_topological_seed_{seed}",
                est_tum=out_dir / "trajectory_0.5.tum",
                gt_tum=out_dir / "gps_pose.tum",
                out_dir=out_dir / "eval",
                rows=rows_map,
                evo_ape_bin=evo_ape_bin,
                evo_rpe_bin=evo_rpe_bin,
                env=env,
            )
            metrics.update({"method": "SLPF-topological", "seed": seed, "runtime_sec": runtime})
            per_seed_rows.append(metrics)
            protocol["commands"].append({"method": "SLPF-topological", "seed": seed, "command": cmd, "runtime_sec": runtime})

    write_csv(run_dir / "trajectory_metrics_multiseed_per_seed.csv", per_seed_rows)
    agg_rows = aggregate_by_method(per_seed_rows)
    write_csv(run_dir / "trajectory_metrics_multiseed_aggregate.csv", agg_rows)

    if not args.skip_plots:
        rep = int(args.representative_seed)
        map_out = run_dir / "topological_map_semantics.png"
        if not args.skip_topological:
            topo_dir = run_dir / "slpf_topological" / f"seed_{rep}"
            if topo_dir.exists():
                run_plot_cmd(
                    python_exec,
                    geojson_path,
                    graph_json,
                    topo_dir / "trajectory_0.5.tum",
                    topo_dir / "gps_pose.tum",
                    map_out,
                    run_dir / f"topological_trajectory_error_seed_{rep}.png",
                    f"Topological SLPF Trajectory Error (seed {rep})",
                    run_dir / "plot_topological.log",
                    env,
                )
        if not args.skip_standard:
            slpf_dir = run_dir / "slpf" / f"seed_{rep}"
            if slpf_dir.exists():
                run_plot_cmd(
                    python_exec,
                    geojson_path,
                    graph_json,
                    slpf_dir / "trajectory_0.5.tum",
                    slpf_dir / "gps_pose.tum",
                    map_out,
                    run_dir / f"slpf_trajectory_error_seed_{rep}.png",
                    f"Standard SLPF Trajectory Error (seed {rep})",
                    run_dir / "plot_standard.log",
                    env,
                )

    (run_dir / "run_protocol.json").write_text(json.dumps(protocol, indent=2, default=str), encoding="utf-8")
    print(f"[INFO] Riseholme-2 comparison complete: {run_dir}")


def WORKSPACE_DIR_DEFAULT() -> Path:
    return BASE_DIR.parent / "topological_pf"


if __name__ == "__main__":
    main()
