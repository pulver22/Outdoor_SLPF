#!/usr/bin/env python3
"""Run fixed localisation follow-up experiments and evaluate their trajectories."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from scripts.run_ab_validation import (
        DEFAULT_GEOJSON,
        aligned_estimate,
        classify_headland_mask,
        compute_row_metrics,
        compute_smoothness_metrics,
        evaluate_run,
        interpolate_positions,
        load_rows_from_geojson,
        read_tum_file,
    )
    from scripts.experiment_runtime import build_experiment_env
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution.
    from run_ab_validation import (
        DEFAULT_GEOJSON,
        aligned_estimate,
        classify_headland_mask,
        compute_row_metrics,
        compute_smoothness_metrics,
        evaluate_run,
        interpolate_positions,
        load_rows_from_geojson,
        read_tum_file,
    )
    from experiment_runtime import build_experiment_env


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_EXTERNAL_PYTHON = Path("/home/pulver/projects/vineyard_pf/Outdoor_SLPF/.venv/bin/python")
DEFAULT_OUTPUT_ROOT = BASE_DIR / "results/localization_improvement_followup"
FULL_DATA_ROOT = Path("/home/pulver/projects/vineyard_pf/Outdoor_SLPF/data/2025")
DEFAULT_DATA_ROOT = FULL_DATA_ROOT if FULL_DATA_ROOT.exists() else BASE_DIR / "data/2025"
KEY_METRICS = [
    "ape_raw_rmse",
    "ape_align_rmse",
    "rpe_2m_align_rmse",
    "rpe_5m_align_rmse",
    "rpe_10m_align_rmse",
    "cross_track_mean",
    "row_correct_fraction",
    "wrong_row_duration_sec",
    "headland_cross_track_mean",
    "inrow_cross_track_mean",
]


@dataclass(frozen=True)
class CommandResult:
    cmd: list[str]
    log_path: Path
    duration_sec: float
    returncode: int


def candidate_matrix(include_gtsam: bool = False) -> list[dict[str, object]]:
    """Return the fixed candidate matrix agreed for the follow-up pass."""
    candidates: list[dict[str, object]] = [
        {
            "id": "accepted_alpha_huber3_cap20",
            "description": "Accepted alpha smoother with Huber GNSS threshold 3.0 and semantic cap 20.",
            "args": [
                "--pose-backend",
                "alpha",
                "--gnss-robust-mode",
                "huber",
                "--gnss-outlier-threshold",
                "3.0",
                "--semantic-penalty-cap",
                "20",
            ],
        },
        {
            "id": "alpha_huber5_nocap",
            "description": "Alpha smoother with relaxed Huber threshold 5.0 and no semantic cap.",
            "args": [
                "--pose-backend",
                "alpha",
                "--gnss-robust-mode",
                "huber",
                "--gnss-outlier-threshold",
                "5.0",
            ],
        },
        {
            "id": "alpha_huber3_cap50",
            "description": "Alpha smoother with Huber threshold 3.0 and looser semantic cap 50.",
            "args": [
                "--pose-backend",
                "alpha",
                "--gnss-robust-mode",
                "huber",
                "--gnss-outlier-threshold",
                "3.0",
                "--semantic-penalty-cap",
                "50",
            ],
        },
        {
            "id": "alpha_cauchy3_cap20",
            "description": "Alpha smoother with Cauchy GNSS robust loss and semantic cap 20.",
            "args": [
                "--pose-backend",
                "alpha",
                "--gnss-robust-mode",
                "cauchy",
                "--gnss-outlier-threshold",
                "3.0",
                "--semantic-penalty-cap",
                "20",
            ],
        },
        {
            "id": "alpha_huber3_cap20_no_smoothing",
            "description": "Accepted robust measurement settings with final pose smoothing disabled.",
            "args": [
                "--pose-backend",
                "alpha",
                "--gnss-robust-mode",
                "huber",
                "--gnss-outlier-threshold",
                "3.0",
                "--semantic-penalty-cap",
                "20",
                "--disable-pose-smoothing",
            ],
        },
        {
            "id": "fixedlag_huber3_cap20",
            "description": "Pure-Python fixed-lag smoother with accepted robust measurement settings.",
            "args": [
                "--pose-backend",
                "fixed-lag",
                "--fixed-lag-window",
                "8",
                "--gnss-robust-mode",
                "huber",
                "--gnss-outlier-threshold",
                "3.0",
                "--semantic-penalty-cap",
                "20",
            ],
        },
    ]
    if include_gtsam:
        candidates.append(
            {
                "id": "gtsam_huber3_cap20",
                "description": "Optional GTSAM smoother smoke with accepted robust measurement settings.",
                "args": [
                    "--pose-backend",
                    "gtsam",
                    "--fixed-lag-window",
                    "4",
                    "--gnss-robust-mode",
                    "huber",
                    "--gnss-outlier-threshold",
                    "3.0",
                    "--semantic-penalty-cap",
                    "20",
                ],
            }
        )
    return candidates


def parse_csv_list(value: str, cast=str) -> list:
    return [cast(part.strip()) for part in value.split(",") if part.strip()]


def build_spf_command(
    python_exec: Path,
    data_root: Path,
    geojson: Path,
    output_root: Path,
    traversal: str,
    seed: int,
    candidate: dict[str, object],
    max_frames: int | None = None,
    require_cuda: bool = True,
) -> list[str]:
    output_folder = output_root / traversal / str(candidate["id"]) / f"seed_{seed}"
    cmd = [
        str(python_exec),
        str(BASE_DIR / "scripts/spf_lidar.py"),
        "--seed",
        str(seed),
        "--data-path",
        str(data_root / traversal),
        "--geojson-path",
        str(geojson),
        "--output-folder",
        str(output_folder),
        "--no-visualization",
    ]
    cmd.extend(str(arg) for arg in candidate["args"])
    if max_frames is not None:
        cmd.extend(["--max-frames", str(max_frames)])
    if require_cuda:
        cmd.append("--require-cuda")
    return cmd


def command_output_folder(cmd: list[str]) -> Path:
    return Path(cmd[cmd.index("--output-folder") + 1])


def run_cmd(cmd: list[str], log_path: Path, env: dict[str, str]) -> CommandResult:
    start = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        cmd,
        cwd=str(BASE_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    duration = time.time() - start
    log_path.write_text(
        "$ " + " ".join(cmd) + "\n\n" + (proc.stdout or "") + f"\n[exit_code={proc.returncode} duration_sec={duration:.2f}]\n",
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit {proc.returncode}: {' '.join(cmd)} (see {log_path})")
    return CommandResult(cmd=cmd, log_path=log_path, duration_sec=duration, returncode=proc.returncode)


def _safe_float(value: object) -> float:
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _read_csv(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def candidate_by_id(candidates: Iterable[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(candidate["id"]): candidate for candidate in candidates}


def select_promoted_candidates(
    screen_rows: list[dict[str, object]],
    candidates: list[dict[str, object]],
    top_n: int = 2,
) -> list[dict[str, object]]:
    by_candidate: dict[str, list[float]] = defaultdict(list)
    for row in screen_rows:
        candidate_id = str(row.get("candidate_id", ""))
        value = _safe_float(row.get("ape_align_rmse"))
        if candidate_id and np.isfinite(value):
            by_candidate[candidate_id].append(value)

    means = {
        candidate_id: float(np.mean(values))
        for candidate_id, values in by_candidate.items()
        if values
    }
    gtsam_ids = {candidate_id for candidate_id in means if candidate_id.startswith("gtsam")}
    non_gtsam = {candidate_id: score for candidate_id, score in means.items() if candidate_id not in gtsam_ids}
    best_non_gtsam = min(non_gtsam.values()) if non_gtsam else float("inf")
    eligible = {
        candidate_id: score
        for candidate_id, score in means.items()
        if not candidate_id.startswith("gtsam") or score < best_non_gtsam
    }
    ordered_ids = sorted(eligible, key=lambda candidate_id: (eligible[candidate_id], candidate_id))[:top_n]
    lookup = candidate_by_id(candidates)
    return [lookup[candidate_id] for candidate_id in ordered_ids if candidate_id in lookup]


def classify_outcome(seed11_ape: float, full_rh1_mean: float | None = None) -> str:
    if seed11_ape > 1.5:
        return "failure"
    if seed11_ape < 1.2 and full_rh1_mean is not None and full_rh1_mean < 1.0:
        return "success"
    return "partial"


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("stage", "")), str(row.get("traversal", "")), str(row.get("candidate_id", "")))].append(row)

    aggregates: list[dict[str, object]] = []
    for (stage, traversal, candidate_id), group_rows in sorted(groups.items()):
        aggregate: dict[str, object] = {
            "stage": stage,
            "traversal": traversal,
            "candidate_id": candidate_id,
            "seed_count": len(group_rows),
        }
        for metric in KEY_METRICS:
            values = np.asarray([_safe_float(row.get(metric)) for row in group_rows], dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size:
                aggregate[f"{metric}_mean"] = float(np.mean(values))
                aggregate[f"{metric}_min"] = float(np.min(values))
                aggregate[f"{metric}_max"] = float(np.max(values))
        aggregates.append(aggregate)
    return aggregates


def evaluate_candidate_run(
    stage: str,
    traversal: str,
    seed: int,
    candidate: dict[str, object],
    run_dir: Path,
    geojson: Path,
    env: dict[str, str],
    python_exec: Path,
) -> dict[str, object]:
    rows = load_rows_from_geojson(geojson.resolve())
    evo_ape = python_exec.parent / "evo_ape"
    evo_rpe = python_exec.parent / "evo_rpe"
    if not evo_ape.exists():
        evo_ape = BASE_DIR / ".venv/bin/evo_ape"
    if not evo_rpe.exists():
        evo_rpe = BASE_DIR / ".venv/bin/evo_rpe"
    if not evo_ape.exists():
        evo_ape = Path("evo_ape")
    if not evo_rpe.exists():
        evo_rpe = Path("evo_rpe")

    run_name = f"{stage}_{traversal}_{candidate['id']}_seed_{seed}"
    est_tum = run_dir / "trajectory_0.5.tum"
    gt_tum = run_dir / "gps_pose.tum"
    try:
        metrics = evaluate_run(
            name=run_name,
            est_tum=est_tum,
            gt_tum=gt_tum,
            out_dir=run_dir / "eval",
            rows=rows,
            evo_ape_bin=evo_ape,
            evo_rpe_bin=evo_rpe,
            env=env,
            start_pose_anchor=False,
        )
    except Exception as exc:
        metrics = evaluate_short_run(run_name, est_tum, gt_tum, rows)
        metrics["eval_warning"] = f"repo_evo_bundle_failed: {type(exc).__name__}: {exc}"
    metrics.update(
        {
            "stage": stage,
            "traversal": traversal,
            "seed": seed,
            "candidate_id": candidate["id"],
            "candidate_description": candidate.get("description", ""),
            "run_dir": str(run_dir),
        }
    )
    return metrics


def evaluate_short_run(
    name: str,
    est_tum: Path,
    gt_tum: Path,
    rows: dict[str, np.ndarray],
) -> dict[str, object]:
    """Evaluate a short smoke run when evo RPE windows have too few pairs."""
    est = read_tum_file(est_tum)
    gt = read_tum_file(gt_tum)
    gt_interp_raw = interpolate_positions(gt.timestamps, gt.positions, est.timestamps)
    raw_errors = np.linalg.norm(est.positions - gt_interp_raw, axis=1)
    aligned = aligned_estimate(est_tum, gt_tum)
    row_metrics = compute_row_metrics(
        aligned["est_aligned"],
        aligned["gt_interp"],
        rows,
        timestamps=aligned["timestamps"],
        headland_mask=classify_headland_mask(aligned["gt_interp"], rows),
    )
    smoothness = compute_smoothness_metrics(aligned["timestamps"], aligned["est_aligned"])
    return {
        "run_name": name,
        "est_tum": str(est_tum),
        "eval_est_tum": str(est_tum),
        "gt_tum": str(gt_tum),
        "start_pose_anchor": 0,
        "ape_raw_rmse": float(np.sqrt(np.mean(raw_errors**2))),
        "ape_align_rmse": float(np.sqrt(np.mean(aligned["errors"] ** 2))),
        "rpe_2m_align_rmse": float("nan"),
        "rpe_5m_align_rmse": float("nan"),
        "rpe_10m_align_rmse": float("nan"),
        **row_metrics,
        **smoothness,
    }


def run_experiments(
    stage: str,
    traversals: list[str],
    seeds: list[int],
    candidates: list[dict[str, object]],
    python_exec: Path,
    data_root: Path,
    geojson: Path,
    output_root: Path,
    max_frames: int | None,
    require_cuda: bool,
    cuda_visible_devices: str | None,
) -> tuple[list[dict[str, object]], list[str]]:
    env = build_experiment_env(BASE_DIR)
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    env.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".tmp_mpl"))
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = _read_csv(output_root / "followup_metrics_per_seed.csv")
    commands: list[str] = []
    for traversal in traversals:
        for candidate in candidates:
            for seed in seeds:
                cmd = build_spf_command(
                    python_exec=python_exec,
                    data_root=data_root,
                    geojson=geojson,
                    output_root=output_root / stage,
                    traversal=traversal,
                    seed=seed,
                    candidate=candidate,
                    max_frames=max_frames,
                    require_cuda=require_cuda,
                )
                run_dir = command_output_folder(cmd)
                log_path = run_dir / "spf_lidar.log"
                result = run_cmd(cmd, log_path, env=env)
                commands.append(" ".join(result.cmd))
                metrics = evaluate_candidate_run(stage, traversal, seed, candidate, run_dir, geojson, env, python_exec)
                metrics["duration_sec"] = result.duration_sec
                metrics["log_path"] = str(log_path)
                rows.append(metrics)
                write_csv(output_root / "followup_metrics_per_seed.csv", rows)
                write_csv(output_root / "followup_metrics_aggregate.csv", aggregate_rows(rows))
    return rows, commands


def write_protocol(
    path: Path,
    stage: str,
    traversals: list[str],
    seeds: list[int],
    candidates: list[dict[str, object]],
    commands: list[str],
    max_frames: int | None,
    require_cuda: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "stage": stage,
                "traversals": traversals,
                "seeds": seeds,
                "candidate_ids": [candidate["id"] for candidate in candidates],
                "candidates": candidates,
                "commands": commands,
                "max_frames": max_frames,
                "require_cuda": require_cuda,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-exec", type=Path, default=DEFAULT_EXTERNAL_PYTHON)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--geojson", type=Path, default=DEFAULT_GEOJSON)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stage", choices=["screen", "full"], default="screen")
    parser.add_argument("--traversals", type=str, default="rh_run1")
    parser.add_argument("--seeds", type=str, default="11")
    parser.add_argument("--candidate-ids", type=str, default="")
    parser.add_argument("--include-gtsam", action="store_true")
    parser.add_argument("--promote-from", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--require-cuda", dest="require_cuda", action="store_true", default=True)
    parser.add_argument("--allow-cpu", dest="require_cuda", action="store_false")
    parser.add_argument("--cuda-visible-devices", type=str, default=None)
    args = parser.parse_args()

    traversals = parse_csv_list(args.traversals, str)
    seeds = parse_csv_list(args.seeds, int)
    candidates = candidate_matrix(include_gtsam=args.include_gtsam)
    lookup = candidate_by_id(candidates)

    if args.candidate_ids:
        requested_ids = parse_csv_list(args.candidate_ids, str)
        candidates = [lookup[candidate_id] for candidate_id in requested_ids]
    elif args.stage == "full" and args.promote_from is not None:
        candidates = select_promoted_candidates(_read_csv(args.promote_from), candidates, top_n=args.top_n)

    rows, commands = run_experiments(
        stage=args.stage,
        traversals=traversals,
        seeds=seeds,
        candidates=candidates,
        python_exec=args.python_exec,
        data_root=args.data_root,
        geojson=args.geojson,
        output_root=args.output_root,
        max_frames=args.max_frames,
        require_cuda=args.require_cuda,
        cuda_visible_devices=args.cuda_visible_devices,
    )
    write_protocol(
        args.output_root / f"followup_protocol_{args.stage}.json",
        stage=args.stage,
        traversals=traversals,
        seeds=seeds,
        candidates=candidates,
        commands=commands,
        max_frames=args.max_frames,
        require_cuda=args.require_cuda,
    )
    print(f"Wrote {len(rows)} evaluated runs to {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
