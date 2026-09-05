#!/usr/bin/env python3
"""Tune SPF localisation robustness and smoothing parameters.

This runner keeps GTSAM and Optuna optional. It always supports a deterministic
grid so smoke tuning remains available in minimal environments.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

try:
    from run_ab_validation import BASE_DIR, evaluate_run, load_rows_from_geojson, run_cmd
    from run_spfpp_ablation import primary_score, validate_data_path
except ModuleNotFoundError:
    from scripts.run_ab_validation import BASE_DIR, evaluate_run, load_rows_from_geojson, run_cmd
    from scripts.run_spfpp_ablation import primary_score, validate_data_path


DEFAULT_OUTPUT_ROOT = BASE_DIR / "results" / "spf_lidar++" / "tuning_optuna"
DEFAULT_DATA_PATH = BASE_DIR / "data" / "2025" / "rh_run2"
DEFAULT_GEOJSON = BASE_DIR / "data" / "riseholme_poles_trunk.geojson"
SPF_SCRIPT = BASE_DIR / "scripts" / "spf_lidar.py"


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def safe_float(value, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def dedupe_candidates(candidates: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    seen = set()
    out = []
    for candidate in candidates:
        key = tuple(sorted(candidate.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(candidate))
    return out


def grid_candidates(limit: int, include_gtsam: bool = False) -> List[Dict[str, object]]:
    base = [
        {
            "pose_backend": "alpha",
            "fixed_lag_window": 8,
            "gnss_robust_mode": "huber",
            "gnss_outlier_threshold": 3.0,
            "semantic_penalty_cap": 20.0,
        },
        {
            "pose_backend": "alpha",
            "fixed_lag_window": 8,
            "gnss_robust_mode": "huber",
            "gnss_outlier_threshold": 5.0,
            "semantic_penalty_cap": None,
        },
        {
            "pose_backend": "alpha",
            "fixed_lag_window": 8,
            "gnss_robust_mode": "cauchy",
            "gnss_outlier_threshold": 3.0,
            "semantic_penalty_cap": 20.0,
        },
        {
            "pose_backend": "alpha",
            "fixed_lag_window": 8,
            "gnss_robust_mode": "off",
            "gnss_outlier_threshold": 5.0,
            "semantic_penalty_cap": None,
        },
    ]
    for window in (4, 6, 8):
        base.append(
            {
                "pose_backend": "fixed-lag",
                "fixed_lag_window": window,
                "gnss_robust_mode": "huber",
                "gnss_outlier_threshold": 3.0,
                "semantic_penalty_cap": 20.0,
            }
        )
        if include_gtsam:
            base.append(
                {
                    "pose_backend": "gtsam",
                    "fixed_lag_window": window,
                    "gnss_robust_mode": "huber",
                    "gnss_outlier_threshold": 3.0,
                    "semantic_penalty_cap": 20.0,
                }
            )
    candidates = dedupe_candidates(base)
    return candidates[: max(0, limit)]


def build_trial_command(
    args,
    candidate: Dict[str, object],
    seed: int,
    seed_dir: Path,
) -> List[str]:
    cmd = [
        str(args.python_exec),
        str(SPF_SCRIPT),
        "--seed", str(seed),
        "--output-folder", str(seed_dir),
        "--data-path", str(args.data_path),
        "--geojson-path", str(args.geojson),
        "--frame-stride", str(args.frame_stride),
        "--max-frames", str(args.max_frames),
        "--particle-count", str(args.particle_count),
        "--miss-penalty", str(args.miss_penalty),
        "--wrong-hit-penalty", str(args.wrong_hit_penalty),
        "--gps-weight", str(args.gps_weight),
        "--pose-backend", str(candidate["pose_backend"]),
        "--fixed-lag-window", str(candidate["fixed_lag_window"]),
        "--gnss-robust-mode", str(candidate["gnss_robust_mode"]),
        "--gnss-outlier-threshold", str(candidate["gnss_outlier_threshold"]),
        "--no-visualization",
    ]
    cap = candidate.get("semantic_penalty_cap")
    if cap is not None:
        cmd.extend(["--semantic-penalty-cap", str(cap)])
    if args.require_cuda:
        cmd.append("--require-cuda")
    return cmd


def evaluate_candidate(
    args,
    candidate: Dict[str, object],
    trial_dir: Path,
    rows_map: Dict[str, np.ndarray],
    env: Dict[str, str],
) -> Dict[str, object]:
    per_seed_rows: List[Dict[str, object]] = []
    evo_ape_bin = args.python_exec.parent / "evo_ape"
    evo_rpe_bin = args.python_exec.parent / "evo_rpe"
    for seed in args.search_seeds:
        seed_dir = trial_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_trial_command(args, candidate, seed, seed_dir)
        runtime = run_cmd(cmd, seed_dir / "run.log", cwd=BASE_DIR, env=env)
        metrics = evaluate_run(
            name=f"{trial_dir.name}_seed_{seed}",
            est_tum=seed_dir / f"trajectory_{args.gps_weight}.tum",
            gt_tum=seed_dir / "gps_pose.tum",
            out_dir=seed_dir / "eval",
            rows=rows_map,
            evo_ape_bin=evo_ape_bin,
            evo_rpe_bin=evo_rpe_bin,
            env=env,
        )
        metrics.update(candidate)
        metrics["seed"] = seed
        metrics["runtime_sec"] = runtime
        metrics["command"] = " ".join(cmd)
        per_seed_rows.append(metrics)

    metric_means = {}
    for metric in [
        "ape_align_rmse",
        "rpe_2m_align_rmse",
        "rpe_5m_align_rmse",
        "rpe_10m_align_rmse",
        "cross_track_mean",
        "row_correct_fraction",
        "row_switch_events",
        "jerk_rms",
        "heading_accel_rms",
    ]:
        vals = np.asarray([safe_float(row.get(metric)) for row in per_seed_rows], dtype=np.float64)
        metric_means[metric] = float(np.nanmean(vals))

    score = primary_score(metric_means)
    summary: Dict[str, object] = {
        **candidate,
        "trial_dir": str(trial_dir),
        "n_seeds": len(per_seed_rows),
        "primary_score": float(score),
        "runtime_sec_mean": float(np.mean([safe_float(row["runtime_sec"], 0.0) for row in per_seed_rows])),
    }
    for key, value in metric_means.items():
        summary[f"{key}_mean"] = value
    write_csv(trial_dir / "metrics_per_seed.csv", per_seed_rows)
    (trial_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def suggest_candidate(trial, include_gtsam: bool) -> Dict[str, object]:
    backends = ["alpha", "fixed-lag"]
    if include_gtsam:
        backends.append("gtsam")
    robust_mode = trial.suggest_categorical("gnss_robust_mode", ["huber", "cauchy", "gate", "off"])
    cap = trial.suggest_categorical("semantic_penalty_cap", [None, 10.0, 20.0, 50.0])
    return {
        "pose_backend": trial.suggest_categorical("pose_backend", backends),
        "fixed_lag_window": trial.suggest_categorical("fixed_lag_window", [4, 6, 8, 10]),
        "gnss_robust_mode": robust_mode,
        "gnss_outlier_threshold": trial.suggest_float("gnss_outlier_threshold", 2.0, 6.0),
        "semantic_penalty_cap": cap,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune SPF++ localisation robustness parameters.")
    parser.add_argument("--python-exec", type=Path, default=BASE_DIR / ".venv" / "bin" / "python")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--geojson", type=Path, default=DEFAULT_GEOJSON)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--search-seeds", type=str, default="22")
    parser.add_argument("--grid-max-trials", type=int, default=8)
    parser.add_argument("--bayes-trials", type=int, default=0)
    parser.add_argument("--topk-enqueue", type=int, default=0, help="Accepted for compatibility; grid trials run before Bayesian trials.")
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--particle-count", type=int, default=100)
    parser.add_argument("--miss-penalty", type=float, default=4.0)
    parser.add_argument("--wrong-hit-penalty", type=float, default=4.0)
    parser.add_argument("--gps-weight", type=float, default=0.5)
    parser.add_argument("--cuda-visible-devices", type=str, default="0")
    parser.add_argument("--include-gtsam", action="store_true")
    parser.add_argument("--require-cuda", dest="require_cuda", action="store_true", default=True)
    parser.add_argument("--allow-cpu", dest="require_cuda", action="store_false")
    parser.add_argument("--no-visualization", action="store_true", help="Accepted for compatibility; visualisation is always disabled.")
    args = parser.parse_args()

    args.python_exec = args.python_exec.expanduser()
    if not args.python_exec.is_absolute():
        args.python_exec = (BASE_DIR / args.python_exec).resolve()
    args.data_path = args.data_path.expanduser()
    if not args.data_path.is_absolute():
        args.data_path = (BASE_DIR / args.data_path).resolve()
    args.geojson = args.geojson.expanduser()
    if not args.geojson.is_absolute():
        args.geojson = (BASE_DIR / args.geojson).resolve()
    args.search_seeds = parse_int_list(args.search_seeds)
    if not args.search_seeds:
        raise ValueError("--search-seeds must contain at least one seed")
    validate_data_path(args.data_path)

    if not (args.python_exec.parent / "evo_ape").exists() or not (args.python_exec.parent / "evo_rpe").exists():
        raise FileNotFoundError("evo_ape/evo_rpe not found in virtualenv bin directory.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root.resolve() / f"{timestamp}_spfpp_tuning"
    run_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(BASE_DIR / ".tmp_mpl")
    env["HOME"] = str(BASE_DIR)
    (BASE_DIR / ".tmp_mpl").mkdir(parents=True, exist_ok=True)
    rows_map = load_rows_from_geojson(args.geojson)

    summaries: List[Dict[str, object]] = []
    for trial_idx, candidate in enumerate(grid_candidates(args.grid_max_trials, include_gtsam=args.include_gtsam), start=1):
        trial_dir = run_dir / f"grid_{trial_idx:03d}_{candidate['pose_backend']}_{candidate['gnss_robust_mode']}"
        print(f"[grid {trial_idx}] {candidate}")
        summary = evaluate_candidate(args, candidate, trial_dir, rows_map, env)
        summary["trial_type"] = "grid"
        summaries.append(summary)

    if args.bayes_trials > 0:
        try:
            import optuna  # type: ignore
        except Exception as exc:
            print(f"[WARN] Optuna unavailable; skipping Bayesian trials: {exc}")
        else:
            def objective(trial) -> float:
                candidate = suggest_candidate(trial, include_gtsam=args.include_gtsam)
                trial_dir = run_dir / f"optuna_{trial.number:03d}_{candidate['pose_backend']}_{candidate['gnss_robust_mode']}"
                print(f"[optuna {trial.number}] {candidate}")
                summary = evaluate_candidate(args, candidate, trial_dir, rows_map, env)
                summary["trial_type"] = "optuna"
                summary["trial_number"] = trial.number
                summaries.append(summary)
                return safe_float(summary["primary_score"], 1e9)

            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=args.bayes_trials)

    summaries.sort(key=lambda row: safe_float(row["primary_score"], 1e9))
    write_csv(run_dir / "tuning_trials.csv", summaries)
    protocol = {
        "run_dir": str(run_dir),
        "timestamp": timestamp,
        "search_seeds": args.search_seeds,
        "grid_max_trials": args.grid_max_trials,
        "bayes_trials": args.bayes_trials,
        "include_gtsam": args.include_gtsam,
        "data_path": str(args.data_path),
        "geojson": str(args.geojson),
        "python_exec": str(args.python_exec),
        "best": summaries[0] if summaries else None,
    }
    (run_dir / "tuning_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    if summaries:
        best = summaries[0]
        print(
            "[done] best "
            f"score={safe_float(best['primary_score']):.6f} "
            f"ape_align_mean={safe_float(best['ape_align_rmse_mean']):.6f} "
            f"backend={best['pose_backend']} robust={best['gnss_robust_mode']}"
        )
    print(f"[done] artifacts in {run_dir}")


if __name__ == "__main__":
    main()
