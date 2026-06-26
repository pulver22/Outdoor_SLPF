#!/usr/bin/env python3
"""Diagnostics for the Outdoor SLPF localisation follow-up runs."""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is best-effort in headless smoke tests.
    plt = None

try:
    from scripts.run_ab_validation import aligned_estimate, read_tum_file
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution.
    from run_ab_validation import aligned_estimate, read_tum_file


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RH2_ROOT = BASE_DIR / "results/localization_improvement_full/alpha_huber_cap"
DEFAULT_RH1_ROOT = BASE_DIR / "results/localization_improvement_full/alpha_huber_cap_rh1"
DEFAULT_OUTPUT_DIR = BASE_DIR / "results/localization_improvement_report/diagnostics"


@dataclass(frozen=True)
class RunSpec:
    traversal: str
    variant: str
    root: Path


def _safe_float(value: object, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _coerce_csv_value(value: str) -> object:
    if value == "":
        return value
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer():
        return int(number)
    return number


def _read_csv(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [
            {key: _coerce_csv_value(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else math.nan


def _median(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(median(vals)) if vals else math.nan


def _min(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(min(vals)) if vals else math.nan


def _max(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(max(vals)) if vals else math.nan


def _numeric_column(rows: list[dict[str, object]], key: str) -> list[float]:
    return [_safe_float(row.get(key)) for row in rows]


def summarize_frame_stats(stats_path: Path) -> dict[str, object]:
    """Summarise per-frame SPF diagnostics from a ``stats.csv`` file."""
    rows = _read_csv(stats_path)
    if not rows:
        return {"frame_count": 0}

    semantic_ratios = []
    for row in rows:
        correct = _safe_float(row.get("correct_hits"), 0.0)
        incorrect = _safe_float(row.get("incorrect_hits"), 0.0)
        no_hits = _safe_float(row.get("no_hits"), 0.0)
        denom = correct + incorrect + no_hits
        if denom > 0:
            semantic_ratios.append(correct / denom)

    ess_values = _numeric_column(rows, "ess")
    first_50_ess = ess_values[:50]
    smoother_values = [str(row.get("smoother_status", "")) for row in rows if row.get("smoother_status")]
    backend_values = [str(row.get("pose_backend_used", "")) for row in rows if row.get("pose_backend_used")]

    return {
        "frame_count": len(rows),
        "gnss_innovation_median": _median(_numeric_column(rows, "gnss_innovation")),
        "gnss_innovation_max": _max(_numeric_column(rows, "gnss_innovation")),
        "gnss_robust_scale_median": _median(_numeric_column(rows, "gnss_robust_scale")),
        "ess_median": _median(ess_values),
        "ess_min": _min(ess_values),
        "ess_min_first_50": _min(first_50_ess),
        "max_weight_median": _median(_numeric_column(rows, "max_weight")),
        "max_weight_max": _max(_numeric_column(rows, "max_weight")),
        "correct_hits_mean": _mean(_numeric_column(rows, "correct_hits")),
        "incorrect_hits_mean": _mean(_numeric_column(rows, "incorrect_hits")),
        "no_hits_mean": _mean(_numeric_column(rows, "no_hits")),
        "semantic_hit_ratio_mean": _mean(semantic_ratios),
        "smoother_status_last": smoother_values[-1] if smoother_values else "",
        "pose_backend_used": backend_values[-1] if backend_values else "",
    }


def collect_diagnostics(run_specs: list[RunSpec]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Collect seed-level and frame-level diagnostic rows for existing runs."""
    diagnostic_rows: list[dict[str, object]] = []
    frame_rows: list[dict[str, object]] = []

    for spec in run_specs:
        metrics_path = spec.root / "trajectory_metrics_per_seed.csv"
        for metric_row in _read_csv(metrics_path):
            seed = _safe_int(metric_row.get("seed"))
            stats_path = spec.root / f"seed_{seed}" / "stats.csv"
            stats = summarize_frame_stats(stats_path)
            merged: dict[str, object] = {
                "traversal": spec.traversal,
                "variant": spec.variant,
                "seed": seed,
            }
            merged.update(metric_row)
            merged["traversal"] = spec.traversal
            merged["variant"] = spec.variant
            merged["seed"] = seed
            merged.update(stats)
            merged["stats_csv"] = str(stats_path)
            diagnostic_rows.append(merged)

            for frame_row in _read_csv(stats_path):
                prefixed = {
                    "traversal": spec.traversal,
                    "variant": spec.variant,
                    "seed": seed,
                }
                prefixed.update(frame_row)
                frame_rows.append(prefixed)

    return diagnostic_rows, frame_rows


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


def _plot_trajectory(row: dict[str, object], out_dir: Path) -> Path | None:
    if plt is None:
        return None
    est_path = Path(str(row.get("est_tum", "")))
    gt_path = Path(str(row.get("gt_tum", "")))
    if not est_path.exists() or not gt_path.exists():
        return None
    try:
        aligned = aligned_estimate(est_path, gt_path)
    except Exception:
        return None

    out_path = out_dir / f"trajectory_{row['traversal']}_seed_{row['seed']}.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    gt_xy = aligned["gt_interp"][:, :2]
    est_xy = aligned["est_aligned"][:, :2]
    ax.plot(gt_xy[:, 0], gt_xy[:, 1], color="black", linestyle="--", linewidth=1.8, label="Ground truth")
    ax.plot(est_xy[:, 0], est_xy[:, 1], color="tab:blue", linewidth=1.8, label=str(row.get("variant", "estimate")))
    ax.set_title(f"{row['traversal']} seed {row['seed']} trajectory")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_timeline(row: dict[str, object], frame_rows: list[dict[str, object]], out_dir: Path) -> Path | None:
    if plt is None:
        return None
    rows = [
        frame
        for frame in frame_rows
        if str(frame.get("traversal")) == str(row.get("traversal"))
        and str(frame.get("variant")) == str(row.get("variant"))
        and int(_safe_int(frame.get("seed"))) == int(_safe_int(row.get("seed")))
    ]
    if not rows:
        return None
    x = np.arange(len(rows), dtype=np.float64)
    semantic_ratio = []
    for frame in rows:
        correct = _safe_float(frame.get("correct_hits"), 0.0)
        incorrect = _safe_float(frame.get("incorrect_hits"), 0.0)
        no_hits = _safe_float(frame.get("no_hits"), 0.0)
        denom = correct + incorrect + no_hits
        semantic_ratio.append(correct / denom if denom > 0 else math.nan)

    series = [
        ("gnss_innovation", "GNSS innovation (m)", _numeric_column(rows, "gnss_innovation")),
        ("ess", "ESS", _numeric_column(rows, "ess")),
        ("max_weight", "Max weight", _numeric_column(rows, "max_weight")),
        ("semantic_hit_ratio", "Semantic hit ratio", semantic_ratio),
    ]
    out_path = out_dir / f"diagnostics_{row['traversal']}_seed_{row['seed']}.png"
    fig, axes = plt.subplots(len(series), 1, figsize=(10, 8), sharex=True)
    for ax, (_, ylabel, values) in zip(axes, series):
        ax.plot(x, values, linewidth=1.5)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Processed frame index")
    fig.suptitle(f"{row['traversal']} seed {row['seed']} frame diagnostics")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_row_error_timeline(row: dict[str, object], out_dir: Path) -> Path | None:
    if plt is None:
        return None
    est_path = Path(str(row.get("est_tum", "")))
    gt_path = Path(str(row.get("gt_tum", "")))
    if not est_path.exists() or not gt_path.exists():
        return None
    try:
        aligned = aligned_estimate(est_path, gt_path)
    except Exception:
        return None
    err = np.linalg.norm(aligned["est_aligned"][:, :2] - aligned["gt_interp"][:, :2], axis=1)
    out_path = out_dir / f"row_error_{row['traversal']}_seed_{row['seed']}.png"
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(aligned["timestamps"], err, color="tab:red", linewidth=1.4)
    ax.set_title(f"{row['traversal']} seed {row['seed']} aligned position error")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Error (m)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def write_plots(
    diagnostic_rows: list[dict[str, object]],
    frame_rows: list[dict[str, object]],
    output_dir: Path,
) -> list[Path]:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for row in diagnostic_rows:
        for path in (
            _plot_trajectory(row, plot_dir),
            _plot_row_error_timeline(row, plot_dir),
            _plot_timeline(row, frame_rows, plot_dir),
        ):
            if path is not None:
                paths.append(path)
    return paths


def parse_run_spec(value: str) -> RunSpec:
    parts = value.split("=", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Run specs must use traversal=variant=/path/to/root")
    return RunSpec(traversal=parts[0], variant=parts[1], root=Path(parts[2]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rh2-root", type=Path, default=DEFAULT_RH2_ROOT)
    parser.add_argument("--rh1-root", type=Path, default=DEFAULT_RH1_ROOT)
    parser.add_argument("--run-spec", action="append", type=parse_run_spec, default=[])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    run_specs = args.run_spec or [
        RunSpec("rh_run2", "alpha_huber_cap", args.rh2_root),
        RunSpec("rh_run1", "alpha_huber_cap", args.rh1_root),
    ]
    diagnostic_rows, frame_rows = collect_diagnostics(run_specs)
    write_csv(args.output_dir / "diagnostic_metrics.csv", diagnostic_rows)
    write_csv(args.output_dir / "frame_diagnostics.csv", frame_rows)
    if not args.skip_plots:
        write_plots(diagnostic_rows, frame_rows, args.output_dir)
    print(f"Wrote {len(diagnostic_rows)} diagnostic rows and {len(frame_rows)} frame rows to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
