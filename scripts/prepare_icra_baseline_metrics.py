#!/usr/bin/env python3
"""Normalize the auditable AMCL/RTAB-Map baseline runs for Table I.

The historical multiseed runner copied one fixed trajectory into three seed
folders for the ``rh_run1`` AMCL and RTAB-Map rows.  This utility creates one
canonical CSV from the dedicated three-run artifacts and the already-real
``rh_run2`` per-seed bundle.  It preserves source paths and output hashes so
the evidence builder can audit the replicate identity.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


SEEDS = (11, 22, 33)
BASE_DIR = Path(__file__).resolve().parent.parent
RUN_TO_SEED = {"run1": 11, "run2": 22, "run3": 33}
AMCL_RUN_TO_SEED = {
    "amcl_pose": 11,
    "amcl_pose1_2": 22,
    "amcl_pose1_3": 33,
    "amcl_pose_ngps_s11": 11,
    "amcl_pose1_2_ngps_s22": 22,
    "amcl_pose1_3_ngps_s33": 33,
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_artifact(value: str) -> Path | None:
    """Resolve stale absolute paths from sibling checkouts into this checkout."""
    if not value:
        return None
    candidate = Path(value)
    if candidate.exists():
        return candidate.resolve()
    marker = "/results/"
    if marker in value:
        mapped = BASE_DIR / "results" / value.split(marker, 1)[1]
        if mapped.exists():
            return mapped.resolve()
    return None


def _require_three(rows: list[dict[str, str]], label: str) -> None:
    if len(rows) != 3:
        raise ValueError(f"{label}: expected exactly 3 rows, found {len(rows)}")
    seeds = [int(row["seed"]) for row in rows]
    if sorted(seeds) != list(SEEDS):
        raise ValueError(f"{label}: expected seeds {SEEDS}, found {seeds}")


def _base_row(
    *,
    method: str,
    traversal: str,
    seed: int,
    source_file: Path,
    source_run: str,
    source_output: Path | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "method": method,
        "traversal": traversal,
        "seed": seed,
        "profile": "full",
        "replicate_kind": "dedicated_run",
        "source_file": str(source_file.resolve()),
        "source_file_sha256": _sha256(source_file),
        "source_run": source_run,
    }
    if source_output is not None:
        resolved_output = _resolve_artifact(str(source_output)) or source_output
        row["source_output"] = str(resolved_output)
        if resolved_output.exists():
            row["source_output_sha256"] = _sha256(resolved_output)
    return row


def _copy_metrics(target: dict[str, object], source: dict[str, str], mapping: dict[str, str]) -> None:
    for destination, origin in mapping.items():
        if origin in source and source[origin] != "":
            target[destination] = source[origin]


def _dedicated_amcl_rows(metrics_path: Path, data_root: Path) -> list[dict[str, object]]:
    rows = _read_csv(metrics_path)
    selected = [row for row in rows if row.get("run") in {"amcl_pose", "amcl_pose1_2", "amcl_pose1_3"}]
    if len(selected) != 3:
        raise ValueError(f"{metrics_path}: expected the three AMCL pose rows")
    output_names = {
        "amcl_pose": "amcl_pose_clean.tum",
        "amcl_pose1_2": "amcl_pose1_2_clean.tum",
        "amcl_pose1_3": "amcl_pose1_3_clean.tum",
    }
    output_dir = metrics_path.parent
    result: list[dict[str, object]] = []
    for source in selected:
        source_run = str(source["run"])
        output = output_dir / output_names[source_run]
        target = _base_row(
            method="amcl",
            traversal="rh_run1",
            seed=AMCL_RUN_TO_SEED[source_run],
            source_file=metrics_path,
            source_run=source_run,
            source_output=output,
        )
        _copy_metrics(
            target,
            source,
            {
                "ape_raw_rmse": "ape_raw_rmse",
                "ape_align_rmse": "ape_align_rmse",
                "rpe_2m_align_rmse": "rpe_2m_align_rmse",
                "rpe_5m_align_rmse": "rpe_5m_align_rmse",
                "rpe_10m_align_rmse": "rpe_10m_align_rmse",
                "cross_track_mean": "cross_track_mean",
                "row_correct_fraction": "row_correct_fraction",
                "row_switch_events": "row_switch_events",
            },
        )
        result.append(target)
    return result


def _dedicated_amcl_ngps_rows(metrics_path: Path) -> list[dict[str, object]]:
    rows = _read_csv(metrics_path)
    selected = [row for row in rows if row.get("run") in {"amcl_pose_ngps_s11", "amcl_pose1_2_ngps_s22", "amcl_pose1_3_ngps_s33"}]
    if len(selected) != 3:
        raise ValueError(f"{metrics_path}: expected the three AMCL+NoisyGNSS rows")
    output_dir = metrics_path.parent / "amcl_ngps_fused"
    result: list[dict[str, object]] = []
    for source in selected:
        source_run = str(source["run"])
        output = output_dir / f"{source_run}.tum"
        target = _base_row(
            method="amcl_ngps",
            traversal="rh_run1",
            seed=AMCL_RUN_TO_SEED[source_run],
            source_file=metrics_path,
            source_run=source_run,
            source_output=output,
        )
        _copy_metrics(
            target,
            source,
            {
                "ape_raw_rmse": "ape_raw_rmse",
                "ape_align_rmse": "ape_align_rmse",
                "rpe_2m_align_rmse": "rpe_2m_align_rmse",
                "rpe_5m_align_rmse": "rpe_5m_align_rmse",
                "rpe_10m_align_rmse": "rpe_10m_align_rmse",
                "cross_track_mean": "cross_track_mean",
                "row_correct_fraction": "row_correct_fraction",
                "row_switch_events": "row_switch_events",
            },
        )
        result.append(target)
    return result


def _dedicated_rtab_rows(metrics_path: Path, method: str, output_root: Path) -> list[dict[str, object]]:
    rows = _read_csv(metrics_path)
    selected = []
    for row in rows:
        seed = RUN_TO_SEED.get(str(row.get("run", "")))
        if seed is not None:
            selected.append((seed, row))
    if len(selected) != 3 or sorted(seed for seed, _ in selected) != list(SEEDS):
        raise ValueError(f"{metrics_path}: expected run1/run2/run3 for seeds {SEEDS}")
    result: list[dict[str, object]] = []
    for seed, source in selected:
        source_run = str(source["run"])
        output = output_root / source_run / "rtabmap" / ("rgb" if method == "rtab_rgb" else "rgbd") / "tum1" / (
            "rtabmap_rgb_filtered.tum" if method == "rtab_rgb" else "rtabmap_rgbd_filtered.tum"
        )
        target = _base_row(
            method=method,
            traversal="rh_run1",
            seed=seed,
            source_file=metrics_path,
            source_run=source_run,
            source_output=output,
        )
        _copy_metrics(
            target,
            source,
            {
                "ape_raw_rmse": "ape_raw_rmse",
                "ape_align_rmse": "ape_umey_rmse",
                "rpe_2m_align_rmse": "rpe_2m_rmse",
                "rpe_5m_align_rmse": "rpe_5m_rmse",
                "rpe_10m_align_rmse": "rpe_10m_rmse",
                "cross_track_mean": "cross_track_mean",
                "row_correct_fraction": "row_correct_fraction",
                "row_switch_events": "row_switch_events",
            },
        )
        result.append(target)
    return result


def _historical_rows(path: Path, traversal: str, allowed_methods: set[str] | None = None) -> list[dict[str, object]]:
    """Retain selected methods from a validated three-seed CSV."""
    result: list[dict[str, object]] = []
    for source in _read_csv(path):
        method = str(source.get("method", "")).strip().lower()
        if method not in {
            "ngps", "noisy gps", "noisygnss", "amcl", "amcl+noisygps", "amcl+noisygnss",
            "rtab_rgb", "rtab_rgbd", "rtabmap rgb", "rtabmap rgbd",
        }:
            continue
        if allowed_methods is not None and method not in allowed_methods:
            continue
        seed_text = str(source.get("seed", ""))
        if not seed_text.isdigit() or int(seed_text) not in SEEDS:
            continue
        canonical = {
            "ngps": "ngps", "noisy gps": "ngps", "noisygnss": "ngps",
            "amcl+noisygps": "amcl_ngps", "amcl+noisygnss": "amcl_ngps",
            "rtabmap rgb": "rtab_rgb", "rtabmap rgbd": "rtab_rgbd",
        }.get(method, method)
        target = _base_row(
            method=canonical,
            traversal=traversal,
            seed=int(seed_text),
            source_file=path,
            source_run=str(source.get("run_name", "")),
        )
        for key in (
            "ape_raw_rmse", "ape_align_rmse", "rpe_2m_align_rmse", "rpe_5m_align_rmse", "rpe_10m_align_rmse",
            "cross_track_mean", "row_correct_fraction", "row_switch_events",
        ):
            if source.get(key, "") != "":
                target[key] = source[key]
        target["replicate_kind"] = "historical_seed_run"
        resolved_output = _resolve_artifact(str(source.get("est_tum", "")))
        target["source_output"] = str(resolved_output or source.get("est_tum", ""))
        if resolved_output is not None:
            target["source_output_sha256"] = _sha256(resolved_output)
        result.append(target)
    return result


def _rtab_ngps_rows(path: Path) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    method_map = {
        "rtab rgb+noisygnss": "rtab_rgb_ngps",
        "rtab rgbd+noisygnss": "rtab_rgbd_ngps",
    }
    for source in _read_csv(path):
        method = method_map.get(str(source.get("method", "")).strip().lower())
        if method is None:
            continue
        traversal = {"exp1": "rh_run1", "exp2": "rh_run2"}.get(str(source.get("traversal", "")).strip().lower())
        seed_text = str(source.get("seed", ""))
        if traversal is None or not seed_text.isdigit() or int(seed_text) not in SEEDS:
            continue
        target = _base_row(
            method=method,
            traversal=traversal,
            seed=int(seed_text),
            source_file=path,
            source_run=str(source.get("run_name", "")),
        )
        for key in (
            "ape_raw_rmse", "ape_align_rmse", "rpe_2m_align_rmse", "rpe_5m_align_rmse", "rpe_10m_align_rmse",
            "cross_track_mean", "row_correct_fraction", "row_switch_events",
        ):
            if source.get(key, "") != "":
                target[key] = source[key]
        target["replicate_kind"] = "historical_seed_run"
        resolved_output = _resolve_artifact(str(source.get("est_tum", "")))
        target["source_output"] = str(resolved_output or source.get("est_tum", ""))
        if resolved_output is not None:
            target["source_output_sha256"] = _sha256(resolved_output)
        result.append(target)
    return result


def prepare(
    *,
    amcl_metrics: Path,
    amcl_data_root: Path,
    rtab_rgb_metrics: Path,
    rtab_rgb_root: Path,
    rtab_rgbd_metrics: Path,
    rtab_rgbd_root: Path,
    rh1_metrics: Path,
    rh2_metrics: Path,
    rtab_ngps_metrics: Path,
    output: Path,
) -> list[dict[str, object]]:
    rows = []
    rows.extend(_historical_rows(rh1_metrics, "rh_run1", {"ngps", "noisy gps", "noisygnss"}))
    rows.extend(_historical_rows(rh2_metrics, "rh_run2"))
    rows.extend(_dedicated_amcl_rows(amcl_metrics, amcl_data_root))
    rows.extend(_dedicated_amcl_ngps_rows(amcl_metrics))
    rows.extend(_dedicated_rtab_rows(rtab_rgb_metrics, "rtab_rgb", rtab_rgb_root))
    rows.extend(_dedicated_rtab_rows(rtab_rgbd_metrics, "rtab_rgbd", rtab_rgbd_root))
    rows.extend(_rtab_ngps_rows(rtab_ngps_metrics))

    keys = [(str(row["method"]), str(row["traversal"]), int(row["seed"]), str(row["profile"])) for row in rows]
    if len(keys) != len(set(keys)):
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        raise ValueError(f"duplicate canonical baseline keys: {duplicates}")
    for method in sorted({str(row["method"]) for row in rows}):
        for traversal in ("rh_run1", "rh_run2"):
            present = {int(row["seed"]) for row in rows if str(row["method"]) == method and str(row["traversal"]) == traversal}
            if present and present != set(SEEDS):
                raise ValueError(f"{method}/{traversal}: expected seeds {SEEDS}, found {sorted(present)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (str(row["method"]), str(row["traversal"]), int(row["seed"]))))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amcl-metrics", type=Path, required=True)
    parser.add_argument("--amcl-data-root", type=Path, required=True)
    parser.add_argument("--rtab-rgb-metrics", type=Path, required=True)
    parser.add_argument("--rtab-rgb-root", type=Path, required=True)
    parser.add_argument("--rtab-rgbd-metrics", type=Path, required=True)
    parser.add_argument("--rtab-rgbd-root", type=Path, required=True)
    parser.add_argument("--rh1-metrics", type=Path, required=True)
    parser.add_argument("--rh2-metrics", type=Path, required=True)
    parser.add_argument("--rtab-ngps-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = prepare(**vars(args))
    print(f"Wrote {len(rows)} canonical baseline rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
