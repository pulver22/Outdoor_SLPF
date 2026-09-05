#!/usr/bin/env python3
"""Evaluate the vineyard landmark detector on a SemanticBLT validation YAML."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = BASE_DIR / "results" / "iros_revision" / "detector_eval"


def build_conf_thresholds(text: str) -> list[float]:
    thresholds = sorted({float(item.strip()) for item in text.split(",") if item.strip()})
    if not thresholds:
        raise ValueError("At least one confidence threshold is required.")
    invalid = [value for value in thresholds if value < 0.0 or value > 1.0]
    if invalid:
        raise ValueError(f"Confidence thresholds must be in [0, 1], got {invalid}")
    return thresholds


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=BASE_DIR / "models" / "yolo.pt")
    parser.add_argument("--data-yaml", type=Path, required=True, help="SemanticBLT validation YAML.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--conf-thresholds", default="0.10,0.25,0.50")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument(
        "--distance-bins-csv",
        type=Path,
        default=None,
        help="Optional CSV with distance_m and tp/fp/fn or correct columns for binned summaries.",
    )
    return parser.parse_args(argv)


def validate_inputs(args: argparse.Namespace) -> None:
    if not args.data_yaml.exists():
        raise FileNotFoundError(f"Missing data YAML: {args.data_yaml}")
    if not args.model.exists():
        raise FileNotFoundError(f"Missing detector model: {args.model}")
    if args.distance_bins_csv is not None and not args.distance_bins_csv.exists():
        raise FileNotFoundError(f"Missing distance bins CSV: {args.distance_bins_csv}")


def safe_float(value: object, default: float = float("nan")) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def f1_score(precision: float, recall: float) -> float:
    if not math.isfinite(precision) or not math.isfinite(recall):
        return float("nan")
    denom = precision + recall
    if denom <= 0.0:
        return 0.0
    return float(2.0 * precision * recall / denom)


def metric_attr(obj: object, attr: str) -> float:
    return safe_float(getattr(obj, attr, float("nan")))


def extract_ultralytics_metrics(results: object, conf_threshold: float) -> dict[str, object]:
    box = getattr(results, "box", None)
    result_dict = getattr(results, "results_dict", {}) or {}
    precision = metric_attr(box, "mp") if box is not None else safe_float(result_dict.get("metrics/precision(B)"))
    recall = metric_attr(box, "mr") if box is not None else safe_float(result_dict.get("metrics/recall(B)"))
    map50 = metric_attr(box, "map50") if box is not None else safe_float(result_dict.get("metrics/mAP50(B)"))
    map5095 = metric_attr(box, "map") if box is not None else safe_float(result_dict.get("metrics/mAP50-95(B)"))
    row = {
        "conf_threshold": conf_threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1_score(precision, recall),
        "map50": map50,
        "map50_95": map5095,
    }
    for key, value in result_dict.items():
        safe_key = str(key).replace("/", "_").replace("(", "").replace(")", "")
        row[f"ultralytics_{safe_key}"] = value
    return row


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


def build_distance_bins(distance_bins_csv: Path) -> list[dict[str, object]]:
    rows = []
    with distance_bins_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    if not rows:
        return []

    bins = [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, float("inf"))]
    out: list[dict[str, object]] = []
    for lo, hi in bins:
        selected = []
        for row in rows:
            distance = safe_float(row.get("distance_m"))
            if not math.isfinite(distance):
                continue
            if distance >= lo and distance < hi:
                selected.append(row)
        if not selected:
            continue

        if {"tp", "fp", "fn"}.issubset(selected[0].keys()):
            tp = sum(safe_float(row.get("tp"), 0.0) for row in selected)
            fp = sum(safe_float(row.get("fp"), 0.0) for row in selected)
            fn = sum(safe_float(row.get("fn"), 0.0) for row in selected)
            precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
            recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            out.append(
                {
                    "distance_bin_m": f"{lo:g}-{hi:g}",
                    "n": len(selected),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1_score(precision, recall),
                }
            )
        elif "correct" in selected[0]:
            correct = [safe_float(row.get("correct"), 0.0) for row in selected]
            out.append(
                {
                    "distance_bin_m": f"{lo:g}-{hi:g}",
                    "n": len(selected),
                    "accuracy": sum(correct) / len(correct),
                }
            )
    return out


def run_evaluation(args: argparse.Namespace) -> list[dict[str, object]]:
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Ultralytics is required for detector evaluation. Install requirements/experiments.txt first."
        ) from exc

    thresholds = build_conf_thresholds(args.conf_thresholds)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))
    rows = []

    for conf in thresholds:
        run_name = f"conf_{conf:.2f}".replace(".", "p")
        val_kwargs = {
            "data": str(args.data_yaml),
            "conf": conf,
            "imgsz": args.imgsz,
            "split": args.split,
            "project": str(output_root),
            "name": run_name,
            "plots": bool(args.plots),
            "save_json": bool(args.save_json),
        }
        if args.device is not None:
            val_kwargs["device"] = args.device
        results = model.val(**val_kwargs)
        rows.append(extract_ultralytics_metrics(results, conf))

    write_csv(output_root / "detector_metrics.csv", rows)
    if args.distance_bins_csv is not None:
        write_csv(output_root / "distance_binned_metrics.csv", build_distance_bins(args.distance_bins_csv))

    protocol = {
        "model": str(args.model.resolve()),
        "data_yaml": str(args.data_yaml.resolve()),
        "output_root": str(output_root),
        "conf_thresholds": thresholds,
        "imgsz": args.imgsz,
        "split": args.split,
        "device": args.device,
        "distance_bins_csv": str(args.distance_bins_csv.resolve()) if args.distance_bins_csv else None,
    }
    (output_root / "run_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    return rows


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_inputs(args)
    rows = run_evaluation(args)
    print(f"[INFO] Wrote detector metrics for {len(rows)} threshold(s) to {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
