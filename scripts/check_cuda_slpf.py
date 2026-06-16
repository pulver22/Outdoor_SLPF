#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = BASE_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


def import_canonical_slpf():
    module_path = SCRIPTS_DIR / "spf_lidar.py"
    spec = importlib.util.spec_from_file_location("canonical_spf_lidar_cuda_check", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import canonical SLPF module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def yolo_model_device(slpf) -> str:
    try:
        param = next(slpf.yolo.model.parameters())
        return str(param.device)
    except Exception as exc:
        return f"unknown:{type(exc).__name__}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test CUDA for the full SLPF pipeline.")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    summary = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "torch_cuda_available_before_slpf": bool(torch.cuda.is_available()),
        "torch_device_count": int(torch.cuda.device_count()),
    }
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is not available before importing SLPF: {summary}")

    tensor = torch.zeros((4,), device="cuda" if torch.cuda.is_available() else "cpu")
    summary["torch_allocation_device"] = str(tensor.device)

    slpf = import_canonical_slpf()
    summary["slpf_device"] = str(slpf.device)
    summary["yolo_model_device"] = yolo_model_device(slpf)
    if args.require_cuda and slpf.device != "cuda":
        raise RuntimeError(f"Canonical SLPF did not select CUDA: {summary}")
    if args.require_cuda and "cuda" not in summary["yolo_model_device"]:
        raise RuntimeError(f"YOLO model is not on CUDA: {summary}")

    grouped_map_points = {
        "smoke_row": [
            {"coords": np.array([0.0, 0.0], dtype=np.float64), "class": 2},
            {"coords": np.array([0.0, 5.0], dtype=np.float64), "class": 4},
        ]
    }
    seg_p1, seg_p2, seg_v2, seg_cls = slpf.build_segment_tensors(grouped_map_points, device=slpf.device)
    particles = np.array(
        [
            [0.0, 0.0, np.pi / 2.0],
            [1.0, 0.0, np.pi / 2.0],
            [0.0, 1.0, np.pi / 2.0],
        ],
        dtype=np.float64,
    )
    weights, stats = slpf.measurement_likelihood_gpu(
        grouped_map_points,
        np.array([[0.0, 1.0]], dtype=np.float32),
        np.empty((0, 2), dtype=np.float32),
        np.empty((0, 2), dtype=np.float32),
        particles,
        miss_penalty=4.0,
        wrong_hit_penalty=4.0,
        gps_weight=0.5,
        gps_xy=(0.0, 0.0),
        gps_sigma=1.1,
        seg_p1=seg_p1,
        seg_p2=seg_p2,
        seg_v2=seg_v2,
        seg_cls=seg_cls,
        sensor_range=slpf.SENSOR_RANGE,
        class_weights=slpf.CLASS_WEIGHTS,
        device=slpf.device,
        segment_chunk=32,
    )
    summary["likelihood_weight_sum"] = float(np.sum(weights))
    summary["likelihood_stats_keys"] = sorted(stats.keys())
    if args.require_cuda and not np.isfinite(weights).all():
        raise RuntimeError(f"Non-finite likelihood weights: {summary}")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
