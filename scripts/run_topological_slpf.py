#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = BASE_DIR.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
os.chdir(BASE_DIR)


def import_canonical_slpf():
    module_path = SCRIPTS_DIR / "spf_lidar.py"
    spec = importlib.util.spec_from_file_location("canonical_spf_lidar", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import canonical SLPF module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


slpf = import_canonical_slpf()


def import_topological_pf(topological_pf_root: Path):
    root = topological_pf_root.expanduser()
    if not root.is_absolute():
        root = (BASE_DIR / root).resolve()
    sys.path.insert(0, str(root))
    from topological_particle_filter import TopologicalMap, TopologicalParticleFilter

    return TopologicalMap, TopologicalParticleFilter


def resolve_path(path: Path, base: Path = BASE_DIR) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def configure_slpf_globals(args: argparse.Namespace) -> None:
    slpf.set_global_seed(args.seed)
    slpf.FRAME_STRIDE = max(1, int(args.frame_stride))
    slpf.SEMANTIC_SIGMA = float(args.semantic_sigma)
    slpf.GPS_SIGMA = float(args.gps_sigma)
    slpf.CORRIDOR_WEIGHT = float(args.corridor_weight)
    slpf.CORRIDOR_DIST_SIGMA = float(args.corridor_dist_sigma)
    slpf.CORRIDOR_HEADING_SIGMA = float(args.corridor_heading_sigma)
    slpf.BACKGROUND_CLASS_WEIGHT = float(args.background_class_weight)
    slpf.BACKGROUND_OBS_MAX = max(0, int(args.max_background_obs))
    slpf.POSE_SMOOTH_ALPHA_POS = float(args.pose_smooth_alpha_pos)
    slpf.POSE_SMOOTH_ALPHA_THETA = float(args.pose_smooth_alpha_theta)
    slpf.ODOM_YAW_FILTER_ALPHA = float(args.odom_yaw_filter_alpha)
    slpf.EXPECTED_OBS_COUNT = max(1.0, float(args.expected_obs_count))
    slpf.PARTICLE_COUNT = max(10, int(args.particle_count))
    slpf.SEMANTIC_MODEL = str(args.semantic_model)
    slpf.SEMANTIC_CLASSES_MODE = str(args.semantic_classes)
    slpf.POINT_ANG_SIGMA = max(1e-6, float(args.point_ang_sigma))
    slpf.POINT_RANGE_SIGMA = max(1e-6, float(args.point_range_sigma))
    slpf.POINT_ANG_GATE = max(0.0, float(args.point_ang_gate))
    slpf.POINT_MAX_RANGE_DIFF = max(0.0, float(args.point_max_range_diff))
    slpf.geojson_path = resolve_path(args.geojson_path)


def extract_semantic_lidar_observations(
    row,
    data_path: Path,
    *,
    semantic_classes: str,
    detection_drop_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    rgb_path = data_path / str(row["rgb_image"])
    depth_path = data_path / str(row["depth_image"])
    lidar_path = data_path / str(row["lidar_csv"])
    if not rgb_path.exists() or not depth_path.exists() or not lidar_path.exists():
        return empty_obs(), empty_obs(), empty_obs(), {
            "detections_raw": 0,
            "detections_kept": 0,
            "detections_dropped": 0,
        }

    color_img = cv2.imread(str(rgb_path))
    depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if color_img is None or depth_img is None:
        return empty_obs(), empty_obs(), empty_obs(), {
            "detections_raw": 0,
            "detections_kept": 0,
            "detections_dropped": 0,
        }

    lidar_frame = slpf.load_lidar_frame_from_csv(str(lidar_path), slpf.LIDAR_RANGE)
    results = slpf.yolo.predict(color_img, conf=0.2, classes=slpf.CLASS_IDS, verbose=False)[0]
    semantic_centers: list[tuple[float, float, int]] = []

    if results.masks is not None:
        masks = results.masks.data.cpu().numpy()
        for index, mask in enumerate(masks):
            class_id = int(results.boxes.cls[index].item())
            if class_id not in (2, 4):
                continue

            mask_resized_color = cv2.resize(
                mask,
                (color_img.shape[1], color_img.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            mask_indices_color = np.argwhere(mask_resized_color > 0.5)
            if mask_indices_color.size == 0:
                continue

            mask_resized_depth = cv2.resize(
                mask,
                (depth_img.shape[1], depth_img.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            mask_indices_depth = np.argwhere(mask_resized_depth > 0.5)
            if mask_indices_depth.size == 0:
                continue

            depth_values_mm = depth_img[mask_indices_depth[:, 0], mask_indices_depth[:, 1]]
            valid_depths_mm = depth_values_mm[depth_values_mm > 0]
            if valid_depths_mm.size == 0:
                continue

            min_depth_m = float(np.min(valid_depths_mm)) * 0.001
            if min_depth_m == 0.0 or min_depth_m > 10.0:
                continue

            u, v = np.mean(mask_indices_color, axis=0).astype(int)
            x_cam = (v - slpf.intr.ppx) / slpf.intr.fx * min_depth_m
            rel_x = -x_cam
            rel_z = min_depth_m
            semantic_centers.append((rel_x, rel_z, class_id))

    detections_raw = int(len(semantic_centers))
    if detection_drop_rate > 0.0 and detections_raw > 0:
        keep_mask = np.random.random(detections_raw) >= float(detection_drop_rate)
        semantic_centers = [semantic_centers[i] for i, keep in enumerate(keep_mask) if keep]

    detections_kept = int(len(semantic_centers))
    detections_dropped = int(detections_raw - detections_kept)
    diagnostics = {
        "detections_raw": detections_raw,
        "detections_kept": detections_kept,
        "detections_dropped": detections_dropped,
    }

    if not semantic_centers or lidar_frame is None or "xy" not in lidar_frame:
        return empty_obs(), empty_obs(), empty_obs(), diagnostics

    xy_lidar = lidar_frame["xy"]
    mask_valid = lidar_frame.get("mask_valid", np.isfinite(lidar_frame["ranges"]))
    xy_lidar = xy_lidar[mask_valid]
    if xy_lidar.size == 0:
        return empty_obs(), empty_obs(), empty_obs(), diagnostics

    x_fwd = xy_lidar[:, 0] - slpf.LIDAR_TO_CAMERA_DX
    y_left = xy_lidar[:, 1] - slpf.LIDAR_TO_CAMERA_DY
    lidar_bev = np.stack([y_left, x_fwd], axis=1)

    centers = np.array([[cx, cz] for (cx, cz, _) in semantic_centers], dtype=float)
    classes = np.array([cid for (_, _, cid) in semantic_centers], dtype=int)
    diffs = lidar_bev[:, None, :] - centers[None, :, :]
    d2 = np.sum(diffs * diffs, axis=2)
    nearest = np.argmin(d2, axis=1)
    nearest_dist = np.sqrt(d2[np.arange(d2.shape[0]), nearest])

    inside = nearest_dist <= slpf.SEMANTIC_RADIUS
    semantic_points = lidar_bev[inside]
    semantic_point_classes = classes[nearest[inside]]
    background_points = lidar_bev[~inside]

    poles = semantic_points[semantic_point_classes == 2]
    trunks = semantic_points[semantic_point_classes == 4]
    if semantic_classes == "poles":
        trunks = empty_obs()
    elif semantic_classes == "trunks":
        poles = empty_obs()

    return as_obs(poles), as_obs(trunks), as_obs(background_points), diagnostics


def empty_obs() -> np.ndarray:
    return np.empty((0, 2), dtype=np.float32)


def as_obs(values: np.ndarray) -> np.ndarray:
    if values is None or values.size == 0:
        return empty_obs()
    return np.asarray(values, dtype=np.float32).reshape(-1, 2)


def write_stats_csv(path: Path, rows: list[dict]) -> None:
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


def frame_timestamp(row, frame_idx: int) -> float:
    if "timestamp" in row and slpf.pd.notna(row["timestamp"]):
        return float(row["timestamp"])
    if "timestamp_sec" in row and slpf.pd.notna(row["timestamp_sec"]):
        return float(row["timestamp_sec"])
    return float(frame_idx)


def semantic_enabled_classes(mode: str) -> set[int]:
    if mode == "poles":
        return {2}
    if mode == "trunks":
        return {4}
    return {2, 4}


def run(args: argparse.Namespace) -> Path:
    configure_slpf_globals(args)
    if args.require_cuda and slpf.device != "cuda":
        raise RuntimeError("CUDA was required, but canonical scripts/spf_lidar.py selected a non-CUDA device.")

    rng = np.random.default_rng(args.seed)

    data_path = resolve_path(args.data_path)
    geojson_path = resolve_path(args.geojson_path)
    graph_json = resolve_path(args.graph_json)
    output_folder = resolve_path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    for required_path, label in (
        (data_path / "data.csv", "dataset CSV"),
        (geojson_path, "GeoJSON map"),
        (graph_json, "topological graph JSON"),
    ):
        if not required_path.exists():
            raise FileNotFoundError(f"Missing {label}: {required_path}")
    for subdir_name in ("rgb", "depth", "lidar"):
        subdir = data_path / subdir_name
        if not subdir.exists():
            raise FileNotFoundError(f"Missing dataset folder: {subdir}")

    TopologicalMap, TopologicalParticleFilter = import_topological_pf(resolve_path(args.topological_pf_root))
    topological_map = TopologicalMap.from_json(graph_json)
    pf = TopologicalParticleFilter(
        topological_map,
        num_particles=args.particle_count,
        rng=rng,
        unconnected_jump_distance=args.unconnected_jump_distance,
    )

    df_data = slpf.load_csv_with_utm(str(data_path / "data.csv"))
    grouped_map_points, center = slpf.load_landmarks_as_lines(str(geojson_path))
    grouped_semantic_points = slpf.filter_grouped_map_points_by_classes(
        grouped_map_points,
        semantic_enabled_classes(args.semantic_classes),
    )

    seg_p1, seg_p2, seg_v2, seg_cls = slpf.build_segment_tensors(grouped_map_points, device=slpf.device)
    sem_seg_p1, sem_seg_p2, sem_seg_v2, sem_seg_cls = slpf.build_segment_tensors(
        grouped_semantic_points,
        device=slpf.device,
    )
    point_poles, point_trunks = slpf.build_point_tensors(grouped_semantic_points, device=slpf.device)

    class_weights = dict(slpf.CLASS_WEIGHTS)
    class_weights[0] = float(args.background_class_weight)
    if args.semantic_classes == "poles":
        class_weights[4] = 0.0
    elif args.semantic_classes == "trunks":
        class_weights[2] = 0.0

    trajectory: list[tuple[float, float, float, float]] = []
    gps_gt_trajectory: list[tuple[float, float, float, float]] = []
    stats_rows: list[dict] = []
    prev_odom_pos_x = prev_odom_pos_y = prev_odom_yaw = None
    pose_smoothed_theta: float | None = None
    processed = 0
    start_time = time.time()

    for frame_idx, row in slpf.tqdm(df_data.iterrows(), total=df_data.shape[0], desc="Topological SLPF"):
        if frame_idx % args.frame_stride != 0:
            continue
        if args.max_frames is not None and processed >= args.max_frames:
            break

        poles, trunks, background, detection_stats = extract_semantic_lidar_observations(
            row,
            data_path,
            semantic_classes=args.semantic_classes,
            detection_drop_rate=args.detection_drop_rate,
        )

        gps_x = float(row["utm_easting"] - center[0])
        gps_y = float(row["utm_northing"] - center[1])
        gps_x_noisy = float(row["utm_easting_noisy"] - center[0])
        gps_y_noisy = float(row["utm_northing_noisy"] - center[1])

        current_odom_pos_x = float(row["odom_pos_x"])
        current_odom_pos_y = float(row["odom_pos_y"])
        current_odom_yaw = float(
            slpf.quaternion_to_yaw(
                row["odom_orient_x"],
                row["odom_orient_y"],
                row["odom_orient_z"],
                row["odom_orient_w"],
            )
        )

        had_prev_odom = prev_odom_pos_x is not None
        delta_distance = 0.0
        delta_theta = 0.0
        if pf.particles is None:
            init_x = gps_x_noisy if args.init_from_noisy_gps else gps_x
            init_y = gps_y_noisy if args.init_from_noisy_gps else gps_y
            pf.update_pose_observation(
                init_x,
                init_y,
                args.init_pose_sigma**2,
                args.init_pose_sigma**2,
                identifying=True,
                delta_secs=0.0,
            )
        elif had_prev_odom:
            dx_odom = current_odom_pos_x - float(prev_odom_pos_x)
            dy_odom = current_odom_pos_y - float(prev_odom_pos_y)
            delta_distance = float(np.hypot(dx_odom, dy_odom))
            filtered_odom_yaw = slpf.circular_lerp(float(prev_odom_yaw), current_odom_yaw, slpf.ODOM_YAW_FILTER_ALPHA)
            delta_theta = float(slpf.angle_diff(filtered_odom_yaw, float(prev_odom_yaw)))
            pf.predict_from_odometry(
                delta_distance=delta_distance,
                delta_theta=delta_theta,
                noise_std=(args.motion_xy_std, args.motion_xy_std, args.motion_theta_std),
            )

        prev_odom_pos_x = current_odom_pos_x
        prev_odom_pos_y = current_odom_pos_y
        prev_odom_yaw = current_odom_yaw

        if pf.particles is None:
            raise RuntimeError("Topological particle filter did not initialize.")

        weights, frame_stats = slpf.measurement_likelihood_gpu(
            grouped_map_points,
            poles,
            trunks,
            background,
            pf.particles[:, :3],
            miss_penalty=args.miss_penalty,
            wrong_hit_penalty=args.wrong_hit_penalty,
            gps_weight=args.gps_weight,
            gps_xy=(gps_x_noisy, gps_y_noisy) if not args.disable_gps else None,
            gps_sigma=slpf.GPS_SIGMA,
            seg_p1=seg_p1,
            seg_p2=seg_p2,
            seg_v2=seg_v2,
            seg_cls=seg_cls,
            sem_seg_p1=sem_seg_p1,
            sem_seg_p2=sem_seg_p2,
            sem_seg_v2=sem_seg_v2,
            sem_seg_cls=sem_seg_cls,
            point_poles=point_poles,
            point_trunks=point_trunks,
            sensor_range=slpf.SENSOR_RANGE,
            class_weights=class_weights,
            device=slpf.device,
            segment_chunk=max(32, int(args.segment_chunk)),
            max_background_obs=slpf.BACKGROUND_OBS_MAX,
            background_class_weight=slpf.BACKGROUND_CLASS_WEIGHT,
            corridor_weight=slpf.CORRIDOR_WEIGHT,
            corridor_dist_sigma=slpf.CORRIDOR_DIST_SIGMA,
            corridor_heading_sigma=slpf.CORRIDOR_HEADING_SIGMA,
            disable_gps=args.disable_gps,
            disable_semantic=args.disable_semantic,
            disable_corridor=args.disable_corridor,
            disable_background=args.disable_background,
            disable_dynamic_gps_weight=args.disable_dynamic_gps_weight,
            semantic_model=args.semantic_model,
            point_ang_sigma=args.point_ang_sigma,
            point_range_sigma=args.point_range_sigma,
            point_ang_gate=args.point_ang_gate,
            point_max_range_diff=args.point_max_range_diff,
        )

        result = pf.apply_particle_weights(
            weights,
            diagnostics={
                "frame_idx": int(frame_idx),
                "num_graph_nodes": int(topological_map.num_nodes),
                **frame_stats,
            },
        )

        weighted_pose = result.estimated_pose
        est_xy = topological_map.node_coords[result.estimated_node_index]
        est_theta_raw = float(weighted_pose[2])
        if args.disable_pose_smoothing:
            est_theta = est_theta_raw
        elif pose_smoothed_theta is None:
            pose_smoothed_theta = est_theta_raw
            est_theta = est_theta_raw
        else:
            predicted_theta = pose_smoothed_theta
            if had_prev_odom:
                predicted_theta = float(slpf.wrap_to_pi(pose_smoothed_theta + delta_theta))
            pose_smoothed_theta = float(slpf.circular_lerp(predicted_theta, est_theta_raw, slpf.POSE_SMOOTH_ALPHA_THETA))
            est_theta = pose_smoothed_theta

        ts = frame_timestamp(row, int(frame_idx))
        trajectory.append((ts, float(est_xy[0]), float(est_xy[1]), float(est_theta)))
        gps_gt_trajectory.append((ts, gps_x, gps_y, 0.0))

        stats_row = dict(result.diagnostics)
        stats_row.update(detection_stats)
        stats_row.update(
            {
                "frame_idx": int(frame_idx),
                "timestamp": float(ts),
                "estimated_node": result.estimated_node_name,
                "estimated_node_index": int(result.estimated_node_index),
                "estimated_x": float(est_xy[0]),
                "estimated_y": float(est_xy[1]),
                "estimated_theta": float(est_theta),
                "weighted_mean_x": float(weighted_pose[0]),
                "weighted_mean_y": float(weighted_pose[1]),
                "weighted_mean_theta": float(weighted_pose[2]),
                "gps_x": gps_x,
                "gps_y": gps_y,
                "gps_x_noisy": gps_x_noisy,
                "gps_y_noisy": gps_y_noisy,
                "pose_smoothing_enabled": int(not args.disable_pose_smoothing),
                "semantic_classes_mode": str(args.semantic_classes),
                "detection_drop_rate": float(args.detection_drop_rate),
                "delta_distance": float(delta_distance),
                "delta_theta": float(delta_theta),
                "graph_node_pose": 1,
            }
        )
        stats_rows.append(stats_row)
        processed += 1

    slpf.save_tum_trajectory(trajectory, str(output_folder / "trajectory_0.5.tum"))
    slpf.save_tum_trajectory(gps_gt_trajectory, str(output_folder / "gps_pose.tum"))
    write_stats_csv(output_folder / "stats.csv", stats_rows)

    protocol = {
        "method": "SLPF-topological",
        "runtime_sec": time.time() - start_time,
        "processed_frames": processed,
        "data_path": str(data_path),
        "geojson_path": str(geojson_path),
        "graph_json": str(graph_json),
        "topological_pf_root": str(resolve_path(args.topological_pf_root)),
        "canonical_slpf": str(SCRIPTS_DIR / "spf_lidar.py"),
        "args": vars(args),
    }
    (output_folder / "run_protocol.json").write_text(json.dumps(protocol, indent=2, default=str), encoding="utf-8")
    return output_folder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run paper-protocol SLPF with graph-constrained topological particles.")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--geojson-path", type=Path, required=True)
    parser.add_argument("--graph-json", type=Path, required=True)
    parser.add_argument("--topological-pf-root", type=Path, default=WORKSPACE_DIR / "topological_pf")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output-folder", type=Path, required=True)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-visualization", action="store_true", help="Accepted for CLI parity; this runner does not render frames.")
    parser.add_argument("--semantic-sigma", type=float, default=slpf.SEMANTIC_SIGMA)
    parser.add_argument("--gps-sigma", type=float, default=slpf.GPS_SIGMA)
    parser.add_argument("--miss-penalty", type=float, default=4.0)
    parser.add_argument("--wrong-hit-penalty", type=float, default=4.0)
    parser.add_argument("--gps-weight", type=float, default=0.5)
    parser.add_argument("--corridor-weight", type=float, default=slpf.CORRIDOR_WEIGHT)
    parser.add_argument("--corridor-dist-sigma", type=float, default=slpf.CORRIDOR_DIST_SIGMA)
    parser.add_argument("--corridor-heading-sigma", type=float, default=slpf.CORRIDOR_HEADING_SIGMA)
    parser.add_argument("--background-class-weight", type=float, default=slpf.BACKGROUND_CLASS_WEIGHT)
    parser.add_argument("--max-background-obs", type=int, default=slpf.BACKGROUND_OBS_MAX)
    parser.add_argument("--pose-smooth-alpha-pos", type=float, default=slpf.POSE_SMOOTH_ALPHA_POS)
    parser.add_argument("--pose-smooth-alpha-theta", type=float, default=slpf.POSE_SMOOTH_ALPHA_THETA)
    parser.add_argument("--odom-yaw-filter-alpha", type=float, default=slpf.ODOM_YAW_FILTER_ALPHA)
    parser.add_argument("--expected-obs-count", type=float, default=slpf.EXPECTED_OBS_COUNT)
    parser.add_argument("--particle-count", type=int, default=slpf.PARTICLE_COUNT)
    parser.add_argument("--disable-gps", action="store_true")
    parser.add_argument("--disable-semantic", action="store_true")
    parser.add_argument("--disable-corridor", action="store_true")
    parser.add_argument("--disable-background", action="store_true")
    parser.add_argument("--disable-dynamic-gps-weight", action="store_true")
    parser.add_argument("--disable-pose-smoothing", action="store_true")
    parser.add_argument("--semantic-model", choices=["wall", "point"], default=slpf.SEMANTIC_MODEL)
    parser.add_argument("--semantic-classes", choices=["both", "poles", "trunks"], default=slpf.SEMANTIC_CLASSES_MODE)
    parser.add_argument("--detection-drop-rate", type=float, default=0.0)
    parser.add_argument("--point-ang-sigma", type=float, default=slpf.POINT_ANG_SIGMA)
    parser.add_argument("--point-range-sigma", type=float, default=slpf.POINT_RANGE_SIGMA)
    parser.add_argument("--point-ang-gate", type=float, default=slpf.POINT_ANG_GATE)
    parser.add_argument("--point-max-range-diff", type=float, default=slpf.POINT_MAX_RANGE_DIFF)
    parser.add_argument("--segment-chunk", type=int, default=4096)
    parser.add_argument("--motion-xy-std", type=float, default=0.1)
    parser.add_argument("--motion-theta-std", type=float, default=float(np.deg2rad(10.0)))
    parser.add_argument("--unconnected-jump-distance", type=float, default=3.0)
    parser.add_argument("--init-pose-sigma", type=float, default=slpf.PARTICLE_STD)
    parser.add_argument("--init-from-noisy-gps", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = run(args)
    print(f"[INFO] Topological SLPF outputs written to {out_dir}")


if __name__ == "__main__":
    main()
