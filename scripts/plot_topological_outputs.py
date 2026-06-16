#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


def extract_row_id(row) -> str | None:
    if row["feature_type"] == "vine" and row.get("vine_vine_row_id"):
        return str(row["vine_vine_row_id"])
    if row["feature_type"] == "row_post" and row.get("feature_name"):
        parts = str(row["feature_name"]).split("_")
        return f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else str(row["feature_name"])
    return None


def load_semantic_map(geojson_path: Path):
    gdf = gpd.read_file(geojson_path)
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[gdf.geometry.geom_type == "Point"].copy()
    gdf["row_id"] = gdf.apply(extract_row_id, axis=1)
    gdf.dropna(subset=["row_id"], inplace=True)
    if not gdf.crs or not gdf.crs.is_projected:
        gdf = gdf.to_crs(gdf.estimate_utm_crs())

    try:
        center = gdf.geometry.union_all().centroid
    except AttributeError:
        center = gdf.geometry.unary_union.centroid

    gdf["x_centered"] = gdf.geometry.x - center.x
    gdf["y_centered"] = gdf.geometry.y - center.y
    gdf["semantic_class"] = np.where(gdf["feature_type"] == "row_post", "pole", "trunk")

    row_segments = []
    for _, group in gdf.groupby("row_id"):
        coords = group[["x_centered", "y_centered"]].to_numpy(dtype=float)
        if len(coords) < 2:
            continue
        sort_axis = 1 if np.ptp(coords[:, 0]) < np.ptp(coords[:, 1]) else 0
        coords = coords[np.argsort(coords[:, sort_axis])]
        row_segments.extend([[a, b] for a, b in zip(coords[:-1], coords[1:])])

    return gdf, np.asarray(row_segments, dtype=float), np.array([center.x, center.y], dtype=float)


def load_graph(graph_json: Path):
    payload = json.loads(graph_json.read_text(encoding="utf-8"))
    nodes = payload["nodes"]
    coords = {node["name"]: np.array([float(node["x"]), float(node["y"])], dtype=float) for node in nodes}
    edges = []
    seen = set()
    for node in nodes:
        a = node["name"]
        for b in node.get("neighbors", []):
            key = tuple(sorted((a, b)))
            if key in seen or b not in coords:
                continue
            seen.add(key)
            edges.append([coords[a], coords[b]])
    return payload, np.asarray(list(coords.values()), dtype=float), np.asarray(edges, dtype=float)


def read_tum(path: Path):
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
    return np.asarray(stamps, dtype=float), np.asarray(xy, dtype=float)


def interpolate_xy(src_t: np.ndarray, src_xy: np.ndarray, dst_t: np.ndarray) -> np.ndarray:
    if len(src_t) == 1:
        return np.repeat(src_xy, len(dst_t), axis=0)
    return np.column_stack([np.interp(dst_t, src_t, src_xy[:, i]) for i in range(2)])


def set_equal_axes(ax, all_xy: list[np.ndarray], pad: float = 3.0) -> None:
    points = np.vstack([xy.reshape(-1, 2) for xy in all_xy if xy.size])
    xmin, ymin = np.min(points, axis=0) - pad
    xmax, ymax = np.max(points, axis=0) + pad
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")


def draw_base(ax, gdf, row_segments, graph_xy, graph_edges, title: str) -> None:
    if len(row_segments):
        ax.add_collection(LineCollection(row_segments, colors="#6B7280", linewidths=2.0, alpha=0.55, label="semantic walls"))
    if len(graph_edges):
        ax.add_collection(LineCollection(graph_edges, colors="#2563EB", linewidths=0.8, alpha=0.45, label="topological edges"))
    if len(graph_xy):
        ax.scatter(graph_xy[:, 0], graph_xy[:, 1], s=7, c="#1D4ED8", alpha=0.85, linewidths=0, label="graph nodes")

    poles = gdf[gdf["semantic_class"] == "pole"]
    trunks = gdf[gdf["semantic_class"] == "trunk"]
    ax.scatter(trunks["x_centered"], trunks["y_centered"], s=28, c="#16A34A", edgecolors="white", linewidths=0.4, label="trunks")
    ax.scatter(poles["x_centered"], poles["y_centered"], s=44, c="#DC2626", marker="s", edgecolors="white", linewidths=0.5, label="poles")
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel("x from map center [m]")
    ax.set_ylabel("y from map center [m]")
    ax.grid(True, color="#E5E7EB", linewidth=0.8)


def plot_map(args) -> None:
    gdf, row_segments, _ = load_semantic_map(args.geojson_path)
    graph_payload, graph_xy, graph_edges = load_graph(args.graph_json)

    fig, ax = plt.subplots(figsize=(9.5, 8.5), dpi=args.dpi)
    draw_base(
        ax,
        gdf,
        row_segments,
        graph_xy,
        graph_edges,
        args.map_title,
    )
    set_equal_axes(ax, [gdf[["x_centered", "y_centered"]].to_numpy(), graph_xy])
    metadata = graph_payload.get("metadata", {})
    subtitle = (
        f"{len(gdf[gdf['semantic_class'] == 'pole'])} poles, "
        f"{len(gdf[gdf['semantic_class'] == 'trunk'])} trunks, "
        f"{metadata.get('num_nodes', len(graph_xy))} graph nodes"
    )
    ax.text(0.01, 0.99, subtitle, transform=ax.transAxes, va="top", ha="left", fontsize=10, color="#374151")
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=9)
    fig.tight_layout()
    args.map_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.map_out, bbox_inches="tight")


def plot_trajectory(args) -> None:
    gdf, row_segments, _ = load_semantic_map(args.geojson_path)
    _, graph_xy, graph_edges = load_graph(args.graph_json)
    est_t, est_xy = read_tum(args.trajectory_tum)
    gt_t, gt_xy = read_tum(args.reference_tum)
    gt_interp = interpolate_xy(gt_t, gt_xy, est_t)
    errors = np.linalg.norm(est_xy - gt_interp, axis=1)

    segments = np.stack([est_xy[:-1], est_xy[1:]], axis=1) if len(est_xy) > 1 else np.empty((0, 2, 2))
    segment_errors = (errors[:-1] + errors[1:]) * 0.5 if len(errors) > 1 else np.asarray([])

    fig, ax = plt.subplots(figsize=(9.5, 8.5), dpi=args.dpi)
    draw_base(ax, gdf, row_segments, graph_xy, graph_edges, args.trajectory_title)
    if len(segments):
        lc = LineCollection(segments, cmap="viridis", linewidths=3.2, alpha=0.95)
        lc.set_array(segment_errors)
        lc.set_clim(0.0, max(float(np.nanpercentile(errors, 95)), 1e-6))
        ax.add_collection(lc)
        cbar = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("position error to reference [m]")
    ax.scatter(est_xy[0, 0], est_xy[0, 1], s=90, marker="o", c="#111827", edgecolors="white", linewidths=0.8, label="start")
    ax.scatter(est_xy[-1, 0], est_xy[-1, 1], s=110, marker="X", c="#F97316", edgecolors="white", linewidths=0.8, label="end")
    ax.plot(gt_interp[:, 0], gt_interp[:, 1], color="#111827", linewidth=1.4, linestyle="--", alpha=0.65, label="reference")
    set_equal_axes(ax, [gdf[["x_centered", "y_centered"]].to_numpy(), graph_xy, est_xy, gt_interp])
    ax.text(
        0.01,
        0.99,
        f"ATE RMSE {np.sqrt(np.mean(errors * errors)):.2f} m | max {np.max(errors):.2f} m | {len(est_xy)} samples",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        color="#374151",
    )
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=9)
    fig.tight_layout()
    args.trajectory_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.trajectory_out, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Riseholme topological graph, semantic walls, and trajectory error.")
    parser.add_argument("--geojson-path", type=Path, required=True)
    parser.add_argument("--graph-json", type=Path, required=True)
    parser.add_argument("--trajectory-tum", type=Path, required=True)
    parser.add_argument("--reference-tum", type=Path, required=True)
    parser.add_argument("--map-out", type=Path, required=True)
    parser.add_argument("--trajectory-out", type=Path, required=True)
    parser.add_argument("--map-title", type=str, default="Riseholme Semantic Walls and Densified Topological Graph")
    parser.add_argument("--trajectory-title", type=str, default="Topological SLPF Trajectory Error on Graph")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    plot_map(args)
    plot_trajectory(args)


if __name__ == "__main__":
    main()
