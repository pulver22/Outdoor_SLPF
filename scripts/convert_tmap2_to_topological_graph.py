#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import yaml


def sanitize_name(value: object) -> str:
    text = re.sub(r"[^0-9A-Za-z_]+", "_", str(value).strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "node"


def load_tmap2_nodes(path: Path) -> tuple[dict[str, np.ndarray], set[tuple[str, str]], dict]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "nodes" not in payload:
        raise ValueError(f"{path} is not a tmap2 YAML file with a top-level 'nodes' entry.")

    coords: dict[str, np.ndarray] = {}
    raw_edges: set[tuple[str, str]] = set()
    for entry in payload["nodes"]:
        node = entry.get("node", {})
        name = str(node["name"])
        position = node["pose"]["position"]
        coords[name] = np.array([float(position["x"]), float(position["y"])], dtype=np.float64)

    for entry in payload["nodes"]:
        node = entry.get("node", {})
        src = str(node["name"])
        for edge in node.get("edges", []):
            dst = str(edge["node"])
            if src not in coords or dst not in coords:
                continue
            raw_edges.add(tuple(sorted((src, dst))))

    metadata = {
        "source_format": "tmap2",
        "source_tmap2": str(path),
        "metric_map": payload.get("metric_map"),
        "name": payload.get("name"),
        "pointset": payload.get("pointset"),
        "num_tmap_nodes": len(coords),
        "num_tmap_edges": len(raw_edges),
    }
    return coords, raw_edges, metadata


def add_edge(neighbors: dict[str, set[str]], a: str, b: str) -> None:
    if a == b:
        return
    neighbors.setdefault(a, set()).add(b)
    neighbors.setdefault(b, set()).add(a)


def is_lane_edge(a: np.ndarray, b: np.ndarray, min_dy_dx_ratio: float = 2.0) -> bool:
    delta = b - a
    return abs(float(delta[1])) > abs(float(delta[0])) * float(min_dy_dx_ratio)


def lane_components(
    coords: dict[str, np.ndarray],
    edges: set[tuple[str, str]],
    *,
    min_dy_dx_ratio: float = 2.0,
) -> list[list[str]]:
    adjacency: dict[str, set[str]] = {name: set() for name in coords}
    for src, dst in edges:
        if is_lane_edge(coords[src], coords[dst], min_dy_dx_ratio=min_dy_dx_ratio):
            adjacency[src].add(dst)
            adjacency[dst].add(src)

    seen: set[str] = set()
    components: list[list[str]] = []
    for name in coords:
        if name in seen or not adjacency[name]:
            continue
        stack = [name]
        seen.add(name)
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        component.sort(key=lambda node_name: coords[node_name][1], reverse=True)
        components.append(component)

    components.sort(key=lambda component: float(np.mean([coords[name][0] for name in component])))
    return components


def add_outer_lanes_to_tmap_edges(
    coords: dict[str, np.ndarray],
    edges: set[tuple[str, str]],
    *,
    min_dy_dx_ratio: float = 2.0,
) -> tuple[dict[str, np.ndarray], set[tuple[str, str]], dict]:
    components = lane_components(coords, edges, min_dy_dx_ratio=min_dy_dx_ratio)
    if len(components) < 2:
        raise ValueError("At least two lane components are required to extrapolate outer lanes.")
    if len(components[0]) != len(components[1]) or len(components[-1]) != len(components[-2]):
        raise ValueError("Boundary lane components must have matching node counts.")

    augmented_coords = dict(coords)
    augmented_edges = set(edges)
    added_lanes: list[dict] = []

    def add_lane(prefix: str, boundary: list[str], neighbor: list[str], direction: int) -> list[str]:
        names = []
        for index, (boundary_name, neighbor_name) in enumerate(zip(boundary, neighbor)):
            name = f"{prefix}_{index:02d}"
            xy = coords[boundary_name] + direction * (coords[boundary_name] - coords[neighbor_name])
            augmented_coords[name] = xy.astype(np.float64)
            names.append(name)

        for a, b in zip(names[:-1], names[1:]):
            augmented_edges.add(tuple(sorted((a, b))))
        augmented_edges.add(tuple(sorted((names[0], boundary[0]))))
        augmented_edges.add(tuple(sorted((names[-1], boundary[-1]))))

        added_lanes.append(
            {
                "prefix": prefix,
                "node_count": len(names),
                "connected_to_top": boundary[0],
                "connected_to_bottom": boundary[-1],
            }
        )
        return names

    add_lane("OuterLaneLeft", components[0], components[1], direction=1)
    add_lane("OuterLaneRight", components[-1], components[-2], direction=1)
    return augmented_coords, augmented_edges, {"outer_lanes": added_lanes, "detected_lane_count": len(components)}


def build_payload_from_tmap2(path: Path, spacing_m: float, *, add_outer_lanes: bool = False) -> dict:
    coords, tmap_edges, metadata = load_tmap2_nodes(path)
    spacing_m = max(float(spacing_m), 1e-6)
    if add_outer_lanes:
        coords, tmap_edges, outer_lane_metadata = add_outer_lanes_to_tmap_edges(coords, tmap_edges)
        metadata.update(outer_lane_metadata)
        metadata["added_outer_lanes"] = True
    else:
        metadata["added_outer_lanes"] = False

    nodes: dict[str, dict] = {}
    neighbors: dict[str, set[str]] = {}

    def ensure_node(name: str, xy: np.ndarray) -> str:
        safe = sanitize_name(name)
        nodes.setdefault(safe, {"name": safe, "x": float(xy[0]), "y": float(xy[1])})
        neighbors.setdefault(safe, set())
        return safe

    for name, xy in coords.items():
        ensure_node(name, xy)

    for src, dst in sorted(tmap_edges):
        src_xy = coords[src]
        dst_xy = coords[dst]
        delta = dst_xy - src_xy
        length = float(np.linalg.norm(delta))
        steps = max(1, int(math.ceil(length / spacing_m)))

        chain = [ensure_node(src, src_xy)]
        edge_prefix = f"{sanitize_name(src)}__{sanitize_name(dst)}"
        for step in range(1, steps):
            xy = src_xy + (step / steps) * delta
            chain.append(ensure_node(f"{edge_prefix}_{step:04d}", xy))
        chain.append(ensure_node(dst, dst_xy))

        for a, b in zip(chain[:-1], chain[1:]):
            add_edge(neighbors, a, b)

    out_nodes = []
    for name in sorted(nodes):
        node = dict(nodes[name])
        node["neighbors"] = sorted(neighbors.get(name, ()))
        out_nodes.append(node)

    metadata.update(
        {
            "spacing_m": float(spacing_m),
            "num_nodes": len(out_nodes),
            "num_edges": sum(len(node["neighbors"]) for node in out_nodes) // 2,
        }
    )
    return {"metadata": metadata, "nodes": out_nodes}


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a ROS tmap2 YAML graph into TopologicalMap JSON.")
    parser.add_argument("--tmap2-yaml", type=Path, required=True)
    parser.add_argument("--spacing-m", type=float, default=0.5)
    parser.add_argument("--add-outer-lanes", action="store_true")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    payload = build_payload_from_tmap2(
        args.tmap2_yaml,
        spacing_m=args.spacing_m,
        add_outer_lanes=args.add_outer_lanes,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"[INFO] Wrote {payload['metadata']['num_nodes']} nodes and "
        f"{payload['metadata']['num_edges']} edges to {args.output_json}"
    )


if __name__ == "__main__":
    main()
