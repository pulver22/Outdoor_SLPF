from __future__ import annotations


def filter_grouped_map_points_by_classes(grouped_map_points, enabled_classes):
    enabled = {int(c) for c in enabled_classes}
    filtered = {}
    for row_id, points_in_row in grouped_map_points.items():
        kept = [p for p in points_in_row if int(p.get("class", -1)) in enabled]
        if kept:
            filtered[row_id] = kept
    return filtered
