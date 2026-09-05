#!/usr/bin/env python3
"""Merge per-traversal experiment CSVs into release-facing matrix views."""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


SEEDS = (11, 22, 33)


def read_rows(paths: Iterable[Path]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                item: dict[str, object] = dict(row)
                item["source_csv"] = str(path)
                try:
                    item["seed"] = int(str(item["seed"]))
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid seed in {path}: {row}") from exc
                if str(item.get("traversal", "")).strip() not in {"rh_run1", "rh_run2"}:
                    raise ValueError(f"{path}: every row must carry traversal=rh_run1 or rh_run2")
                result.append(item)
    return result


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _number(value: object) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def _summarize(values: Iterable[object]) -> tuple[float, float, float]:
    arr = np.asarray([_number(value) for value in values], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr)), float(np.median(arr))


def aggregate_ablation(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["variant"])].append(row)
    ignored = {"variant", "traversal", "seed", "command", "source_csv"}
    out: list[dict[str, object]] = []
    for variant, group in sorted(groups.items()):
        item: dict[str, object] = {"variant": variant, "n_runs": len(group), "traversals": ",".join(sorted({str(r["traversal"]) for r in group}))}
        numeric_fields = sorted({field for row in group for field, value in row.items() if field not in ignored and math.isfinite(_number(value))})
        for field in numeric_fields:
            mean, std, median = _summarize(row.get(field) for row in group)
            item[f"{field}_mean"] = mean
            item[f"{field}_std"] = std
            item[f"{field}_median"] = median
        out.append(item)
    full = next((row for row in out if row["variant"] == "full"), None)
    if full is not None:
        base = _number(full.get("ape_align_rmse_mean"))
        for row in out:
            row["delta_vs_full_ape_align_rmse"] = _number(row.get("ape_align_rmse_mean")) - base
    return out


def aggregate_robustness(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["option"]), str(row["variant"]))].append(row)
    ignored = {"option", "variant", "traversal", "seed", "command", "source_csv", "map_geojson"}
    out: list[dict[str, object]] = []
    for (option, variant), group in sorted(groups.items()):
        item: dict[str, object] = {"option": option, "variant": variant, "n_runs": len(group), "traversals": ",".join(sorted({str(r["traversal"]) for r in group}))}
        numeric_fields = sorted({field for row in group for field, value in row.items() if field not in ignored and math.isfinite(_number(value))})
        for field in numeric_fields:
            mean, std, median = _summarize(row.get(field) for row in group)
            item[f"{field}_mean"] = mean
            item[f"{field}_std"] = std
            item[f"{field}_median"] = median
        out.append(item)
    baseline = next((row for row in out if row["option"] == "baseline" and row["variant"] == "full_map"), None)
    if baseline is not None:
        base_ape = _number(baseline.get("ape_align_rmse_mean"))
        base_corr = _number(baseline.get("row_correct_fraction_mean"))
        for row in out:
            row["delta_ape_align_vs_baseline"] = _number(row.get("ape_align_rmse_mean")) - base_ape
            row["delta_row_correct_vs_baseline"] = _number(row.get("row_correct_fraction_mean")) - base_corr
    return out


def merge_recovery(paths: Iterable[Path]) -> list[dict[str, object]]:
    rows = read_rows(paths)
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["variant"])].append(row)
    out: list[dict[str, object]] = []
    for variant, group in sorted(groups.items()):
        item: dict[str, object] = {"variant": variant, "n_runs": len(group), "traversals": ",".join(sorted({str(r["traversal"]) for r in group}))}
        ignored = {"variant", "traversal", "seed", "source_csv"}
        for field in sorted({field for row in group for field, value in row.items() if field not in ignored and math.isfinite(_number(value))}):
            mean, std, median = _summarize(row.get(field) for row in group)
            item[f"{field}_mean"] = mean
            item[f"{field}_std"] = std
            item[f"{field}_median"] = median
        out.append(item)
    return out


def validate_complete(rows: list[dict[str, object]], kind: str) -> None:
    if kind == "ablation":
        keys = {(str(r["traversal"]), str(r["variant"])) for r in rows}
        for traversal, variant in sorted(keys):
            present = {int(r["seed"]) for r in rows if str(r["traversal"]) == traversal and str(r["variant"]) == variant}
            if present != set(SEEDS):
                raise ValueError(f"Incomplete {kind} group {traversal}/{variant}: {sorted(present)}")
    else:
        keys = {(str(r["traversal"]), str(r["option"]), str(r["variant"])) for r in rows}
        for traversal, option, variant in sorted(keys):
            present = {int(r["seed"]) for r in rows if str(r["traversal"]) == traversal and str(r["option"]) == option and str(r["variant"]) == variant}
            if present != set(SEEDS):
                raise ValueError(f"Incomplete {kind} group {traversal}/{option}/{variant}: {sorted(present)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["ablation", "robustness"], required=True)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output-per-seed", type=Path, required=True)
    parser.add_argument("--output-aggregate", type=Path, required=True)
    parser.add_argument("--recovery-input", type=Path, nargs="*", default=[])
    parser.add_argument("--recovery-output", type=Path, default=None)
    args = parser.parse_args()
    rows = read_rows(args.input)
    validate_complete(rows, args.kind)
    rows.sort(key=lambda row: (str(row["traversal"]), str(row.get("option", "")), str(row.get("variant", "")), int(row["seed"])))
    write_rows(args.output_per_seed, rows)
    aggregate = aggregate_ablation(rows) if args.kind == "ablation" else aggregate_robustness(rows)
    write_rows(args.output_aggregate, aggregate)
    if args.kind == "robustness" and args.recovery_input:
        if args.recovery_output is None:
            raise ValueError("--recovery-output is required with --recovery-input")
        write_rows(args.recovery_output, merge_recovery(args.recovery_input))
    print(f"[INFO] Merged {len(rows)} {args.kind} rows into {args.output_per_seed}")


if __name__ == "__main__":
    main()
