#!/usr/bin/env python3
"""Create the machine-readable descriptive/statistical ICRA evidence summary.

The unit of analysis is a seed within a traversal/profile.  Pairwise tests are
only emitted when both methods have genuinely rerun, non-dedicated rows for
the same units.  Dedicated AMCL/RTAB-Map references are summarized separately
as fixed-reference deltas and never receive a significance value.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


EXPECTED_SEEDS = (11, 22, 33)
DEFAULT_METRICS = (
    "ape_raw_rmse",
    "ape_align_rmse",
    "wrong_row_duration_sec",
    "cross_track_mean",
    "row_correct_fraction",
    "recovery_distance_m",
)


def _float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def read_rows(paths: Iterable[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                item: dict[str, object] = dict(row)
                item["source_csv"] = str(path)
                # Matrix views use ``variant`` (ablation) or ``option`` plus
                # ``variant`` (robustness) instead of a method column.  Keep
                # those rows identifiable in the same descriptive summary.
                method = str(item.get("method", "")).strip()
                if not method:
                    method = str(item.get("variant", "")).strip()
                    option = str(item.get("option", "")).strip()
                    if option and method:
                        method = f"{option}:{method}"
                item["method"] = method
                item["traversal"] = str(item.get("traversal", "pooled")).strip() or "pooled"
                item["profile"] = str(item.get("profile", "full")).strip() or "full"
                try:
                    item["seed"] = int(str(item["seed"]))
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid seed in {path}: {row}") from exc
                rows.append(item)
    return rows


def descriptive(values: Iterable[float]) -> dict[str, float | int]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "std_population": float("nan"), "std_sample": float("nan")}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std_population": float(np.std(arr, ddof=0)),
        "std_sample": float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan"),
    }


def exact_paired_sign_pvalue(differences: Iterable[float]) -> float:
    """Two-sided exact sign p-value, excluding zero differences."""
    values = np.asarray(list(differences), dtype=np.float64)
    values = values[np.isfinite(values) & (np.abs(values) > 1e-12)]
    n = int(values.size)
    if n == 0:
        return float("nan")
    positives = int(np.sum(values > 0))
    tail = sum(math.comb(n, k) for k in range(0, min(positives, n - positives) + 1)) / (2**n)
    return float(min(1.0, 2.0 * tail))


def bootstrap_mean_ci(values: Iterable[float], *, seed: int = 20260902, samples: int = 20000) -> tuple[float, float]:
    """Deterministic percentile bootstrap CI for a mean effect."""
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, arr.size, size=(samples, arr.size))
    means = np.mean(arr[sample_idx], axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _metric_value(row: dict[str, object], metric: str) -> float:
    # Recovery CSVs use recovery_distance_m; evaluation CSVs use the same
    # metric names as the paper-facing per-seed outputs.
    return _float(row.get(metric))


def _group_rows(rows: list[dict[str, object]]) -> dict[tuple[str, str], list[dict[str, object]]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["traversal"]), str(row["profile"])), []).append(row)
    return groups


def build_summary(rows: list[dict[str, object]], metrics: Iterable[str] = DEFAULT_METRICS) -> dict[str, object]:
    metrics = tuple(metrics)
    seed_set = sorted({int(row["seed"]) for row in rows})
    summary: dict[str, object] = {
        "schema_version": 1,
        "unit_of_analysis": "seed within traversal/profile",
        "registered_seeds": list(EXPECTED_SEEDS),
        "observed_seeds": seed_set,
        "estimands": list(metrics),
        "n3_results_are_descriptive": True,
        "paired_comparisons": [],
        "fixed_reference_deltas": [],
    }

    for (traversal, profile), group in sorted(_group_rows(rows).items()):
        by_method: dict[str, list[dict[str, object]]] = {}
        for row in group:
            by_method.setdefault(str(row["method"]), []).append(row)
        methods = sorted(by_method)
        for method_a, method_b in itertools.combinations(methods, 2):
            a_rows = {int(row["seed"]): row for row in by_method[method_a]}
            b_rows = {int(row["seed"]): row for row in by_method[method_b]}
            common = sorted(set(a_rows) & set(b_rows))
            if not common:
                continue
            dedicated = any(
                str(a_rows[s].get("replicate_kind", "")) == "dedicated_run"
                or str(b_rows[s].get("replicate_kind", "")) == "dedicated_run"
                for s in common
            )
            if dedicated:
                for metric in metrics:
                    a_values = [_metric_value(a_rows[s], metric) for s in common]
                    b_values = [_metric_value(b_rows[s], metric) for s in common]
                    a_desc = descriptive(a_values)
                    b_desc = descriptive(b_values)
                    if a_desc["n"] == 0 or b_desc["n"] == 0:
                        continue
                    ref_method, ref_desc, eval_method, eval_desc = (
                        (method_a, a_desc, method_b, b_desc)
                        if any(str(a_rows[s].get("replicate_kind", "")) == "dedicated_run" for s in common)
                        else (method_b, b_desc, method_a, a_desc)
                    )
                    summary["fixed_reference_deltas"].append({
                        "traversal": traversal,
                        "profile": profile,
                        "metric": metric,
                        "reference_method": ref_method,
                        "evaluation_method": eval_method,
                        "reference_mean": ref_desc["mean"],
                        "reference_std_population": ref_desc["std_population"],
                        "evaluation_mean": eval_desc["mean"],
                        "evaluation_std_population": eval_desc["std_population"],
                        "evaluation_minus_reference": float(eval_desc["mean"] - ref_desc["mean"]),
                        "n_reference": ref_desc["n"],
                        "n_evaluation": eval_desc["n"],
                        "inference": "descriptive_fixed_reference_no_p_value",
                    })
                continue

            record: dict[str, object] = {
                "traversal": traversal,
                "profile": profile,
                "method_a": method_a,
                "method_b": method_b,
                "seeds": common,
                "n": len(common),
                "inference": "exact_paired_sign_and_bootstrap_mean_ci",
                "metrics": {},
            }
            for metric in metrics:
                differences = np.asarray(
                    [_metric_value(b_rows[s], metric) - _metric_value(a_rows[s], metric) for s in common],
                    dtype=np.float64,
                )
                differences = differences[np.isfinite(differences)]
                if differences.size == 0:
                    continue
                ci_low, ci_high = bootstrap_mean_ci(differences)
                record["metrics"][metric] = {
                    "method_b_minus_method_a_mean": float(np.mean(differences)),
                    "difference_std_population": float(np.std(differences)),
                    "difference_ci95_low": ci_low,
                    "difference_ci95_high": ci_high,
                    "exact_paired_sign_p_two_sided": exact_paired_sign_pvalue(differences),
                    "n_pairs": int(differences.size),
                }
            if record["metrics"]:
                summary["paired_comparisons"].append(record)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="One or more per-seed CSV files.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path.")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    metrics = tuple(x.strip() for x in args.metrics.split(",") if x.strip())
    result = build_summary(read_rows(args.input), metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"[INFO] Wrote statistics summary to {args.output}")


if __name__ == "__main__":
    main()
