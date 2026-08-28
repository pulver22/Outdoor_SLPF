#!/usr/bin/env python3
"""Generate a shareable Markdown/JSON report for localisation follow-up work."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Mapping
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "results/localization_improvement_report"
DEFAULT_CURRENT_RH2 = BASE_DIR / "results/localization_improvement_full/alpha_huber_cap/summary.json"
DEFAULT_CURRENT_RH1 = BASE_DIR / "results/localization_improvement_full/alpha_huber_cap_rh1/summary.json"
DEFAULT_DIAGNOSTICS = DEFAULT_OUTPUT_DIR / "diagnostics/diagnostic_metrics.csv"
DEFAULT_FOLLOWUP = BASE_DIR / "results/localization_improvement_followup/followup_metrics_per_seed.csv"
DEFAULT_FOLLOWUP_AGGREGATE = BASE_DIR / "results/localization_improvement_followup/followup_metrics_aggregate.csv"
DEFAULT_GTSAM = BASE_DIR / "results/localization_improvement_gtsam_smoke/gtsam_seed22_rh2/metric_summary.json"
ROW_IDENTITY_CANDIDATES = {
    "row_mixture",
    "delayed_correction",
    "row_mixture_delayed",
    "row_mixture_delayed_gnss_gating",
}


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, object]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _git_value(args: list[str], default: str = "unknown") -> str:
    try:
        return subprocess.check_output(args, cwd=BASE_DIR, text=True).strip()
    except Exception:
        return default


def _safe_float(value: object, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _first_float(row: dict[str, object], *keys: str) -> float | None:
    for key in keys:
        value = _safe_float(row.get(key))
        if value is not None:
            return value
    return None


def row_metric_view(row: dict[str, object]) -> dict[str, float | None]:
    """Expose publication-facing row metrics with explicit region semantics."""
    return {
        "inrow_wrong_sec": _first_float(
            row,
            "inrow_wrong_row_duration_sec_mean",
            "inrow_wrong_row_duration_sec",
        ),
        "headland_wrong_sec": _first_float(
            row,
            "headland_wrong_row_duration_sec_mean",
            "headland_wrong_row_duration_sec",
        ),
        "inrow_row_correct": _first_float(
            row,
            "inrow_row_correct_fraction_mean",
            "inrow_row_correct_fraction",
        ),
        "inrow_cross_track": _first_float(
            row,
            "inrow_cross_track_mean_mean",
            "inrow_cross_track_mean",
        ),
        "headland_cross_track": _first_float(
            row,
            "headland_cross_track_mean_mean",
            "headland_cross_track_mean",
        ),
        "headland_recovery_m": _first_float(
            row,
            "headland_mean_recovery_distance_m_mean",
            "headland_mean_recovery_distance_m",
        ),
    }


def _format_float(value: object) -> str:
    number = _safe_float(value)
    if number is None:
        return "n/a"
    return f"{number:.3f}"


def _summary_rows(current_rh2: dict, current_rh1: dict) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for traversal, summary in [("rh_run2", current_rh2), ("rh_run1", current_rh1)]:
        rows.append(
            {
                "traversal": traversal,
                "mean_aligned_ape_m": summary.get("ape_align_rmse_mean", summary.get("ape_align_rmse", "")),
                "seed11_aligned_ape_m": (summary.get("per_seed_ape_align_rmse") or {}).get("11", ""),
                "seed22_aligned_ape_m": (summary.get("per_seed_ape_align_rmse") or {}).get("22", ""),
                "seed33_aligned_ape_m": (summary.get("per_seed_ape_align_rmse") or {}).get("33", ""),
            }
        )
    return rows


def _markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    if not rows:
        return "_No rows available._"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join([header, sep, *body])


def _compact_metric_rows(rows: list[dict[str, object]], limit: int = 40) -> list[dict[str, object]]:
    compact = []
    for row in rows[:limit]:
        compact.append(
            {
                "stage": row.get("stage", ""),
                "traversal": row.get("traversal", ""),
                "candidate_id": row.get("candidate_id", row.get("variant", "")),
                "seed": row.get("seed", ""),
                "ape_align_rmse": _format_float(row.get("ape_align_rmse")),
                "inrow_cross_track_mean": _format_float(row_metric_view(row)["inrow_cross_track"]),
                "inrow_row_correct_fraction": _format_float(row_metric_view(row)["inrow_row_correct"]),
                "inrow_wrong_row_duration_sec": _format_float(row_metric_view(row)["inrow_wrong_sec"]),
                "all_frame_wrong_row_duration_sec_diagnostic": _format_float(
                    row.get("wrong_row_duration_sec")
                ),
            }
        )
    return compact


def _compact_aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    compact = []
    for row in rows:
        compact.append(
            {
                "stage": row.get("stage", ""),
                "traversal": row.get("traversal", ""),
                "candidate_id": row.get("candidate_id", ""),
                "seed_count": row.get("seed_count", ""),
                "ape_align_rmse_mean": _format_float(row.get("ape_align_rmse_mean")),
                "inrow_cross_track_mean": _format_float(row_metric_view(row)["inrow_cross_track"]),
                "inrow_row_correct_fraction": _format_float(row_metric_view(row)["inrow_row_correct"]),
                "inrow_wrong_row_duration_sec": _format_float(row_metric_view(row)["inrow_wrong_sec"]),
                "all_frame_wrong_row_duration_sec_diagnostic": _format_float(
                    row.get("wrong_row_duration_sec_mean")
                ),
            }
        )
    return compact


def _compact_row_identity_rows(rows: list[dict[str, object]], limit: int = 40) -> list[dict[str, object]]:
    compact = []
    for row in rows[:limit]:
        compact.append(
            {
                "traversal": row.get("traversal", ""),
                "variant": row.get("variant", row.get("candidate_id", "")),
                "seed": row.get("seed", ""),
                "ape_align_rmse": _format_float(row.get("ape_align_rmse")),
                "row_switch_count": row.get("row_switch_count", ""),
                "row_entropy_mean": _format_float(row.get("row_hypothesis_entropy_mean")),
                "row_gap_median": _format_float(row.get("row_hypothesis_gap_median")),
                "gnss_row_switch_corr": _format_float(row.get("gnss_row_switch_corr")),
                "gnss_gate_scale_median": _format_float(row.get("gnss_row_gate_scale_median")),
            }
        )
    return compact


def _controlled_ablation_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    wanted = {
        "baseline_alpha_huber3_cap50",
        "row_mixture",
        "delayed_correction",
        "row_mixture_delayed",
        "row_mixture_delayed_gnss_gating",
    }
    compact = []
    for row in rows:
        candidate_id = str(row.get("candidate_id", ""))
        if candidate_id not in wanted:
            continue
        compact.append(
            {
                "stage": row.get("stage", ""),
                "traversal": row.get("traversal", ""),
                "candidate_id": candidate_id,
                "seed_count": row.get("seed_count", ""),
                "ape_align_rmse_mean": _format_float(row.get("ape_align_rmse_mean")),
                "inrow_cross_track_mean": _format_float(row_metric_view(row)["inrow_cross_track"]),
                "inrow_wrong_row_duration_sec": _format_float(row_metric_view(row)["inrow_wrong_sec"]),
            }
        )
    return compact


def _acceptance_summary(
    followup_rows: list[dict[str, object]],
    aggregate_rows: list[dict[str, object]],
) -> dict[str, object]:
    aggregate_by_key = {
        (str(row.get("traversal", "")), str(row.get("candidate_id", ""))): row
        for row in aggregate_rows
        if row.get("stage") == "full"
    }
    seed_by_key = {
        (str(row.get("traversal", "")), str(row.get("candidate_id", "")), str(row.get("seed", ""))): row
        for row in followup_rows
        if row.get("stage") == "full"
    }
    baseline_rh1 = aggregate_by_key.get(("rh_run1", "baseline_alpha_huber3_cap50"), {})
    baseline_rh2 = aggregate_by_key.get(("rh_run2", "baseline_alpha_huber3_cap50"), {})
    baseline_rh1_ape = _safe_float(baseline_rh1.get("ape_align_rmse_mean"))
    baseline_rh2_ape = _safe_float(baseline_rh2.get("ape_align_rmse_mean"))
    baseline_rh1_view = row_metric_view(baseline_rh1)
    baseline_rh2_view = row_metric_view(baseline_rh2)
    baseline_rh1_wrong = baseline_rh1_view["inrow_wrong_sec"]
    baseline_rh2_wrong = baseline_rh2_view["inrow_wrong_sec"]
    baseline_rh1_ct = baseline_rh1_view["inrow_cross_track"]
    baseline_rh2_ct = baseline_rh2_view["inrow_cross_track"]

    checks: list[dict[str, object]] = []
    for candidate_id in sorted(ROW_IDENTITY_CANDIDATES):
        rh1 = aggregate_by_key.get(("rh_run1", candidate_id), {})
        rh2 = aggregate_by_key.get(("rh_run2", candidate_id), {})
        seed11 = seed_by_key.get(("rh_run1", candidate_id, "11"), {})
        rh1_seed11_ape = _safe_float(seed11.get("ape_align_rmse"))
        rh1_ape = _safe_float(rh1.get("ape_align_rmse_mean"))
        rh2_ape = _safe_float(rh2.get("ape_align_rmse_mean"))
        rh1_view = row_metric_view(rh1)
        rh2_view = row_metric_view(rh2)
        rh1_wrong = rh1_view["inrow_wrong_sec"]
        rh2_wrong = rh2_view["inrow_wrong_sec"]
        rh1_ct = rh1_view["inrow_cross_track"]
        rh2_ct = rh2_view["inrow_cross_track"]
        check = {
            "candidate_id": candidate_id,
            "rh_run1_seed11_lt_1p2": rh1_seed11_ape is not None and rh1_seed11_ape < 1.2,
            "rh_run1_mean_lt_1p0": rh1_ape is not None and rh1_ape < 1.0,
            "rh_run2_no_ape_regression": (
                rh2_ape is not None and baseline_rh2_ape is not None and rh2_ape <= baseline_rh2_ape
            ),
            "rh_run1_wrong_row_reduced": (
                rh1_wrong is not None and baseline_rh1_wrong is not None and rh1_wrong < baseline_rh1_wrong
            ),
            "rh_run2_wrong_row_reduced": (
                rh2_wrong is not None and baseline_rh2_wrong is not None and rh2_wrong < baseline_rh2_wrong
            ),
            "rh_run1_cross_track_reduced": (
                rh1_ct is not None and baseline_rh1_ct is not None and rh1_ct < baseline_rh1_ct
            ),
            "rh_run2_cross_track_reduced": (
                rh2_ct is not None and baseline_rh2_ct is not None and rh2_ct < baseline_rh2_ct
            ),
            "rh_run1_seed11_ape": rh1_seed11_ape,
            "rh_run1_mean_ape": rh1_ape,
            "rh_run2_mean_ape": rh2_ape,
            "rh_run1_wrong_row_sec": rh1_wrong,
            "rh_run2_wrong_row_sec": rh2_wrong,
            "rh_run1_cross_track_mean": rh1_ct,
            "rh_run2_cross_track_mean": rh2_ct,
        }
        check["accepted"] = all(
            bool(check[key])
            for key in [
                "rh_run1_seed11_lt_1p2",
                "rh_run1_mean_lt_1p0",
                "rh_run2_no_ape_regression",
                "rh_run1_wrong_row_reduced",
                "rh_run2_wrong_row_reduced",
                "rh_run1_cross_track_reduced",
                "rh_run2_cross_track_reduced",
            ]
        )
        checks.append(check)

    accepted = [check["candidate_id"] for check in checks if check["accepted"]]
    ape_improved = [
        check["candidate_id"]
        for check in checks
        if check["rh_run1_mean_ape"] is not None
        and baseline_rh1_ape is not None
        and check["rh_run1_mean_ape"] < baseline_rh1_ape
    ]
    outcome = "success" if accepted else ("partial" if ape_improved else "failure")
    return {
        "baseline": {
            "rh_run1_ape": baseline_rh1_ape,
            "rh_run2_ape": baseline_rh2_ape,
            "rh_run1_wrong_row_sec": baseline_rh1_wrong,
            "rh_run2_wrong_row_sec": baseline_rh2_wrong,
            "rh_run1_cross_track_mean": baseline_rh1_ct,
            "rh_run2_cross_track_mean": baseline_rh2_ct,
        },
        "accepted_candidate": accepted[0] if accepted else "",
        "outcome": outcome,
        "checks": checks,
    }


def _compact_acceptance_rows(acceptance: dict[str, object]) -> list[dict[str, object]]:
    compact = []
    for row in acceptance.get("checks", []):
        compact.append(
            {
                "candidate_id": row.get("candidate_id", ""),
                "accepted": row.get("accepted", False),
                "rh1_seed11": _format_float(row.get("rh_run1_seed11_ape")),
                "rh1_mean": _format_float(row.get("rh_run1_mean_ape")),
                "rh2_mean": _format_float(row.get("rh_run2_mean_ape")),
                "rh1_wrong_s": _format_float(row.get("rh_run1_wrong_row_sec")),
                "rh2_wrong_s": _format_float(row.get("rh_run2_wrong_row_sec")),
                "rh1_ct": _format_float(row.get("rh_run1_cross_track_mean")),
                "rh2_ct": _format_float(row.get("rh_run2_cross_track_mean")),
            }
        )
    return compact


def _best_full_candidate(
    followup_rows: list[dict[str, object]],
    aggregate_rows: list[dict[str, object]],
) -> dict[str, object] | None:
    grouped: dict[str, dict[str, object]] = {}
    for row in aggregate_rows:
        if row.get("stage") != "full":
            continue
        candidate_id = str(row.get("candidate_id", ""))
        mean_ape = _safe_float(row.get("ape_align_rmse_mean"))
        seed_count = int(_safe_float(row.get("seed_count"), 0) or 0)
        if not candidate_id or mean_ape is None or seed_count <= 0:
            continue
        entry = grouped.setdefault(
            candidate_id,
            {
                "candidate_id": candidate_id,
                "weighted_ape_sum": 0.0,
                "seed_count": 0,
                "traversal_means": {},
                "rh_run1_seed11_ape": None,
                "rh_run1_mean_ape": None,
            },
        )
        entry["weighted_ape_sum"] = float(entry["weighted_ape_sum"]) + mean_ape * seed_count
        entry["seed_count"] = int(entry["seed_count"]) + seed_count
        entry["traversal_means"][row.get("traversal", "")] = mean_ape
        if row.get("traversal") == "rh_run1":
            entry["rh_run1_mean_ape"] = mean_ape

    for row in followup_rows:
        if row.get("stage") == "full" and row.get("traversal") == "rh_run1" and str(row.get("seed")) == "11":
            candidate_id = str(row.get("candidate_id", ""))
            if candidate_id in grouped:
                grouped[candidate_id]["rh_run1_seed11_ape"] = _safe_float(row.get("ape_align_rmse"))

    candidates = []
    for entry in grouped.values():
        if int(entry["seed_count"]) > 0:
            entry["overall_mean_ape"] = float(entry["weighted_ape_sum"]) / int(entry["seed_count"])
            candidates.append(entry)
    return min(candidates, key=lambda entry: float(entry["overall_mean_ape"])) if candidates else None


def _outcome_branch(best_full: dict[str, object] | None) -> str:
    if best_full is None:
        return "pending"
    seed11 = _safe_float(best_full.get("rh_run1_seed11_ape"))
    rh1_mean = _safe_float(best_full.get("rh_run1_mean_ape"))
    if seed11 is not None and seed11 > 1.5:
        return "failure"
    if seed11 is not None and rh1_mean is not None and seed11 < 1.2 and rh1_mean < 1.0:
        return "success"
    return "partial"


def _interpretation(
    current_rh2: dict,
    current_rh1: dict,
    followup_rows: list[dict[str, object]],
    aggregate_rows: list[dict[str, object]],
    gtsam: dict,
) -> list[str]:
    notes: list[str] = []
    rh2_mean = _safe_float(current_rh2.get("ape_align_rmse_mean"))
    rh1_mean = _safe_float(current_rh1.get("ape_align_rmse_mean"))
    rh1_seed11 = _safe_float((current_rh1.get("per_seed_ape_align_rmse") or {}).get("11"))
    if rh2_mean is not None:
        notes.append(f"Current accepted rh_run2 configuration mean aligned APE is {rh2_mean:.3f} m.")
    if rh1_mean is not None and rh1_seed11 is not None:
        notes.append(f"Current rh_run1 mean aligned APE is {rh1_mean:.3f} m, with seed 11 at {rh1_seed11:.3f} m.")
    if gtsam:
        gtsam_ape = _safe_float(gtsam.get("ape_align_rmse"))
        if gtsam_ape is not None:
            notes.append(f"GTSAM smoke completed at {gtsam_ape:.3f} m aligned APE; it is only promoted if it beats alpha in the screen.")
    if followup_rows:
        best = min(followup_rows, key=lambda row: _safe_float(row.get("ape_align_rmse"), float("inf")) or float("inf"))
        best_ape = _safe_float(best.get("ape_align_rmse"))
        if best_ape is not None:
            notes.append(f"Best generated follow-up row is {best.get('candidate_id')} on {best.get('traversal')} seed {best.get('seed')} at {best_ape:.3f} m aligned APE.")
        best_full = _best_full_candidate(followup_rows, aggregate_rows)
        if best_full is not None:
            traversal_means = best_full.get("traversal_means", {})
            rh1_mean = _safe_float(traversal_means.get("rh_run1"))
            rh2_mean = _safe_float(traversal_means.get("rh_run2"))
            notes.append(
                "Best promoted full configuration by weighted mean is "
                f"{best_full['candidate_id']} at {float(best_full['overall_mean_ape']):.3f} m overall "
                f"(rh_run1 {_format_float(rh1_mean)} m, rh_run2 {_format_float(rh2_mean)} m)."
            )
            notes.append(f"Outcome branch: {_outcome_branch(best_full)}.")
    else:
        notes.append("Follow-up runner output was not present when this report was generated.")
    notes.append(
        "Scientific hypothesis under test: multi-hypothesis row reasoning should reduce worst-case wrong-row recovery errors beyond scalar robust-loss tuning."
    )
    return notes


def _next_experiments() -> list[str]:
    return [
        "Failure branch: if rh_run1 seed 11 remains above 1.5 m, diagnose map/GNSS frame consistency, early particle collapse, headland-only error, and wrong-row duration before changing paper claims.",
        "Success branch: if rh_run1 seed 11 drops below 1.2 m and full rh_run1 mean is below 1.0 m, freeze one config and regenerate full ablation, runtime, trajectory plots, and manuscript tables.",
        "Partial branch: if rh_run1 improves but remains 1.0-1.5 m, keep rh_run2 accepted, report rh_run1 as a robustness limitation, and run one additional seed-stability check.",
    ]


def generate_report(
    output_markdown: Path,
    output_json: Path,
    current_rh2_summary: Path,
    current_rh1_summary: Path,
    diagnostics_csv: Path,
    followup_csv: Path,
    branch: str,
    commit: str,
    commands: list[str],
    gtsam_summary: Path = DEFAULT_GTSAM,
    followup_aggregate_csv: Path | None = None,
) -> dict:
    current_rh2 = _read_json(current_rh2_summary)
    current_rh1 = _read_json(current_rh1_summary)
    diagnostics_rows = _read_csv(diagnostics_csv)
    followup_rows = _read_csv(followup_csv)
    aggregate_path = followup_aggregate_csv or followup_csv.with_name("followup_metrics_aggregate.csv")
    aggregate_rows = _read_csv(aggregate_path)
    gtsam = _read_json(gtsam_summary)
    summary_rows = _summary_rows(current_rh2, current_rh1)
    interpretation = _interpretation(current_rh2, current_rh1, followup_rows, aggregate_rows, gtsam)
    next_experiments = _next_experiments()
    best_full = _best_full_candidate(followup_rows, aggregate_rows)
    row_identity_rows = _compact_row_identity_rows(diagnostics_rows)
    controlled_ablation_rows = _controlled_ablation_rows(aggregate_rows)
    acceptance = _acceptance_summary(followup_rows, aggregate_rows)
    if acceptance["accepted_candidate"]:
        interpretation.append(f"Full acceptance passed for {acceptance['accepted_candidate']}.")
    else:
        interpretation.append(
            "No row-identity candidate passed the full acceptance gate; the current result is partial because APE improvements trade off against rh_run2 regression or wrong-row duration."
        )

    data = {
        "git": {"branch": branch, "commit": commit},
        "current_results": {
            "rh_run2": current_rh2,
            "rh_run1": current_rh1,
            "summary_rows": summary_rows,
        },
        "diagnostics": {
            "csv": str(diagnostics_csv),
            "row_count": len(diagnostics_rows),
            "rows": diagnostics_rows,
            "row_identity_rows": row_identity_rows,
        },
        "followup": {
            "csv": str(followup_csv),
            "row_count": len(followup_rows),
            "rows": followup_rows,
        },
        "followup_aggregate": {
            "csv": str(aggregate_path),
            "row_count": len(aggregate_rows),
            "rows": aggregate_rows,
            "controlled_row_identity_ablation": controlled_ablation_rows,
        },
        "best_full_candidate": best_full,
        "acceptance": acceptance,
        "outcome_branch": acceptance["outcome"],
        "gtsam_smoke": gtsam,
        "commands": commands,
        "artifacts": {
            "report_markdown": str(output_markdown),
            "report_json": str(output_json),
            "diagnostics_csv": str(diagnostics_csv),
            "followup_csv": str(followup_csv),
            "followup_aggregate_csv": str(aggregate_path),
        },
        "interpretation_notes": interpretation,
        "next_experiments": next_experiments,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    current_table_rows = [
        {
            "traversal": row["traversal"],
            "mean_aligned_ape_m": _format_float(row["mean_aligned_ape_m"]),
            "seed11": _format_float(row["seed11_aligned_ape_m"]),
            "seed22": _format_float(row["seed22_aligned_ape_m"]),
            "seed33": _format_float(row["seed33_aligned_ape_m"]),
        }
        for row in summary_rows
    ]
    diagnostic_table = _compact_metric_rows(diagnostics_rows)
    row_identity_table = _compact_row_identity_rows(diagnostics_rows)
    followup_table = _compact_metric_rows(followup_rows)
    followup_aggregate_table = _compact_aggregate_rows(aggregate_rows)
    controlled_ablation_table = _controlled_ablation_rows(aggregate_rows)
    acceptance_table = _compact_acceptance_rows(acceptance)
    command_text = "\n".join(f"- `{command}`" for command in commands) if commands else "_No commands recorded._"
    interpretation_text = "\n".join(f"- {note}" for note in interpretation)
    next_text = "\n".join(f"- {note}" for note in next_experiments)

    markdown = f"""# Outdoor SLPF Localisation Improvement Report

## Summary
Branch `{branch}` at commit `{commit}`. This report captures the current localisation improvement state, the diagnostic artifacts generated for the rh_run1 seed-11 failure, and the follow-up experiment decision tree for external review.

## Changes Implemented
- Added diagnostics that compare traversal/seed metrics with frame-level GNSS innovation, ESS, max weight, semantic-hit, smoother-status, row-hypothesis, row-switch, and GNSS-row-gating signals.
- Added opt-in SPF switches for top-k row-mixture likelihood, lightweight delayed row correction, and adaptive GNSS row gating while freezing `alpha_huber3_cap50` as the reference baseline.
- Added a controlled row-identity ablation runner matrix for baseline, row mixture, delayed correction, row mixture plus delayed correction, and row mixture plus delayed correction plus GNSS gating.
- Added this Markdown plus JSON report for sharing with ChatGPT or parsing programmatically.

## Current Metric Summary
{_markdown_table(current_table_rows, ["traversal", "mean_aligned_ape_m", "seed11", "seed22", "seed33"])}

## Diagnostics
{_markdown_table(diagnostic_table, ["traversal", "candidate_id", "seed", "ape_align_rmse", "inrow_cross_track_mean", "inrow_row_correct_fraction", "inrow_wrong_row_duration_sec", "all_frame_wrong_row_duration_sec_diagnostic"])}

## Row-Identity Diagnostics
{_markdown_table(row_identity_table, ["traversal", "variant", "seed", "ape_align_rmse", "row_switch_count", "row_entropy_mean", "row_gap_median", "gnss_row_switch_corr", "gnss_gate_scale_median"])}

## Follow-Up Experiments
{_markdown_table(followup_table, ["stage", "traversal", "candidate_id", "seed", "ape_align_rmse", "inrow_cross_track_mean", "inrow_row_correct_fraction", "inrow_wrong_row_duration_sec", "all_frame_wrong_row_duration_sec_diagnostic"])}

### Follow-Up Aggregates
{_markdown_table(followup_aggregate_table, ["stage", "traversal", "candidate_id", "seed_count", "ape_align_rmse_mean", "inrow_cross_track_mean", "inrow_row_correct_fraction", "inrow_wrong_row_duration_sec", "all_frame_wrong_row_duration_sec_diagnostic"])}

## Controlled Row-Identity Ablation
{_markdown_table(controlled_ablation_table, ["stage", "traversal", "candidate_id", "seed_count", "ape_align_rmse_mean", "inrow_cross_track_mean", "inrow_wrong_row_duration_sec"])}

## Acceptance Check
Outcome: `{acceptance["outcome"]}`. Accepted candidate: `{acceptance["accepted_candidate"] or "none"}`.

Row-identity acceptance uses in-row frames. Headland frames are evaluated with cross-track and transition-recovery metrics because nearest-row identity is ambiguous outside a corridor. Total wrong-row duration is retained only as an all-frame diagnostic.

{_markdown_table(acceptance_table, ["candidate_id", "accepted", "rh1_seed11", "rh1_mean", "rh2_mean", "rh1_wrong_s", "rh2_wrong_s", "rh1_ct", "rh2_ct"])}

## Commands
{command_text}

## Artifact Paths
- Markdown report: `{output_markdown}`
- JSON data: `{output_json}`
- Diagnostics CSV: `{diagnostics_csv}`
- Follow-up CSV: `{followup_csv}`
- Follow-up aggregate CSV: `{aggregate_path}`

## Failures
- rh_run1 seed 11 is the current generalisation concern unless follow-up results show it below the configured thresholds.
- GTSAM remains smoke-only unless its screened aligned APE beats the alpha baseline.

## Successes
- rh_run2 accepted alpha+Huber+cap result remains the current positive result.
- The workflow now produces repeatable diagnostics, candidate commands, metrics, and report artifacts.

## Interpretation Notes
{interpretation_text}

## Next Experiments
{next_text}
"""
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(markdown, encoding="utf-8")
    return data


def _manifest_rows(manifest: Mapping[str, object]) -> list[dict[str, object]]:
    for key in ("baseline_rows", "rows"):
        value = manifest.get(key)
        if isinstance(value, list):
            return [dict(row) for row in value if isinstance(row, Mapping)]
    baseline = manifest.get("baseline")
    if isinstance(baseline, Mapping):
        rows = []
        for traversal in ("rh_run1", "rh_run2"):
            value = baseline.get(f"{traversal}_ape")
            if value is None:
                value = baseline.get(f"{traversal}_ape_align_rmse")
            if value is not None:
                rows.append({"traversal": traversal, "ape_align_rmse": value})
        return rows
    outputs = manifest.get("outputs")
    if isinstance(outputs, Mapping):
        aggregate = outputs.get("aggregate_csv")
        if aggregate:
            return _read_csv(Path(str(aggregate)))
    return []


def generate_icra_report(manifest_path: Path, output_dir: Path) -> dict[str, object]:
    """Generate the shareable report directly from a canonical evidence manifest."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Evidence manifest must contain an object: {manifest_path}")
    rows = _manifest_rows(manifest)
    commit = str((manifest.get("git") or {}).get("commit", "unknown"))
    output_dir.mkdir(parents=True, exist_ok=True)
    report_data: dict[str, object] = {
        "git": dict(manifest.get("git") or {}),
        "configuration": manifest.get("configuration"),
        "seeds": manifest.get("seeds", [11, 22, 33]),
        "traversals": manifest.get("traversals", ["rh_run1", "rh_run2"]),
        "sources": manifest.get("sources", []),
        "rows": rows,
        "comparison": manifest.get("comparison", {}),
        "claim_checks": manifest.get("claim_checks", {}),
        "manifest": str(manifest_path),
    }
    table_rows = []
    for row in rows:
        table_rows.append(
            {
                "traversal": row.get("traversal", ""),
                "method": row.get("method", row.get("candidate_id", "")),
                "ape_align_rmse": _format_float(row.get("ape_align_rmse", row.get("ape_align_rmse_mean"))),
                "ape_raw_rmse": _format_float(row.get("ape_raw_rmse", row.get("ape_raw_rmse_mean"))),
                "inrow_cross_track": _format_float(row_metric_view(row).get("inrow_cross_track")),
                "inrow_wrong_sec": _format_float(row_metric_view(row).get("inrow_wrong_sec")),
            }
        )
    markdown = f"""# Evidence-First ICRA Submission Report

## Provenance

Canonical evidence manifest: `{manifest_path}`. Git commit: `{commit}`. Configuration: `{manifest.get('configuration', 'n/a')}`.

## Baseline Evidence

{_markdown_table(table_rows, ['traversal', 'method', 'ape_raw_rmse', 'ape_align_rmse', 'inrow_cross_track', 'inrow_wrong_sec'])}

Row-identity acceptance uses in-row frames. Headland frames are evaluated with cross-track and transition-recovery metrics because nearest-row identity is ambiguous outside a corridor. Total wrong-row duration is retained only as an all-frame diagnostic.

## Claim Boundary

Detector accuracy is not claimed without a supplied SemanticBLT validation YAML. GTSAM and row-mixture/delayed-correction variants are not promoted as paper-facing methods.

## Source Hashes

{_markdown_table([{'path': item.get('path', ''), 'sha256': item.get('sha256', '')} for item in report_data['sources'] if isinstance(item, Mapping)], ['path', 'sha256'])}
"""
    report_data["table_rows"] = table_rows
    output_json = output_dir / "report_data.json"
    output_markdown = output_dir / "report.md"
    output_json.write_text(json.dumps(report_data, indent=2, sort_keys=True), encoding="utf-8")
    output_markdown.write_text(markdown, encoding="utf-8")
    return report_data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--evidence-manifest", type=Path, default=BASE_DIR / "results/icra_submission/evidence/manifest.json")
    parser.add_argument("--allow-historical-commit", action="store_true")
    parser.add_argument("--current-rh2-summary", type=Path, default=DEFAULT_CURRENT_RH2)
    parser.add_argument("--current-rh1-summary", type=Path, default=DEFAULT_CURRENT_RH1)
    parser.add_argument("--diagnostics-csv", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument("--followup-csv", type=Path, default=DEFAULT_FOLLOWUP)
    parser.add_argument("--followup-aggregate-csv", type=Path, default=DEFAULT_FOLLOWUP_AGGREGATE)
    parser.add_argument("--gtsam-summary", type=Path, default=DEFAULT_GTSAM)
    parser.add_argument("--command", action="append", default=[])
    args = parser.parse_args()

    legacy_args = any(
        value is not None
        for value in (args.current_rh2_summary, args.current_rh1_summary, args.diagnostics_csv, args.followup_csv, args.followup_aggregate_csv, args.gtsam_summary)
    )
    if args.evidence_manifest.exists():
        manifest = json.loads(args.evidence_manifest.read_text(encoding="utf-8"))
        manifest_commit = str((manifest.get("git") or {}).get("commit", ""))
        current_commit = _git_value(["git", "rev-parse", "HEAD"], default="")
        if manifest_commit and current_commit and manifest_commit != current_commit and not args.allow_historical_commit:
            raise RuntimeError(
                f"Evidence manifest commit {manifest_commit} differs from current HEAD {current_commit}; "
                "pass --allow-historical-commit to inspect historical evidence."
            )
        if legacy_args and any(option not in {None, DEFAULT_CURRENT_RH2, DEFAULT_CURRENT_RH1, DEFAULT_DIAGNOSTICS, DEFAULT_FOLLOWUP, DEFAULT_FOLLOWUP_AGGREGATE, DEFAULT_GTSAM} for option in (args.current_rh2_summary, args.current_rh1_summary, args.diagnostics_csv, args.followup_csv, args.followup_aggregate_csv, args.gtsam_summary)):
            raise RuntimeError("Legacy summary inputs are deprecated; use --evidence-manifest instead.")
        generate_icra_report(args.evidence_manifest, args.output_dir)
        print(f"Wrote ICRA report to {args.output_dir / 'report.md'}")
        return 0

    branch = _git_value(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    commit = _git_value(["git", "rev-parse", "--short", "HEAD"])
    commands = args.command or [
        "python3 scripts/analyze_localization_followup.py",
        "python3 scripts/run_localization_followup_experiments.py --stage screen --traversals rh_run1 --seeds 11",
        "python3 scripts/run_localization_followup_experiments.py --stage full --traversals rh_run1,rh_run2 --seeds 11,22,33 --promote-from results/localization_improvement_followup/followup_metrics_per_seed.csv",
        "python3 scripts/generate_localization_improvement_report.py",
    ]
    generate_report(
        output_markdown=args.output_dir / "report.md",
        output_json=args.output_dir / "report_data.json",
        current_rh2_summary=args.current_rh2_summary,
        current_rh1_summary=args.current_rh1_summary,
        diagnostics_csv=args.diagnostics_csv,
        followup_csv=args.followup_csv,
        branch=branch,
        commit=commit,
        commands=commands,
        gtsam_summary=args.gtsam_summary,
        followup_aggregate_csv=args.followup_aggregate_csv,
    )
    print(f"Wrote report to {args.output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
