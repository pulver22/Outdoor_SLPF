#!/usr/bin/env python3
"""Generate a shareable Markdown/JSON report for localisation follow-up work."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "results/localization_improvement_report"
DEFAULT_CURRENT_RH2 = BASE_DIR / "results/localization_improvement_full/alpha_huber_cap/summary.json"
DEFAULT_CURRENT_RH1 = BASE_DIR / "results/localization_improvement_full/alpha_huber_cap_rh1/summary.json"
DEFAULT_DIAGNOSTICS = DEFAULT_OUTPUT_DIR / "diagnostics/diagnostic_metrics.csv"
DEFAULT_FOLLOWUP = BASE_DIR / "results/localization_improvement_followup/followup_metrics_per_seed.csv"
DEFAULT_FOLLOWUP_AGGREGATE = BASE_DIR / "results/localization_improvement_followup/followup_metrics_aggregate.csv"
DEFAULT_GTSAM = BASE_DIR / "results/localization_improvement_gtsam_smoke/gtsam_seed22_rh2/metric_summary.json"


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


def _compact_metric_rows(rows: list[dict[str, object]], limit: int = 24) -> list[dict[str, object]]:
    compact = []
    for row in rows[:limit]:
        compact.append(
            {
                "stage": row.get("stage", ""),
                "traversal": row.get("traversal", ""),
                "candidate_id": row.get("candidate_id", row.get("variant", "")),
                "seed": row.get("seed", ""),
                "ape_align_rmse": _format_float(row.get("ape_align_rmse")),
                "cross_track_mean": _format_float(row.get("cross_track_mean")),
                "row_correct_fraction": _format_float(row.get("row_correct_fraction")),
                "wrong_row_duration_sec": _format_float(row.get("wrong_row_duration_sec")),
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
                "cross_track_mean_mean": _format_float(row.get("cross_track_mean_mean")),
                "row_correct_fraction_mean": _format_float(row.get("row_correct_fraction_mean")),
                "wrong_row_duration_sec_mean": _format_float(row.get("wrong_row_duration_sec_mean")),
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
        },
        "best_full_candidate": best_full,
        "outcome_branch": _outcome_branch(best_full),
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
    followup_table = _compact_metric_rows(followup_rows)
    followup_aggregate_table = _compact_aggregate_rows(aggregate_rows)
    command_text = "\n".join(f"- `{command}`" for command in commands) if commands else "_No commands recorded._"
    interpretation_text = "\n".join(f"- {note}" for note in interpretation)
    next_text = "\n".join(f"- {note}" for note in next_experiments)

    markdown = f"""# Outdoor SLPF Localisation Improvement Report

## Summary
Branch `{branch}` at commit `{commit}`. This report captures the current localisation improvement state, the diagnostic artifacts generated for the rh_run1 seed-11 failure, and the follow-up experiment decision tree for external review.

## Changes Implemented
- Added diagnostics that compare traversal/seed metrics with frame-level GNSS innovation, ESS, max weight, semantic-hit, and smoother-status signals.
- Added a fixed candidate matrix runner for the accepted alpha setting, robust-loss/cap variants, no-smoothing, fixed-lag, and optional GTSAM smoke.
- Added this Markdown plus JSON report for sharing with ChatGPT or parsing programmatically.

## Current Metric Summary
{_markdown_table(current_table_rows, ["traversal", "mean_aligned_ape_m", "seed11", "seed22", "seed33"])}

## Diagnostics
{_markdown_table(diagnostic_table, ["traversal", "candidate_id", "seed", "ape_align_rmse", "cross_track_mean", "row_correct_fraction", "wrong_row_duration_sec"])}

## Follow-Up Experiments
{_markdown_table(followup_table, ["stage", "traversal", "candidate_id", "seed", "ape_align_rmse", "cross_track_mean", "row_correct_fraction", "wrong_row_duration_sec"])}

### Follow-Up Aggregates
{_markdown_table(followup_aggregate_table, ["stage", "traversal", "candidate_id", "seed_count", "ape_align_rmse_mean", "cross_track_mean_mean", "row_correct_fraction_mean", "wrong_row_duration_sec_mean"])}

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--current-rh2-summary", type=Path, default=DEFAULT_CURRENT_RH2)
    parser.add_argument("--current-rh1-summary", type=Path, default=DEFAULT_CURRENT_RH1)
    parser.add_argument("--diagnostics-csv", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument("--followup-csv", type=Path, default=DEFAULT_FOLLOWUP)
    parser.add_argument("--followup-aggregate-csv", type=Path, default=DEFAULT_FOLLOWUP_AGGREGATE)
    parser.add_argument("--gtsam-summary", type=Path, default=DEFAULT_GTSAM)
    parser.add_argument("--command", action="append", default=[])
    args = parser.parse_args()

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
