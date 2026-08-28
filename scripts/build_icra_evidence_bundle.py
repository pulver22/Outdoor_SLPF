#!/usr/bin/env python3
"""Build a provenance-checked, paper-ready ICRA evidence bundle."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping


BASE_DIR = Path(__file__).resolve().parent.parent
EXPECTED_SEEDS = (11, 22, 33)
EXPECTED_TRAVERSALS = ("rh_run1", "rh_run2")
PRIMARY_METRICS = ("ape_raw_rmse", "ape_align_rmse", "rpe_2m_align_rmse", "rpe_5m_align_rmse", "rpe_10m_align_rmse")
TABLE_METRICS = (
    "ape_raw_rmse",
    "ape_align_rmse",
    "rpe_2m_align_rmse",
    "rpe_5m_align_rmse",
    "rpe_10m_align_rmse",
    "inrow_cross_track_mean",
    "inrow_row_correct_fraction",
    "inrow_wrong_row_duration_sec",
    "headland_cross_track_mean",
    "headland_mean_recovery_distance_m",
)


@dataclass(frozen=True)
class EvidenceInputs:
    """Input metric files and protocols used to construct one evidence bundle."""

    followup_metrics: Path
    localization_metrics: Path | None = None
    rtab_metrics: Path | None = None
    protocols: tuple[Path, ...] = field(default_factory=tuple)
    reference_metrics: Path | None = None

    def metric_paths(self) -> tuple[Path, ...]:
        return tuple(path for path in (self.followup_metrics, self.localization_metrics, self.rtab_metrics) if path)

    def source_paths(self) -> tuple[Path, ...]:
        return self.metric_paths() + self.protocols + ((self.reference_metrics,) if self.reference_metrics else ())


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field_name in row:
            if str(field_name).startswith("_"):
                continue
            if field_name not in fields:
                fields.append(field_name)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _normalise_row(row: Mapping[str, object], source: Path) -> dict[str, object]:
    method = str(row.get("method") or row.get("candidate_id") or row.get("variant") or "").strip()
    if not method:
        raise ValueError(f"{source}: row has no method, candidate_id, or variant")
    try:
        seed = int(str(row.get("seed", "")).strip())
    except ValueError as exc:
        raise ValueError(f"{source}: invalid seed {row.get('seed')!r}") from exc
    traversal = str(row.get("traversal", "")).strip()
    profile = str(row.get("profile") or row.get("stage") or "full").strip()
    normalised = dict(row)
    normalised.update(
        {
            "method": method,
            "method_key": str(row.get("method_key") or method.lower().replace(" ", "_")),
            "traversal": traversal,
            "seed": seed,
            "profile": profile,
            "source_file": str(source),
        }
    )
    return normalised


def _validate_rows(
    rows: list[dict[str, object]],
    source: Path,
    *,
    require_complete_matrix: bool,
) -> None:
    keys: dict[tuple[str, str, int, str], int] = defaultdict(int)
    commits: set[str] = set()
    for row in rows:
        key = (str(row["method"]), str(row["traversal"]), int(row["seed"]), str(row["profile"]))
        keys[key] += 1
        if keys[key] > 1:
            raise ValueError(f"{source}: duplicate key {key}")
        commit = str(row.get("commit") or row.get("git_commit") or "").strip()
        if commit:
            commits.add(commit)
        if _float(row.get("ape_align_rmse")) is None:
            raise ValueError(f"{source}: non-finite primary metric ape_align_rmse for {key}")
        for metric in PRIMARY_METRICS:
            if metric in row and str(row.get(metric, "")).strip().lower() not in {"", "nan", "none", "null"}:
                if _float(row.get(metric)) is None:
                    raise ValueError(f"{source}: non-finite primary metric {metric} for {key}")
    if len(commits) > 1:
        raise ValueError(f"{source}: mixed commits {sorted(commits)}")
    if not require_complete_matrix:
        return
    methods_profiles = {(str(row["method"]), str(row["profile"])) for row in rows}
    for method, profile in sorted(methods_profiles):
        present = {(str(row["traversal"]), int(row["seed"])) for row in rows if str(row["method"]) == method and str(row["profile"]) == profile}
        for traversal in EXPECTED_TRAVERSALS:
            missing = [seed for seed in EXPECTED_SEEDS if (traversal, seed) not in present]
            if missing:
                raise ValueError(f"{source}: missing seeds {','.join(map(str, missing))} for {method}/{profile}/{traversal}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=BASE_DIR, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _protocol_check(protocols: Iterable[Path]) -> None:
    for path in protocols:
        data = json.loads(path.read_text(encoding="utf-8"))
        commands = data.get("commands", [])
        if not commands:
            raise ValueError(f"{path}: protocol has no commands")
        missing = [command for command in commands if "--config-yaml" not in str(command)]
        if missing:
            raise ValueError(f"{path}: protocol commands omit the frozen --config-yaml option")


def format_mean_std(values: Iterable[object], digits: int = 2) -> str:
    finite = [number for value in values if (number := _float(value)) is not None]
    if not finite:
        return "n/a"
    mean = sum(finite) / len(finite)
    variance = sum((number - mean) ** 2 for number in finite) / len(finite)
    return f"{mean:.{digits}f} \\pm {math.sqrt(variance):.{digits}f}"


def _group_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["method"]), str(row["traversal"]), str(row["profile"]))].append(row)
    result: list[dict[str, object]] = []
    for (method, traversal, profile), group in sorted(groups.items()):
        aggregate: dict[str, object] = {"method": method, "traversal": traversal, "profile": profile, "seed_count": len(group)}
        for metric in TABLE_METRICS:
            values = [row.get(metric) for row in group]
            finite = [value for value in values if _float(value) is not None]
            if finite:
                aggregate[f"{metric}_mean"] = sum(float(value) for value in finite) / len(finite)
                aggregate[f"{metric}_std"] = math.sqrt(
                    sum((float(value) - float(aggregate[f"{metric}_mean"])) ** 2 for value in finite) / len(finite)
                )
        aggregate["_values"] = {metric: [row.get(metric) for row in group] for metric in TABLE_METRICS}
        result.append(aggregate)
    return result


def _tex_table(path: Path, rows: list[dict[str, object]], columns: tuple[tuple[str, str], ...]) -> None:
    lines = ["% Generated by scripts/build_icra_evidence_bundle.py. Do not edit manually.", "\\begin{tabular}{ll" + "r" * (len(columns) - 2) + "}", "\\toprule"]
    lines.append(" & ".join(label for _, label in columns) + r" \\ ")
    lines.append("\\midrule")
    if rows:
        for row in rows:
            cells = []
            for key, _ in columns:
                if key in {"method", "traversal"}:
                    cells.append(str(row.get(key, "")).replace("&", r"\&"))
                else:
                    values = row.get("_values", {}).get(key, [row.get(f"{key}_mean")])
                    cells.append(format_mean_std(values, digits=2))
            lines.append(" & ".join(cells) + r" \\ ")
    else:
        lines.append(r"\multicolumn{" + str(len(columns)) + r"}{c}{No validated rows available} \\ ")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_tables(rows: list[dict[str, object]], output_dir: Path) -> dict[str, str]:
    grouped = _group_rows(rows)
    main = [row for row in grouped if str(row["method"]).lower() in {"slpf", "baseline", "baseline_alpha_huber3_cap50"}]
    stress = [row for row in grouped if "gnss" in str(row["method"]).lower() or "amcl" in str(row["method"]).lower() or "rtab" in str(row["method"]).lower()]
    operational = [row for row in grouped if row in main]
    paths = {
        "main": output_dir / "paper_main_table.tex",
        "gnss_stress": output_dir / "paper_gnss_stress_table.tex",
        "operational": output_dir / "paper_operational_table.tex",
    }
    _tex_table(paths["main"], main, (("method", "Method"), ("traversal", "Traversal"), ("ape_raw_rmse", "Raw APE (m)"), ("ape_align_rmse", "Aligned APE (m)"), ("rpe_2m_align_rmse", "RPE 2m (m)")))
    _tex_table(paths["gnss_stress"], stress, (("method", "Method"), ("traversal", "Traversal"), ("ape_raw_rmse", "Raw APE (m)"), ("ape_align_rmse", "Aligned APE (m)"), ("rpe_5m_align_rmse", "RPE 5m (m)")))
    _tex_table(paths["operational"], operational, (("method", "Method"), ("traversal", "Traversal"), ("inrow_cross_track_mean", "In-row XT (m)"), ("inrow_row_correct_fraction", "In-row correct"), ("inrow_wrong_row_duration_sec", "In-row wrong (s)"), ("headland_cross_track_mean", "Headland XT (m)")))
    return {key: str(path) for key, path in paths.items()}


def _comparison(rows: list[dict[str, object]], reference_path: Path | None) -> dict[str, object]:
    if reference_path is None:
        return {"available": False, "deltas": []}
    reference = [_normalise_row(row, reference_path) for row in _read_csv(reference_path)]
    _validate_rows(reference, reference_path, require_complete_matrix=True)
    current_by_key = {(str(row["traversal"]), int(row["seed"]), str(row["profile"])): row for row in rows}
    deltas: list[dict[str, object]] = []
    for row in reference:
        key = (str(row["traversal"]), int(row["seed"]), str(row["profile"]))
        current = current_by_key.get(key)
        if current is None:
            continue
        old = _float(row.get("ape_align_rmse"))
        new = _float(current.get("ape_align_rmse"))
        if old is not None and new is not None:
            deltas.append({"traversal": key[0], "seed": key[1], "profile": key[2], "reference_ape_align_rmse": old, "baseline_ape_align_rmse": new, "delta_m": new - old})
    return {"available": True, "reference": str(reference_path), "deltas": deltas, "max_abs_ape_delta_m": max((abs(float(row["delta_m"])) for row in deltas), default=0.0)}


def build_bundle(inputs: EvidenceInputs, output_dir: Path) -> dict[str, object]:
    """Validate inputs and write the canonical evidence bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    for path in inputs.metric_paths():
        rows = [_normalise_row(row, path) for row in _read_csv(path)]
        if path == inputs.followup_metrics:
            _validate_rows(rows, path, require_complete_matrix=True)
        else:
            _validate_rows(rows, path, require_complete_matrix=False)
        all_rows.extend(rows)
    if inputs.protocols:
        _protocol_check(inputs.protocols)
    sources = [{"path": str(path), "sha256": _sha256(path)} for path in inputs.source_paths()]
    commit = _git_commit()
    row_commits = {str(row.get("commit") or row.get("git_commit")) for row in all_rows if row.get("commit") or row.get("git_commit")}
    if len(row_commits) > 1:
        raise ValueError(f"mixed commits in evidence rows: {sorted(row_commits)}")
    comparison = _comparison([row for row in all_rows if row.get("source_file") == str(inputs.followup_metrics)], inputs.reference_metrics)
    normalised_rows = sorted(all_rows, key=lambda row: (str(row["method"]), str(row["traversal"]), int(row["seed"]), str(row["profile"])))
    aggregates = _group_rows(normalised_rows)
    per_seed_path = output_dir / "method_metrics_per_seed.csv"
    aggregate_path = output_dir / "method_metrics_aggregate.csv"
    _write_csv(per_seed_path, normalised_rows)
    _write_csv(aggregate_path, aggregates)
    deltas_path = output_dir / "baseline_deltas.csv"
    _write_csv(deltas_path, comparison.get("deltas", []))
    table_paths = _write_tables(normalised_rows, output_dir)
    claim_checks = {
        "passed": True,
        "detector_accuracy_claim_permitted": False,
        "required_metrics": list(TABLE_METRICS),
        "notes": ["Raw map-frame APE and aligned APE are retained.", "In-row metrics are used for row identity; headland metrics are used for transition behaviour."],
    }
    (output_dir / "claim_checks.json").write_text(json.dumps(claim_checks, indent=2, sort_keys=True), encoding="utf-8")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "git": {"commit": commit},
        "configuration": "configs/icra/alpha_huber3_cap50.yaml",
        "seeds": list(EXPECTED_SEEDS),
        "traversals": list(EXPECTED_TRAVERSALS),
        "sources": sources,
        "inputs": {"followup_metrics": str(inputs.followup_metrics), "localization_metrics": str(inputs.localization_metrics) if inputs.localization_metrics else None, "rtab_metrics": str(inputs.rtab_metrics) if inputs.rtab_metrics else None, "protocols": [str(path) for path in inputs.protocols], "reference_metrics": str(inputs.reference_metrics) if inputs.reference_metrics else None},
        "row_count": len(normalised_rows),
        "aggregate_row_count": len(aggregates),
        "outputs": {"per_seed_csv": str(per_seed_path), "aggregate_csv": str(aggregate_path), "baseline_deltas_csv": str(deltas_path), "tables": table_paths, "claim_checks": str(output_dir / "claim_checks.json")},
        "comparison": comparison,
        "claim_checks": claim_checks,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _path(value: str | None) -> Path | None:
    return Path(value) if value else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Canonical SLPF per-seed CSV")
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--localization", type=Path, default=None)
    parser.add_argument("--rtab", type=Path, default=None)
    parser.add_argument("--protocol", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paper-output-dir", type=Path, default=None)
    args = parser.parse_args()
    manifest = build_bundle(EvidenceInputs(followup_metrics=args.baseline, localization_metrics=args.localization, rtab_metrics=args.rtab, protocols=tuple(args.protocol), reference_metrics=args.reference), args.output_dir)
    if args.paper_output_dir:
        args.paper_output_dir.mkdir(parents=True, exist_ok=True)
        for table_path in manifest["outputs"]["tables"].values():
            destination = args.paper_output_dir / Path(table_path).name
            destination.write_text(Path(table_path).read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Wrote evidence manifest to {args.output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
