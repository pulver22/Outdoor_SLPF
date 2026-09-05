from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.build_icra_evidence_bundle import EvidenceInputs, _tex_gnss_stress_table, build_bundle


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def make_rows(*, traversals: list[str], seeds: list[int]) -> list[dict[str, object]]:
    return [
        {
            "method": "SLPF",
            "method_key": "slpf",
            "traversal": traversal,
            "seed": seed,
            "profile": "full",
            "ape_raw_rmse": 1.0,
            "ape_align_rmse": 0.9,
            "rpe_2m_align_rmse": 0.2,
            "rpe_5m_align_rmse": 0.3,
            "rpe_10m_align_rmse": 0.4,
            "inrow_cross_track_mean": 1.0,
            "inrow_row_correct_fraction": 0.8,
            "inrow_wrong_row_duration_sec": 10.0,
            "headland_cross_track_mean": 1.5,
            "headland_mean_recovery_distance_m": 4.0,
            "commit": "c15803d",
        }
        for traversal in traversals
        for seed in seeds
    ]


def write_inputs(tmp_path: Path, rows: list[dict[str, object]] | None = None) -> EvidenceInputs:
    rows = rows or make_rows(traversals=["rh_run1", "rh_run2"], seeds=[11, 22, 33])
    baseline = _write_csv(tmp_path / "baseline.csv", rows)
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "commit": "c15803d",
                "commands": ["python scripts/spf_lidar.py --config-yaml configs/icra/alpha_huber3_cap50.yaml"],
            }
        ),
        encoding="utf-8",
    )
    return EvidenceInputs(followup_metrics=baseline, protocols=(protocol,))


def test_bundle_rejects_missing_seed(tmp_path: Path) -> None:
    rows = make_rows(traversals=["rh_run1", "rh_run2"], seeds=[11, 22])

    with pytest.raises(ValueError, match="missing.*33"):
        build_bundle(write_inputs(tmp_path, rows), tmp_path / "out")


def test_bundle_records_exact_source_hashes(tmp_path: Path) -> None:
    data = build_bundle(write_inputs(tmp_path), tmp_path / "out")

    assert data["git"]["commit"]
    assert all(item["sha256"] for item in data["sources"])
    assert (tmp_path / "out/manifest.json").exists()
    assert (tmp_path / "out/method_metrics_per_seed.csv").exists()
    assert (tmp_path / "out/method_metrics_aggregate.csv").exists()
    assert (tmp_path / "out/claim_checks.json").exists()
    assert (tmp_path / "out/icra_main_table.tex").exists()
    assert (tmp_path / "out/icra_gnss_stress_table.tex").exists()
    assert (tmp_path / "out/icra_operational_table.tex").exists()
    assert "Headland recovery" in (tmp_path / "out/icra_operational_table.tex").read_text(encoding="utf-8")


def test_gnss_stress_table_highlights_metric_winners_independently(tmp_path: Path) -> None:
    stress_csv = _write_csv(tmp_path / "stress.csv", [{"placeholder": "1"}])
    _write_csv(
        tmp_path / "compact_summary.csv",
        [
            {
                "profile": "nominal",
                "method": "AMCL",
                "ape_align_rmse_across_traversals_mean": 0.50,
                "row_correct_fraction_across_traversals_mean": 0.70,
                "failure_rate_across_traversals_mean": 0.20,
            },
            {
                "profile": "nominal",
                "method": "SLPF",
                "ape_align_rmse_across_traversals_mean": 0.90,
                "row_correct_fraction_across_traversals_mean": 0.80,
                "failure_rate_across_traversals_mean": 0.30,
            },
        ],
    )

    output = tmp_path / "stress_table.tex"
    _tex_gnss_stress_table(output, stress_csv)
    rendered = output.read_text(encoding="utf-8")

    assert "Nominal & AMCL & $\\mathbf{0.50}$ & $0.70$ & $\\mathbf{0.20}$ \\\\" in rendered
    assert "Nominal & \\textbf{SLPF (ours)} & $0.90$ & $\\mathbf{0.80}$ & $0.30$ \\\\" in rendered


def test_bundle_rejects_unexpected_seed_in_baseline(tmp_path: Path) -> None:
    inputs = write_inputs(tmp_path)
    extra = make_rows(traversals=["rh_run1"], seeds=[44])
    baseline = _write_csv(tmp_path / "extra.csv", extra)
    inputs = EvidenceInputs(followup_metrics=inputs.followup_metrics, baseline_metrics=(baseline,), protocols=inputs.protocols)

    with pytest.raises(ValueError, match="unexpected seed 44"):
        build_bundle(inputs, tmp_path / "out")


def test_bundle_rejects_reused_dedicated_output_hash(tmp_path: Path) -> None:
    inputs = write_inputs(tmp_path)
    rows = []
    for seed in [11, 22, 33]:
        row = make_rows(traversals=["rh_run1"], seeds=[seed])[0]
        row.update(
            {
                "method": "AMCL",
                "replicate_kind": "dedicated_run",
                "source_output_sha256": "same-output",
            }
        )
        rows.append(row)
    baseline = _write_csv(tmp_path / "dedicated.csv", rows)
    inputs = EvidenceInputs(followup_metrics=inputs.followup_metrics, baseline_metrics=(baseline,), protocols=inputs.protocols)

    with pytest.raises(ValueError, match="reuses an output hash"):
        build_bundle(inputs, tmp_path / "out")
