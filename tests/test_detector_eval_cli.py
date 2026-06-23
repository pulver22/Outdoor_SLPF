from __future__ import annotations

from pathlib import Path

import pytest

from scripts.evaluate_detector_semanticblt import build_conf_thresholds, parse_args, validate_inputs


def test_detector_eval_cli_rejects_missing_external_data_yaml(tmp_path: Path):
    args = parse_args(
        [
            "--model",
            str(tmp_path / "model.pt"),
            "--data-yaml",
            str(tmp_path / "missing.yaml"),
            "--output-root",
            str(tmp_path / "out"),
        ]
    )

    with pytest.raises(FileNotFoundError, match="data YAML"):
        validate_inputs(args)


def test_detector_eval_cli_accepts_existing_paths_and_sorts_thresholds(tmp_path: Path):
    model = tmp_path / "model.pt"
    data_yaml = tmp_path / "semanticblt.yaml"
    model.write_bytes(b"placeholder")
    data_yaml.write_text("path: .\nval: images\nnames: [pole, trunk]\n", encoding="utf-8")
    args = parse_args(
        [
            "--model",
            str(model),
            "--data-yaml",
            str(data_yaml),
            "--conf-thresholds",
            "0.5,0.1,0.25",
            "--output-root",
            str(tmp_path / "out"),
        ]
    )

    validate_inputs(args)

    assert build_conf_thresholds(args.conf_thresholds) == [0.1, 0.25, 0.5]
