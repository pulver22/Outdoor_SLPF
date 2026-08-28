from __future__ import annotations

from pathlib import Path

from outdoor_slpf.config import load_yaml_overlay
from scripts import run_localization_followup_experiments as followup


def test_icra_baseline_config_is_complete_and_frozen() -> None:
    values = load_yaml_overlay("configs/icra/alpha_huber3_cap50.yaml")

    assert values["pose_backend"] == "alpha"
    assert values["gnss_robust_mode"] == "huber"
    assert values["gnss_outlier_threshold"] == 3.0
    assert values["semantic_penalty_cap"] == 50
    assert values["row_likelihood_mode"] == "single"
    assert values["delayed_row_correction"] is False
    assert values["gnss_row_gating"] is False


def test_icra_baseline_candidate_uses_only_the_config_overlay() -> None:
    config_path = Path("configs/icra/alpha_huber3_cap50.yaml")
    candidate = followup.icra_baseline_candidate(config_path)

    assert candidate["id"] == "baseline_alpha_huber3_cap50"
    assert candidate["args"] == ["--config-yaml", str(config_path)]
