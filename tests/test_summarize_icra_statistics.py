from __future__ import annotations

import math

from scripts.summarize_icra_statistics import (
    build_summary,
    descriptive,
    exact_paired_sign_pvalue,
)


def test_descriptive_three_runs_has_nonzero_population_std() -> None:
    result = descriptive([1.0, 2.0, 4.0])
    assert result["n"] == 3
    assert result["mean"] == 7.0 / 3.0
    assert result["std_population"] > 0.0


def test_exact_sign_handles_zero_and_insufficient_pairs() -> None:
    assert math.isnan(exact_paired_sign_pvalue([0.0, 0.0]))
    assert exact_paired_sign_pvalue([1.0, 1.0, -1.0]) == 1.0


def test_dedicated_comparison_has_no_p_value() -> None:
    rows = [
        {"method": "AMCL", "traversal": "rh1", "profile": "full", "seed": 11, "replicate_kind": "dedicated_run", "ape_align_rmse": 1.0},
        {"method": "AMCL", "traversal": "rh1", "profile": "full", "seed": 22, "replicate_kind": "dedicated_run", "ape_align_rmse": 2.0},
        {"method": "AMCL", "traversal": "rh1", "profile": "full", "seed": 33, "replicate_kind": "dedicated_run", "ape_align_rmse": 3.0},
        {"method": "SLPF", "traversal": "rh1", "profile": "full", "seed": 11, "replicate_kind": "stochastic_rerun", "ape_align_rmse": 0.5},
        {"method": "SLPF", "traversal": "rh1", "profile": "full", "seed": 22, "replicate_kind": "stochastic_rerun", "ape_align_rmse": 0.6},
        {"method": "SLPF", "traversal": "rh1", "profile": "full", "seed": 33, "replicate_kind": "stochastic_rerun", "ape_align_rmse": 0.7},
    ]
    result = build_summary(rows, metrics=("ape_align_rmse",))
    assert result["paired_comparisons"] == []
    fixed = result["fixed_reference_deltas"]
    assert len(fixed) == 1
    assert "p_value" not in fixed[0]
    assert fixed[0]["evaluation_minus_reference"] < 0.0
