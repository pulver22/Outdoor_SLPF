from __future__ import annotations

from scripts.check_paper_claims import CLAIMS, EVIDENCE, check_claims


def test_claim_checker_rejects_unbounded_deployment_claim() -> None:
    errors = check_claims("SLPF guarantees reliable deployment in vineyards.", CLAIMS, EVIDENCE)
    assert any("guarantees" in error for error in errors)


def test_claim_checker_accepts_bounded_row_level_claim() -> None:
    errors = check_claims(
        "SLPF improves row-level localisation under the evaluated map and GNSS assumptions.",
        CLAIMS,
        EVIDENCE,
    )
    assert errors == []

