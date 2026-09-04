from __future__ import annotations

import pytest

from scripts.run_localization_followup_experiments import validate_experiment_seeds


def test_paper_experiments_require_exact_three_seeds() -> None:
    validate_experiment_seeds([11, 22, 33])

    with pytest.raises(ValueError, match="exactly seeds"):
        validate_experiment_seeds([11])
    with pytest.raises(ValueError, match="exactly seeds"):
        validate_experiment_seeds([11, 22, 44])
