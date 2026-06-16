"""Live regression gate for the extract scenario.

Excluded by default (needs LangSmith + provider credentials and a seeded dataset).
Run via ``make gate`` / ``pytest -m live`` after ``make seed``. Fails the build if
mean field-extraction accuracy regresses below the bar.
"""

from __future__ import annotations

import pytest

from _gate_helpers import feedback_scores
from evals_showcase.scenarios.extract.experiment import run_experiment

# Minimum acceptable mean field_accuracy across the dataset.
FIELD_ACCURACY_THRESHOLD = 0.85


@pytest.mark.live
def test_extract_field_accuracy_gate():
    scores = feedback_scores(run_experiment(), "field_accuracy")
    assert scores, "no 'field_accuracy' feedback found — is the dataset seeded?"
    mean = sum(scores) / len(scores)
    assert mean >= FIELD_ACCURACY_THRESHOLD, (
        f"extract field_accuracy {mean:.2f} < {FIELD_ACCURACY_THRESHOLD}"
    )
