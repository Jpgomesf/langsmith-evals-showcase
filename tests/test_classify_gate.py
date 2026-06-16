"""Live regression gate for the classify scenario.

Excluded from the default test run (it needs LangSmith + provider credentials and
a seeded dataset). Run explicitly with ``make gate`` / ``pytest -m live`` in CI,
after ``make seed``. The gate fails the build if accuracy regresses below the bar.
"""

from __future__ import annotations

import pytest

from _gate_helpers import feedback_scores
from evals_showcase.scenarios.classify.experiment import run_experiment

# Minimum acceptable accuracy on the classify dataset.
ACCURACY_THRESHOLD = 0.80


@pytest.mark.live
def test_classify_accuracy_gate():
    scores = feedback_scores(run_experiment(), "correct")
    assert scores, "no 'correct' feedback found — is the dataset seeded?"
    acc = sum(scores) / len(scores)
    assert acc >= ACCURACY_THRESHOLD, f"classify accuracy {acc:.2f} < {ACCURACY_THRESHOLD}"
