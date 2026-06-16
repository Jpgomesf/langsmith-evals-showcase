"""Live regression gate for the classify scenario.

Excluded from the default test run (it needs LangSmith + provider credentials and
a seeded dataset). Run explicitly with ``make gate`` / ``pytest -m live`` in CI,
after ``make seed``. The gate fails the build if accuracy regresses below the bar.
"""

from __future__ import annotations

from typing import Any

import pytest

from evals_showcase.scenarios.classify.experiment import run_experiment

# Minimum acceptable macro accuracy on the classify dataset.
ACCURACY_THRESHOLD = 0.80


def _correct_scores(results: Any) -> list[float]:
    """Pull every row-level ``correct`` feedback score out of the experiment results."""
    scores: list[float] = []
    for row in results:
        for result in row["evaluation_results"]["results"]:
            key = result.get("key") if isinstance(result, dict) else getattr(result, "key", None)
            if key != "correct":
                continue
            score = (
                result.get("score")
                if isinstance(result, dict)
                else getattr(result, "score", None)
            )
            if score is not None:
                scores.append(float(score))
    return scores


@pytest.mark.live
def test_classify_accuracy_gate():
    results = run_experiment()
    scores = _correct_scores(results)
    assert scores, "no 'correct' feedback found — is the dataset seeded?"
    acc = sum(scores) / len(scores)
    assert acc >= ACCURACY_THRESHOLD, f"classify accuracy {acc:.2f} < {ACCURACY_THRESHOLD}"
