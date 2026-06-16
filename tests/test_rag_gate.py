"""Live regression gate for the rag scenario.

Excluded by default (needs LangSmith + provider credentials and a seeded dataset).
Run via ``make gate`` / ``pytest -m live`` after ``make seed``. Gates on mean
answer-correctness from the LLM-judge.
"""

from __future__ import annotations

import pytest

from _gate_helpers import feedback_scores
from evals_showcase.scenarios.rag.experiment import run_experiment

# Minimum acceptable mean correctness across the dataset.
CORRECTNESS_THRESHOLD = 0.80


@pytest.mark.live
def test_rag_correctness_gate():
    scores = feedback_scores(run_experiment(), "correctness")
    assert scores, "no 'correctness' feedback found — is the dataset seeded?"
    mean = sum(scores) / len(scores)
    assert mean >= CORRECTNESS_THRESHOLD, f"rag correctness {mean:.2f} < {CORRECTNESS_THRESHOLD}"
