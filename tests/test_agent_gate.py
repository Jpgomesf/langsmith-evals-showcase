"""Live regression gate for the agent scenario.

Excluded by default (needs LangSmith + provider credentials and a seeded dataset).
Run via ``make gate`` / ``pytest -m live`` after ``make seed``. Gates on mean
final-answer correctness and tool-selection.
"""

from __future__ import annotations

import pytest

from _gate_helpers import feedback_scores
from evals_showcase.scenarios.agent.experiment import run_experiment

CORRECTNESS_THRESHOLD = 0.80
TOOL_SELECTION_THRESHOLD = 0.80


@pytest.mark.live
def test_agent_gate():
    rows = run_experiment()
    correctness = feedback_scores(rows, "correctness")
    tool_selection = feedback_scores(rows, "tool_selection")
    assert correctness, "no 'correctness' feedback found — is the dataset seeded?"

    mean_correct = sum(correctness) / len(correctness)
    mean_tools = sum(tool_selection) / len(tool_selection)
    assert mean_correct >= CORRECTNESS_THRESHOLD, f"agent correctness {mean_correct:.2f} too low"
    assert mean_tools >= TOOL_SELECTION_THRESHOLD, f"agent tool_selection {mean_tools:.2f} too low"
