"""Live regression gate for the generate scenario.

Excluded by default (needs LangSmith + provider credentials, a seeded dataset,
and the prompts pushed to the Hub). Run via ``make gate`` / ``pytest -m live``
after ``make seed``. Gates on the v2 prompt's mean rubric ``overall`` score.
"""

from __future__ import annotations

import pytest

from _gate_helpers import feedback_scores
from evals_showcase.scenarios.generate.experiment import run_experiment

# Rubric scores are 1-5; require a solid mean overall on the stronger prompt.
OVERALL_THRESHOLD = 3.5


@pytest.mark.live
def test_generate_v2_quality_gate():
    result = run_experiment()
    scores = feedback_scores(result["v2"], "overall")
    assert scores, "no 'overall' rubric feedback found — is the dataset seeded?"
    mean = sum(scores) / len(scores)
    assert mean >= OVERALL_THRESHOLD, f"generate v2 overall {mean:.2f} < {OVERALL_THRESHOLD}"
