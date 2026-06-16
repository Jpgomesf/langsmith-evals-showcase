"""Trajectory evaluators for agents — score the *process*, not just the answer.

Compares the sequence of tools an agent actually called against the expected
sequence, in one of three modes. Pure and unit-tested; the agent emits its tool
trajectory in the experiment output so these need no run-tree parsing (LangSmith
also exposes the run tree directly if you prefer to inspect it there).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

TrajectoryMode = Literal["exact", "in_order", "set"]
RowEvaluator = Callable[..., dict[str, Any]]


def is_subsequence(expected: Sequence[Any], actual: Sequence[Any]) -> bool:
    """True if ``expected`` appears in ``actual`` in order (gaps allowed)."""
    it = iter(actual)
    return all(item in it for item in expected)


def make_trajectory_match(
    *,
    mode: TrajectoryMode = "in_order",
    expected_field: str = "trajectory",
    actual_field: str = "trajectory",
    feedback_key: str = "trajectory_match",
) -> RowEvaluator:
    """Build a trajectory evaluator.

    - ``exact``: actual tool sequence equals expected.
    - ``in_order``: expected tools appear as an ordered subsequence of actual.
    - ``set``: every expected tool was used (order/extras ignored) — tool selection.
    """

    def trajectory(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> dict[str, Any]:
        actual = list(outputs.get(actual_field) or [])
        expected = list(reference_outputs.get(expected_field) or [])
        if mode == "exact":
            score = int(actual == expected)
        elif mode == "set":
            score = int(set(expected) <= set(actual))
        else:
            score = int(is_subsequence(expected, actual))
        return {"key": feedback_key, "score": score}

    return trajectory
