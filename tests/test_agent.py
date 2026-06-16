"""Deterministic unit tests for the agent tools and trajectory evaluators."""

from __future__ import annotations

import pytest

from evals_showcase.evaluators.trajectory import is_subsequence, make_trajectory_match
from evals_showcase.scenarios.agent.app import calculate


def test_calculate_basic_arithmetic():
    assert calculate("47 * 13") == "611"
    assert calculate("256 / 8") == "32"  # integral result formatted as int
    assert calculate("9 ** 3") == "729"
    assert calculate("100 - 37") == "63"


def test_calculate_rejects_non_arithmetic():
    with pytest.raises((ValueError, KeyError, SyntaxError)):
        calculate("__import__('os').system('echo hi')")


def test_calculate_zero_division_is_friendly():
    assert calculate("1 / 0") == "undefined (division by zero)"


def test_calculate_rejects_huge_exponent():
    # Guards against huge-integer DoS like 10**10**10.
    with pytest.raises(ValueError):
        calculate("10 ** 999")


def test_is_subsequence():
    assert is_subsequence(["a", "b"], ["a", "x", "b"]) is True
    assert is_subsequence(["b", "a"], ["a", "x", "b"]) is False


def test_trajectory_exact():
    ev = make_trajectory_match(mode="exact")
    assert ev({"trajectory": ["calculator"]}, {"trajectory": ["calculator"]})["score"] == 1
    assert (
        ev({"trajectory": ["calculator", "word_count"]}, {"trajectory": ["calculator"]})["score"]
        == 0
    )


def test_trajectory_in_order_allows_extra_tools():
    ev = make_trajectory_match(mode="in_order")
    out = ev({"trajectory": ["calculator", "word_count"]}, {"trajectory": ["calculator"]})
    assert out["score"] == 1


def test_trajectory_set_is_order_agnostic():
    ev = make_trajectory_match(mode="set", feedback_key="tool_selection")
    out = ev(
        {"trajectory": ["word_count", "calculator"]}, {"trajectory": ["calculator", "word_count"]}
    )
    assert out == {"key": "tool_selection", "score": 1}
