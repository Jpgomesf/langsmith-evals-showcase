"""Heuristic (rule-based) evaluators — cheap, deterministic, no LLM.

Best when a correct answer is exactly checkable. See ``docs/EVALUATION_STRATEGY.md``
for when to prefer these over an LLM-as-judge.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from . import normalize_label

RowEvaluator = Callable[..., dict[str, Any]]


def make_exact_match(field: str = "label", *, feedback_key: str = "correct") -> RowEvaluator:
    """Build an evaluator that scores 1 when ``outputs[field] == reference[field]``.

    Comparison is case/whitespace-insensitive via :func:`normalize_label`.
    """

    def exact_match(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> dict[str, Any]:
        pred = normalize_label(outputs.get(field))
        gold = normalize_label(reference_outputs.get(field))
        return {"key": feedback_key, "score": int(pred == gold)}

    exact_match.__name__ = f"exact_match_{field}"
    return exact_match
