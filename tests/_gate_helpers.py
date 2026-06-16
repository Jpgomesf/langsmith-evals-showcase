"""Shared helpers for the live regression gates."""

from __future__ import annotations

from typing import Any


def feedback_scores(results: Any, key: str) -> list[float]:
    """Collect every feedback score with ``key`` from LangSmith experiment results."""
    scores: list[float] = []
    for row in results:
        for result in row["evaluation_results"]["results"]:
            k = result.get("key") if isinstance(result, dict) else getattr(result, "key", None)
            if k != key:
                continue
            s = result.get("score") if isinstance(result, dict) else getattr(result, "score", None)
            if s is not None:
                scores.append(float(s))
    return scores
