"""Reusable evaluator library.

Evaluators follow the LangSmith contract: a *row-level* evaluator receives any
subset of ``inputs, outputs, reference_outputs, run, example`` and returns a
dict ``{"key", "score"|"value", "comment"?}``; a *summary* evaluator receives
``runs, examples`` and may return ``{"results": [ ... ]}`` to emit several
dataset-level metrics at once.
"""

from __future__ import annotations


def normalize_label(value: object) -> str:
    """Lowercase and strip a label so comparisons ignore casing/whitespace."""
    return str(value if value is not None else "").strip().lower()
