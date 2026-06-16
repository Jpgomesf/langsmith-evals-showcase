"""Custom code evaluators — programmatic checks beyond a single exact match.

The headline pattern here is *one evaluator emitting many feedback keys*: a field
extraction is graded per-field (with the right comparison per field type) plus an
aggregate, all returned as ``{"results": [...]}`` so each surfaces as its own
metric in LangSmith. The scoring math is pure and unit-tested without the API.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from . import normalize_label

FieldKind = Literal["exact", "text", "numeric"]


@dataclass(frozen=True)
class FieldSpec:
    """How one extracted field should be compared to its reference.

    - ``exact``: strict equality (dates, enums).
    - ``text``: case/whitespace-insensitive equality (free-text like merchant).
    - ``numeric``: equal within ``tolerance`` (amounts).
    """

    name: str
    kind: FieldKind = "exact"
    tolerance: float = 0.0


def compare_field(spec: FieldSpec, pred: Any, gold: Any) -> int:
    """Return 1 if the predicted value matches the reference under ``spec``, else 0."""
    if spec.kind == "numeric":
        try:
            return int(abs(float(pred) - float(gold)) <= spec.tolerance)
        except (TypeError, ValueError):
            return 0
    if spec.kind == "text":
        return int(normalize_label(pred) == normalize_label(gold))
    return int(pred == gold)


def score_fields(
    fields: list[FieldSpec],
    outputs: dict[str, Any] | None,
    reference_outputs: dict[str, Any] | None,
) -> dict[str, Any]:
    """Grade an extraction, returning a multi-key ``{"results": [...]}`` payload."""
    out = outputs or {}
    ref = reference_outputs or {}

    present = all(out.get(f.name) not in (None, "") for f in fields)
    results: list[dict[str, Any]] = [{"key": "all_fields_present", "score": int(present)}]

    matches = []
    for field in fields:
        match = compare_field(field, out.get(field.name), ref.get(field.name))
        matches.append(match)
        results.append({"key": f"{field.name}_match", "score": match})

    field_accuracy = sum(matches) / len(matches) if matches else 0.0
    results.append({"key": "field_accuracy", "score": field_accuracy})
    return {"results": results}


def make_field_extraction_evaluator(
    fields: list[FieldSpec],
) -> Callable[..., dict[str, Any]]:
    """Build a row-level evaluator that grades each field and the aggregate at once."""

    def field_extraction(
        outputs: dict[str, Any], reference_outputs: dict[str, Any]
    ) -> dict[str, Any]:
        return score_fields(fields, outputs, reference_outputs)

    return field_extraction
