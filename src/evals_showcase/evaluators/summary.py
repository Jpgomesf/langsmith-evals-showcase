"""Summary (dataset-level) evaluators for classification.

A summary evaluator sees every run+example at once, so it can compute aggregate
metrics — precision/recall/F1, accuracy — that a row-level evaluator cannot.
The metric math is kept as pure functions so it is unit-testable without the API.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from . import normalize_label

# A (prediction, gold) pair of normalized labels.
Pair = tuple[str, str]


def accuracy(pairs: list[Pair]) -> float:
    """Fraction of pairs where prediction equals gold."""
    if not pairs:
        return 0.0
    return sum(1 for pred, gold in pairs if pred == gold) / len(pairs)


def macro_precision_recall_f1(pairs: list[Pair]) -> tuple[float, float, float]:
    """Macro-averaged precision, recall and F1 over the union of observed labels.

    Averaging over predicted ∪ gold labels means a hallucinated class (predicted
    but never correct) is still penalized through its precision term.
    """
    if not pairs:
        return 0.0, 0.0, 0.0
    labels = sorted({gold for _, gold in pairs} | {pred for pred, _ in pairs})
    precisions, recalls, f1s = [], [], []
    for label in labels:
        tp = sum(1 for pred, gold in pairs if pred == label and gold == label)
        fp = sum(1 for pred, gold in pairs if pred == label and gold != label)
        fn = sum(1 for pred, gold in pairs if pred != label and gold == label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    n = len(labels)
    return sum(precisions) / n, sum(recalls) / n, sum(f1s) / n


def make_classification_summary(field: str = "label") -> Callable[..., dict[str, Any]]:
    """Build a summary evaluator emitting accuracy + macro precision/recall/F1."""

    def classification_summary(runs: list[Any], examples: list[Any]) -> dict[str, Any]:
        pairs: list[Pair] = [
            (
                normalize_label((run.outputs or {}).get(field)),
                normalize_label((example.outputs or {}).get(field)),
            )
            for run, example in zip(runs, examples, strict=False)
        ]
        precision, recall, f1 = macro_precision_recall_f1(pairs)
        return {
            "results": [
                {"key": "accuracy", "score": accuracy(pairs)},
                {"key": "macro_precision", "score": precision},
                {"key": "macro_recall", "score": recall},
                {"key": "macro_f1", "score": f1},
            ]
        }

    return classification_summary
